# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file exceam in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "accelerate>=0.12.0",
#     "torch>=1.5.0",
#     "torchvision>=0.6.0",
#     "datasets>=2.14.0",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

import logging
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoImageProcessor,
    HfArgumentParser,
    TimmWrapperImageProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_parameter_names
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.versions import require_version

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


def subset_dataset(dataset_split, max_samples: Optional[int], seed: int, output_dir: str, split_name: str, shuffle_before_subsample: bool = True):
    """Create a small subset for debug runs.

    When using datasets loaded from disk, `.shuffle()` may try to write temporary index files next to the
    dataset shards. If the dataset directory is read-only, we redirect the shuffle index cache into `output_dir`.
    If that still fails, we gracefully fall back to taking the first `max_samples` examples without shuffling.
    """
    if max_samples is None:
        return dataset_split

    max_samples = min(max_samples, len(dataset_split))
    if max_samples <= 0:
        return dataset_split.select([])

    if not shuffle_before_subsample:
        logger.warning(f"Subsampling {split_name} without shuffling: taking first {max_samples} samples.")
        return dataset_split.select(range(max_samples))

    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, f"{split_name}_shuffle_indices_{max_samples}_{seed}.arrow")
    try:
        return dataset_split.shuffle(seed=seed, indices_cache_file_name=cache_file).select(range(max_samples))
    except (PermissionError, OSError) as e:
        logger.warning(
            f"Shuffle-based subsampling for {split_name} failed ({e}). "
            f"Falling back to first {max_samples} samples without shuffling."
        )
        return dataset_split.select(range(max_samples))


class EnhancedTrainer(Trainer):
    def __init__(
        self,
        *args,
        embed_lr=None,
        ema_decay=0.9999,
        use_ema=True,
        raw_loss_log_steps=20,
        raw_loss_log_file=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embed_lr = embed_lr
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema_model = None
        self.raw_loss_log_steps = max(0, int(raw_loss_log_steps or 0))
        self.raw_loss_log_file = raw_loss_log_file
        self._raw_loss_last_logged_step = None
        self._raw_loss_log_announced = False

    @staticmethod
    def _rank():
        try:
            return int(os.environ.get("RANK", "0"))
        except Exception:
            return 0

    def _maybe_log_raw_loss(self, loss):
        if self.raw_loss_log_steps <= 0 or self._rank() != 0:
            return
        if not torch.is_tensor(loss):
            return

        global_step = int(getattr(self.state, "global_step", 0) or 0)
        if self._raw_loss_last_logged_step == global_step:
            return
        if self._raw_loss_last_logged_step is not None and global_step % self.raw_loss_log_steps != 0:
            return

        loss_float = loss.detach().float()
        scalar_loss = loss_float.mean() if loss_float.dim() != 0 else loss_float
        finite_all = torch.isfinite(loss_float).all().item()
        scalar_finite = torch.isfinite(scalar_loss).item()
        scalar_value = scalar_loss.item()

        output_path = self.raw_loss_log_file
        if output_path is None:
            output_path = os.path.join(self.args.output_dir, "raw_outputs_loss.jsonl")
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)

        if not self._raw_loss_log_announced:
            print(f"[RAW-LOSS] rank0 logging raw outputs.loss to {output_path}", flush=True)
            self._raw_loss_log_announced = True

        record = {
            "trainer_global_step_before_update": global_step,
            "epoch": getattr(self.state, "epoch", None),
            "raw_outputs_loss": scalar_value if scalar_finite else None,
            "raw_outputs_loss_repr": str(scalar_value),
            "scalar_finite": bool(scalar_finite),
            "finite_all": bool(finite_all),
            "loss_shape": list(loss.shape),
            "loss_dtype": str(loss.dtype),
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "logging_note": "raw model outputs.loss before Trainer gradient-accumulation scaling/aggregation",
        }
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        self._raw_loss_last_logged_step = global_step

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        def _is_debug_enabled():
            return os.environ.get("NEPA_NAN_DEBUG", "0") == "1"

        def _rank():
            return self._rank()

        def _tensor_stat(name, x):
            if not torch.is_tensor(x):
                print(f"[NAN-DEBUG][rank={_rank()}] {name}: type={type(x)}")
                return

            msg = (
                f"[NAN-DEBUG][rank={_rank()}] {name}: "
                f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
            )

            if torch.is_floating_point(x):
                finite = torch.isfinite(x)
                msg += (
                    f", finite_all={finite.all().item()}"
                    f", nan_any={torch.isnan(x).any().item()}"
                    f", inf_any={torch.isinf(x).any().item()}"
                )
                if x.numel() > 0 and finite.any().item():
                    xf = x.detach().float()
                    msg += (
                        f", min={xf[finite].min().item():.6g}"
                        f", max={xf[finite].max().item():.6g}"
                        f", mean={xf[finite].mean().item():.6g}"
                    )
            else:
                msg += f", min={x.min().item() if x.numel() else 'NA'}, max={x.max().item() if x.numel() else 'NA'}"

            print(msg, flush=True)

        def _walk_outputs(prefix, obj, depth=0):
            if depth > 3:
                return
            if torch.is_tensor(obj):
                _tensor_stat(prefix, obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _walk_outputs(f"{prefix}.{k}", v, depth + 1)
            elif hasattr(obj, "items"):
                try:
                    for k, v in obj.items():
                        _walk_outputs(f"{prefix}.{k}", v, depth + 1)
                except Exception:
                    pass
            elif isinstance(obj, (tuple, list)):
                for i, v in enumerate(obj):
                    _walk_outputs(f"{prefix}[{i}]", v, depth + 1)
            else:
                # ModelOutput may expose keys through to_tuple / to_dict
                if hasattr(obj, "to_tuple"):
                    try:
                        for i, v in enumerate(obj.to_tuple()):
                            _walk_outputs(f"{prefix}.to_tuple[{i}]", v, depth + 1)
                    except Exception:
                        pass

        debug = _is_debug_enabled()

        if debug:
            print(f"[NAN-DEBUG][rank={_rank()}] compute_loss enter: trainer_global_step={self.state.global_step}", flush=True)
            for k, v in inputs.items():
                _tensor_stat(f"inputs[{k}]", v)

        outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs.get("loss", None)
        else:
            loss = outputs[0] if isinstance(outputs, (tuple, list)) else getattr(outputs, "loss", None)

        if loss is None:
            raise ValueError("Model did not return a loss. Cannot train NEPA pretraining model.")

        self._maybe_log_raw_loss(loss)

        if debug:
            _walk_outputs("outputs", outputs)
            _tensor_stat("loss_before_scalar_fix", loss)

        # DeepSpeed requires a 0-dim scalar loss tensor for backward().
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Expected `loss` to be a torch.Tensor, got {type(loss)!r}.")
        if loss.dim() != 0:
            loss = loss.mean()
        if loss.dim() != 0:
            loss = loss.reshape(())

        if debug:
            _tensor_stat("loss_after_scalar_fix", loss)

        if torch.is_tensor(loss) and torch.is_floating_point(loss) and not torch.isfinite(loss).all():
            print(f"[NAN-DEBUG][rank={_rank()}] Non-finite loss detected. Raising RuntimeError to stop.", flush=True)
            raise RuntimeError("NEPA loss became NaN/Inf. See [NAN-DEBUG] logs above.")

        return (loss, outputs) if return_outputs else loss

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"layer_scale", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if self.embed_lr is None:
            return super().create_optimizer()

        decay_params = set(self.get_decay_parameter_names(self.model))
        backbone_prefix = getattr(self.model, "base_model_prefix", None)
        backbone = getattr(self.model, backbone_prefix, None) if isinstance(backbone_prefix, str) else None
        if backbone is None and hasattr(self.model, "vit_nepa"):
            backbone_prefix = "vit_nepa"
            backbone = self.model.vit_nepa
        if backbone is None or backbone_prefix is None:
            raise ValueError(
                "Could not resolve backbone module for embed_lr grouping. "
                f"Got base_model_prefix={getattr(self.model, 'base_model_prefix', None)!r}."
            )

        embed_params = set(f"{backbone_prefix}.embeddings.{n}" for n, _ in backbone.embeddings.named_parameters())

        wd = self.args.weight_decay
        base_lr = self.args.learning_rate

        groups = [
            {"params": [], "weight_decay": wd,  "lr": self.embed_lr},
            {"params": [], "weight_decay": 0.0, "lr": self.embed_lr},
            {"params": [], "weight_decay": wd,  "lr": base_lr},
            {"params": [], "weight_decay": 0.0, "lr": base_lr},
        ]

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_decay = name in decay_params
            is_embed = name in embed_params

            if is_embed and is_decay:
                groups[0]["params"].append(p)
            elif is_embed and not is_decay:
                groups[1]["params"].append(p)
            elif (not is_embed) and is_decay:
                groups[2]["params"].append(p)
            else:
                groups[3]["params"].append(p)

        optimizer_grouped_parameters = [g for g in groups if g["params"]]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _init_ema_model(self):
        if self.ema_model is None:
            import copy
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            self.ema_model = self.ema_model.float()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def _update_ema(self):
        if not self.use_ema:
            return
        if self.ema_model is None:
            self._init_ema_model()
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    model_param = msd[k].float()
                    v.mul_(self.ema_decay).add_(model_param, alpha=1.0 - self.ema_decay)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.state.global_step > getattr(self, "_ema_global_step", 0):
            self._update_ema()
            self._ema_global_step = self.state.global_step
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):
        out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

        if self.use_ema and self.ema_model is not None:
            backup = self.model
            self.model = self.ema_model
            ema_out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix + "_ema", **gen_kwargs)
            self.model = backup

            out.update(ema_out)
        return out

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test", **gen_kwargs):
        out = super().predict(test_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

        if self.use_ema and self.ema_model is not None:
            backup = self.model
            self.model = self.ema_model
            ema_out = super().predict(test_dataset, ignore_keys, metric_key_prefix + "_ema", **gen_kwargs)
            self.model = backup

            from transformers.trainer_utils import PredictionOutput
            if isinstance(out, PredictionOutput) and isinstance(ema_out, PredictionOutput):
                out.metrics.update({k: v for k, v in ema_out.metrics.items()})
            else:
                out.update(ema_out)
        return out

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.use_ema and self.ema_model is not None and self.args.should_save:
            ema_path = f"{output_dir}/pytorch_model_ema.bin"
            os.makedirs(os.path.dirname(ema_path), exist_ok=True)
            torch.save(self.ema_model.state_dict(), ema_path)
            self.log({"ema_model_saved": ema_path})

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)

        # Debug RWKV non-persistent runtime state after checkpoint load.
        base_model = self.model
        if hasattr(base_model, "module"):
            base_model = base_model.module

        rwkv_core = None
        try:
            rwkv_core = base_model.rwkv_nepa.rwkv
        except Exception:
            rwkv_core = None

        if rwkv_core is not None:
            print(
                "[RWKV-RESUME] after super load: "
                f"training={rwkv_core.training}, "
                f"layers_are_rescaled={getattr(rwkv_core, 'layers_are_rescaled', 'NA')}"
            )

            # HF RWKV uses a non-persistent layers_are_rescaled flag.
            # If the model is in train mode but still marked as rescaled,
            # call train(True) once to force the internal train-state transition.
            if getattr(rwkv_core, "layers_are_rescaled", False):
                print("[RWKV-RESUME] layers_are_rescaled=True; calling rwkv_core.train(True) to restore train-time scaling.")
                rwkv_core.train(True)
                print(
                    "[RWKV-RESUME] after train(True): "
                    f"training={rwkv_core.training}, "
                    f"layers_are_rescaled={getattr(rwkv_core, 'layers_are_rescaled', 'NA')}"
                )

        if self.use_ema:
            ema_ckpt = os.path.join(resume_from_checkpoint, "pytorch_model_ema.bin")
            if os.path.exists(ema_ckpt):
                if self.ema_model is None:
                    import copy
                    self.ema_model = copy.deepcopy(self.model)

                self.ema_model.eval()
                self.ema_model = self.ema_model.float()
                for p in self.ema_model.parameters():
                    p.requires_grad_(False)

                state_dict = torch.load(ema_ckpt, map_location="cpu")
                incompatible = self.ema_model.load_state_dict(state_dict, strict=True)
                print(f"[EMA] Loaded EMA checkpoint from {ema_ckpt}")
                print(f"[EMA] load_state_dict result: {incompatible}")

                # Prevent an accidental EMA update before the first resumed optimizer step.
                # TrainerState may not be restored yet inside _load_from_checkpoint,
                # so read global_step directly from checkpoint/trainer_state.json.
                trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
                if os.path.exists(trainer_state_path):
                    import json
                    with open(trainer_state_path, "r") as f:
                        trainer_state = json.load(f)
                    self._ema_global_step = int(trainer_state.get("global_step", 0))
                else:
                    self._ema_global_step = int(getattr(self.state, "global_step", 0))
                print(f"[EMA] _ema_global_step restored to {self._ema_global_step}")
            else:
                raise FileNotFoundError(f"[EMA] Expected EMA checkpoint not found: {ema_ckpt}")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    # label_column_name: str = field(
    #     default="label",
    #     metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    # )
    load_from_disk: bool = field(default=False, metadata={"help": "Load from disk"})
    keep_in_memory: bool = field(default=False, metadata={"help": "keep_in_memory"})
    resize_size: int = field(default=256, metadata={"help": "Resize shorter side for validation when needed."})
    shuffle_before_subsample: bool = field(default=True, metadata={"help": "Whether to shuffle before applying max_train_samples/max_eval_samples."})

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    embed_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for embedding layer parameters."},
    )
    trainable: Optional[str] = field(
        default=None,
        metadata={"help": "Freeze/unfreeze parameters via model._set_trainable (timemix|all|none)."},
    )
    raw_loss_log_steps: int = field(
        default=20,
        metadata={"help": "Log raw model outputs.loss every N optimizer steps on rank0. Set 0 to disable."},
    )
    raw_loss_log_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSONL path for raw outputs.loss monitor. Defaults to output_dir/raw_outputs_loss.jsonl."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        if data_args.load_from_disk:
            dataset = load_from_disk(data_args.dataset_name, keep_in_memory=data_args.keep_in_memory)
        else:
            dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if data_args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {data_args.image_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    # if data_args.label_column_name not in dataset_column_names:
    #     raise ValueError(
    #         f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
    #         "Make sure to set `--label_column_name` to the correct text column - one of "
    #         f"{', '.join(dataset_column_names)}."
    #     )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        return {"pixel_values": pixel_values}
        # labels = torch.tensor([example[data_args.label_column_name] for example in examples])
        # return {"pixel_values": pixel_values, "labels": labels}

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    # labels = dataset["train"].features[data_args.label_column_name].names
    # label2id, id2label = {}, {}
    # for i, label in enumerate(labels):
    #     label2id[label] = str(i)
    #     id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    # metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    # def compute_metrics(p):
    #     """Computes accuracy on a batch of predictions"""
    #     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config_path = model_args.config_name or model_args.model_name_or_path
    if config_path is None:
        raise ValueError("You must provide --config_name (train from scratch) or --model_name_or_path (resume).")

    nepa_model_type = model_args.model_type
    if nepa_model_type is None:
        cfg_dict, _ = PretrainedConfig.get_config_dict(
            config_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        nepa_model_type = cfg_dict.get("model_type")

    if nepa_model_type == "rwkv_nepa":
        from models.rwkv_nepa import RwkvNepaConfig as NepaConfig
        from models.rwkv_nepa import RwkvNepaForPreTraining as NepaForPreTraining
    elif nepa_model_type == "vit_nepa":
        from models.vit_nepa import ViTNepaConfig as NepaConfig
        from models.vit_nepa import ViTNepaForPreTraining as NepaForPreTraining
    else:
        raise ValueError(
            f"Unsupported NEPA model_type={nepa_model_type!r}. Expected 'vit_nepa' or 'rwkv_nepa'. "
            "Set it in config.json or pass --model_type."
        )

    config = NepaConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        # num_labels=len(labels),
        # label2id=label2id,
        # id2label=id2label,
        # finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if model_args.model_name_or_path:
        model = NepaForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        logger.info("Training new model from scratch")
        model = NepaForPreTraining(config)

    if model_args.trainable is not None:
        if not hasattr(model, "_set_trainable"):
            raise ValueError(f"--trainable is set but model {type(model)} has no _set_trainable() method.")
        model._set_trainable(model_args.trainable)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable}/{total} ({trainable/total:.2%})")
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )

    # Define torchvision transforms to be applied to each image.
    if isinstance(image_processor, TimmWrapperImageProcessor):
        # Pretraining: only random crop + random flip + normalize
        _train_transforms = Compose([
            RandomResizedCrop(image_processor.input_size, interpolation=3),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.mean, std=image_processor.std),
        ])
        # Validation: resize + center crop + normalize
        _val_transforms = Compose([
            Resize(data_args.resize_size, interpolation=3),
            CenterCrop(image_processor.input_size),
            ToTensor(),
            Normalize(mean=image_processor.mean, std=image_processor.std),
        ])
    else:
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])

        # Create normalization transform
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
            normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        else:
            normalize = Lambda(lambda x: x)
        _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = subset_dataset(
                dataset["train"],
                data_args.max_train_samples,
                training_args.seed,
                training_args.output_dir,
                "train",
                data_args.shuffle_before_subsample,
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = subset_dataset(
                dataset["validation"],
                data_args.max_eval_samples,
                training_args.seed,
                training_args.output_dir,
                "validation",
                data_args.shuffle_before_subsample,
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    # Initialize our trainer
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        # compute_metrics=compute_metrics,
        processing_class=image_processor,
        data_collator=collate_fn,
        embed_lr=model_args.embed_lr,
        raw_loss_log_steps=model_args.raw_loss_log_steps,
        raw_loss_log_file=model_args.raw_loss_log_file,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate()
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "embedded-prediction",
        "dataset": data_args.dataset_name,
        "tags": ["embedded-prediction", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
