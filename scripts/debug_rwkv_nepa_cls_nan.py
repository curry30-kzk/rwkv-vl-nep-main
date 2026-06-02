#!/usr/bin/env python
"""Single-GPU RWKV-NEPA classification NaN gradient probe.

This script intentionally avoids Trainer/DDP. It loads one fixed batch, runs a
few classification backward passes, and reports the first non-finite activation
or gradient it can observe.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any


DEFAULT_PROJECT_DIR = "/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main"
DEFAULT_MODEL_DIR = (
    "/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main/"
    "outputs_init/rwkv_nepa_full_ep100_cls_init_from_model_seed42"
)
DEFAULT_DATASET = "/public/home/ssjxkzk/rwkv-vl-nep-main/data/imagenet-1k-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--image_processor_name", default=None)
    parser.add_argument("--load_from_disk", action="store_true", default=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer used by the probe.",
    )
    parser.add_argument(
        "--train_mode",
        choices=["all", "head_only", "backbone_no_special"],
        default="all",
        help="Which parameters should require gradients.",
    )
    parser.add_argument(
        "--force_torch_wkv",
        action="store_true",
        help="Monkey-patch HF RWKV to use the differentiable torch fallback instead of the custom CUDA WKV kernel.",
    )
    parser.add_argument(
        "--disable_rescale_every",
        action="store_true",
        help="Set config.rescale_every=0 before loading the model.",
    )
    parser.add_argument(
        "--ensure_train_rescale",
        action="store_true",
        help="Force the RWKV core back to train-time scaling before the probe.",
    )
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use a fixed synthetic image batch instead of the dataset.")
    parser.add_argument("--num_labels", type=int, default=None, help="Only used with --synthetic if config.num_labels is absent.")
    parser.add_argument("--no_step", action="store_true", help="Run backward but skip optimizer.step().")
    parser.add_argument("--max_report_params", type=int, default=20)
    return parser.parse_args()


def setup_imports(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    sys.path.insert(0, str(project_dir))
    os.chdir(project_dir)
    if args.force_torch_wkv:
        os.environ["RWKV_NEPA_FORCE_TORCH_WKV"] = "1"


def iter_tensors(value: Any):
    if value is None:
        return
    if isinstance(value, tuple) or isinstance(value, list):
        for item in value:
            yield from iter_tensors(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
        return
    try:
        import torch

        if torch.is_tensor(value):
            yield value
    except Exception:
        return


def tensor_summary(tensor) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        x = tensor.detach()
        finite = torch.isfinite(x)
        finite_count = int(finite.sum().item())
        total = int(x.numel())
        out = {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "numel": total,
            "finite": finite_count,
            "nan": int(torch.isnan(x).sum().item()),
            "posinf": int(torch.isposinf(x).sum().item()),
            "neginf": int(torch.isneginf(x).sum().item()),
        }
        if finite_count:
            y = x[finite].float()
            out.update(
                {
                    "min": float(y.min().item()),
                    "max": float(y.max().item()),
                    "mean": float(y.mean().item()),
                    "absmax": float(y.abs().max().item()),
                }
            )
        return out


class FirstBadRecorder:
    def __init__(self) -> None:
        self.first: dict[str, Any] | None = None

    def check(self, step: int, kind: str, name: str, value: Any) -> None:
        if self.first is not None:
            return
        import torch

        for tensor in iter_tensors(value):
            if tensor.numel() == 0:
                continue
            if not torch.isfinite(tensor.detach()).all().item():
                self.first = {
                    "step": step,
                    "kind": kind,
                    "name": name,
                    "summary": tensor_summary(tensor),
                }
                print("[FIRST_NONFINITE]", json.dumps(self.first, ensure_ascii=False, sort_keys=True), flush=True)
                return


def module_is_leaf(module) -> bool:
    return len(list(module.children())) == 0


def attach_hooks(model, recorder: FirstBadRecorder, step_box: dict[str, int]):
    handles = []

    for name, module in model.named_modules():
        if not module_is_leaf(module):
            continue

        def make_forward_hook(module_name: str):
            def hook(_module, _inputs, output):
                recorder.check(step_box["step"], "forward_output", module_name, output)

            return hook

        def make_backward_hook(module_name: str):
            def hook(_module, grad_input, grad_output):
                recorder.check(step_box["step"], "backward_grad_output", module_name, grad_output)
                recorder.check(step_box["step"], "backward_grad_input", module_name, grad_input)

            return hook

        handles.append(module.register_forward_hook(make_forward_hook(name)))
        handles.append(module.register_full_backward_hook(make_backward_hook(name)))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        def make_param_hook(param_name: str):
            def hook(grad):
                recorder.check(step_box["step"], "param_grad_hook", param_name, grad)
                return grad

            return hook

        handles.append(param.register_hook(make_param_hook(name)))

    return handles


def set_train_mode(model, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad_(mode != "head_only")

    if mode == "head_only":
        for name, param in model.named_parameters():
            if name.startswith("classifier.") or name.startswith("fc_norm."):
                param.requires_grad_(True)
        return

    if mode == "backbone_no_special":
        special = ("time_decay", "time_first", "time_mix")
        for name, param in model.named_parameters():
            if name.startswith("rwkv_nepa.rwkv.") and (
                any(token in name for token in special)
                or ".ln" in name
                or ".pre_ln" in name
                or "norm" in name.lower()
                or name.endswith(".bias")
            ):
                param.requires_grad_(False)


def print_trainable_summary(model, max_report: int) -> None:
    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    frozen = [(name, p.numel()) for name, p in model.named_parameters() if not p.requires_grad]
    print(f"trainable params: tensors={len(trainable)} numel={sum(x[1] for x in trainable):,}")
    print(f"frozen params:    tensors={len(frozen)} numel={sum(x[1] for x in frozen):,}")
    print("first trainable names:")
    for name, numel in trainable[:max_report]:
        print(f"  {name} ({numel:,})")


def load_fixed_batch(args: argparse.Namespace, image_processor, config):
    import numpy as np
    import torch
    from datasets import load_dataset, load_from_disk
    from PIL import Image
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

    if args.synthetic:
        num_labels = args.num_labels or int(getattr(config, "num_labels", 1000) or 1000)
        image_size = getattr(config, "image_size", 224)
        if isinstance(image_size, (tuple, list)):
            height, width = int(image_size[0]), int(image_size[1])
        else:
            height = width = int(image_size)
        gen = torch.Generator().manual_seed(args.seed)
        pixel_values = torch.randn(args.batch_size, 3, height, width, generator=gen)
        labels = torch.arange(args.batch_size, dtype=torch.long) % num_labels
        return pixel_values, labels

    dataset = load_from_disk(args.dataset_name) if args.load_from_disk else load_dataset(args.dataset_name)
    if hasattr(dataset, "keys"):
        dataset = dataset[args.split]

    size_cfg = getattr(image_processor, "size", None) or {}
    if "shortest_edge" in size_cfg:
        size = int(size_cfg["shortest_edge"])
        crop_size = size
    elif "height" in size_cfg and "width" in size_cfg:
        size = (int(size_cfg["height"]), int(size_cfg["width"]))
        crop_size = size
    else:
        image_size = getattr(config, "image_size", 224)
        size = image_size if isinstance(image_size, tuple) else int(image_size)
        crop_size = size

    mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
    transform = Compose([Resize(size), CenterCrop(crop_size), ToTensor(), Normalize(mean=mean, std=std)])

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.batch_size, replace=False).tolist()
    pixels = []
    labels = []
    for index in indices:
        row = dataset[int(index)]
        image = row[args.image_column]
        if not isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = image.convert("RGB")
        pixels.append(transform(image))
        labels.append(int(row[args.label_column]))

    print("fixed dataset indices:", indices)
    return torch.stack(pixels, dim=0), torch.tensor(labels, dtype=torch.long)


def scan_nonfinite_grads(model, step: int, max_report: int) -> list[dict[str, Any]]:
    bad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not param.grad.isfinite().all().item():
            bad.append({"step": step, "name": name, "summary": tensor_summary(param.grad)})
            if len(bad) >= max_report:
                break
    return bad


def scan_nonfinite_params(model, step: int, max_report: int) -> list[dict[str, Any]]:
    bad = []
    for name, param in model.named_parameters():
        if not param.data.isfinite().all().item():
            bad.append({"step": step, "name": name, "summary": tensor_summary(param.data)})
            if len(bad) >= max_report:
                break
    return bad


def ensure_train_rescale(model) -> None:
    core = getattr(getattr(model, "rwkv_nepa", None), "rwkv", None)
    if core is None:
        return
    core.train(True)
    if getattr(core, "layers_are_rescaled", False):
        print("[RESCALE] layers_are_rescaled=True before probe; restoring train-time weights via _rescale_layers().")
        core._rescale_layers()
    print(
        "[RESCALE] core.training=",
        core.training,
        "layers_are_rescaled=",
        getattr(core, "layers_are_rescaled", "NA"),
    )


def main() -> None:
    args = parse_args()
    setup_imports(args)

    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoImageProcessor
    from transformers.models.rwkv import modeling_rwkv as hf_rwkv

    from models.rwkv_nepa import RwkvNepaConfig, RwkvNepaForImageClassification

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("=" * 100)
    print("RWKV-NEPA classification NaN probe")
    print("project_dir:", Path(args.project_dir).resolve())
    print("model_dir:", args.model_dir)
    print("dataset:", args.dataset_name)
    print("device:", device)
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("hf rwkv source:", getattr(hf_rwkv, "__file__", "NA"))
    print("force_torch_wkv:", args.force_torch_wkv or os.environ.get("RWKV_NEPA_FORCE_TORCH_WKV") == "1")

    config = RwkvNepaConfig.from_pretrained(args.model_dir)
    if args.disable_rescale_every:
        print(f"[CONFIG] overriding rescale_every {getattr(config, 'rescale_every', None)} -> 0")
        config.rescale_every = 0

    image_processor = AutoImageProcessor.from_pretrained(args.image_processor_name or args.model_dir, use_fast=True)
    model = RwkvNepaForImageClassification.from_pretrained(
        args.model_dir,
        config=config,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.train()

    if args.ensure_train_rescale:
        ensure_train_rescale(model)

    set_train_mode(model, args.train_mode)
    print_trainable_summary(model, args.max_report_params)

    pixel_values, labels = load_fixed_batch(args, image_processor, config)
    pixel_values = pixel_values.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device)
    print("batch pixel_values:", tensor_summary(pixel_values))
    print("batch labels:", labels.detach().cpu().tolist())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable_params:
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    recorder = FirstBadRecorder()
    step_box = {"step": 0}
    handles = attach_hooks(model, recorder, step_box)

    try:
        for step in range(1, args.steps + 1):
            step_box["step"] = step
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            context = torch.autograd.detect_anomaly(check_nan=True) if args.detect_anomaly else torch.enable_grad()
            with context:
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = F.cross_entropy(logits.float(), labels)
                print(
                    f"[STEP {step}] forward loss={float(loss.detach().cpu())} "
                    f"logits_finite={bool(torch.isfinite(logits).all().item())} "
                    f"loss_finite={bool(torch.isfinite(loss).all().item())}",
                    flush=True,
                )
                recorder.check(step, "logits", "model.logits", logits)
                recorder.check(step, "loss", "cross_entropy", loss)
                loss.backward()

            if device.type == "cuda":
                torch.cuda.synchronize()

            bad_grads = scan_nonfinite_grads(model, step, args.max_report_params)
            if bad_grads:
                print("[BAD_GRADS]", json.dumps(bad_grads, ensure_ascii=False, sort_keys=True), flush=True)
                break

            bad_params_after_backward = scan_nonfinite_params(model, step, args.max_report_params)
            if bad_params_after_backward:
                print(
                    "[BAD_PARAMS_AFTER_BACKWARD]",
                    json.dumps(bad_params_after_backward, ensure_ascii=False, sort_keys=True),
                    flush=True,
                )
                break

            if optimizer is not None and not args.no_step:
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()

            bad_params = scan_nonfinite_params(model, step, args.max_report_params)
            if bad_params:
                print("[BAD_PARAMS_AFTER_STEP]", json.dumps(bad_params, ensure_ascii=False, sort_keys=True), flush=True)
                break

            print(f"[STEP {step}] backward grads finite; optimizer_step={'no' if args.no_step else 'yes'}", flush=True)

        if recorder.first is None:
            print("[RESULT] No non-finite activation/gradient observed by hooks.")
        else:
            print("[RESULT] First non-finite record above.")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
