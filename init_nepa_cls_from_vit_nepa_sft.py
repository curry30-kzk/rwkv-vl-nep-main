import argparse
import os
import shutil
from typing import Iterable

import torch


def _copy_if_exists(src_dir: str, filename: str, dst_dir: str) -> bool:
    src = os.path.join(src_dir, filename)
    if not os.path.isfile(src):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, filename))
    return True


def _load_tensors_from_checkpoint(path: str, keys: Iterable[str]) -> dict[str, torch.Tensor]:
    keys = list(keys)
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            missing = [k for k in keys if k not in available]
            if missing:
                preview = "\n".join(missing[:50])
                raise KeyError(f"Missing {len(missing)} keys in {path}:\n{preview}")
            for k in keys:
                tensors[k] = f.get_tensor(k)
        return tensors

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint format at {path}: expected a state_dict dict.")
    missing = [k for k in keys if k not in obj]
    if missing:
        preview = "\n".join(missing[:50])
        raise KeyError(f"Missing {len(missing)} keys in {path}:\n{preview}")
    return {k: obj[k] for k in keys}


def _load_full_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint format at {path}: expected a state_dict dict.")
    return obj


def _parse_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    v = str(value).strip().lower()
    if v.startswith("torch."):
        v = v[len("torch.") :]
    if v in ("auto", ""):
        return None
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    if v in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported --dtype={value!r}. Use auto|bf16|fp16|fp32.")


def _build_vit_sft_to_rwkv_cls_mapping(num_hidden_layers: int, include_fc_norm: bool) -> dict[str, str]:
    mapping = {
        # embeddings
        "rwkv_nepa.embeddings.cls_token": "vit_nepa.embeddings.cls_token",
        "rwkv_nepa.embeddings.patch_embeddings.projection.weight": "vit_nepa.embeddings.patch_embeddings.projection.weight",
        "rwkv_nepa.embeddings.patch_embeddings.projection.bias": "vit_nepa.embeddings.patch_embeddings.projection.bias",
        # classifier head
        "classifier.weight": "classifier.weight",
        "classifier.bias": "classifier.bias",
    }
    if include_fc_norm:
        mapping["fc_norm.weight"] = "fc_norm.weight"
        mapping["fc_norm.bias"] = "fc_norm.bias"

    # ViT FFN -> RWKV ChannelMix (feed_forward)
    for i in range(num_hidden_layers):
        mapping[f"rwkv_nepa.rwkv.blocks.{i}.feed_forward.key.weight"] = (
            f"vit_nepa.encoder.layer.{i}.intermediate.up_proj.weight"
        )
        mapping[f"rwkv_nepa.rwkv.blocks.{i}.feed_forward.value.weight"] = (
            f"vit_nepa.encoder.layer.{i}.output.dense.weight"
        )

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize a RWKV-NEPA image-classification checkpoint by merging:\n"
            "- embeddings + FFN + (fc_norm/)classifier head from a ViT-NEPA SFT checkpoint\n"
            "- RWKV TimeMix (and the rest of RWKV backbone) from an existing RWKV-NEPA trained checkpoint\n"
        )
    )
    parser.add_argument(
        "--vit_nepa_sft_checkpoint",
        type=str,
        required=True,
        help="Path to ViT-NEPA SFT checkpoint (.safetensors or torch state_dict).",
    )
    parser.add_argument(
        "--rwkv_nepa_trained_checkpoint",
        type=str,
        required=True,
        help="Path to RWKV-NEPA trained checkpoint (typically a HF model.safetensors).",
    )
    parser.add_argument(
        "--rwkv_nepa_config_dir",
        type=str,
        default=None,
        help="Path to RWKV-NEPA config directory (contains config.json). Defaults to dirname of --rwkv_nepa_trained_checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to write RWKV-NEPA image-classification checkpoint (HF save_pretrained format).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Output dtype for the merged model weights: auto|bf16|fp16|fp32. (auto uses rwkv config 'dtype' if present).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_dir if it exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate weight transfer, but do not write output_dir.",
    )
    args = parser.parse_args()

    rwkv_config_dir = args.rwkv_nepa_config_dir
    if rwkv_config_dir is None:
        rwkv_config_dir = os.path.dirname(args.rwkv_nepa_trained_checkpoint)

    if not args.dry_run and os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to overwrite it.")
        shutil.rmtree(args.output_dir)

    if args.vit_nepa_sft_checkpoint.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(args.vit_nepa_sft_checkpoint, framework="pt", device="cpu") as f:
            available = set(f.keys())
            if "classifier.weight" not in available:
                raise KeyError(
                    f"{args.vit_nepa_sft_checkpoint} has no 'classifier.weight'. "
                    "Expected a ViT-NEPA ImageClassification SFT checkpoint."
                )
            num_labels = int(f.get_tensor("classifier.weight").shape[0])
            has_fc_norm = ("fc_norm.weight" in available) and ("fc_norm.bias" in available)
    else:
        obj = torch.load(args.vit_nepa_sft_checkpoint, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise TypeError(
                f"Unsupported checkpoint format at {args.vit_nepa_sft_checkpoint}: expected a state_dict dict."
            )
        if "classifier.weight" not in obj:
            raise KeyError(
                f"{args.vit_nepa_sft_checkpoint} has no 'classifier.weight'. "
                "Expected a ViT-NEPA ImageClassification SFT checkpoint."
            )
        num_labels = int(obj["classifier.weight"].shape[0])
        has_fc_norm = ("fc_norm.weight" in obj) and ("fc_norm.bias" in obj)

    import sys

    sys.path.append(".")
    from models.rwkv_nepa import RwkvNepaConfig, RwkvNepaForImageClassification

    config = RwkvNepaConfig.from_pretrained(rwkv_config_dir)
    config.num_labels = num_labels
    config.architectures = ["RwkvNepaForImageClassification"]
    if has_fc_norm:
        config.add_pooling_layer = True

    model = RwkvNepaForImageClassification(config)

    dtype = _parse_dtype(args.dtype)
    if dtype is None:
        dtype = _parse_dtype(getattr(config, "dtype", None))
    if dtype is not None:
        model = model.to(dtype=dtype)

    rwkv_sd = _load_full_state_dict(args.rwkv_nepa_trained_checkpoint)
    missing, unexpected = model.load_state_dict(rwkv_sd, strict=False)
    expected_missing = {"classifier.weight", "classifier.bias"}
    if config.add_pooling_layer:
        expected_missing |= {"fc_norm.weight", "fc_norm.bias"}
    missing_set = set(missing)
    unexpected_set = set(unexpected)
    if unexpected_set:
        preview = "\n".join(sorted(unexpected_set)[:50])
        raise KeyError(
            f"Unexpected {len(unexpected_set)} keys when loading RWKV checkpoint into classification model:\n{preview}"
        )
    if not missing_set.issubset(expected_missing):
        preview = "\n".join(sorted(missing_set - expected_missing)[:50])
        raise KeyError(
            "RWKV checkpoint load has missing keys beyond classifier/fc_norm.\n"
            f"Unexpected missing ({len(missing_set - expected_missing)}):\n{preview}"
        )

    mapping = _build_vit_sft_to_rwkv_cls_mapping(config.num_hidden_layers, include_fc_norm=config.add_pooling_layer)
    vit_tensors = _load_tensors_from_checkpoint(args.vit_nepa_sft_checkpoint, mapping.values())

    state_dict = model.state_dict()
    for dst_key, src_key in mapping.items():
        if dst_key not in state_dict:
            raise KeyError(f"Target key not found in RWKV-NEPA state_dict: {dst_key}")
        src_tensor = vit_tensors[src_key]
        dst_tensor = state_dict[dst_key]
        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            raise ValueError(
                "Shape mismatch for weight transfer:\n"
                f"- src ({src_key}): {tuple(src_tensor.shape)}\n"
                f"- dst ({dst_key}): {tuple(dst_tensor.shape)}"
            )
        state_dict[dst_key] = src_tensor.to(dtype=dst_tensor.dtype)

    model.load_state_dict(state_dict, strict=True)

    if args.dry_run:
        print("[OK] dry_run: weights merged successfully (no files written).")
        return

    model.save_pretrained(args.output_dir, safe_serialization=True)
    _copy_if_exists(rwkv_config_dir, "preprocessor_config.json", args.output_dir)

    print(f"[OK] Saved RWKV-NEPA image-classification checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
