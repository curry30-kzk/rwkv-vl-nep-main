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


def _build_vit_to_rwkv_mapping(num_hidden_layers: int) -> dict[str, str]:
    mapping = {
        # NEPA-specific embeddings (shared implementation)
        "rwkv_nepa.embeddings.cls_token": "vit_nepa.embeddings.cls_token",
        "rwkv_nepa.embeddings.patch_embeddings.projection.weight": "vit_nepa.embeddings.patch_embeddings.projection.weight",
        "rwkv_nepa.embeddings.patch_embeddings.projection.bias": "vit_nepa.embeddings.patch_embeddings.projection.bias",
    }

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
            "Initialize a RWKV-NEPA checkpoint from a ViT-NEPA pretrain checkpoint:\n"
            "- copy patch embeddings + CLS token\n"
            "- copy ViT FFN weights into RWKV ChannelMix (feed_forward key/value)\n"
            "- keep RWKV TimeMix weights from Transformers initialization"
        )
    )
    parser.add_argument(
        "--vit_nepa_checkpoint",
        type=str,
        required=True,
        help="Path to ViT-NEPA checkpoint (.safetensors or torch state_dict).",
    )
    parser.add_argument(
        "--rwkv_nepa_config_dir",
        type=str,
        required=True,
        help="Path to RWKV-NEPA config directory (contains config.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to write RWKV-NEPA checkpoint (HF save_pretrained format).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_dir if it exists.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to overwrite it.")
        shutil.rmtree(args.output_dir)

    import sys

    sys.path.append(".")
    from models.rwkv_nepa import RwkvNepaConfig, RwkvNepaForPreTraining

    config = RwkvNepaConfig.from_pretrained(args.rwkv_nepa_config_dir)
    model = RwkvNepaForPreTraining(config)

    mapping = _build_vit_to_rwkv_mapping(config.num_hidden_layers)
    vit_tensors = _load_tensors_from_checkpoint(args.vit_nepa_checkpoint, mapping.values())

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

    model.save_pretrained(args.output_dir, safe_serialization=True)
    _copy_if_exists(args.rwkv_nepa_config_dir, "preprocessor_config.json", args.output_dir)

    print(f"[OK] Saved RWKV-NEPA checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()

