import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def _load_state_dict(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    ckpt_dir = Path(checkpoint_dir)
    safetensors_path = ckpt_dir / "model.safetensors"
    bin_path = ckpt_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")

    if bin_path.exists():
        obj = torch.load(bin_path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise TypeError(f"Unsupported checkpoint payload in {bin_path}")
        return obj

    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}")


def _copy_if_exists(src_dir: str, filename: str, dst_dir: str) -> None:
    src = Path(src_dir) / filename
    if src.is_file():
        shutil.copy2(src, Path(dst_dir) / filename)


def _init_classifier(model, mode: str, std: float, seed: int) -> None:
    if not hasattr(model, "classifier") or not isinstance(model.classifier, torch.nn.Linear):
        return

    torch.manual_seed(seed)
    with torch.no_grad():
        if mode == "zeros":
            model.classifier.weight.zero_()
        elif mode == "normal":
            model.classifier.weight.normal_(mean=0.0, std=std)
        else:
            raise ValueError(f"Unsupported classifier_init={mode!r}")

        if model.classifier.bias is not None:
            model.classifier.bias.zero_()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize RwkvNepaForImageClassification from a RwkvNepaForPreTraining checkpoint."
    )
    parser.add_argument("--pretrain_checkpoint", required=True, help="Directory containing pretraining config/model.")
    parser.add_argument("--output_dir", required=True, help="Where to save the classification model.")
    parser.add_argument("--num_labels", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier_init", choices=["normal", "zeros"], default="normal")
    parser.add_argument("--classifier_std", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    args = parser.parse_args()

    pretrain_dir = Path(args.pretrain_checkpoint)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to overwrite it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from models.rwkv_nepa import RwkvNepaConfig, RwkvNepaForImageClassification

    config = RwkvNepaConfig.from_pretrained(str(pretrain_dir))
    config.num_labels = int(args.num_labels)
    config.architectures = ["RwkvNepaForImageClassification"]
    config.add_pooling_layer = True
    config.id2label = {str(i): str(i) for i in range(args.num_labels)}
    config.label2id = {str(i): i for i in range(args.num_labels)}

    model = RwkvNepaForImageClassification(config)
    pretrain_sd = _load_state_dict(str(pretrain_dir))
    missing, unexpected = model.load_state_dict(pretrain_sd, strict=False)

    expected_missing = {"classifier.weight", "classifier.bias", "fc_norm.weight", "fc_norm.bias"}
    missing_set = set(missing)
    unexpected_set = set(unexpected)
    if unexpected_set:
        preview = "\n".join(sorted(unexpected_set)[:50])
        raise KeyError(f"Unexpected keys when loading pretrain checkpoint:\n{preview}")
    if not missing_set.issubset(expected_missing):
        preview = "\n".join(sorted(missing_set - expected_missing)[:50])
        raise KeyError(f"Unexpected missing keys beyond classifier/fc_norm:\n{preview}")

    if model.fc_norm is not None:
        with torch.no_grad():
            model.fc_norm.weight.fill_(1.0)
            model.fc_norm.bias.zero_()
    _init_classifier(model, args.classifier_init, args.classifier_std, args.seed)

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    model.to(dtype=dtype)

    bad = []
    for name, tensor in model.state_dict().items():
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and not torch.isfinite(tensor.float()).all().item():
            bad.append(name)
    if bad:
        raise RuntimeError(f"Non-finite tensors after classification init: {bad[:20]}")

    model.save_pretrained(str(output_dir), safe_serialization=True)
    config.save_pretrained(str(output_dir))
    _copy_if_exists(str(pretrain_dir), "preprocessor_config.json", str(output_dir))

    summary = {
        "pretrain_checkpoint": str(pretrain_dir),
        "output_dir": str(output_dir),
        "num_labels": args.num_labels,
        "classifier_init": args.classifier_init,
        "classifier_std": args.classifier_std,
        "dtype": args.dtype,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    with open(output_dir / "classification_init_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Saved RWKV-NEPA classification init:", output_dir)
    print("[OK] missing keys:", missing)
    print("[OK] unexpected keys:", unexpected)


if __name__ == "__main__":
    main()
