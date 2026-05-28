import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_from_disk
from PIL import Image


def _resample_bicubic():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def _to_rgb_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, str):
        with Image.open(value) as image:
            return image.convert("RGB")

    if isinstance(value, dict):
        path = value.get("path")
        if path:
            with Image.open(path) as image:
                return image.convert("RGB")
        data = value.get("bytes")
        if data is not None:
            with Image.open(io.BytesIO(data)) as image:
                return image.convert("RGB")

    if hasattr(value, "convert"):
        return value.convert("RGB")

    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def _resolve_split(dataset, split: str):
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split {split!r} not found. Available splits: {list(dataset.keys())}")
        return dataset[split]
    return dataset


def _select_subset(dataset, sample_fraction: float | None, max_samples: int | None, seed: int, shuffle: bool, cache_dir=None):
    total = len(dataset)

    if sample_fraction is not None:
        num_samples = int(total * sample_fraction)
    elif max_samples is not None:
        num_samples = min(int(max_samples), total)
    else:
        num_samples = total

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    if shuffle:
        if cache_dir is not None:
            dataset = dataset.shuffle(
                seed=seed,
                indices_cache_file_name=str(cache_dir / "shuffle_indices.arrow"),
            )
        else:
            dataset = dataset.shuffle(seed=seed)

    if num_samples < total:
        if cache_dir is not None:
            dataset = dataset.select(
                range(num_samples),
                indices_cache_file_name=str(cache_dir / "subset_indices.arrow"),
            )
        else:
            dataset = dataset.select(range(num_samples))

    return dataset, num_samples


def preprocess_to_uint8_shards(
    dataset_path: str,
    output_dir: str,
    split: str,
    image_column: str,
    label_column: str,
    image_size: int,
    shard_size: int,
    sample_fraction: float | None,
    max_samples: int | None,
    seed: int,
    shuffle: bool,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = _resolve_split(load_from_disk(dataset_path), split)
    if image_column not in dataset.column_names:
        raise ValueError(f"Image column {image_column!r} not found in {dataset.column_names}")
    if label_column not in dataset.column_names:
        raise ValueError(f"Label column {label_column!r} not found in {dataset.column_names}")

    dataset, num_samples = _select_subset(dataset, sample_fraction, max_samples, seed, shuffle, cache_dir=output_dir)
    if num_samples <= 0:
        raise ValueError("No samples selected for preprocessing.")

    resample = _resample_bicubic()
    shards = []
    processed = 0
    shard_idx = 0

    while processed < num_samples:
        current_size = min(shard_size, num_samples - processed)
        image_name = f"images_{shard_idx:05d}.npy"
        label_name = f"labels_{shard_idx:05d}.npy"
        image_path = output / image_name
        label_path = output / label_name

        images = np.lib.format.open_memmap(
            image_path,
            mode="w+",
            dtype=np.uint8,
            shape=(current_size, 3, image_size, image_size),
        )
        labels = np.lib.format.open_memmap(
            label_path,
            mode="w+",
            dtype=np.int64,
            shape=(current_size,),
        )

        for local_idx in range(current_size):
            row = dataset[processed + local_idx]
            image = _to_rgb_image(row[image_column])
            image = image.resize((image_size, image_size), resample=resample)
            array = np.asarray(image, dtype=np.uint8)
            images[local_idx] = np.transpose(array, (2, 0, 1))
            labels[local_idx] = int(row[label_column])

        images.flush()
        labels.flush()
        del images
        del labels

        shards.append(
            {
                "images": image_name,
                "labels": label_name,
                "num_samples": current_size,
            }
        )
        processed += current_size
        shard_idx += 1
        print(f"[{processed}/{num_samples}] wrote {image_name} and {label_name}", flush=True)

    metadata = {
        "format": "rwkv_nepa_uint8_npy_shards",
        "version": 1,
        "dataset_path": os.path.abspath(dataset_path),
        "split": split,
        "image_column": image_column,
        "label_column": label_column,
        "num_samples": num_samples,
        "shard_size": shard_size,
        "image_size": [image_size, image_size],
        "image_shape": [3, image_size, image_size],
        "image_dtype": "uint8",
        "label_dtype": "int64",
        "shards": shards,
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"[OK] wrote metadata to {output / 'metadata.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess HF load_from_disk ImageNet data into uint8 npy shards.")
    parser.add_argument("--dataset_path", required=True, help="Path passed to datasets.load_from_disk().")
    parser.add_argument("--output_dir", required=True, help="Directory for images_*.npy, labels_*.npy, metadata.json.")
    parser.add_argument("--split", default="train", help="Dataset split to preprocess.")
    parser.add_argument("--image_column", default="image", help="Image column name.")
    parser.add_argument("--label_column", default="label", help="Label column name.")
    parser.add_argument("--image_size", type=int, default=224, help="Output square image size.")
    parser.add_argument("--shard_size", type=int, default=4096, help="Samples per npy shard.")
    parser.add_argument("--sample_fraction", type=float, default=None, help="Optional fraction, e.g. 0.1 for 10%%.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional absolute sample limit.")
    parser.add_argument("--seed", type=int, default=1337, help="Shuffle seed.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before selecting a subset.")
    args = parser.parse_args()

    preprocess_to_uint8_shards(**vars(args))


if __name__ == "__main__":
    main()
