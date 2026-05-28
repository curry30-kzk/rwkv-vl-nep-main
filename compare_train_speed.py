import argparse
import json
import os


def _load_metrics(output_dir: str) -> dict:
    candidates = [
        os.path.join(output_dir, "train_results.json"),
        os.path.join(output_dir, "all_results.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No train_results.json or all_results.json found in {output_dir}")


def _metric(metrics: dict, name: str) -> float:
    if name not in metrics:
        raise KeyError(f"Metric {name!r} not found. Available keys: {sorted(metrics)}")
    return float(metrics[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Trainer speed metrics between Arrow and uint8 npy runs.")
    parser.add_argument("--arrow_output_dir", required=True)
    parser.add_argument("--uint8_output_dir", required=True)
    args = parser.parse_args()

    arrow = _load_metrics(args.arrow_output_dir)
    uint8 = _load_metrics(args.uint8_output_dir)

    keys = ("train_samples_per_second", "train_steps_per_second")
    print("metric,arrow,uint8,uint8_over_arrow")
    for key in keys:
        arrow_value = _metric(arrow, key)
        uint8_value = _metric(uint8, key)
        speedup = uint8_value / arrow_value if arrow_value else 0.0
        print(f"{key},{arrow_value:.6f},{uint8_value:.6f},{speedup:.4f}x")


if __name__ == "__main__":
    main()
