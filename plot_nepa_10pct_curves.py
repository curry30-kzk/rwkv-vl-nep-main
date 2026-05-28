import json
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main")
OUT_DIR = BASE / "figures_nepa_10pct_loss"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = [
    {
        "group": "pretrain",
        "name": "RWKV-NEPA scratch pretrain",
        "exact_dirs": [
            BASE / "outputs_pretrain/rwkv_nepa_scratch_10pct_pretrain_ep100_bs1536_seed42",
        ],
        "include": ["outputs_pretrain", "rwkv", "nepa", "scratch", "10pct", "pretrain", "ep100"],
        "exclude": ["full", "smoke", "transfer", "supervised"],
    },
    {
        "group": "pretrain",
        "name": "ViT-NEPA scratch pretrain",
        "exact_dirs": [
            BASE / "outputs_pretrain/vit_nepa_scratch_10pct_pretrain_ep100_bs1536_4gpu_seed42",
        ],
        "include": ["outputs_pretrain", "vit", "nepa", "scratch", "10pct", "pretrain", "ep100"],
        "exclude": ["full", "smoke", "transfer", "supervised"],
    },
    {
        "group": "finetune",
        "name": "RWKV-NEPA scratch pretrain -> finetune",
        "exact_dirs": [
            BASE / "outputs_pilot10pct/rwkv_nepa_scratch_pre100_ft100_6gpu_bs1152_seed42",
        ],
        "include": ["outputs_pilot10pct", "rwkv", "nepa", "scratch", "pre100", "ft100"],
        "exclude": ["supervised", "transfer", "tuned", "safev2", "smoke"],
    },
    {
        "group": "finetune",
        "name": "ViT-NEPA scratch pretrain -> finetune",
        "exact_dirs": [
            BASE / "outputs_pilot10pct/vit_nepa_scratch_pre100_ft100_4gpu_bs1152_seed42",
        ],
        "include": ["outputs_pilot10pct", "vit", "nepa", "scratch", "pre100", "ft100"],
        "exclude": ["supervised", "transfer", "tuned", "safev2", "smoke"],
    },
]


def checkpoint_number(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)", str(path))
    return int(m.group(1)) if m else -1


def state_score(path: Path) -> int:
    """Prefer trainer_state with the largest global_step / checkpoint step."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("global_step"), int):
            return data["global_step"]
        steps = [
            int(x["step"])
            for x in data.get("log_history", [])
            if isinstance(x, dict) and "step" in x
        ]
        if steps:
            return max(steps)
    except Exception:
        pass
    return checkpoint_number(path)


def states_under_dir(run_dir: Path):
    if not run_dir.exists():
        return []
    states = []
    root_state = run_dir / "trainer_state.json"
    if root_state.exists():
        states.append(root_state)
    states.extend(run_dir.glob("checkpoint-*/trainer_state.json"))
    return states


def path_matches(path: Path, include, exclude):
    s = str(path).lower()
    return all(k.lower() in s for k in include) and not any(k.lower() in s for k in exclude)


def find_best_state(run):
    candidates = []

    # 1) first try exact dirs
    for d in run["exact_dirs"]:
        candidates.extend(states_under_dir(d))

    if candidates:
        candidates = sorted(set(candidates), key=state_score)
        return candidates[-1]

    # 2) fallback: recursive search by keywords
    all_states = list(BASE.rglob("trainer_state.json"))
    matched = [
        p for p in all_states
        if path_matches(p, run["include"], run["exclude"])
    ]

    if matched:
        matched = sorted(set(matched), key=state_score)
        return matched[-1]

    return None


def load_loss_points(state_path: Path):
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data.get("log_history", []):
        if not isinstance(item, dict):
            continue
        if "loss" not in item or "step" not in item:
            continue

        row = {
            "step": int(item["step"]),
            "loss": float(item["loss"]),
            "epoch": item.get("epoch", ""),
            "learning_rate": item.get("learning_rate", ""),
            "grad_norm": item.get("grad_norm", ""),
        }
        rows.append(row)

    # deduplicate by step, keep the last record
    by_step = {}
    for row in rows:
        by_step[row["step"]] = row

    rows = [by_step[k] for k in sorted(by_step.keys())]
    return rows


def smooth(values, window=5):
    if window <= 1 or len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i+1]) / (i - lo + 1))
    return out


selected = {}
all_rows_by_group = {"pretrain": [], "finetune": []}

print("=" * 100)
print("Selecting trainer_state.json files")
print("=" * 100)

for run in RUNS:
    state = find_best_state(run)
    if state is None:
        print(f"[MISSING] {run['name']}")
        selected[run["name"]] = None
        continue

    rows = load_loss_points(state)
    selected[run["name"]] = state

    print(f"[OK] {run['name']}")
    print(f"     state: {state}")
    print(f"     points: {len(rows)}")
    if rows:
        print(f"     step range: {rows[0]['step']} -> {rows[-1]['step']}")
        print(f"     loss range: {rows[0]['loss']} -> {rows[-1]['loss']}")

    for row in rows:
        all_rows_by_group[run["group"]].append({
            "run": run["name"],
            "state_path": str(state),
            **row,
        })


with open(OUT_DIR / "selected_trainer_states.txt", "w", encoding="utf-8") as f:
    for name, state in selected.items():
        f.write(f"{name}\t{state}\n")


def save_csv(group):
    csv_path = OUT_DIR / f"{group}_loss_points.csv"
    rows = all_rows_by_group[group]
    if not rows:
        return

    keys = ["run", "step", "epoch", "loss", "learning_rate", "grad_norm", "state_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(f"[SAVED] {csv_path}")


def plot_group(group, title, filename):
    plt.figure(figsize=(12, 6))

    found_any = False
    for run in RUNS:
        if run["group"] != group:
            continue

        state = selected.get(run["name"])
        if state is None:
            continue

        rows = load_loss_points(state)
        if not rows:
            continue

        found_any = True
        steps = [r["step"] for r in rows]
        losses = [r["loss"] for r in rows]
        losses_smooth = smooth(losses, window=5)

        # raw line + smoothed line
        plt.plot(steps, losses, linewidth=0.8, alpha=0.35)
        plt.plot(steps, losses_smooth, linewidth=2.0, label=run["name"])

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if found_any:
        out_path = OUT_DIR / filename
        plt.savefig(out_path, dpi=300)
        print(f"[SAVED] {out_path}")
    else:
        print(f"[SKIP] no data for {group}")
    plt.close()


def plot_two_panel():
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    panels = [
        ("pretrain", "10% ImageNet NEPA Scratch Pretraining Loss"),
        ("finetune", "10% ImageNet Fine-tuning Loss"),
    ]

    for ax, (group, title) in zip(axes, panels):
        found_any = False
        for run in RUNS:
            if run["group"] != group:
                continue

            state = selected.get(run["name"])
            if state is None:
                continue

            rows = load_loss_points(state)
            if not rows:
                continue

            found_any = True
            steps = [r["step"] for r in rows]
            losses = [r["loss"] for r in rows]
            losses_smooth = smooth(losses, window=5)

            ax.plot(steps, losses, linewidth=0.8, alpha=0.35)
            ax.plot(steps, losses_smooth, linewidth=2.0, label=run["name"])

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        if found_any:
            ax.legend()

    plt.tight_layout()
    out_path = OUT_DIR / "nepa_10pct_loss_compare_2panel.png"
    plt.savefig(out_path, dpi=300)
    print(f"[SAVED] {out_path}")
    plt.close()


save_csv("pretrain")
save_csv("finetune")

plot_group(
    "pretrain",
    "10% ImageNet NEPA Scratch Pretraining Loss",
    "pretrain_loss_10pct_100ep.png",
)

plot_group(
    "finetune",
    "10% ImageNet Fine-tuning Loss",
    "finetune_loss_10pct_100ep.png",
)

plot_two_panel()

print("=" * 100)
print("Done. Output directory:")
print(OUT_DIR)
print("=" * 100)
