"""Merge achievement checkpoint evaluations and write paper-friendly outputs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from concept_mapping.evaluate_achievements_at_checkpoints import ACHIEVEMENTS


def _label(name: str) -> str:
    return name.replace("_", " ").title()


def _step_label(step: int) -> str:
    if step < 1_000_000:
        return f"{step / 1_000:.0f}K"
    if step < 1_000_000_000:
        return f"{step / 1_000_000:g}M"
    return f"{step / 1_000_000_000:g}B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    shard_root, out_dir = Path(args.shard_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(shard_root.glob("shard_*/achievement_success.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["checkpoint"]))
    if not rows:
        raise RuntimeError(f"No shard CSV files found under {shard_root}")
    with (out_dir / "achievement_success_over_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    steps = [int(row["checkpoint"]) for row in rows]
    rates = np.asarray([[100 * float(row[name]) for row in rows] for name in ACHIEVEMENTS])
    # A heatmap is far more readable than 22 overlapping learning curves.
    fig, ax = plt.subplots(figsize=(8.2, 8.2), constrained_layout=True)
    image = ax.imshow(rates, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(steps)), [_step_label(step) for step in steps])
    ax.set_yticks(range(len(ACHIEVEMENTS)), [_label(name) for name in ACHIEVEMENTS])
    ax.set_xlabel("Training environment steps (log-spaced checkpoints)")
    ax.set_title("Achievement success over PPO training")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78)
    colorbar.set_label("Success rate (%)")
    fig.savefig(out_dir / "achievement_success_heatmap.png", dpi=300)
    fig.savefig(out_dir / "achievement_success_heatmap.pdf")
    headers = " & ".join(_step_label(step) for step in steps)
    lines = [
        "\\begin{sidewaystable}",
        "\\centering",
        "\\caption{Achievement success rates (\\%) over PPO training. Columns are the closest available saved checkpoints to logarithmically spaced training milestones; each checkpoint is evaluated for 59 episodes.}",
        "\\label{tab:achievement_success_over_training}",
        "\\scriptsize",
        "\\begin{tabular}{l" + "r" * len(steps) + "}",
        "\\toprule",
        "\\textbf{Achievement} & " + headers + " \\\\",
        "\\midrule",
    ]
    for name, values in zip(ACHIEVEMENTS, rates):
        lines.append(_label(name) + " & " + " & ".join(f"{value:.1f}" for value in values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{sidewaystable}", ""])
    (out_dir / "achievement_success_over_checkpoints.tex").write_text("\n".join(lines))
    print(f"Merged {len(rows)} checkpoints into {out_dir}")


if __name__ == "__main__":
    main()
