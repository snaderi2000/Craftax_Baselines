"""Measure convergence of diamond--pickaxe top eigenvectors over training.

Reads the per-checkpoint eigendecomposition outputs produced by
``analyze_diamond_representation_coherence.py``.  It does not reload PPO
checkpoints or recompute hidden representations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUT_DIR = (
    "concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
    "merged/representation_coherence_over_checkpoints"
)
DEFAULT_ORDERING_CSV = (
    "concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
    "merged/ordering_over_checkpoints/ordering_over_checkpoints.csv"
)
DEFAULT_SUCCESS_CSV = (
    "concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
    "merged/diamond_success_curve/diamond_success_smoothed.csv"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_checkpoint_vectors(per_checkpoint_dir: Path):
    records = []
    for path in per_checkpoint_dir.glob("checkpoint_*.npz"):
        with np.load(path) as data:
            checkpoint = int(data["checkpoint"])
            vector = np.asarray(data["top_eigenvector"], dtype=np.float64)
            eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Top eigenvector in {path} is not unit norm: {norm}")
        records.append({
            "checkpoint": checkpoint,
            "top_eigenvector": vector,
            "lambda_1": float(eigenvalues[0]),
        })
    if not records:
        raise FileNotFoundError(f"No checkpoint_*.npz files found in {per_checkpoint_dir}")
    return sorted(records, key=lambda record: record["checkpoint"])


def _plot_alignment(records: list[dict], output_path: Path):
    steps = np.asarray([record["checkpoint"] for record in records], dtype=float)
    alignment = np.asarray([record["alignment_to_final"] for record in records])
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(steps, alignment, color="#0f766e", marker="o", markersize=4, linewidth=2.2)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(r"Alignment $|v_1(t)^\top v_{\mathrm{final}}|$")
    ax.set_title("Convergence toward the final diamond--pickaxe direction")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_combined(records: list[dict], ordering_csv: Path, success_csv: Path, output_path: Path):
    """Use two aligned panels to retain interpretable lambda and % scales."""
    steps = np.asarray([record["checkpoint"] for record in records], dtype=float)
    lambda_1 = np.asarray([record["lambda_1"] for record in records])
    alignment = np.asarray([record["alignment_to_final"] for record in records])
    fig, (ax_lambda, ax_concept) = plt.subplots(
        2, 1, figsize=(8.3, 7.0), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.15)},
    )
    ax_lambda.plot(steps, lambda_1, color="#7c3aed", marker="o", markersize=3.8,
                   linewidth=2.1, label=r"$\lambda_1(M_t)$")
    ax_lambda.set_xscale("log")
    ax_lambda.set_ylabel(r"$\lambda_1(M_t)$")
    ax_lambda.set_title("Strength, stability, and behavioural emergence of diamond collectability")
    ax_lambda.grid(True, which="both", alpha=0.25)
    ax_lambda.legend(frameon=False)

    ax_concept.plot(steps, 100 * alignment, color="#0f766e", marker="o", markersize=3.8,
                    linewidth=2.1, label=r"Direction alignment ($100a(t)$)")
    ordering = _read_csv(ordering_csv)
    if ordering:
        x = np.asarray([float(row["checkpoint"]) for row in ordering])
        y = 100 * np.asarray([float(row["full_ordering_satisfied"]) for row in ordering])
        ax_concept.plot(x, y, color="#dc2626", marker="o", markersize=3.8,
                        linewidth=2.0, label="Diamond-pickaxe ordering")
    success = _read_csv(success_csv)
    if success:
        x = np.asarray([float(row["environment_steps"]) for row in success])
        y = np.asarray([float(row["diamond_success_smoothed_pct"]) for row in success])
        ax_concept.plot(x, y, color="#2563eb", linewidth=2.0,
                        label="Diamond collection success (smoothed)")
    ax_concept.set_xscale("log")
    ax_concept.set_ylim(-2, 102)
    ax_concept.set_xlabel("Training environment steps")
    ax_concept.set_ylabel("Alignment / success (%)")
    ax_concept.grid(True, which="both", alpha=0.25)
    ax_concept.legend(frameon=False, loc="best")
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--ordering_csv", default=DEFAULT_ORDERING_CSV)
    parser.add_argument("--success_csv", default=DEFAULT_SUCCESS_CSV)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    records = _load_checkpoint_vectors(out_dir / "per_checkpoint")
    final_record = records[-1]
    final_vector = final_record["top_eigenvector"]  # [d=512], unit norm
    for record in records:
        # Both vectors have shape [512] and unit norm, so this is cosine
        # alignment. The absolute value removes the arbitrary eigenvector sign.
        record["alignment_to_final"] = float(abs(record["top_eigenvector"] @ final_vector))

    alignments = np.asarray([record["alignment_to_final"] for record in records])
    steps = np.asarray([record["checkpoint"] for record in records])
    alignment_changes = np.abs(np.diff(alignments))
    # The last point is the reference vector itself and can be separated from
    # ordinary adjacent changes, particularly when the final checkpoint is far
    # from the preceding saved checkpoint.
    nonfinal_changes = alignment_changes[:-1] if len(alignment_changes) > 1 else alignment_changes
    nonfinal_index = int(np.argmax(nonfinal_changes))
    rows = []
    for index, record in enumerate(records):
        rows.append({
            "checkpoint": record["checkpoint"],
            "lambda_1": record["lambda_1"],
            "alignment_to_final": record["alignment_to_final"],
            "absolute_change_from_previous": "" if index == 0 else alignment_changes[index - 1],
        })
    with (out_dir / "eigenvector_alignment_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "reference_checkpoint": int(final_record["checkpoint"]),
        "n_checkpoints": len(records),
        "initial_alignment": float(alignments[0]),
        "final_alignment": float(alignments[-1]),
        "minimum_alignment": float(alignments.min()),
        "minimum_alignment_checkpoint": int(steps[np.argmin(alignments)]),
        "largest_adjacent_alignment_change": float(alignment_changes.max()),
        "largest_adjacent_change_from_checkpoint": int(steps[np.argmax(alignment_changes)]),
        "largest_adjacent_change_to_checkpoint": int(steps[np.argmax(alignment_changes) + 1]),
        "largest_adjacent_change_excluding_final_interval": float(nonfinal_changes[nonfinal_index]),
        "largest_nonfinal_change_from_checkpoint": int(steps[nonfinal_index]),
        "largest_nonfinal_change_to_checkpoint": int(steps[nonfinal_index + 1]),
    }
    (out_dir / "eigenvector_alignment_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    _plot_alignment(records, out_dir / "eigenvector_alignment_vs_steps")
    _plot_combined(records, Path(args.ordering_csv), Path(args.success_csv),
                   out_dir / "strength_alignment_ordering_collection")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
