"""Test whether the iron-sword transition direction reuses earlier sword directions.

This analysis consumes the saved per-checkpoint top eigenvectors from the
natural achievement-transition coherence sweep; it performs no network
inference.  All metrics are sign invariant because eigenvector orientation is
arbitrary.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from concept_mapping.plot_achievement_value_effects_with_success import log_bin_mean


ROOT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "achievement_transition_coherence_stride3"
)
SUCCESS_CSV = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "achievement_stabilized_effect_over_checkpoints_stride3/"
    "wandb_achievement_success_history.csv"
)
CONCEPTS = ("make_wood_sword", "make_stone_sword", "make_iron_sword")
DIMENSION = 512


def checkpoint_files(concept: str) -> dict[int, Path]:
    result = {}
    for path in (ROOT / concept / "per_checkpoint").glob("checkpoint_*.npz"):
        with np.load(path) as data:
            result[int(data["checkpoint"])] = path
    return result


def unit_vector(path: Path) -> np.ndarray:
    with np.load(path) as data:
        vector = np.asarray(data["top_eigenvector"], dtype=np.float64)
    return vector / np.linalg.norm(vector)


def span_projection(vector: np.ndarray, basis_vectors: np.ndarray) -> float:
    """Norm of vector's projection into span(rows(basis_vectors)), in [0,1]."""
    # QR creates an orthonormal basis even though wood and stone directions need
    # not be orthogonal.  Shapes: basis_vectors [2,512], Q [512,2].
    q, _ = np.linalg.qr(basis_vectors.T)
    return float(np.linalg.norm(q.T @ vector))


def main() -> None:
    files = {concept: checkpoint_files(concept) for concept in CONCEPTS}
    checkpoints = sorted(set.intersection(*(set(paths) for paths in files.values())))
    if not checkpoints:
        raise RuntimeError("No shared checkpoints found")

    records = []
    for checkpoint in checkpoints:
        wood, stone, iron = (
            unit_vector(files[concept][checkpoint]) for concept in CONCEPTS
        )
        records.append({
            "checkpoint": checkpoint,
            "alignment_iron_wood": abs(float(iron @ wood)),
            "alignment_iron_stone": abs(float(iron @ stone)),
            "alignment_wood_stone": abs(float(wood @ stone)),
            "iron_projection_into_wood_stone_span": span_projection(
                iron, np.stack([wood, stone])
            ),
        })

    csv_path = ROOT / "sword_direction_compositionality.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    with SUCCESS_CSV.open(newline="") as handle:
        success_rows = list(csv.DictReader(handle))
    sx, sy = [], []
    for row in success_rows:
        value = row.get("make_iron_sword", "")
        if value not in (None, ""):
            sx.append(float(row["environment_steps"])); sy.append(float(value))
    success_x, success_y = log_bin_mean(np.asarray(sx), np.asarray(sy), bins=500)

    x = np.asarray([row["checkpoint"] for row in records], dtype=float)
    iron_wood = np.asarray([row["alignment_iron_wood"] for row in records])
    iron_stone = np.asarray([row["alignment_iron_stone"] for row in records])
    wood_stone = np.asarray([row["alignment_wood_stone"] for row in records])
    span = np.asarray([row["iron_projection_into_wood_stone_span"] for row in records])

    # Approximate 95th percentiles under independent isotropic unit vectors.
    # |dot| uses 1.96/sqrt(d); projection into a 2-D subspace uses
    # sqrt(chi2_2(.95)/d), where chi2_2(.95)=5.991.
    pairwise_chance_95 = 1.96 / np.sqrt(DIMENSION)
    span_chance_95 = np.sqrt(5.991 / DIMENSION)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True,
                             constrained_layout=True)
    top, bottom = axes
    top.plot(x, iron_wood, color="#2563eb", linewidth=2.0,
             label="Iron sword vs. wood sword")
    top.plot(x, iron_stone, color="#7c3aed", linewidth=2.0,
             label="Iron sword vs. stone sword")
    top.plot(x, wood_stone, color="#059669", linewidth=1.7, alpha=.85,
             label="Wood sword vs. stone sword")
    top.axhline(pairwise_chance_95, color="#64748b", linestyle="--", linewidth=1.2,
                label="Isotropic 95% chance level")
    top.set_ylim(0, 1)
    top.set_ylabel("Absolute eigenvector alignment")
    top.set_title("Pairwise alignment of sword achievement-transition directions")
    top.grid(True, which="both", alpha=.20)
    top.legend(frameon=False, ncol=2)

    bottom.plot(x, span, color="#1d4ed8", linewidth=2.2,
                label="Iron direction in span(wood, stone)")
    bottom.axhline(span_chance_95, color="#64748b", linestyle="--", linewidth=1.2,
                   label="Isotropic 95% chance level")
    success_axis = bottom.twinx()
    success_axis.plot(success_x, success_y, color="#dc2626", linewidth=1.7,
                      alpha=.9, label="Iron-sword success")
    bottom.set_ylim(0, 1)
    success_axis.set_ylim(0, 100)
    success_axis.tick_params(axis="y", colors="#b91c1c")
    success_axis.set_ylabel("Make Iron Sword success rate (%)", color="#b91c1c")
    bottom.set_ylabel("Subspace projection")
    bottom.set_title("Does the iron-sword direction lie in the earlier sword subspace?")
    bottom.grid(True, which="both", alpha=.20)
    handles1, labels1 = bottom.get_legend_handles_labels()
    handles2, labels2 = success_axis.get_legend_handles_labels()
    bottom.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")

    bottom.set_xscale("log")
    ticks = [1e5, 1e6, 1e7, 1e8, 1e9]
    bottom.set_xticks(ticks, [r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$"])
    bottom.set_xlabel("Training environment steps")
    fig.suptitle("Compositional reuse of sword representations", fontsize=17)

    png = ROOT / "sword_direction_compositionality.png"
    pdf = ROOT / "sword_direction_compositionality.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(csv_path); print(png); print(pdf)


if __name__ == "__main__":
    main()
