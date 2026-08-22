"""Write LaTex tables for the two diamond-ordering conventions in this repo."""

import argparse
import csv
from pathlib import Path

import numpy as np


def _latex_table(caption: str, label: str, rows: list[tuple[str, np.ndarray]]) -> str:
    n = len(rows[0][1])
    body = [
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{{caption} ($n={n:,}$).}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Ordering Constraint} & \textbf{Success Rate} \\",
        r"\midrule",
    ]
    for name, satisfied in rows[:-1]:
        body.append(f"{name} & {100 * satisfied.mean():.1f}\\% " + r"\\")
    body.extend([
        r"\midrule",
        f"{rows[-1][0]} & \\textbf{{{100 * rows[-1][1].mean():.1f}\\%}} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    with open(args.values_csv, newline="") as handle:
        data = list(csv.DictReader(handle))
    if not data:
        raise RuntimeError("No counterfactual values found")
    a = np.asarray([float(row["A_no_diamond_no_pickaxe"]) for row in data])
    b = np.asarray([float(row["B_no_diamond_pickaxe"]) for row in data])
    c = np.asarray([float(row["C_diamond_no_pickaxe"]) for row in data])
    d = np.asarray([float(row["D_diamond_pickaxe"]) for row in data])

    requested = [
        (r"Diamond + Pickaxe $>$ No Diamond + No Pickaxe", d > a),
        (r"Diamond + Pickaxe $>$ Diamond + No Pickaxe", d > c),
        (r"No Diamond + Pickaxe $>$ No Diamond + No Pickaxe", b > a),
        (r"Diamond + No Pickaxe $>$ No Diamond + Pickaxe", c > b),
    ]
    requested.append((r"Full ordering satisfied", np.logical_and.reduce([row[1] for row in requested])))

    middle_free = [
        (r"Diamond + Pickaxe $>$ No Diamond + No Pickaxe", d > a),
        (r"Diamond + Pickaxe $>$ Diamond + No Pickaxe", d > c),
        (r"No Diamond + No Pickaxe $>$ No Diamond + Pickaxe", a > b),
        (r"Diamond + No Pickaxe $>$ No Diamond + Pickaxe", c > b),
    ]
    middle_free.append((r"Full ordering satisfied", np.logical_and.reduce([row[1] for row in middle_free])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diamond_ordering_requested.tex").write_text(
        _latex_table(
            "Success rates for the requested theoretical ordering on the diamond counterfactual task",
            "tab:diamond_ordering_requested",
            requested,
        )
    )
    (out_dir / "diamond_ordering_middle_free.tex").write_text(
        _latex_table(
            "Success rates for the implemented middle-free diamond ordering",
            "tab:diamond_ordering_middle_free",
            middle_free,
        )
    )
    print(f"Wrote LaTex tables to {out_dir}")


if __name__ == "__main__":
    main()
