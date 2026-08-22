"""Write a paper-ready LaTeX table for the sword counterfactual ordering."""

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--values_csv",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/sword_counterfactual_values.csv",
    )
    parser.add_argument(
        "--out_path",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/sword_ordering.tex",
    )
    args = parser.parse_args()
    with Path(args.values_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    iron = np.asarray([float(row["iron_only"]) for row in rows])
    stone = np.asarray([float(row["stone_only"]) for row in rows])
    wood = np.asarray([float(row["wood_only"]) for row in rows])
    rates = (
        ("Stone sword $>$ Wood sword", (stone > wood).mean()),
        ("Iron sword $>$ Wood sword", (iron > wood).mean()),
        ("Iron sword $>$ Stone sword", (iron > stone).mean()),
        ("Full ordering: Iron $>$ Stone $>$ Wood", ((iron > stone) & (stone > wood)).mean()),
    )
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Success rates for the theoretical ordering of sword types in hostile-visible states. Each counterfactual contains exactly one indicated sword, with all other state features fixed.}",
        "\\label{tab:sword_ordering}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "\\textbf{Ordering Constraint} & \\textbf{Success Rate} \\\\",
        "\\midrule",
    ]
    for label, rate in rates[:-1]:
        lines.append(f"{label} & {100 * rate:.1f}\\% \\\\")
    lines.extend([
        "\\midrule",
        f"{rates[-1][0]} & \\textbf{{{100 * rates[-1][1]:.1f}\\%}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    out_path = Path(args.out_path)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path} from n={len(rows)} states")


if __name__ == "__main__":
    main()
