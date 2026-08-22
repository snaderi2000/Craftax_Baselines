import argparse
import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from craftax.craftax_classic.constants import load_all_textures

from concept_mapping.diamond_affordance_over_time import _make_counterfactuals
from concept_mapping.visualize_symbolic_trajectory import decode_obs, render_local_view


VALUE_KEYS = [
    "A_no_diamond_no_pickaxe",
    "B_no_diamond_pickaxe",
    "C_diamond_no_pickaxe",
    "D_diamond_pickaxe",
]


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _load_values(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _middle_free_correct(row: dict) -> bool:
    a = float(row["A_no_diamond_no_pickaxe"])
    b = float(row["B_no_diamond_pickaxe"])
    c = float(row["C_diamond_no_pickaxe"])
    d = float(row["D_diamond_pickaxe"])
    return d > a and d > c and a > b and c > b


def _write_corrected_summary(rows: list[dict], out_path: str) -> None:
    vals = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        for key in VALUE_KEYS
    }
    a = vals["A_no_diamond_no_pickaxe"]
    b = vals["B_no_diamond_pickaxe"]
    c = vals["C_diamond_no_pickaxe"]
    d = vals["D_diamond_pickaxe"]
    middle_free = (d > a) & (d > c) & (a > b) & (c > b)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_states", len(rows)])
        writer.writerow(["middle_free_success_rate", float(middle_free.mean())])
        writer.writerow(["D_gt_A_rate", float((d > a).mean())])
        writer.writerow(["D_gt_C_rate", float((d > c).mean())])
        writer.writerow(["A_gt_B_rate", float((a > b).mean())])
        writer.writerow(["C_gt_B_rate", float((c > b).mean())])
        writer.writerow(["mean_A", float(a.mean())])
        writer.writerow(["mean_B", float(b.mean())])
        writer.writerow(["mean_C", float(c.mean())])
        writer.writerow(["mean_D", float(d.mean())])


def _text_panel(width: int, lines: list[str]) -> Image.Image:
    panel = Image.new("RGB", (width, 42 + 18 * len(lines)), "white")
    draw = ImageDraw.Draw(panel)
    for i, line in enumerate(lines):
        draw.text((8, 8 + 18 * i), line, fill=(0, 0, 0))
    return panel


def _render_sheet(
    cf: dict[str, np.ndarray],
    row: dict,
    out_path: str,
    block_size: int,
    display_scale: int,
    textures: dict,
    panel_textures: dict,
    minimal: bool = False,
) -> None:
    titles = [
        ("A_no_diamond_no_pickaxe", "A: no diamond, no pickaxe"),
        ("B_no_diamond_pickaxe", "B: no diamond, pickaxe"),
        ("C_diamond_no_pickaxe", "C: diamond, no pickaxe"),
        ("D_diamond_pickaxe", "D: diamond, pickaxe"),
    ]
    frames = []
    for key, title in titles:
        value = float(row[key])
        view = render_local_view(
            cf[key],
            block_size=block_size,
            include_inventory=True,
            display_scale=display_scale,
            textures=textures,
            panel_textures=panel_textures,
        )
        if minimal:
            label_h = max(60, 14 * display_scale)
            frame = Image.new("RGB", (view.width, view.height + label_h), "white")
            frame.paste(view, (0, 0))
            draw = ImageDraw.Draw(frame)
            font = _font(max(28, 6 * display_scale))
            draw.text((4, view.height + 8), f"value: {value:.4f}", fill=(0, 0, 0), font=font)
        else:
            header = _text_panel(view.width, [title, f"V={value:.4f}"])
            frame = Image.new("RGB", (view.width, header.height + view.height), "white")
            frame.paste(header, (0, 0))
            frame.paste(view, (0, header.height))
        frames.append(frame)

    a = float(row["A_no_diamond_no_pickaxe"])
    b = float(row["B_no_diamond_pickaxe"])
    c = float(row["C_diamond_no_pickaxe"])
    d = float(row["D_diamond_pickaxe"])
    ok = d > a and d > c and a > b and c > b
    gap = 16
    cell_w = max(frame.width for frame in frames)
    cell_h = max(frame.height for frame in frames)
    footer = None
    footer_h = 0
    if not minimal:
        footer = _text_panel(
            2 * cell_w + gap,
            [
                f"sample={row['sample_index']} episode={row['episode']} base_step={row['base_step']} collect_step={row['collect_step']}",
                "Rule: D > {A, C} > B; A/C order does not matter",
                f"pass={ok} | D={d:.3f}, A={a:.3f}, C={c:.3f}, B={b:.3f}",
            ],
        )
        footer_h = footer.height
    sheet = Image.new("RGB", (2 * cell_w + gap, 2 * cell_h + gap + footer_h), "white")
    positions = [(0, 0), (cell_w + gap, 0), (0, cell_h + gap), (cell_w + gap, cell_h + gap)]
    for frame, pos in zip(frames, positions):
        sheet.paste(frame, pos)
    if footer is not None:
        sheet.paste(footer, (0, 2 * cell_h + gap))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render examples and corrected summary for diamond affordance middle-free rule."
    )
    parser.add_argument("--run_dir", default="concept_mapping/runs/diamond_affordance_platonic_500_final_m0mw4end")
    parser.add_argument("--values_csv", default=None)
    parser.add_argument("--states_npz", default=None)
    parser.add_argument("--out_subdir", default="middle_free_examples")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument(
        "--sample_indices",
        default="",
        help="Comma-separated row indices to render exactly, overriding random selection.",
    )
    parser.add_argument("--failures_only", action="store_true")
    parser.add_argument("--all_failures", action="store_true")
    parser.add_argument(
        "--max_health",
        type=float,
        default=None,
        help="Only render states whose base-observation health is at most this value.",
    )
    parser.add_argument(
        "--all_matching",
        action="store_true",
        help="Render every state matching --max_health, including passes and failures.",
    )
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()

    values_path = args.values_csv or os.path.join(args.run_dir, "counterfactual_values.csv")
    states_path = args.states_npz or os.path.join(args.run_dir, "diamond_tminus_states.npz")
    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    rows = _load_values(values_path)
    states = np.load(states_path)
    base_obs = states["base_obs"]

    summary_path = os.path.join(args.run_dir, "middle_free_summary.csv")
    _write_corrected_summary(rows, summary_path)

    eligible = list(range(len(rows)))
    if args.max_health is not None:
        eligible = [
            i for i in eligible
            if float(decode_obs(base_obs[i]).intrinsics[0]) <= args.max_health
        ]
    correct = [i for i in eligible if _middle_free_correct(rows[i])]
    incorrect = [i for i in eligible if not _middle_free_correct(rows[i])]
    rng = np.random.default_rng(args.seed)
    requested_indices = [
        int(raw.strip())
        for raw in args.sample_indices.split(",")
        if raw.strip()
    ]
    if requested_indices:
        if any(index < 0 or index >= len(rows) for index in requested_indices):
            raise ValueError("--sample_indices contains an out-of-range row index")
        chosen = requested_indices
    elif args.all_matching:
        chosen = eligible
    elif args.all_failures:
        chosen = incorrect
    elif args.failures_only:
        chosen = rng.choice(incorrect, size=min(args.examples, len(incorrect)), replace=False).tolist()
    else:
        chosen = []
        if correct:
            chosen.extend(rng.choice(correct, size=min(args.examples // 2, len(correct)), replace=False).tolist())
        if incorrect:
            chosen.extend(rng.choice(incorrect, size=min(args.examples - len(chosen), len(incorrect)), replace=False).tolist())
        chosen = chosen[: args.examples]

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    for rank, idx in enumerate(chosen):
        cf = _make_counterfactuals(base_obs[idx])
        suffix = "pass" if _middle_free_correct(rows[idx]) else "fail"
        _render_sheet(
            cf,
            rows[idx],
            os.path.join(out_dir, f"example_{rank:02d}_idx_{idx:03d}_{suffix}.png"),
            args.block_size,
            args.display_scale,
            textures,
            panel_textures,
            args.minimal,
        )

    print(f"Matched states: {len(eligible)}")
    print(f"Saved corrected summary: {summary_path}")
    print(f"Saved examples: {out_dir}")


if __name__ == "__main__":
    main()
