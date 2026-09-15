"""Format existing lava-suicide frames as paper-style, value-labeled panels."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="concept_mapping/runs/lava_suicide_episode00_last5")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    with (run_dir / "last_5_steps.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    frames = sorted(run_dir.glob("step_*.png"))
    if len(frames) != len(records):
        raise ValueError("Saved frame count does not match last_5_steps.csv")
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    cells = []
    for frame_path, record in zip(frames, records):
        frame = Image.open(frame_path).convert("RGB")
        # The source video has a 42-pixel action footer; replace it with a
        # larger, consistent paper-style caption below the rendered state.
        view = frame.crop((0, 0, frame.width, frame.height - 42))
        footer_h = 49
        cell = Image.new("RGB", (view.width, view.height + footer_h), "white")
        cell.paste(view, (0, 0))
        draw = ImageDraw.Draw(cell)
        title = f"step {record['step']}" + (" (terminal)" if record["done"] == "True" else "")
        draw.text((5, view.height + 4), title, font=bold, fill="black")
        draw.text((5, view.height + 27), f"value: {float(record['value']):.4f}", font=bold, fill="black")
        cells.append(cell)
    gap = 18
    output = Image.new("RGB", (sum(cell.width for cell in cells) + gap * (len(cells) - 1), max(cell.height for cell in cells)), "white")
    x = 0
    for cell in cells:
        output.paste(cell, (x, 0)); x += cell.width + gap
    output.save(run_dir / "last_5_steps_paper.png")
    print(run_dir / "last_5_steps_paper.png")


if __name__ == "__main__":
    main()
