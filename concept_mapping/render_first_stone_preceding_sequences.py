"""Render t-10 through t+1 around first Collect Stone in ten sampled episodes."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from craftax.craftax_classic.constants import Achievement, Action, load_all_textures
from PIL import Image, ImageDraw, ImageFont, ImageOps

from concept_mapping.visualize_symbolic_trajectory import render_local_view


DEFAULT_EPISODES = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/episodes"
)
DEFAULT_OUT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "first_collect_stone_preceding_states_10episodes"
)


def font(size: int, bold: bool = True):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = f"/usr/share/fonts/truetype/dejavu/{filename}"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def first_stone_event(path: Path):
    with np.load(path) as data:
        achievements = np.asarray(data["achievements"])
        terminal = np.asarray(data["terminal_achievements"])
        actions = np.asarray(data["actions"])
        episode_id = int(data["episode_id"])
    post = np.concatenate([achievements[1:], terminal[None]], axis=0)
    index = Achievement.COLLECT_STONE.value
    hits = np.flatnonzero((achievements[:, index] == 0) & (post[:, index] > 0))
    if not len(hits):
        return None
    event_step = int(hits[0])
    if event_step < 10:
        return None
    if int(actions[event_step]) != Action.DO.value:
        raise RuntimeError(
            f"Episode {episode_id} first Collect Stone is not caused by DO at {event_step}"
        )
    return episode_id, event_step


def action_name(action: int) -> str:
    try:
        return Action(action).name.replace("_", " ").title()
    except ValueError:
        return str(action)


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def make_frame(observation: np.ndarray, absolute_step: int, offset: int,
               action: int | None, reward: float | None, block_size: int,
               display_scale: int, textures, panel_textures) -> Image.Image:
    view = render_local_view(
        observation, block_size=block_size, include_inventory=True,
        display_scale=display_scale, textures=textures, panel_textures=panel_textures,
    )
    footer_height = 76
    frame = Image.new("RGB", (view.width, view.height + footer_height), "white")
    frame.paste(view, (0, 0))
    draw = ImageDraw.Draw(frame)
    if offset == 0:
        relative = "t: pre"
    elif offset == 1:
        relative = "t+1: post"
    else:
        relative = f"t{offset:+d}"
    draw.text((6, view.height + 4), f"{relative}  |  step {absolute_step}",
              fill="#111827", font=font(15))
    if offset == 0:
        detail = f"action: {action_name(action)}  |  reward: {reward:+.1f}"
    elif action is not None:
        detail = f"next action: {action_name(action)}"
    else:
        detail = "terminal state"
    draw.text((6, view.height + 34), detail, fill="#475569", font=font(13, bold=False))
    if offset == 0:
        frame = ImageOps.expand(frame, border=5, fill="#dc2626")
    elif offset == 1:
        frame = ImageOps.expand(frame, border=5, fill="#059669")
    else:
        frame = ImageOps.expand(frame, border=5, fill="#cbd5e1")
    return frame


def render_sequence(data: dict[str, np.ndarray], episode_id: int, event_step: int,
                    block_size: int, display_scale: int, textures,
                    panel_textures) -> Image.Image:
    observations = np.asarray(data["observations"], dtype=np.float32)
    terminal_observation = np.asarray(data["terminal_observation"], dtype=np.float32)
    actions = np.asarray(data["actions"])
    rewards = np.asarray(data["rewards"])
    frames = []
    for absolute_step in range(event_step - 10, event_step + 2):
        if absolute_step < len(observations):
            observation = observations[absolute_step]
            action = int(actions[absolute_step])
            reward = float(rewards[absolute_step])
        elif absolute_step == len(observations):
            observation = terminal_observation
            action = None
            reward = None
        else:
            raise RuntimeError(f"Episode {episode_id} lacks state step {absolute_step}")
        frames.append(make_frame(
            observation, absolute_step, absolute_step - event_step, action, reward,
            block_size, display_scale, textures, panel_textures,
        ))

    columns, rows, gap, margin, header = 4, 3, 14, 18, 58
    frame_width = max(frame.width for frame in frames)
    frame_height = max(frame.height for frame in frames)
    sheet = Image.new(
        "RGB",
        (2 * margin + columns * frame_width + (columns - 1) * gap,
         header + margin + rows * frame_height + (rows - 1) * gap + margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 10),
              f"Episode {episode_id}: first Collect Stone at action step t={event_step}",
              fill="#111827", font=font(22))
    for index, frame in enumerate(frames):
        sheet.paste(frame, (
            margin + (index % columns) * (frame_width + gap),
            header + (index // columns) * (frame_height + gap),
        ))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes_dir", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eligible = []
    for path in sorted(args.episodes_dir.glob("episode_*.npz")):
        event = first_stone_event(path)
        if event is not None:
            eligible.append((path, *event))
    if len(eligible) < args.count:
        raise RuntimeError(f"Only {len(eligible)} eligible episodes for count={args.count}")
    chosen_indices = np.sort(
        np.random.default_rng(args.seed).choice(len(eligible), args.count, replace=False)
    )
    chosen = [eligible[index] for index in chosen_indices]
    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )

    sheets, audit = [], []
    for rank, (path, episode_id, event_step) in enumerate(chosen, start=1):
        data = load_episode(path)
        sheet = render_sequence(
            data, episode_id, event_step, args.block_size, args.display_scale,
            textures, panel_textures,
        )
        output = args.out_dir / f"episode_{episode_id:04d}_first_stone_t{event_step:04d}.png"
        sheet.save(output)
        sheets.append(sheet.convert("RGB"))
        audit.append({
            "rank": rank, "episode_id": episode_id, "event_step_t": event_step,
            "first_rendered_step": event_step - 10,
            "last_rendered_step": event_step + 1,
            "event_action": action_name(int(data["actions"][event_step])),
            "event_reward": float(data["rewards"][event_step]),
            "source_npz": str(path), "output_file": output.name,
        })
    pdf = args.out_dir / "ten_random_first_stone_sequences.pdf"
    sheets[0].save(pdf, save_all=True, append_images=sheets[1:], resolution=150)

    selection = args.out_dir / "selection.csv"
    temporary = selection.with_name(f".{selection.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    temporary.replace(selection)
    print(
        f"Saved {len(sheets)} sequences sampled from {len(eligible)} eligible episodes "
        f"(seed={args.seed}) to {args.out_dir}"
    )
    for row in audit:
        print(f"  episode={row['episode_id']} t={row['event_step_t']} {row['output_file']}")
    print(pdf)


if __name__ == "__main__":
    main()
