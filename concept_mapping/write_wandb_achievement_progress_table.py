"""Extract W&B achievement metrics at logarithmically spaced training steps."""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
)


def _config_value(config: dict, key: str):
    value = config[key]
    return value["value"] if isinstance(value, dict) and "value" in value else value


def _step_label(steps: int) -> str:
    if steps < 1_000_000:
        return f"{steps / 1_000:.0f}K"
    if steps < 1_000_000_000:
        return f"{steps / 1_000_000:g}M"
    return f"{steps / 1_000_000_000:g}B"


def _pretty(name: str) -> str:
    return name.replace("_", " ").title()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument("--wandb_run", default="synaderi-uc-davis/Craftax_Baselines/m0mw4end")
    parser.add_argument("--out_dir", default="concept_mapping/runs/achievement_progress_wandb_m0mw4end")
    args = parser.parse_args()
    with (Path(args.run_path) / "files" / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    steps_per_update = int(_config_value(config, "NUM_ENVS")) * int(_config_value(config, "NUM_STEPS"))
    import wandb

    metric_keys = [f"Achievements/{name}" for name in ACHIEVEMENTS]
    run = wandb.Api(timeout=120).run(args.wandb_run)
    rows = []
    for history in run.scan_history(keys=["_step", *metric_keys], page_size=10_000):
        if history.get("_step") is None or not all(history.get(key) is not None for key in metric_keys):
            continue
        rows.append((int(history["_step"] + 1) * steps_per_update, [float(history[key]) for key in metric_keys]))
    if not rows:
        raise RuntimeError("No complete achievement rows found in W&B history")
    actual_steps = np.asarray([row[0] for row in rows], dtype=np.int64)
    values = np.asarray([row[1] for row in rows], dtype=np.float64)
    requested = np.asarray([10**power for power in range(3, 11)], dtype=np.int64)
    selected = []
    for target in requested:
        index = int(np.argmin(np.abs(actual_steps - target)))
        if not selected or index != selected[-1]:
            selected.append(index)
    selected_steps, selected_values = actual_steps[selected], values[selected]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "achievement_success_wandb_logsteps.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["achievement", *selected_steps])
        for name, series in zip(ACHIEVEMENTS, selected_values.T):
            writer.writerow([name, *series])
    headers = " & ".join(_step_label(int(step)) for step in selected_steps)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Training-time achievement success rates (\\%) logged by W\\&B at the nearest available checkpoints to logarithmically spaced environment-step milestones.}",
        "\\label{tab:achievement_success_over_training}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "r" * len(selected_steps) + "}",
        "\\toprule",
        "\\textbf{Achievement} & " + headers + " \\\\",
        "\\midrule",
    ]
    for name, series in zip(ACHIEVEMENTS, selected_values.T):
        lines.append(_pretty(name) + " & " + " & ".join(f"{value:.1f}" for value in series) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    (out_dir / "achievement_success_wandb_logsteps.tex").write_text("\n".join(lines))
    print("Selected logged environment steps:", ", ".join(str(int(step)) for step in selected_steps))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
