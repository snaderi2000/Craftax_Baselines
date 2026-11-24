"""Sweep a fixed collectability probe bank and plot it against training success.

Run one task at a time. The plot uses a compact, consistent panel format so the
diamond, iron, and plant PDFs can later be placed side by side.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import numpy as np

plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "mathtext.default": "bf",
})

from concept_mapping.analyze_wood_sword_combat_followup import (
    _init_network,
    _load_wandb_config,
    _restore_train_state,
    evaluate_values,
    make_craftax_env_from_name,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = Path("wandb/run-20260630_214658-m0mw4end")
DEFAULT_WANDB_RUN = "synaderi-uc-davis/Craftax_Baselines/m0mw4end"
BEHAVIOR_METRICS = {
    "diamond": "Achievements/collect_diamond",
    "iron": "Achievements/collect_iron",
    "plant": "Achievements/eat_plant",
}
BANKS = {
    "diamond": ROOT / "runs/diamond_collectability_10b_10000episodes_all_visible_stone",
    "iron": ROOT / "runs/iron_collectability_10b_10000episodes_all_visible_stone",
    "plant": ROOT / "runs/plant_collectability_10b_10000episodes_ripe_vs_unripe",
}
DIAMOND_BEHAVIOR = ROOT / "runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/diamond_success_curve/diamond_success_smoothed.csv"


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoints(run):
    folder = run / "files/policies" if run.name != "files" else run / "policies"
    steps = sorted(int(path.name) for path in folder.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"No checkpoint directories in {folder}")
    return steps


def extract_local_behavior(run, tasks, window_updates=1024):
    """Cache the requested smoothed W&B history curves in each task's bank."""
    import yaml
    from google.protobuf.message import DecodeError
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    from concept_mapping.plot_smoothed_diamond_success import _centered_rolling_mean

    run_dir = run.parent if run.name == "files" else run
    files_dir = run if run.name == "files" else run / "files"
    with (files_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    def setting(key):
        value = config[key]
        return value["value"] if isinstance(value, dict) and "value" in value else value
    steps_per_update = int(setting("NUM_ENVS")) * int(setting("NUM_STEPS"))
    logs = list(run_dir.glob("run-*.wandb"))
    if len(logs) != 1:
        raise FileNotFoundError(f"Expected one run-*.wandb in {run_dir}, found {len(logs)}")
    wanted = {task: BEHAVIOR_METRICS[task] for task in tasks}
    points = {task: [] for task in tasks}
    store = DataStore()
    store.open_for_scan(str(logs[0]))
    print(f"Reading W&B history for {', '.join(tasks)} from {logs[0]}", flush=True)
    while True:
        try:
            scanned = store.scan_record()
        except (IndexError, OSError):
            break  # A local W&B file may end with a partial record.
        if scanned is None:
            break
        _, encoded = scanned
        record = wandb_internal_pb2.Record()
        try:
            record.ParseFromString(encoded)
        except DecodeError:
            continue
        if record.WhichOneof("record_type") != "history":
            continue
        values = {item.key or "/".join(item.nested_key): json.loads(item.value_json)
                  for item in record.history.item}
        update = values.get("_step")
        if update is None:
            continue
        for task, metric in wanted.items():
            if metric in values:
                points[task].append((int(update), float(values[metric])))
    for task, samples in points.items():
        if not samples:
            raise RuntimeError(f"No {wanted[task]} samples in local W&B history")
        samples.sort()
        update, raw = np.asarray(samples, dtype=np.float64).T
        smoothed = _centered_rolling_mean(raw, window_updates)
        output = BANKS[task] / "emergence_over_checkpoints/behavior_success_smoothed.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["wandb_update", "environment_steps", "success_raw_pct", "success_smoothed_pct"])
            writer.writerows(zip(update.astype(int), ((update + 1) * steps_per_update).astype(np.int64), raw, smoothed))
        print(f"Saved {len(samples):,} W&B history points to {output}", flush=True)


def extract_wandb_behavior(run_ref, run, tasks, window_updates=1024):
    """Use synced W&B history; the local archive can stop before training ends."""
    import wandb
    import yaml

    files_dir = run if run.name == "files" else run / "files"
    with (files_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    def setting(key):
        value = config[key]
        return value["value"] if isinstance(value, dict) and "value" in value else value
    steps_per_update = int(setting("NUM_ENVS")) * int(setting("NUM_STEPS"))
    wanted = {task: BEHAVIOR_METRICS[task] for task in tasks}
    points = {task: [] for task in tasks}
    api_run = wandb.Api(timeout=60).run(run_ref)
    print(f"Reading synced W&B history for {', '.join(tasks)} from {run_ref}", flush=True)
    for row in api_run.scan_history(keys=["_step", *wanted.values()], page_size=10_000):
        update = row.get("_step")
        if update is None:
            continue
        for task, metric in wanted.items():
            if row.get(metric) is not None:
                points[task].append((int(update), float(row[metric])))
    for task, samples in points.items():
        if not samples:
            raise RuntimeError(f"No {wanted[task]} history in synced W&B run {run_ref}")
        samples.sort()
        update, raw = np.asarray(samples, dtype=np.float64).T
        if (update.max() + 1) * steps_per_update < 9.9e9:
            raise RuntimeError(f"Synced W&B {task} history ends too early: update {int(update.max())}")
        from concept_mapping.plot_smoothed_diamond_success import _centered_rolling_mean
        smoothed = _centered_rolling_mean(raw, window_updates)
        output = BANKS[task] / "emergence_over_checkpoints/behavior_success_smoothed.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["wandb_update", "environment_steps", "success_raw_pct", "success_smoothed_pct"])
            writer.writerows(zip(update.astype(int), ((update + 1) * steps_per_update).astype(np.int64), raw, smoothed))
        print(f"Saved {len(samples):,} synced W&B history points to {output}", flush=True)


def make_branches(task, bank):
    with np.load(bank / "base_states.npz") as data:
        factual = np.asarray(data["observations"], dtype=np.float32)
    n = len(factual)
    assert factual.shape[1] == 1345
    if task == "plant":
        records = read_rows(bank / "per_state_results.csv")
        assert len(records) == n
        target = np.asarray([
            (int(row["target_row"]) * 9 + int(row["target_col"])) * 21
            for row in records
        ])
        assert np.all(factual[np.arange(n), target + 16] == 1)
        unripe = factual.copy()
        unripe[np.arange(n), target + 16] = 0
        unripe[np.arange(n), target + 15] = 1
        assert np.all(np.count_nonzero(factual != unripe, axis=1) == 2)
        return {"ripe": factual, "unripe": unripe}

    block, tool_slot = (10, 1331) if task == "diamond" else (9, 1330)
    visible = factual[:, :1323].reshape(n, 63, 21)[:, :, block] == 1
    assert np.all(visible.sum(axis=1) >= 1)
    assert np.all(factual[:, tool_slot] > 0)
    no_tool = factual.copy()
    no_tool[:, tool_slot] = 0
    no_resource = factual.copy()
    view = no_resource[:, :1323].reshape(n, 63, 21)
    view[:, :, block][visible] = 0
    view[:, :, 4][visible] = 1  # Replace every visible target tile with stone.
    neither = no_resource.copy()
    neither[:, tool_slot] = 0
    assert np.all(np.count_nonzero(factual != no_tool, axis=1) == 1)
    assert np.all(np.count_nonzero(factual != no_resource, axis=1) == 2 * visible.sum(axis=1))
    assert not np.any(no_resource[:, :1323].reshape(n, 63, 21)[:, :, block])
    return {"A": factual, "B": no_tool, "C": neither, "D": no_resource}


def sweep(task, bank, run, selected, batch_size, out_csv):
    branches = make_branches(task, bank)
    n = len(next(iter(branches.values())))
    config = _load_wandb_config(str(run))
    env = make_craftax_env_from_name(config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1"), auto_reset=False)
    network = _init_network(config, env, env.default_params)
    rows = []
    reference = read_rows(bank / "per_state_results.csv")
    assert len(reference) == n
    for step in selected:
        print(f"{task}: checkpoint {step:,} ({len(rows)+1}/{len(selected)})", flush=True)
        state, restored = _restore_train_state(str(run), config, network, env, env.default_params, step)
        values = {
            key: evaluate_values(state.params, network, obs, batch_size).astype(np.float64)
            for key, obs in branches.items()
        }
        if task == "plant":
            success = values["ripe"] > values["unripe"]
            checks = (("ripe", "V_ripe"), ("unripe", "V_unripe"))
        else:
            success = values["A"] > np.maximum.reduce([values["B"], values["C"], values["D"]])
            checks = tuple((key, "V_" + key) for key in "ABCD")
        if restored == 9999941632:
            for key, column in checks:
                expected = np.asarray([float(row[column]) for row in reference])
                error = np.max(np.abs(values[key] - expected))
                if not np.allclose(values[key], expected, rtol=1e-5, atol=1e-5):
                    raise AssertionError(f"Final checkpoint {key} differs from saved result; max error={error}")
            expected_success = np.asarray([row["success"].lower() == "true" for row in reference])
            assert np.array_equal(success, expected_success)
        row = dict(checkpoint=restored, n_states=n, success_count=int(success.sum()),
                   success_rate=float(success.mean()), success_pct=100 * float(success.mean()))
        if task == "iron":
            # A=iron+stone pickaxe, B=iron without pickaxe, D=no iron+pickaxe.
            for name, comparison in (("A_gt_D", values["A"] > values["D"]),
                                     ("A_gt_B", values["A"] > values["B"])):
                row[name + "_count"] = int(comparison.sum())
                row[name + "_pct"] = 100 * float(comparison.mean())
        rows.append(row)
        with out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def plot(task, rows, behavior_csv, behavior_x, behavior_y, out_dir):
    behavior = read_rows(behavior_csv)
    x = np.asarray([float(row[behavior_x]) for row in behavior])
    y = np.asarray([float(row[behavior_y]) for row in behavior])
    order = np.argsort(x)
    x, y = x[order], y[order]
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    assert x.min() > 0 and -0.01 <= y.min() <= y.max() <= 100.01
    cx = np.asarray([float(row["checkpoint"]) for row in rows])
    cy = np.asarray([float(row["success_pct"]) for row in rows])
    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    behavior_label = "Plant eating (smoothed)" if task == "plant" else f"{task.capitalize()} collection (smoothed)"
    ax.plot(x, y, color="#2563eb", lw=2.0, label=behavior_label)
    ax.plot(cx, cy, color="#dc2626", lw=1.8, marker="o", ms=3.8,
            label="Full critic ordering" if task == "iron" else "Critic ordering")
    if task == "iron":
        if not all("A_gt_D_pct" in row and "A_gt_B_pct" in row for row in rows):
            raise ValueError("Iron sweep lacks A_gt_D_pct/A_gt_B_pct; rerun without --reuse_sweep")
        ax.plot(cx, [float(row["A_gt_D_pct"]) for row in rows],
                color="#0f766e", lw=1.7, marker="s", ms=3.2,
                label="Iron effect (pickaxe held)")
        ax.plot(cx, [float(row["A_gt_B_pct"]) for row in rows],
                color="#7c3aed", lw=1.7, marker="^", ms=3.4,
                label="Pickaxe effect (iron held)")
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e10)
    ax.xaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(5, 11)]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(task.capitalize())
    ax.grid(alpha=0.22)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    legend_location = {"plant": "center right", "iron": "lower right"}.get(task, "upper left")
    ax.legend(loc=legend_location, frameon=False,
              prop={"size": 7.2 if task == "iron" else 8, "weight": "bold"})
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{task}_emergence_panel.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(BANKS), required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--wandb_run", default=DEFAULT_WANDB_RUN,
                        help="Synced W&B run for the full behavioral history")
    parser.add_argument("--bank", type=Path, help="Override the fixed state-bank directory")
    parser.add_argument("--behavior_csv", type=Path, help="W&B-derived behavioural success history CSV")
    parser.add_argument("--behavior_x", default="environment_steps")
    parser.add_argument("--behavior_y", help="Behavioural success percentage column")
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--checkpoints", help="Optional comma-separated checkpoint steps")
    parser.add_argument("--reuse_sweep", action="store_true", help="Replot existing checkpoint CSV")
    args = parser.parse_args()
    bank = args.bank or BANKS[args.task]
    out_dir = args.out_dir or bank / "emergence_over_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_csv = args.behavior_csv or (DIAMOND_BEHAVIOR if args.task == "diamond"
                                         else out_dir / "behavior_success_smoothed.csv")
    if args.behavior_csv is None and args.task != "diamond":
        history_is_complete = False
        if behavior_csv.exists():
            history = read_rows(behavior_csv)
            history_is_complete = bool(history) and float(history[-1]["environment_steps"]) >= 9.9e9
        if not history_is_complete:
            extract_wandb_behavior(args.wandb_run, args.run, (args.task,))
    behavior_y = args.behavior_y or ("diamond_success_smoothed_pct" if args.task == "diamond"
                                     and behavior_csv == DIAMOND_BEHAVIOR else "success_smoothed_pct")
    out_csv = out_dir / "ordering_over_checkpoints.csv"
    if args.reuse_sweep:
        rows = read_rows(out_csv)
    else:
        selected = ([int(x) for x in args.checkpoints.split(",")] if args.checkpoints
                    else checkpoints(args.run))
        rows = sweep(args.task, bank, args.run, selected, args.batch_size, out_csv)
    plot(args.task, rows, behavior_csv, args.behavior_x, behavior_y, out_dir)
    metadata = dict(task=args.task, n_states=int(rows[0]["n_states"]), bank=str(bank),
                    run=str(args.run), behavior_csv=str(behavior_csv),
                    behavior_x=args.behavior_x, behavior_y=behavior_y,
                    checkpoint_count=len(rows),
                    criterion=("V(ripe)>V(unripe)" if args.task == "plant"
                               else "V(resource, tool)>max{V(resource, no tool), V(no resource, no tool), V(no resource, tool)}"),
                    iron_component_orderings=(
                        {"A_gt_D_pct": "V(iron, stone pickaxe) > V(no iron, stone pickaxe)",
                         "A_gt_B_pct": "V(iron, stone pickaxe) > V(iron, no stone pickaxe)"}
                        if args.task == "iron" else None),
                    behavior_source=f"W&B {BEHAVIOR_METRICS[args.task]}, centered 1024-update rolling mean",
                    note="Fixed final-policy pre-collection state bank evaluated at each saved checkpoint; no new rollouts."
                    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {out_csv} and {out_dir / (args.task + '_emergence_panel.pdf')}")


if __name__ == "__main__":
    main()
