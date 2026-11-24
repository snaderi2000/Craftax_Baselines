"""Lock the delay-20 baseline a new advantage has to beat.

Three conditions, same PPO loop, seeds 0, 1, and 2:

- GAE λ=0.95 is the target. The metric is env steps until the mean of the
  last 100 episode returns reaches 475.
- GAE λ=1 should solve in a similar number of steps.
- One-step λ=0 should not solve. If it does, the delay is no longer testing
  credit assignment.

    conda activate craftax
    python advantage_experiments/run_baseline.py
"""

import json
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SEEDS = (0, 1, 2)
CONDITIONS = (
    {"name": "gae095", "gae_lambda": 0.95, "role": "target"},
    {"name": "gae100", "gae_lambda": 1.0, "role": "monte_carlo"},
    {"name": "gae000", "gae_lambda": 0.0, "role": "one_step"},
)
REWARD_DELAY = 20
TOTAL_TIMESTEPS = 500_000
SOLVE_THRESHOLD = 475.0


def _run_dir() -> Path:
    return Path(__file__).resolve().parent / "runs" / "baseline"


def _one(job):
    name, lam, seed, out_dir = job
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "ppo_delayed_cartpole.py"),
        "--reward-delay",
        str(REWARD_DELAY),
        "--gae-lambda",
        str(lam),
        "--seed",
        str(seed),
        "--total-timesteps",
        str(TOTAL_TIMESTEPS),
        "--log-interval",
        "250",
        "--out-dir",
        str(out_dir),
    ]
    log_path = out_dir / "train.log"
    with log_path.open("w") as handle:
        completed = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
    summary_path = out_dir / "summary.json"
    if completed.returncode != 0 or not summary_path.exists():
        raise RuntimeError(f"{name} seed {seed} failed. See {log_path}")
    summary = json.loads(summary_path.read_text())
    return {
        "name": name,
        "gae_lambda": lam,
        "seed": seed,
        "solved": bool(summary["solved"]),
        "env_steps_to_solve": summary.get("env_steps_to_solve"),
        "episodes_to_solve": summary.get("episodes_to_solve"),
        "wall_seconds_to_solve": summary.get("wall_seconds_to_solve"),
        "final_mean_return_100": summary["final_mean_return_100"],
        "total_env_steps": summary["total_env_steps"],
        "out_dir": str(out_dir),
    }


def _steps(rows):
    return [row["env_steps_to_solve"] for row in rows if row["solved"]]


def _summarize(rows):
    solved = [row for row in rows if row["solved"]]
    steps = _steps(rows)
    return {
        "solved_seeds": [row["seed"] for row in solved],
        "unsolved_seeds": [row["seed"] for row in rows if not row["solved"]],
        "steps_to_solve": {str(row["seed"]): row["env_steps_to_solve"] for row in rows},
        "final_mean_return_100": {
            str(row["seed"]): row["final_mean_return_100"] for row in rows
        },
        "median_steps_to_solve": statistics.median(steps) if steps else None,
        "mean_steps_to_solve": statistics.fmean(steps) if steps else None,
    }


def main():
    root = _run_dir()
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            out_dir = root / f"{condition['name']}_seed{seed}"
            jobs.append((condition["name"], condition["gae_lambda"], seed, out_dir))

    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            status = row["env_steps_to_solve"] if row["solved"] else "not solved"
            print(
                f"{row['name']} seed {row['seed']}: {status}  "
                f"final mean {row['final_mean_return_100']:.1f}",
                flush=True,
            )

    by_name = {condition["name"]: [] for condition in CONDITIONS}
    for row in rows:
        by_name[row["name"]].append(row)
    for group in by_name.values():
        group.sort(key=lambda row: row["seed"])

    target = _summarize(by_name["gae095"])
    monte_carlo = _summarize(by_name["gae100"])
    one_step = _summarize(by_name["gae000"])
    checks = {
        "gae095_solves_every_seed": len(target["unsolved_seeds"]) == 0,
        "one_step_solves_no_seed": len(one_step["solved_seeds"]) == 0,
        "monte_carlo_solves_every_seed": len(monte_carlo["unsolved_seeds"]) == 0,
    }
    report = {
        "metric": "env_steps_to_solve",
        "metric_definition": (
            "Environment steps until the mean return of the last 100 episodes "
            f"reaches {SOLVE_THRESHOLD:g}. Unsolved runs have no steps-to-solve."
        ),
        "reward_delay": REWARD_DELAY,
        "solve_threshold": SOLVE_THRESHOLD,
        "solve_window": 100,
        "seeds": list(SEEDS),
        "total_timesteps": TOTAL_TIMESTEPS,
        "number_to_beat": target["median_steps_to_solve"],
        "number_to_beat_is": "median env_steps_to_solve of GAE λ=0.95 across seeds",
        "checks_passed": all(checks.values()),
        "checks": checks,
        "gae095": target,
        "gae100": monte_carlo,
        "gae000": one_step,
        "runs": rows,
    }
    report_path = root / "baseline.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GAE λ=0.95 median steps to solve: {target['median_steps_to_solve']}")
    print(f"Checks passed: {report['checks_passed']}  {checks}")
    print(f"Wrote {report_path}")
    if not report["checks_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
