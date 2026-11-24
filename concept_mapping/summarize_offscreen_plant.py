"""Summarize the paired offscreen-plant pilot, counting uncertainty at world level."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "runs/sru_12p5b_offscreen_plant_goal"


def read_json(name):
    return json.loads((ROOT / name).read_text())


def outcomes(name):
    with (ROOT / f"{name}.csv").open() as f:
        rows = list(csv.DictReader(f))
    result = {(int(r["world"]), int(r["repeat"])): [int(r[k]) for k in ("harvest", "death")] for r in rows}
    assert len(result) == len(rows)
    return result


def main():
    test = read_json("test_bank.json")
    normal = outcomes("normal")
    plant = outcomes("plant") if (ROOT / "plant.csv").exists() else None
    if plant is None and read_json("normal.json")["harvest"] <= .9 * len(normal):
        raise RuntimeError("The normal baseline is below the ceiling and plant evaluation is pending.")
    if plant is not None:
        assert plant.keys() == normal.keys()
    keys = [(r["world"], i) for r in test["records"] for i in range(4)]
    assert set(keys) == normal.keys()
    result = {name: np.array([arm[k] for k in keys]).reshape(-1, 4, 2)
              for name, arm in (("normal", normal), ("plant", plant)) if arm is not None}
    natural = np.array([r["preparation"] == "natural" for r in test["records"]])
    lines = ["# Return to an out-of-view ripe plant", "",
             "Four paired 64-step continuations per world. Each harvest refers to the exact tile seen before departure.", "",
             "| Start | Worlds | Policy | Target harvested | Deaths |", "|---|---:|---|---:|---:|"]
    for group, mask in (("All", np.ones_like(natural)), ("Natural", natural), ("Movement detour", ~natural)):
        if not mask.any():
            continue
        for name, values in result.items():
            counts = values[mask].sum(axis=(0,1))
            n = 4 * mask.sum()
            lines.append(f"| {group} | {mask.sum()} | {name} | {counts[0]}/{n} ({100*counts[0]/n:.2f}%) | {counts[1]}/{n} ({100*counts[1]/n:.2f}%) |")
    slots = test["simulator_slots"] + read_json("normal.json")["simulator_slots"]
    if plant is not None:
        size = len(natural)
        ix = np.random.default_rng(50).integers(size, size=(10000, size))
        deltas = (result["plant"] - result["normal"]).mean(axis=1) * 100
        ci = np.percentile(deltas[ix].mean(axis=1), [2.5, 97.5], axis=0)
        lines += ["", "Paired differences in percentage points, with 95% world-bootstrap intervals.", ""]
        for i, name in enumerate(("Target harvest", "Death")):
            lines.append(f"- {name} {deltas[:, i].mean():+.2f} [{ci[0,i]:+.2f}, {ci[1,i]:+.2f}].")
        changed = result["plant"][..., 0] - result["normal"][..., 0]
        lines += ["", f"On paired continuations, only the plant goal harvested in {(changed == 1).sum()} trials and only the normal policy harvested in {(changed == -1).sum()}."]
        trained = read_json("training.json")
        slots += sum(read_json(path)["simulator_slots"] for path in
                     ("train_bank.json", "visible_sources/train_bank.json", "training.json", "plant.json"))
        lines += ["", f"The plant head trained for {trained['updates']} PPO updates ({trained['simulator_slots']:,} simulator slots) from {trained['training_worlds']} disjoint training worlds."]
    else:
        lines += ["", "The normal policy exceeded the 90% ceiling and the head was left untrained."]
    lines += ["", f"Total new simulator slots including preparation and evaluation: {slots:,}.",
              "", "The policy saw the ordinary observations, previous action, and displacement. The scorer and the scripted departure could use the target location. The comparison includes recognition and harvesting once the plant re-enters view. The drill does not isolate use of memory or measure whole-episode performance.",
              "", "[Protocol](protocol.md) · [Starts](test_bank.json) · [Normal trials](normal.csv)" + (" · [Plant-goal trials](plant.csv)" if plant else "")]
    (ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
