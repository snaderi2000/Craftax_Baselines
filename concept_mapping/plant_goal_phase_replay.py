"""Paired phase ablation of the frozen offscreen-plant goal head."""
import csv
import json

import jax
import jax.numpy as jnp
import numpy as np


PHASES = {"never": (0, 0), "before": (1, 0), "after": (0, 1), "always": (1, 1)}


def evaluate_phases(root, bank, head, roll):
    out = root / "phase_replay"
    out.mkdir(exist_ok=True)
    records = json.loads((root / "test_bank.json").read_text())["records"]
    coords = jnp.array([r["target"] for r in records])
    offset = np.abs(np.asarray(coords - bank[1].env_state.player_position))
    assert len(set(r["world"] for r in records)) == len(records) == 100
    assert np.all((offset[:, 0] > 3) | (offset[:, 1] > 4))

    @jax.jit
    def replay(snaps, keys, targets, phase):
        seen, harvest, death, live = roll(snaps, keys, head, 0, targets, phase)
        def first(flags):
            return jnp.where(flags.any(0), flags.argmax(0) + 1, 0)
        return first(seen), first(harvest), death.any(0), live.sum(0)

    all_rows = []
    for name, phase in PHASES.items():
        rows = []
        source_ids = np.repeat(np.arange(len(records)), 4)
        for start in range(0, len(source_ids), 40):
            ids = source_ids[start:start+40]
            snaps = jax.tree_util.tree_map(lambda x: x[ids], bank)
            keys = jax.vmap(lambda n: jax.random.fold_in(jax.random.PRNGKey(182194), n))(
                jnp.arange(start, start+len(ids)))
            output = jax.device_get(replay(snaps, keys, coords[ids], jnp.asarray(phase)))
            for i, (seen, harvested, death, steps) in enumerate(zip(*output)):
                assert harvested == 0 or 0 < seen < harvested <= steps
                r = records[ids[i]]
                rows.append(dict(phase=name, world=r["world"], repeat=(start+i)%4,
                                 preparation=r["preparation"], sight_step=int(seen),
                                 harvest_step=int(harvested), death=int(death), steps=int(steps)))
        all_rows.extend(rows)
        print(name, sum(r["sight_step"] > 0 for r in rows), sum(r["harvest_step"] > 0 for r in rows), flush=True)

    with (out / "paired_trials.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    for arm, filename in (("never", "normal.csv"), ("always", "plant.csv")):
        with (root / filename).open() as f:
            reference = {(int(r["world"]), int(r["repeat"])): tuple(int(r[k]) for k in ("harvest", "death", "steps"))
                         for r in csv.DictReader(f)}
        actual = {(r["world"], r["repeat"]): (int(r["harvest_step"] > 0), r["death"], r["steps"])
                  for r in all_rows if r["phase"] == arm}
        assert actual == reference, f"{arm} failed original per-trial replication"
    times = {name: np.array([[r["sight_step"], r["harvest_step"]] for r in all_rows if r["phase"] == name])
             for name in PHASES}
    for x, y in (("never","after"), ("before","always")):
        np.testing.assert_array_equal(times[x][:, 0], times[y][:, 0])

    lines = ["# When the plant goal helps", "",
             "Each arm replayed the same four 64-step continuations in each of 100 held-out worlds. The scorer latched on the first living successor observation showing the exact ripe target tile. The after-sighting phase includes the first action chosen from that observation and continues if the target leaves view again. Target coordinates only entered the scorer and the external switch.", "",
             "| Head active | Target re-sighted | Target harvested | Harvested among sightings | Deaths |",
             "|---|---:|---:|---:|---:|"]
    for name, values in times.items():
        seen, hit = (values > 0).sum(axis=0)
        died = sum(r["death"] for r in all_rows if r["phase"] == name)
        lines.append(f"| {name} | {seen}/400 ({seen/4:.2f}%) | {hit}/400 ({hit/4:.2f}%) | {hit}/{seen} ({100*hit/seen:.2f}%) | {died}/400 |")
    sampled = np.random.default_rng(50).integers(100, size=(10000, 100))
    lines += ["", "Paired differences in percentage points. Each 95% percentile interval resamples whole worlds with their four trials (10,000 draws).", "",
              "| Comparison | Outcome | Difference | 95% CI |", "|---|---|---:|---:|"]
    for x, y, column in (("before","never",0), ("before","never",1), ("after","never",1),
                         ("always","before",1), ("always","after",1), ("always","never",1)):
        delta = 100*((times[x][:,column]>0).astype(float) - (times[y][:,column]>0)).reshape(100,4).mean(axis=1)
        lo, hi = np.percentile(delta[sampled].mean(axis=1), [2.5, 97.5])
        metric = ("Sight", "Harvest")[column]
        lines.append(f"| {x} minus {y} | {metric} | {delta.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "The never and always arms exactly reproduced harvest, death and active steps per trial from the original experiment. First-sighting times matched trial by trial between arms sharing the same pre-sighting policy. The panel contains 98 movement-detour worlds and two natural departures. All weights were frozen. Total simulator slots were 102,400.", "", "[Paired trials](paired_trials.csv)"]
    (out / "results.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)
