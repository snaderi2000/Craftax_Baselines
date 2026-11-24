# Auxiliary-objective search at a 1M interaction budget

Each candidate changes the prediction objective attached to the actor encoder
of ordinary Craftax-Classic PPO. Its network and optimizer start from scratch.
Fitness is mean original-environment evaluation return after **999,424** training
interactions: the largest complete 128×64 PPO rollout count below 1M. No weights
or optimizer states are inherited from a parent.

## Mutation

A recipe has three fields: task, horizon, and loss weight. A child changes exactly
one field. Parentage and the before/after change are saved before training.

- **Task:** switch to a different compatible target. Inverse dynamics requires
  horizon 1, so a horizon-4/16 parent switches between forward target families.
- **Horizon:** move to a neighboring value in `1 → 4 → 16`, or back.
- **Weight:** multiply by 0.5 or 2, staying within `[0.001, 1]`.

When 200 attempts encounter already-tested neighbors, the controller switches
to a small log-uniform weight perturbation between half and twice the parent's
weight, bounded by `[0.001, 1]`. This still changes only the weight. The operator
and actual factor are recorded. It prevents elite parents with exhausted discrete
neighbors from producing duplicate trials or stalling the search.

For example, `event / h=4 / λ=0.1` could become `event / h=16 / λ=0.1`,
`stats_delta / h=4 / λ=0.1`, or `event / h=4 / λ=0.2`.

Parents are selected by a three-candidate tournament on measured return. The
best eight parents/children survive each generation. Duplicate recipes are
excluded across both search branches. Separate RNGs determine auxiliary batches,
head initialization, mutation, random-search controls, and the fixed projection.
Mutation favors weight changes twice as often as either other eligible field.

## Targets and data

| Task | Input | Target |
|---|---|---|
| `inverse` | Actor features before and after | Executed action |
| `observation_delta` | Current actor features + first action | Fixed 32D random projection of `obs[t+h] - obs[t]` |
| `stats_delta` | Current actor features + first action | Changes in 12 inventory and 4 internal-level observation entries |
| `event` | Current actor features + first action | Any scalar reward exactly +1 within the next h transitions |

All tasks use the same 64-unit head and 256 auxiliary examples per PPO rollout,
split across eight minibatches and reused over the four PPO epochs. Regression
losses average over output dimensions. Targets use the agent's own on-policy
rollout, not the mature-policy clustering bank, simulator achievement flags, or
extra data collection. The projection seed is fixed across training seeds.

Targets are built before shuffling. Windows crossing termination/reset or the
rollout end are masked. Terminal transitions themselves are excluded because
the PPO `next_obs` field contains the reset observation there. For horizons >1,
predictions concern the outcome of the first action followed by the current
policy, rather than arbitrary future action sequences. Event sampling remains
uniform over valid windows; positive fraction is logged to reveal imbalance.

The policy and critic architecture, reward, resets, and PPO hyperparameters stay
fixed. Auxiliary gradients enter only the actor encoder and prediction head.
PPO KL, clipping fraction, auxiliary loss, valid fraction, event-positive fraction,
and weighted auxiliary/PPO gradient ratio are logged locally. Large auxiliary
gradients can still change the policy beyond what PPO clipping controls.

## Procedure and budget

1. Eight initial recipes, covering all four families, plus plain PPO. Two matched
   screening seeds (50/51) per recipe. Initial weights are sampled log-uniformly.
2. Four generations, each with eight mutated recipes and eight independent random
   controls. Both branches start from the same initial population and receive
   the same candidate and seed budgets.
3. Confirm three finalists (including the mutation and random-search winners)
   plus PPO on five fresh training seeds (150–154). Screening evaluates 64 episodes
   per agent; confirmation evaluates 256. Separate evaluation RNGs are used.

There are 166 planned 1M training allocations, including the screening baseline,
plus a 2M validation reserve and up to 20M evaluation interactions. The **190M
hard cap** is below the agreed 200M maximum. Every evaluation simulator step is
counted, including steps from environments that already filled their episode
quota. Training reservations stay charged after failures. Interrupted evaluations
retain their reservation until a complete result can reconcile the actual count.

Screening evaluations have a 250K per-agent interaction cap; confirmation has a
1M cap, always bounded by the remaining global evaluation reserve. Incomplete
evaluations never produce a fitness score. The controller stops on a failure,
budget exhaustion, or occupied GPU; it does not interrupt other jobs. Inspect
the stored logs before retrying a failed trial. A completed worker result can be
reconciled on resume even if the controller stopped before recording it.

This is a small search on one environment. Fresh-seed confirmation helps detect
selection overfitting; it does not establish reliable diamond performance or
generalization of the search method to other tasks.

## Run and inspect

From the repository root, with the existing JAX environment:

```bash
JAX_PLATFORMS=cpu /home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -m unittest test_auxiliary_search
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python run_auxiliary_search.py --prepare
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -u validate_auxiliary_search.py --gpu 0
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -u run_auxiliary_search.py --gpu 0 --sanity
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -u run_auxiliary_search.py --gpu 0 --max-hours 10 --background
```

For all eight GPUs, use `--gpus 0,1,2,3,4,5,6,7`. To change the already-running
study from one GPU to this pool, add `--take-over --background --max-hours 10`.
Handoff verifies the existing controller's process identity, stops only that
controller, and adopts its live worker. Completed trials and the original
deadline are retained. No trained agent is restarted for the handoff.

A single controller launches one independent trial per GPU, divides CPU affinity
between workers to bound JAX host thread pools, and serializes all reservations
and manifest writes. Candidate/seed jobs run in parallel within a stage; each
generation waits for every candidate's screening seeds before selecting parents.
The interaction budget and candidate/seed counts are independent of GPU count.

The default output is `concept_mapping/runs/auxiliary_search_1m`. Validation uses
eight 16,384-step real PPO runs, compares all eight actor/critic parameter modules
for baseline vs zero auxiliary weight within float32 GPU tolerance, repeats plain
PPO to measure baseline GPU variation independently, and tests checkpoint evaluation. A successful
validation artifact is required before the controller starts.

`study.json` contains recipes, mutations, processes, scores and the budget ledger.
`controller.log` reports progress; `leaderboard.json` compares search branches
after each generation; `results.json` holds fresh-seed confirmation. Each trial
has a recipe/command, training log, local checkpoint, scalar `metrics.jsonl`,
episode-level evaluation CSV and result JSON. No W&B or external logging is used.

`--sanity` runs one plain-PPO and one nonzero-auxiliary candidate through the full
999,424-step training and 64-episode evaluation pipeline. Their scores/checkpoints
are reused by the subsequent search; `sanity.json` records successful completion.

`--max-hours 10` persists a deadline at the start of the background search. Resuming
does not extend it. The controller stops starting agents when the remaining time
is too short to finish another full trial. Each worker also has a 30-minute timeout
and is bounded by the remaining session time. If a worker must be stopped, only its
dedicated process group is terminated; incomplete trials receive no fitness score
and keep their interaction reservation. Completed results stay available in
`study.json`, `progress.json`, and `leaderboard.json`. Mutation and random-control
trials are interleaved to limit compute imbalance when stopping mid-generation.

## Frozen-recipe replication

After the search completes, replicate the fixed `event / h=16 / weight=0.04210318`
recipe against plain PPO on eight fresh matched training seeds (250–257):

```bash
JAX_PLATFORMS=cpu /home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -m unittest test_auxiliary_search test_auxiliary_replication
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -u run_auxiliary_search.py --replicate --prepare
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -u run_auxiliary_search.py --replicate --gpus 0,1,2,3,4,5,6,7 --max-hours 1
```

The protocol is declared before training in
`concept_mapping/runs/auxiliary_search_1m/replication_event16_8seeds/protocol.json`.
It fixes the recipe, source fingerprints, all eight seeds, 256 evaluation episodes
per agent, matching evaluation seeds (4050–4057), and a two-sided 95% Student-t
interval over the eight auxiliary-minus-baseline seed differences. Earlier search
and confirmation scores never enter this estimate. There is no mutation or
performance-based early stopping. Every declared pair must complete for a result.

The replication includes all prior charged interactions in the original 190M
cap. Each new agent reserves 1M training interactions and at most 112,640 evaluation
steps; actual evaluation steps replace that reservation on completion. Failed or
incomplete evaluations retain their charge and produce no partial score.

The replication directory contains `results.json`, `per_seed.csv`,
`achievement_comparison.csv`, `paired_returns.png`, and `paired_returns.pdf`, plus
the individual trial checkpoints, episode CSVs, and logs. Achievement rates are
diagnostics; the predeclared primary result is original-environment return.
