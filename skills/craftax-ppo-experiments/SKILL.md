---
name: craftax-ppo-experiments
description: Configure, validate, launch, resume, and inspect symbolic Craftax Classic PPO experiments in Craftax_Baselines using shared JSON presets, reward wrappers, architecture options, and wall-clock controls.
---

# Craftax PPO experiments

Work in `/home/shawheen/Craftax_Baselines` (or its current checkout). Use the
shared runner for common experiment changes instead of copying historical
launch scripts.

## Configure an experiment

Copy the closest preset under `experiments/ppo_presets/`, give the new study a
distinct `name`, and change the user's requested fields. `ppo_experiment_config.py`
is the configuration schema and command builder. Supported architectures are
`mlp`, `gru`, `spatial`, `spatial_gru`, `spatial_map_gru`, `map_mlp`, `map_mlp_gru`, and `neural_map_gru`. An omitted section uses the documented
defaults in that module; explicit config fields override them.

Common changes require only JSON:

- `rewards`: positive magnitudes for `death_penalty` and `lava_penalty`; total
  first-achievement rewards for `diamond_reward` and `plant_reward`.
- `seed`, `ppo`, and `architectures`: training settings and explicit
  `name`/`train_gpu`/`evaluation_gpu` entries for each selected architecture.
  Recurrent GRU and vanilla feedforward `map_mlp` can use `train_gpus` with
  2 or 4 devices instead of `train_gpu`. PPO environments, batch size, epochs,
  and minibatches remain global. Auxiliary/practice/replay variants of
  feedforward PPO remain single-GPU. See `experiments/README.md`.
- `ppo.policy_temperature`: strictly positive feedforward training temperature;
  applies consistently to sampling, behavior/current log-probabilities, and
  entropy. Default 1 leaves the original distribution unchanged. Evaluation
  temperature is separate: `evaluation.temperatures` can request `[1, 0.25]`.
- `spatial_map_gru` uses episode-persistent observed terrain and explicit actual
  odometry. `ppo.map_view_size` controls its agent-centered readout (odd >= 9;
  default 33). See `experiments/README.md`; no unseen terrain is supplied.
- `map_mlp` uses the same memory with a feedforward MLP, without CNN or GRU.
  `ppo.map_embedding_size` controls its learned categorical tile embedding
  (default 4); preserve this and the map view size in inference configs.
- `map_mlp_gru` embeds the observed terrain map and local observation with an
  MLP before recurrent memory. `gru_concat_observation` adds the direct path to
  the policy/value heads. It supports two/four GPUs. Optional `ppo.ctec_coef`
  enables temporal contrastive exploration with separate encoders, Adam state,
  episode-safe future pairs, reward warmup and late decay. See
  `experiments/README.md` for the task-reward adaptation and its exact schedule.
- `neural_map_gru` adds a PPO-trained 64×64 spatial memory to terrain-map +
  GRU. Set `ppo.neural_map_channels` (16 in the pilot). Gated writes touch only
  visible cells; local reads and attention over seen cells feed the GRU and
  heads. Preserve channels and `neural_map_reset_every_step` during inference.
  The latter erases only learned map history for ablation. See the neural-map
  section in `experiments/README.md` for capacity, truncation and continuation.
- `ppo.map_survival_features` selects `none` (legacy shape), `enabled`, or
  `masked` (capacity-matched control), for `map_mlp_gru`. The optional 40 inputs
  summarize observed-only global water routes, depletion duration, and action
  outcomes. Preserve this option in inference configs. The paired 500M presets
  use the same initial parameters and a 10B LR horizon for later continuation.
- `ppo.reset_gru_every_step: true` is a matched no-history GRU control. The
  network resets before every observation in collection, PPO replay and inference;
  parameter structure and seed initialization remain identical to the normal GRU.
- `ppo.gru_concat_observation: true` feeds `[current embedding, GRU output]`
  to both heads. GRU/embedding/head widths stay fixed; the wider head inputs add
  parameters. Preserve this flag in inference configs and use a fresh study;
  default false retains the original recurrent checkpoint shapes.
- `ppo.motion_inputs: true` appends `[one_hot(previous_action,17), dx, dy]`
  to the raw symbolic observation (1,345 → 1,364). Flat `gru` only; compatible
  with concatenation and data parallelism. Actual displacement uses simulator
  coordinate differences; all 19 inputs reset to zero on new episodes. Keep
  the flag in inference configs. Use `symbolic_motion_inputs.py` in collection,
  evaluation and videos, inside autoreset wrappers. This adds no terrain memory
  or SRU. Existing 1,345-input checkpoints require explicit migration; fresh
  initialization is the default for this experiment.
- `resume_from`: an explicit saved checkpoint directory. Resume retains weights
  and optimizer state; environment episodes and hidden state reset once.
  Use `--queue` to wait for the configured GPUs before smoke validation and launch.
- `resume_after`: source architecture directory containing `progress.json`.
  Use `--queue` to wait for its successful final checkpoint, smoke-test the
  resume and launch automatically. Set the cumulative target and original LR
  horizon explicitly; `queue_progress.json` records waiting, validation and launch.
- `resume_reset_optimizer: true` explicitly restarts Adam for feedforward
  fine-tuning. With no explicit LR horizon, annealing spans the additional
  budget; use a deliberate nonzero LR for a fully annealed source checkpoint.
- `ppo.reset_seed_bank`: optional JSON `world_seeds` bank for feedforward
  training. Seeds affect resets only; use standard resets for the memorization
  pilot. The bank contents are fingerprinted. Evaluator flags
  `--world-seeds-file` / `--repeats` reproduce worlds with fresh dynamics/actions.
- `budget`: `max_hours`, `evaluation_reserve_minutes`, and `max_steps`. The step
  ceiling determines the learning-rate horizon unless `ppo.lr_horizon_timesteps`
  is supplied. Preserve the original horizon when continuing for a shorter
  additional budget; a wall-clock cap can end training earlier.
- `checkpoint_milestones`: optional absolute step targets; defaults are logarithmic.
- `evaluation`: requested `common_checkpoints`, episode counts, batch, and seed.
  Optional `world_seeds_file` and `repeats` pair trials across policies; the
  final episode count must equal worlds times repeats. Shared evaluation GPUs
  are serialized by an exclusive controller lock.
  `defer_until_training_complete: true` evaluates saved intermediate checkpoints
  after training and permits reusing that job's training GPU for evaluation.

General death and lava-only costs are mutually exclusive. Default rewards are
zero extra costs and +1 for diamond/plant. Lava-only −2 means other deaths
retain their default rewards. A first-achievement bonus does not recur whenever
the item is collected again. Apply reward changes before automatic resets and
episode logging; keep the installed simulator intact.

## Prepare, validate, and run

Use the project's conda Python for JAX execution:

```bash
PY=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only.json --prepare
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only.json --smoke
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only.json --launch
```

`--prepare` is the default and starts no GPU jobs. It emits resolved configs and
commands. `--smoke` runs reward checks, two training updates at the configured
batch size, and checkpoint reload/evaluation. `--launch` requires validation
matching both the config and current source. Run it when the user requested
execution. Skill or config maintenance alone requires no training launch.

Each job uses a distinct training GPU and evaluation GPU. The runner checks
availability and refuses to overwrite a launched study. A request to replace
runs authorizes stopping the specified runs; otherwise preserve existing jobs.
Do not relaunch jobs merely to update the skill or inspect status.

## Runtime invariants

- `cap_reward_shaped_runs.py` owns process supervision, intermediate/final
  evaluations, and the outer deadline. `training_time_limit.py` lets trainers
  finish an update and save the actual completed step count.
- Checkpoints round requested milestones up to completed PPO updates. The final
  checkpoint label is actual progress, not the requested ceiling.
- Multi-GPU milestone callbacks must use the isolated CPU checkpoint writer in
  `ppo_data_parallel.py`; Orbax's nested GPU barriers can deadlock a callback.
  Checkpoints retain full Adam state and support one-/multi-GPU resume with
  fresh environment episodes. Keep a nonzero learning-rate horizon beyond a
  segment's endpoint when further continuation is intended.
- Recurrent minibatches retain temporal sequences and use factual hidden-state
  resets at episode boundaries. CNN input is the symbolic map, not rendered pixels.
- Map memory must wrap a no-auto-reset environment INSIDE log/reset/batch
  wrappers. It survives rollout boundaries and clears on episode resets.
  Evaluation/videos must use `with_map_memory`; their raw-state metrics unwrap
  `MapMemoryState`. Preserve the map view size in inference configs.
- Use explicit `inference_config.json` / `files/study_config.json` during live
  training: W&B may not yet have written `config.yaml`. Support checkpoint
  payloads both directly and inside a `default/` subdirectory.
- Distinguish original return from modified return, training curves from frozen
  evaluation, and per-achievement success from joint completion of all 22.
  Report resets, seeds, episode caps, and model capacity when comparing runs.

For code changes, resume/status work, and the existing module map, read
[references/operations.md](references/operations.md). Keep new mechanics in an
opt-in wrapper; add a model in `models/` and extend the architecture registry
when an existing option cannot express the request.

For MP4 requests, reuse `concept_mapping/render_ppo_paired_videos.py --study
<study-or-architecture-directory> --gpu <idle-physical-id> --episodes 3 --seed
18150`. This discovers the config and latest complete checkpoint, reserves the
GPU, saves clean/annotated copies, and validates the MP4s. Do not create another
per-run renderer or repeat external ffprobe boilerplate.
