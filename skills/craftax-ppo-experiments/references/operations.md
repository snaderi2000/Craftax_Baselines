# Repository map and operations

## Existing components

| Purpose | Source |
| --- | --- |
| Experiment schema, defaults, validation, command builder | `ppo_experiment_config.py` |
| Prepare / smoke / launch from JSON | `run_ppo_study.py` |
| Presets for default, lava-only, and diamond/plant bonuses | `experiments/ppo_presets/` |
| Feedforward PPO / GRU PPO | `ppo.py`, `ppo_rnn_og.py` |
| Global-array 1/2/4-GPU PPO and CPU checkpoint writer | `ppo_data_parallel.py` |
| Matched warmed PPO scaling measurements | `benchmark_ppo_data_parallel.py` |
| Reward wrappers | `wrappers.py` |
| Symbolic spatial CNN and reusable encoder | `models/conditioned_symbolic.py` |
| Observed terrain memory / explicit odometry | `symbolic_map_memory.py` |
| Previous-action and measured-displacement observations | `symbolic_motion_inputs.py`, `test_symbolic_motion_inputs.py` |
| Feedforward map reader without CNN/GRU | `models/actor_critic_map_mlp.py`, `test_map_mlp.py` |
| Fixed-world reset bank / memorization analysis | `fixed_world_seeds.py`, `analyze_failed_world_memorization.py` |
| Wall-clock training loop | `training_time_limit.py` |
| Supervisor, deadline, adoption, final evaluation | `cap_reward_shaped_runs.py` |
| Shared inference config, checkpoint detection, summaries | `run_reward_shaped_architectures.py` |
| Configurable two-update smoke run | `smoke_reward_shaped_ppo.py` |
| Frozen categorical evaluation | `evaluate_reward_shaped_ppo.py` |
| Frozen action-temperature sweep / confirmation | `run_policy_randomness_sweep.py`, `experiments/evaluation_presets/` |
| PPO training temperature / continuation comparisons | `policy_temperature.py`, `analyze_temperature_finetuning.py` |
| Matched clean/annotated videos (MLP, GRU, CNN, CNN + GRU) | `concept_mapping/render_ppo_paired_videos.py` |
| Config consistency tests | `test_ppo_experiment_config.py` |
| Recurrent / feedforward multi-device math and resume checks | `test_ppo_data_parallel.py`, `test_ppo_feedforward_data_parallel.py` |
| Real reward mechanics / recurrent reset tests | `test_reward_shaped_spatial.py`, `test_rnn_death_penalty.py` |
| Observed-only memory / collision / reset / gradient checks | `test_symbolic_map_memory.py` |

Historical launchers remain reproducibility records. Prefer the shared JSON
runner for new studies. The default `layer_size` controls the flat MLP and GRU;
the existing CNN has its own 64 channels, three residual blocks, and 256-wide
heads. CNN + GRU uses the spatial encoder before the configurable GRU. Parameter
counts come from smoke output; architectures differ in sharing and capacity.

`spatial_map_gru` adds deterministic terrain memory and a learned CNN reader
before the GRU. The backing map remembers all observed locations, but the
network sees a configurable agent-centered crop (33 by default). This variant
uses privileged actual displacement from simulator coordinates; never copy
unseen simulator terrain into memory. Keep the wrapper inside autoresets so
memory survives rollout boundaries and clears with episode boundaries.
`experiments/ppo_presets/lava_only_spatial_map_gru.json` configures this variant.

## Editing and validation

For existing reward/seed/budget/GPU options, edit only a preset. Add a new wrapper
or model only for behavior outside the schema. New reward mechanics need a real
environment transition check (including first vs repeated achievement, death,
timeout, and unchanged successor state where relevant). Then exercise the actual
trainer and reload its checkpoint. Avoid rerunning unrelated historical studies.

CPU-only checks and preparation:

```bash
python -m unittest -v test_ppo_experiment_config
python run_ppo_study.py --config experiments/ppo_presets/lava_only.json --prepare --out /tmp/craftax-plan
```

The full smoke command requires available GPUs. It validates the current config
and source fingerprints. `--smoke-gpu N` can validate on an idle device without
changing the planned training GPU; the validation artifact records the device.
For a multi-GPU study use `--smoke-gpus 0,3,4,6`, with the same device count as
the planned group. Reserve all validation devices against concurrent jobs.
Training controllers hold locks for their whole GPU group; shared evaluation
GPUs are serialized separately. Feedforward map-MLP and recurrent PPO share the
global-array helper while retaining their own minibatch semantics.

Changing the config or source invalidates the cached validation;
failed attempts retain their logs. No tests or skill updates imply permission
to start a long training run.

## Inspect a live study

`concept_mapping/runs/README.md` indexes month browsing folders (Pacific
time; recorded start date, otherwise filesystem creation date). Those
folders contain relative shortcuts; the flat legacy paths remain canonical so
active jobs and saved references keep working. Use canonical paths in presets.
Refresh the browser and process snapshot from the server terminal with
`python concept_mapping/organize_runs.py`. Its `--undo` removes only generated
views and restores their Explorer filters; it does not delete experiment data.

Start with `<study>/<architecture>/progress.json`, `run_manifest.json`,
`experiment_metadata.json`, and the latest training/worker log lines. Evaluate
checkpoints only after their completed payload appears. Follow `checkpoint_roots`
when a resumed W&B run has multiple local folders. Estimate steady SPS from
several recent completed checkpoint timestamps, excluding initial compilation.
Treat projected step counts under a time cap as estimates.

Controllers and trainers have separate PIDs. Host PID checks and GPU access may
need an escalated tool execution in this environment. The supervisor uses `/proc`
and PID start signatures; no `psutil` install is needed. Its `--adopt-pid` option
can reattach to an already running matching trainer without restarting it.

When resuming is authorized, retain optimizer state and the intended LR horizon,
record the exact source checkpoint, and record that environment episodes/GRU
hidden state reset. Preserve earlier checkpoints. Add the resumed local run to
`checkpoint_roots`; the W&B run ID can remain the same. Verify the supervisor and
trainer both remain alive and that progress advances after a repair.

The shared JSON runner supports `resume_from` directly. Set `budget.max_steps`
to the source step plus the requested additional interactions, and set
`ppo.lr_horizon_timesteps` to the original annealing horizon. The smoke test
restores the real source optimizer, performs two new updates, and reloads the
result. A continuation gets a new W&B run linked to the original checkpoint;
the previous study remains an immutable comparison. An example is
`experiments/ppo_presets/lava_only_spatial_gru_continue500m.json`. Milestones
are absolute step targets, including any custom `checkpoint_milestones`.

For a source that is still training, use `resume_after` instead of `resume_from`
and run `run_ppo_study.py --config <preset> --queue`. It names the source
architecture directory, not a checkpoint. The detached worker waits for source
completion, resolves its actual final save, checks GPU availability, and runs
smoke validation before launch. It preserves Adam and rejects optimizer restart.
Inspect `queue_progress.json`, `queue_worker.log`, `queue_smoke.log` and
`resolved_continuation.json`. A failed source never silently resumes an older
milestone. Set enough continuation time if the user has lifted a previous cap.

## Result interpretation

For GRU memory ablations, `ppo.reset_gru_every_step` changes resets inside
`ActorCriticRNN`, covering rollout, PPO replay, bootstrap and inference with
the same flag. It does not change actual episode boundaries or GAE done flags.
`test_gru_memory_control.py` checks identical initialization, streamed/replayed
likelihoods and gradients through history. `validate_gru_delayed_cue.py` uses
the real recurrent PPO trainer with only its environment factory substituted;
the normal GRU must solve the delayed-cue task while the no-history control
stays at chance. This positive control does not establish Craftax usefulness.
`ppo.gru_concat_observation` adds a current-embedding bypass into both heads;
streamed/replayed likelihoods, unchanged recurrent dynamics, and the bypass
with zero recurrent weights are checked by the same test module. At width 512,
concatenation adds 524,288 head parameters, so report the capacity difference.
Deferred evaluation can reuse each trainer's GPU after its training lock is
released; all requested intermediate checkpoints are then evaluated in order.

For frozen action-selection comparisons, evaluator `--temperature` defaults to
1 (unchanged categorical sampling); 0 is greedy. Use the JSON evaluation sweep
for paired fixed worlds, world-bootstrap intervals, and independent validation
winner confirmation. Do not report best-of-repeated-attempt success as the
probability of completing all 22 in one episode. `test_policy_temperature.py`
checks baseline equivalence, greedy behavior, and concentration at lower T.

For training, `ppo.policy_temperature` is feedforward-only and must be > 0.
`test_training_temperature.py` checks likelihood-ratio consistency, entropy,
finite gradients, and exact T=1 behavior. Evaluation does not automatically
inherit training temperature; request both values explicitly in
`evaluation.temperatures`. Multiple final outputs have temperature suffixes.
`test_evaluation_scheduling.py` checks temperature scheduling and exclusive
evaluation GPU locks, allowing separate studies to queue on the same evaluator.

Default evaluation uses original maps and independent standard resets. Report
all 22 achievement rates, all-22 completion, return, episode lengths, lava,
non-lava death, and timeout rates. Costs are extra terms on top of the simulator's
health reward. Reduced lava entry can coexist with more deaths elsewhere.

The historical baselines sometimes used optimistic resets; record this rather
than pooling unmatched controls. Compare common checkpoints for sample
efficiency and actual stopping checkpoints for wall-clock efficiency. Episode
confidence intervals do not measure variability across training seeds. Observed
100% on a finite evaluation sample is not a guarantee over all possible maps.

## Render example episodes

Preferred shortcut (no per-run launcher or manual checkpoint/config lookup):

```bash
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python concept_mapping/render_ppo_paired_videos.py --study concept_mapping/runs/STUDY --gpu 1 --episodes 3 --seed 18150
```

The same script reserves the selected physical GPU before JAX imports, resolves
the latest completed checkpoint across resumed run roots, reads inference config,
names the output folder, renders both versions and validates frame counts and
durations with ffprobe. For multi-architecture studies, point `--study` at the
architecture subdirectory. Explicit `--checkpoint` and `--out` remain available.

For fresh episodes or recorded evaluation replays, add `--whole-map` to show the entire world beside the
local view and inventory. Add `--frame-stride 4 --fps 20` for playback at
80 game steps per second. The rollout still executes and records every action.

To replay recorded evaluation failures, add `--evaluation-dir <evaluation-folder>`
and `--episode-ids 13,29,50`. Set `--evaluation-batch` to the original evaluator's
batch size (default 25). This uses the recorded checkpoint, temperature, world
seeds and rollout streams. It verifies selected outcomes against `episodes.csv`
before encoding, records any differences in non-selected batch lanes, retains
compressed traces, and saves `replay_validation.json`. Use `--resume` with the
same settings and `--out` to finish a partially rendered recorded selection.

Reuse `concept_mapping/render_ppo_paired_videos.py` with `--run-path`,
`--config-json <study>/<architecture>/inference_config.json`, `--out`,
`--episodes`, and `--seed`. Omit `--checkpoint` for the latest saved weights,
or pass the final step from the manifest. It detects GRU/CNN configurations,
applies the recorded rewards, and saves clean/annotated MP4s with identical
frames, summaries, and action/value CSVs. Use the same seed for matched starts
across policies. `--max-steps 10000 --fps 20` renders complete episodes at the
usual playback speed. Verify frame counts/durations with ffprobe and matched
initial-observation hashes in the summaries.
