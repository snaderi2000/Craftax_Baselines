# PPO experiment presets

Use `run_ppo_study.py` for new Classic symbolic experiments. Copy a preset,
give it a new `name`, and edit the requested settings. Existing experiment
launchers remain as historical records.

```bash
PY=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only.json --prepare
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only.json --smoke
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only.json --launch
```

Preparation starts no GPU processes. Smoke tests run two configured PPO updates
and reload/evaluate the weights. Launch requires matching config/source
validation. Live studies cannot be overwritten.

A single-model variation can be as small as:

```json
{
  "name": "lava_only_gru_seed51",
  "seed": 51,
  "architectures": [
    {"name": "gru", "train_gpu": 0, "evaluation_gpu": 4}
  ],
  "rewards": {"lava_penalty": 2},
  "budget": {"max_hours": 9}
}
```

Omitted fields use the defaults in [the schema](../ppo_experiment_config.py).
Architectures are `mlp`, `gru`, `spatial`, `spatial_gru`, and `spatial_map_gru`. Each selected model
needs a distinct training and evaluation GPU. Use separate study names/configs
for training seeds.

| Section | Frequently edited fields |
| --- | --- |
| `rewards` | `death_penalty`, `lava_penalty`, `diamond_reward`, `plant_reward` |
| `ppo` | `num_envs`, `num_steps`, `update_epochs`, `num_minibatches`, `gamma`, `gae_lambda`, `lr`, `ent_coef`, `layer_size`, `optimistic_resets` |
| `budget` | `max_hours`, `evaluation_reserve_minutes`, `max_steps` |
| `evaluation` | `common_checkpoints`, `common_episodes`, `final_episodes`, `batch`, `seed` |

Costs are positive magnitudes subtracted from default reward. Diamond/plant
values are total rewards for their first achievement. General death and
lava-only costs cannot both be enabled. `max_steps` is the interaction ceiling
and default LR annealing horizon; `ppo.lr_horizon_timesteps` can preserve an
earlier schedule when the stopping target changes. A time cap can stop earlier. Checkpoint milestones
round up to completed updates, and the final checkpoint records actual steps.

For shorter step budgets, set evaluation targets within that budget, e.g.
`"common_checkpoints": {"1m": 1000000}`. Evaluation counts must be divisible by
the evaluation batch. Unknown keys fail early instead of silently being ignored.

For continuation, set `resume_from` to a saved checkpoint directory and
`budget.max_steps` to the desired cumulative total. Set
`ppo.lr_horizon_timesteps` to the original schedule horizon to retain its decay.
The runner restores the optimizer, starts fresh episodes/hidden state, and
records a separate continuation run. Custom `checkpoint_milestones` are
absolute interaction counts. See `ppo_presets/lava_only_spatial_gru_continue500m.json`.

For deliberate memorization of particular original worlds, set
`ppo.reset_seed_bank` to a JSON file containing `{"world_seeds": [123, 456]}`.
Feedforward training samples these generator seeds uniformly on every reset;
step randomness stays fresh. No seed ID is exposed to the actor. The bank
contents are included in the validation fingerprint. Standard independent
resets (`ppo.optimistic_resets: false`) avoid optimistic reset duplication.
The frozen evaluator accepts `--world-seeds-file` and `--repeats` to test the
same worlds with independent action/dynamics streams.

A fully annealed checkpoint needs a nonzero fine-tuning schedule. For an
explicit optimizer restart, set `resume_reset_optimizer: true`, choose a lower
`ppo.lr`, and omit `ppo.lr_horizon_timesteps`: actor/critic weights are restored,
Adam restarts, and LR anneals over the **additional** training budget. This is
opt-in and currently supported for feedforward models. Ordinary resumes retain
the optimizer. `analyze_failed_world_memorization.py` prepares the selection and
before/after evaluations while `run_ppo_study.py` remains the training launcher.

The [Craftax PPO skill](../skills/craftax-ppo-experiments/SKILL.md) documents the
runtime workflow, shared modules, status checks, and resume behavior.

## Multiple training GPUs for one policy

Recurrent GRU and vanilla feedforward `map_mlp` studies can opt into 2 or 4 training GPUs with a `train_gpus`
list in their architecture entry. The existing `train_gpu` field remains valid
for single-GPU studies. When a list is supplied, its first GPU is the primary
device; `train_gpu` can be omitted or must agree with that first entry.

```json
{
  "name": "spatial_map_gru",
  "train_gpus": [5, 6],
  "evaluation_gpu": 3
}
```

`ppo.num_envs` remains the **global** environment count. With 1,024 environments,
two GPUs handle 512 each, while the 64-step rollout still contains 65,536
transitions. Keep the original number of PPO epochs and global minibatches.
Each global recurrent minibatch must divide evenly across the selected GPUs.
Feedforward `map_mlp` uses shuffled individual transitions and divides each
global flat minibatch across GPUs. Four GPUs handle 256 environments each at
the default settings. Additional auxiliary losses, practice/replay features,
component transfer, fixed-world banks, and optimizer restart remain single-GPU
options for feedforward PPO.
The network, global reset pool, advantage normalization, gradient clipping,
Adam state, and learning-rate horizon retain their global semantics. JAX
partitions array work across the devices. Each environment's entire recurrent
sequence and observed terrain memory stay together.

Use the same shared preparation, smoke, and launch commands. To validate on a
different idle pair, pass `--smoke-gpus 5,6`; its length must match the planned
training GPU count. The smoke gate includes device-count math checks,
two full-size PPO updates, full optimizer-checkpoint roundtrip, and the existing
frozen evaluator. Single-device and multi-device checkpoints use the same
parameter/optimizer payload and can resume with fresh environment episodes.
GPU IDs within a study must be unique, including the separate evaluation GPU.
Multi-GPU milestone callbacks hand their complete host-array state to an
isolated CPU checkpoint writer. This avoids Orbax launching nested GPU work
from inside a callback; writes remain atomic and finish before the callback
returns. Final saves wait for all callbacks to finish.

`ppo_presets/lava_only_spatial_map_gru_dp2_validation.json` is a bounded
1,048,576-step continuation for exercising this integration. Hardware scaling
and sustained learning must be measured before replacing an existing run.

Measure warmed end-to-end PPO updates on the same checkpoint and global
training settings with the shared benchmark tool:

```bash
$PY benchmark_ppo_data_parallel.py \
  --config experiments/ppo_presets/lava_only_spatial_map_gru_dp2_validation.json \
  --gpus 5,6 --device-counts 1,2 \
  --out /tmp/craftax-dp-scaling
```

This reserves the selected devices for the benchmark, performs an initial
compilation/warmup for each device count, synchronizes every measurement,
and records each timing and the speedup. It disables W&B logging and policy
checkpoint output for timing, and never modifies the source checkpoint.
Compilation time and a short throughput result do not establish sustained
learning quality. Use a fresh output directory for each comparison.

For the 5,379,954-parameter `spatial_map_gru` policy on an NVLink-connected
pair of V100-SXM2-16GB GPUs, the matched benchmark measured **15,523
SPS on one GPU and 24,861 SPS on two GPUs (1.60×)**. Each case used the same
330,039,296-step checkpoint, global batch of 65,536, four PPO epochs, and eight
global minibatches; timing covered two warmed four-update repetitions. The
raw timings and source hashes are in
`concept_mapping/runs/lava_only_spatial_map_gru_dp2_validation_seed50_v2/scaling_benchmark/comparison.json`.
This is a short throughput measurement; checkpoint, logging, and evaluation
overhead reduce practical throughput.
The isolated 16-update continuation completed all 1,048,576 additional steps,
saved each milestone once, and passed a 32-episode frozen evaluation. All
checkpoint optimizer counters matched the global updates, and the final
two-GPU checkpoint resumed Adam successfully for two further updates on one
GPU. `integration_validation.json` in that study directory records the checks.

A subsequent matched 1/2/4-GPU benchmark on GPUs 5, 6, 4, and 7 measured:

| GPUs | Warmed SPS | Speedup over one GPU | Projected hours per 1B steps |
| ---: | ---: | ---: | ---: |
| 1 | 15,523 | 1.00× | 17.9 |
| 2 | 24,826 | 1.60× | 11.2 |
| 4 | 37,968 | 2.45× | 7.3 |

Four GPUs were **1.53× faster than two** at the same global batch and PPO
settings, with 256 environments per GPU. The four-device benchmark verified
replicated parameters, four observation shards, finite parameters/optimizer,
and completed update counts. It required no additional trainer changes.
Raw timings and source hashes are in that study's
`scaling_benchmark_4gpu/comparison.json`. These projections exclude compilation,
checkpoint, logging, and evaluation overhead. The checkpoint/evaluation pilot
above used two GPUs; this four-GPU test measures training throughput.

`ppo_presets/lava_only_map_mlp_dp4_5b.json` starts a fresh seed-50 map-MLP
policy on four GPUs with the same global 1,024 environments, 64-step rollout,
four epochs, and eight minibatches. Its target is 5,000,003,584 interactions
(5B rounded up to a complete update), with a 48-hour outer cap. It retains a
10B learning-rate horizon, full optimizer checkpoints, regular milestones,
and independent evaluations at both temperatures 1 and 0.25.

The matched map-MLP benchmark on its assigned GPUs measured 34,489 SPS on one,
54,388 on two, and 62,765 on four (1.82× over one). This projects to about
22.1 hours for 5B before checkpoint, logging, compilation, and evaluation
overhead. Evidence is in
`concept_mapping/runs/lava_only_map_mlp_dp4_5b_seed50/scaling_benchmark/comparison.json`.
GPU assignments preserve the existing runs' training and evaluation devices;
the evaluation GPU is shared through the controller lock.

To explicitly schedule further training after this segment, use:

```bash
$PY run_ppo_study.py \
  --config experiments/ppo_presets/lava_only_map_mlp_dp4_continue_to10b.json \
  --queue
```

The optional preset waits for the successful 5B final checkpoint, restores
the complete optimizer, and validates before starting a new continuation to
10B total steps. It is not queued automatically. Environment episodes start
fresh at a segment boundary; model parameters and optimizer counters resume.

`ppo_presets/lava_only_map_mlp_dp4_continue2b_lr5e5.json` continues the final
10B map-MLP checkpoint for 2,000,027,648 additional transitions, preserving
Adam and schedule counters. To restore useful learning after the original
schedule is exhausted, it deliberately changes the LR base to 1e-4 and the
absolute schedule horizon to 20B. With the restored counter this gives an
**effective LR of 5e-5 at restart, declining to 4e-5 at the new endpoint**;
it does not restart Adam. The target is 11,999,969,280 global steps, with a
10-hour cap including a 30-minute final-evaluation reserve.

The continuation evaluates the exact original 100 worlds/action seeds every
250M steps, tracking both all-22 completion and survival to 10K. Its goal is
at least 96 all-22 successes out of 100. `monitor_ppo_continuation.py` selects
the best paired checkpoint (retaining the original baseline as a fallback)
and evaluates it plus the final checkpoint on a separate set of 100 verified
distinct worlds. The monitoring data and source audit live under
`concept_mapping/runs/lava_only_map_mlp_dp4_continue2b_lr5e5_seed50/`.

After that continuation regressed, the user selected the original 10B
checkpoint for the lower-rate comparison in
`ppo_presets/lava_only_map_mlp_dp4_continue2b_lr5e6.json`. It repeats the same
2B-step budget from that checkpoint, preserving Adam. The effective rate is
5e-6 at restart and about 1e-6 at the endpoint, implemented with a 2.5e-5
schedule base and a 12.5B absolute horizon. On the original 2e-4-to-zero
schedule, those rates correspond to approximately 9.75B and 9.95B steps.
Paired evaluations additionally run at +50M and +100M; all training still
uses the ordinary procedural reset distribution rather than the evaluation
seed bank. A new independent 100-world set is retained for confirmation.

## Previous action and displacement for GRU-concat

`ppo_presets/lava_only_gru_concat_motion_dp2_10b.json` prepares fresh seed-50
training for 10B interactions using GPUs 5 and 7 (evaluation GPU 1). It retains
the existing GRU-concat PPO settings and lava-only −2 cost. No general death
penalty, achievement bonus, terrain memory, CNN or SRU is added.

Set `ppo.motion_inputs: true` to append a 17-way previous-action one-hot and
actual `(dx,dy)` to each 1,345-element observation, producing 1,364 entries.
Simulator `(row,column)` differences are converted to `(dx,dy)`; a blocked
move reports the attempted action and zero displacement. At reset all 19
entries are zero. The wrapper uses no additional random draws, does not change
rewards or transitions, and belongs inside logging/autoreset/batching.
`symbolic_motion_inputs.py` is shared by training, frozen evaluation and the
paired-video renderer. Existing configurations leave this option disabled.

The encoder remains 512 wide and the GRU retains 512 hidden units. Both heads
receive `[current_embedding, hidden]` (1,024 entries). Additional parameters:
19×512 = 9,728; total 3,857,938. Existing raw-observation checkpoints have a
different first-layer shape, so this preset uses fresh initialization.

The global rollout stays 1,024 environments × 64 steps = 65,536 interactions,
split across two devices (512 environments each). Checkpoints are logarithmic;
frozen evaluations use 1,024 episodes at 100M, 1B, 2.5B, 5B, 7.5B and the final
checkpoint, at temperatures 1 and 0.25 with seed 28050. The target rounds down
to 9,999,941,632 completed interactions. The 96-hour outer budget includes a
45-minute final-evaluation reserve. Prepare/smoke do not launch the long run:

```bash
PY=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only_gru_concat_motion_dp2_10b.json --prepare
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only_gru_concat_motion_dp2_10b.json --smoke
$PY run_ppo_study.py --config experiments/ppo_presets/lava_only_gru_concat_motion_dp2_10b.json --launch
```

Compare against `lava_only_gru_concat_dp2_10b_seed50` at matched checkpoints;
its initial 15M interactions used one GPU before migration. This is a joint
previous-action/odometry ablation with one training seed, not an SRU study or
a claim isolating which of the two new inputs is responsible for any gain.

### Spatially enhanced GRU continuation

The [SRU-GRU paper](https://arxiv.org/html/2506.05997v2#S4.SS4) and its
[PyTorch cell](https://github.com/leggedrobotics/sru-pytorch-spatial-learning/blob/main/network/gru_sru.py)
use an affine projection of the current embedding to scale the GRU's candidate
preactivation before `tanh`. Our Flax port keeps its stock GRU reset convention
and multiplies the same preactivation by `W x + b`. The original motion-enabled
GRU-concat checkpoint supplies the encoder, recurrent weights, actor, and
critic. `W = 0` and `b = 1` preserve the starting outputs. The current embedding
already receives the previous action and actual displacement.

`ppo_presets/sru_gru_motion_continue200m.json` starts at 9,999,941,632 and
targets 10,199,957,504 interactions. Adam restarts at the initial transfer.
For the first 763 PPO updates (50,003,968 additional steps), only the new gate
and its optimizer moments change. After that update, the same run unfreezes
the encoder, original GRU, actor, and critic. A later SRU resume restores both
optimizer partitions and the warmup counter. The constant rates are 3e-5 for
the new gate and 3e-6 for the pretrained parameters. A 0.01 penalty on the
gate's squared distance from identity anchors the transfer. The paper's
navigation training uses mutual distillation between two policies and
temporally consistent dropout. This continuation uses a single PPO policy
with the identity penalty and staged unfreezing.

The full-size smoke test checks output parity with the original checkpoint,
exactly frozen pretrained weights during warmup, and updates to the encoder,
GRU, actor, and critic after a shortened warmup in a separate test. Frozen
evaluations at warmup completion, +100M and +200M use the original ablation's
100 worlds, rollout seed, and temperature 1. The frozen baseline for that
bank completed all 22 on 75 worlds and collected diamond on 84.

`ppo_presets/sru_gru_motion_to12p5b.json` resumes the 10.60B checkpoint and
stops at 12,500,008,960 interactions, rounded to a full PPO rollout. The
10.5-hour training cap allows for throughput variation around the roughly
10-hour estimate, and 30 minutes are reserved for frozen evaluation.
`sru_lr_anneal_from` and `sru_lr_anneal_until` are absolute rollout-aligned
steps. They linearly scale both Adam parameter updates to
`sru_lr_final_fraction` by 12.50B while preserving the saved moments and phase
counter. The schedule starts at 3e-6 for pretrained parameters and 3e-5 for the
SRU and keeps a 10% floor. The run evaluates every 400M steps and at the final
save on the same 100 worlds.

## Observed terrain memory with explicit odometry

`ppo_presets/lava_only_spatial_map_gru.json` configures CNN + GRU with an
episode-persistent terrain map. This preset is a ready-to-run configuration,
not a launched study. The installed simulator and historical configurations
are unchanged. Select `spatial_map_gru` and set `ppo.map_view_size` to an odd
integer >= 9 (default 33). CLI equivalent: `ppo_rnn_og.py --symbolic_encoder
spatial_map --map_view_size 33` plus the usual PPO flags.

The memory wrapper (`symbolic_map_memory.py`) writes only the 17 terrain
channels from the existing 7×9 observation into an initially unknown map. It
uses differences of simulator player positions as **privileged actual
odometry**: collisions give zero displacement. No unseen simulator terrain,
achievement flags, absolute coordinates, or future states enter the network.
The backing array uses the configured world dimensions and retains all
observed locations until episode reset. The policy receives an agent-centered
`map_view_size` crop, not the whole backing map; more distant memories become
available again on approach. Out-of-view stored terrain is a last observation,
not a prediction of its current state. A log-scaled observation-age channel
marks staleness; revisiting refreshes changed tiles.

The flat observation is the original 1,345 entries, followed by the remembered
terrain codes (0 unknown, tile ID + 1 otherwise), then normalized ages. At size
33 it has 3,523 entries. Moving mobs/arrows remain in the current local input;
their old positions are not persisted as if they were static terrain.
`models/conditioned_symbolic.py` encodes this extra view with three strided
convolutions and combines it with the existing local CNN before the GRU.
Writes are deterministic; PPO learns the map reader and actor/critic. No
auxiliary labels or map-learning loss are introduced.

The wrapper sits **inside** log/autoreset/batch wrappers. Memory survives PPO
rollout and optimizer-update boundaries; standard and optimistic resets clear
it for each finished environment. The evaluator and paired-video renderer use
the same wrapper. Existing checkpoints without map inputs cannot be resumed
directly into this larger network; initialize a new map agent. The nine-hour
cap, rewards, checkpoint schedule, and evaluation metrics remain configurable
through the shared runner.

### Map memory with a feedforward reader

`map_mlp` reuses the exact observed-terrain memory/odometry wrapper and removes
both the CNN and GRU. It reads the original 1,345 observation entries, a
flattened learned embedding of each remembered terrain code, and flattened
observation ages. Unknown terrain has its own categorical embedding. With a
33×33 view and `ppo.map_embedding_size: 4`, the actor/critic input has 6,790
features. Separate three-layer 512-unit tanh MLPs produce 17 action scores and
one value; the embedding table is shared. Total: **8,013,914 parameters**. There
are no convolutional layers or recurrent hidden states. The external map is
the only persistent memory, and its episode-reset/rollout semantics are unchanged.

For a crop-plus-global checkpoint, enable `map_global_readout`,
`map_single_readout`, and `map_internal_crop` together to use the same weights
with a 9,539-value observation. The environment supplies the 1,345 live values,
the 64×64 remembered terrain and age grids, and two position coordinates. The
policy takes its trained neighborhood from those grids. The
`one_observed_map_12p2b_finetune_100m` preset uses this input for both PPO
and frozen evaluation.

The crop-free `spatial_regions_12p2b_100m` preset reads that same observation.
Each tile supplies a learned four-value terrain embedding, age, a player marker
and two normalized offsets from the player. Two strided convolutions cover
non-overlapping 8×8 tile regions. The actor and critic read all 8×8×32 regional
features in world order alongside the 1,345 live values, for 3,393 inputs apiece.
The shared map writer uses observed terrain and actual displacement.

Run `experiments/distill_single_map.py --spatial-regions` with the 12.2B
crop-plus-global study as `--teacher` and this preset's `warm_start` directory
as `--out`. The first terrain convolution, terrain vocabulary and live/decision
weights transfer. The regional projection learns by imitating the teacher's
action distribution and value on new worlds. The preset starts PPO with fresh
Adam moments and records separate 1,024-world evaluations after imitation and
after the 100M additional PPO steps. The 12.2B checkpoint label denotes teacher
lineage. The student's imitation interaction count is in `distillation.json`.

**V6** sends the actor and critic one fixed 64×64 remembered board. The local
observation updates terrain, observation age, and current occupants at their
world coordinates. The player's cell also carries inventory, vitals, facing,
daylight, and sleep status. Both heads read only features computed from this board.

The [V6 imitation pilot](../concept_mapping/runs/v6_board_only_imitation_10m/train.sh)
starts with random student weights. A frozen 12.2B teacher acts in the environment,
and the student fits the teacher's action probabilities and value estimates on
10,002,432 decisions. The [paired evaluations](../concept_mapping/runs/v6_board_only_imitation_10m/comparison.csv)
give control to each student checkpoint on the same 64 worlds. Checkpoint labels
in `imitation_checkpoints/` count teacher decisions. The final `files/policies/0`
checkpoint starts PPO with zero previous student PPO steps and fresh Adam moments.

**V7** adds a single cross-attention read to V6. The player cell supplies the
query. Observed cells supply terrain, age, occupancy, and their signed offsets
and distance from the player. The resulting weighted average adds a residual to
the shared 1,152 board features before the actor and critic. Its output weights
start at zero so the initial policy matches V6.

The [35M checkpoint comparison](../concept_mapping/runs/v7_board_reader_5m/comparison.csv)
starts all three students from V6 at 30,003,200 imitation decisions and preserves
its Adam history. Each student takes another 5,001,216 actions with the same
frozen teacher providing labels. The control and attention policies use
512-unit decision layers. The width control expands each of those layers to
640 units. The runs share a training seed and 64 evaluation worlds.

The [frozen V1 transfer pilot](../concept_mapping/runs/board_only_frozen_v1_attention_5m/comparison.csv)
teaches a whole-board reader to supply the first hidden activations of the
12.2B crop policy. The student sees the fixed observed board, with player state
at the marker. Eight learned attention queries read seen cells using content
and player-relative positions. The adapter produces one 512-value vector for
each frozen decision head. V1 supplies the terrain embedding, the last two
hidden layers of each head, and both output layers. Only the adapter enters the
optimizer. Half of the 128 training lanes follow the teacher and half follow the
student. All lanes receive V1 action, value and hidden-activation targets.

Preset: `ppo_presets/lava_only_map_mlp_1b.json`. This fresh seed-50 run matches
the CNN + GRU map study's lava-only −2 rewards, 33×33 view, PPO settings,
optimistic resets (ratio 16), and original 10B learning-rate horizon. It stops
at 1B requested steps or nine hours (999,948,288 completed steps if the step
ceiling is reached), saves logarithmic checkpoints, and evaluates at 100M and
the final checkpoint at both action temperatures 1 and 0.25.

When the planned training GPU is busy, `run_ppo_study.py --smoke --smoke-gpu N`
validates the same configuration on an idle GPU; it does not change the
planned training device. Validation records the actual smoke GPU. Launch still
checks availability of the configured training/evaluation devices.

## Frozen-policy action randomness sweeps

`evaluate_reward_shaped_ppo.py --temperature T` changes action selection only:
`T=1` is the original categorical sampler, smaller positive values concentrate
probability on preferred actions, and `T=0` selects argmax. Environment dynamics
remain stochastic. No network weights, rewards, or observation features change.
The default sampler is checked for bitwise agreement in
`test_policy_temperature.py`.

Use `run_policy_randomness_sweep.py --config
experiments/evaluation_presets/death2_10b_randomness_sweep.json --launch` for
a JSON-configured parallel evaluation. It validates sampling, evaluates paired
worlds across temperatures, selects by all-22 single-trial success, and compares
the selected setting with normal sampling on disjoint confirmation worlds.
Three trials per world are averaged, not treated as best-of-three. Confidence
intervals bootstrap worlds and pair improvements against normal sampling.
The worker saves per-trial results, all achievement rates, a plot, CSV tables,
progress, and a report. This command starts evaluation jobs only; it does not
launch or resume PPO training. An existing launched sweep is never overwritten.

## Training temperature and entropy continuations

`ppo.policy_temperature` changes the feedforward PPO distribution consistently
for collection, old/new action likelihoods, and entropy. It must be strictly
positive; 1 preserves the original distribution exactly. Greedy (0) is an
evaluation option only. This is a reparameterized categorical policy, so the
network can change its logit scale during training; temperature need not impose
a fixed long-term entropy.

`death2_10b_50m_{control,low_temperature,low_entropy}.json` launch three separate
continuations from the original 10B general −2-death MLP. Each restores the same
weights, restarts Adam at LR 2e−5, and anneals over 50M additional interactions
on ordinary independent-reset worlds. Training temperatures/entropy coefficients
are (1, 0.01), (0.25, 0.01), and (1, 0.001). Each uses a nine-hour cap and saves
checkpoints near +1M/+5M/+10M/+20M/final. Use the shared runner separately for
each preset; `analyze_temperature_finetuning.py` collects results only.

`evaluation.temperatures` defaults to `[1]` and can request `[1, 0.25]`.
`evaluation.world_seeds_file` plus `repeats` enables paired fixed-world trials;
`final_episodes` must equal worlds × repeats. Final outputs are
`evaluation_wallclock/` and `evaluation_wallclock_t0p25/`. Controllers serialize
shared evaluation GPUs using `/tmp/craftax-ppo-evaluation-gpuN.lock` so multiple
studies can queue safely. The unchanged original and every continuation are
evaluated on the same fresh worlds at both temperatures; compare against the
original at T=0.25 as well as against the matching evaluation temperature.
# Queue an extension of a running study

Set `resume_after` to the source architecture directory containing
`progress.json`, keep the original `ppo.lr_horizon_timesteps`, and set
`budget.max_steps` to the cumulative target. Run the shared entry point with
`--queue`. It waits for a successful final source checkpoint, validates the
optimizer-preserving resume and launches automatically when its GPUs are free.
`queue_progress.json` records status; `resolved_continuation.json` records the
exact checkpoint. Do not combine `resume_after` with `resume_from` or an
optimizer restart. Example: `ppo_presets/lava_only_spatial_map_gru_to2b.json`.

## Map MLP + GRU and temporal contrastive exploration

`map_mlp_gru` learns a tile embedding for the remembered, observed-only terrain
crop (33×33, embedding width 4 by default), concatenates it with local symbolic
observations and map ages, and feeds a 512-wide MLP embedding into a 512-wide GRU.
With `gru_concat_observation`, the heads receive both the current embedding and
recurrent output. Memory and GRU reset at actual episode boundaries, not PPO
rollout boundaries. There is no CNN or privileged unseen map input. Map location
tracking uses measured simulator displacement, as in the existing map-MLP.

The paired `lava_only_map_mlp_gru[_ctec]_dp2_10b_seed50.json` presets train from
scratch with identical policy initialization, global PPO batch and standard
resets. Each uses two GPUs, 10B interactions, linear LR 2e−4 → 0, the existing
achievement/health reward and −2 lava penalty. The 30-day cap is a failsafe;
the step target stops training first if reached. Checkpoints retain Adam state
for explicit future continuation. Frozen evaluations use identical seed streams,
1,024 episodes, and temperatures 1 and 0.25, without exploration rewards.

`ctec.py` independently implements the temporal contrastive distance idea from
[C-TeC](https://arxiv.org/abs/2603.02008) and its
[official implementation](https://github.com/FaisalAhmed0/c-tec).
Separate state-action and future-state MLPs (three 1,024-wide ReLU layers, 64
output dimensions) learn an InfoNCE objective with negative L2 logits. Positive
pairs sample strictly future observations using a truncated geometric horizon
with discount 0.3, never crossing an episode reset. The reward is positive L2
distance, not the InfoNCE loss. Both encoders use the raw 1,345-element local
symbolic observation; the policy additionally receives its terrain memory.
Two Adam updates at 3e−4 use batches of 512 pairs per PPO rollout. Contrastive
batches and negatives are global across devices. No gradient from the auxiliary
objective enters the policy, and bonuses use the encoders before their update.

Our **mixed task-reward adaptation**, rather than a reproduction of the paper's
reward-free experiment, controls mean exploration reward to `ctec_coef` times
the running mean absolute task reward (0.1 in the pilot). Distances are divided
by running RMS and clipped at 5 before mean scaling. Representation learning
starts immediately; bonuses start after 1M interactions, ramp to full strength
over the next 9M, then fade from 8B to zero at 10B. These are deliberate
hyperparameters, not proven optimal settings. Terminal transitions without a
recorded same-episode future receive no bonus. Original episode rewards and
achievement reporting stay separate from this training bonus.

Auxiliary parameters, optimizer, update counters and normalization statistics
are saved inside the full train state. Keep C-TeC settings and the intended
absolute schedule horizon when resuming. `test_ctec_map_gru.py` checks temporal
pair boundaries, representation learning, bonus schedules, checkpoint recovery,
recurrent behavior and one/two-device auxiliary math; the shared smoke gate also
trains full-size batches and reloads them for real-environment evaluation.


## Survival-map pilot following the early-death audit

`audit_ppo_early_deaths.py` reuses the frozen evaluator to reproduce recorded
failures and then replays those actions through the original simulator. It
requires every observation to match before accepting combat versus intrinsic
health-loss accounting. Successful controls use the same checkpoint and world
seeds with independent action/dynamics streams; they are selected diagnostic
controls, not an unbiased success estimate. `summarize_early_death_audit.py`
compares their prefixes at the original failure time and reconstructs routes
from observed tiles only. The audit artifact is under the low-LR 12B map study's
`early_death_audit/` directory: 15 verified failures, 13 successful same-world
controls; 10 terminal depletion losses and 5 terminal combat losses. These
mechanisms are not claims of exclusive causal responsibility.

`ppo.map_survival_features` adds 40 inputs to `map_mlp_gru` when set to `enabled`:
16 global known-water route features, three counters for consecutive food/water/
energy depletion (saturating at 256 steps), previous-action one-hot, actual
movement, and failed-movement feedback. Water planning is computed on the full
observed backing map, even beyond the 33×33 readout. It routes to an adjacent
walkable water-edge tile through observed grass/wood/path/sand, excludes unknown
terrain and lava, and propagates up to 128 walking steps. Its distance/masks are
not a safety guarantee: it does not model dynamic mobs, mining, or unseen changes.
The current facing and adjacent water flags help identify a drinking interaction.
Sleep-ignored movement is not counted as a blocked movement. No action is forced
and no reward is changed. All memory clears at normal episode resets.

`masked` gives the same policy shape and initialization with the 40 inputs set
to zero. `none` preserves legacy observations and parameters. The control is
therefore a capacity-matched ordinary map+GRU policy, rather than the already
running 10B run with different input-kernel initialization. Both new pilots use
seed 50, the same global PPO settings, standard resets, and 500M fresh steps.
Their 10B LR horizon supports continuation; the 120-hour cap is a failsafe, not
an ETA. These are early learning/survival pilots, not a test of final 10B mastery.

The paired `lava_only_map_gru_survival_{control,enabled}_500m_seed50.json` presets
use one GPU each (1 and 2). Evaluation is deferred onto the same GPU after
training; existing studies using these evaluation GPUs queue without stopping
their training. Frozen paired evaluations use seed 88150 at 100M, 250M and final.
`compare_survival_map_pilot.py --watch` compares final checkpoints, reports paired
bootstrap intervals for early death and all-22 rates, and triggers 1,024-episode
confirmation per policy on seed 98150 if its predeclared 2-percentage-point screen
is met. One training seed does not establish architecture-level robustness.


## Learned neural map pilot

`lava_only_neural_map_gru_dp2_1b_seed50.json` starts from scratch, trains to
999,948,288 interactions (the last complete 65,536-step update below 1B),
and uses a **10B-step linear LR horizon**, starting at 2e-4 and reaching about
1.8e-4 at 1B. A later `resume_from` preset must retain that horizon and restore
full Adam state. Environment episodes, GRU and neural memory reset on process
resume; weights, optimizer and training counters persist. No automatic extension
beyond 1B is scheduled. The 120-hour wall cap allows slower-than-baseline training.

Architecture `neural_map_gru` keeps the 33×33 explicit observed terrain/age
readout, four-value terrain embedding, 512-unit MLP and GRU, and adds a
64×64×16 floating-point learned map plus a seen mask per environment. The only
additional environment inputs are two coordinates from the existing odometry.
No hidden simulator terrain, resources, threats, or future state are exposed.
A shared gated writer processes each visible 7×9 cell with its raw 21 channels,
old memory, relative offset and current observation/GRU context. Only in-bounds
visible locations are updated. Learned channels have no predetermined meaning.

A 3×3 local read and content/relative-position attention over all seen cells
produce a 128-value read, fused into the GRU input. Policy/value heads receive
current observation embedding, GRU and neural-map read. All components learn
from the ordinary PPO objective with lava cost −2; no intrinsic reward or
auxiliary prediction loss is added in this first experiment. Neural memory and
GRU reset at episode boundaries and persist across rollout boundaries. Training
backpropagates through 64 steps, truncating gradients at rollout boundaries.
A rematerialized scanned cell reduces activation memory. Only initial neural
carry is stored for PPO replay, not a full map for every environment transition.
This has more parameters/computation than map-MLP+GRU; it is not capacity matched.

Training uses GPUs 4/6, evaluation GPU 7, 1,024 global environments, 64 steps,
four epochs and eight global minibatches. Frozen evaluations at 100M, 250M,
500M and the final checkpoint use 1,024 independent common-seed episodes at
T=1 and T=.25. Existing training and evaluator GPUs are preserved.

`test_neural_map.py` checks streamed/replayed equivalence, episode resets,
border writes, gradients through earlier writes with GRU erased, observation
provenance, and supervised delayed spatial recall with GRU erased. These checks
establish that the memory can learn and retain information, not that it improves
Craftax. Shared smoke validation also trains full-size PPO on two GPUs, saves and
reloads optimizer state, resumes two more updates and runs frozen evaluation.
For a trained checkpoint use `evaluate_reward_shaped_ppo.py --reset-neural-map`
to erase neural history while retaining GRU, parameters, seeds and temperature.
Compare this diagnostic with the unablated policy, and compare achievements and
early deaths against the map-GRU at matched training steps.
