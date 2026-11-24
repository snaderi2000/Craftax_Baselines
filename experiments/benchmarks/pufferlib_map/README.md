# PufferLib map-port migration checks

This isolated experiment tests the actual memory inputs and lava reward used by
our richer-map PPO study. It uses the pinned PufferLib checkout from the earlier
V100 benchmark. It does not change the ongoing JAX runs or their environment.

`prepare.py` generates an environment and trainer derivative under
`third_party/PufferLib/build/map_validation/`, leaving upstream tracked source
unchanged. `map_features.h` supplies the full observed-only terrain memory,
33x33 crop, log ages, and 40 survival inputs. Multi-source BFS computes the same
128-move water-edge distances as the JAX relaxation; results are cached until
observed terrain changes. No unobserved tile values enter memory.

`map_encoder.cu` embeds categorical terrain IDs in a learned 18x4 table, packs
the same local/map/age/survival inputs, and projects to the MinGRU width. The
benchmark uses width 512 and four MinGRU layers. The core, activation structure,
optimizer (Muon), and LR schedule (cosine) still differ from our JAX model.
Thus this is a plausible replacement architecture, not the identical policy.

## Validation

- `validate.py`: feeds the same observed frames to the real JAX wrappers and C
  memory port, then injects identical simulator states into C and JAX for
  deterministic gameplay comparisons. Includes resource exhaustion, sleep,
  lava death (-2), timeout, crafting, mining, repeated achievement reward,
  plants, and zombie attacks. World generation/stochastic parity is not proven.
- `check_encoder.cu`: independently checks CUDA encoder outputs and gradients
  for every table and projection parameter against a double-precision CPU
  reference, including repeated tile indices and all 72 embedding gradients.
- `check_reset.py`: checks autoreset on lava and timeout clears memory,
  depletion/action/blocking history, and ages while preserving terminal reward.
- `check_resume.py`: tests upstream `train --base.load_model_path=...` with
  a real checkpoint and a nonexistent checkpoint against a fresh run, and
  inspects checkpoint format/counters. Uses GPU 6 under the shared lock.
- `timing_config.json`: three 4,194,304-step fresh timing runs with 1024 envs,
  horizon 64, four PPO passes, eight minibatches, synchronous execution and the
  complete map/survival inputs. Timing uses GPU 5 under the shared lock.

The recorded environment mismatches and failed resume support block a valid
learning-per-hour migration comparison. Do not promote these timing checkpoints
or interpret their scores as directly comparable to the JAX experiment.

## Commands on this server

```
PY=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
$PY experiments/benchmarks/pufferlib_map/prepare.py
$PY experiments/benchmarks/pufferlib_map/build.py
# Build commands for the CPU bridge and CUDA gradient check are in build_checks.py.
$PY experiments/benchmarks/pufferlib_map/build_checks.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' $PY experiments/benchmarks/pufferlib_map/validate.py
$PY experiments/benchmarks/pufferlib_map/check_reset.py
CUDA_VISIBLE_DEVICES=5 flock -n /tmp/craftax-ppo-evaluation-gpu5.lock third_party/PufferLib/check_map_encoder
$PY experiments/benchmarks/pufferlib_map/check_resume.py
$PY experiments/benchmarks/pufferlib_v100/run.py \
  --config experiments/benchmarks/pufferlib_map/timing_config.json \
  --out concept_mapping/runs/pufferlib_map_validation/timing
```

Use a new output directory when repeating measurements. The short training runs
save policy-only checkpoints and reload them for headless evaluation; that is
not a full optimizer-state training resume. All current sources, generated files
and the binary are fingerprinted in the result directory.
