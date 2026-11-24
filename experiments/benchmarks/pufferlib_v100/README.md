# PufferLib Classic throughput probe on V100

This benchmark runs upstream PufferLib's C Craftax-Classic environment and native
CUDA trainer, pinned to commit `ea7f989371270d2948f96728bda8aec5e7634a4c`.
It does not replace the ongoing JAX experiments or migrate their checkpoints.

## Reproduce on this server

The source lives in `third_party/PufferLib`. A separate CUDA compiler lives in
`.tools/puffer-cuda`; the active JAX environment was not modified. Dependencies:
CUDA 12.4.1 channel packages `cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`,
`cuda-nvml-dev`, `cuda-nvtx`, `cuda-profiler-api`; Raylib 5.5 Linux static archive;
existing NVIDIA library wheels (paths recorded in `build.py`). System GCC/OpenMP
are used. Build uses float32, sm_70, O3 and fast math; upstream source is unchanged.

```
PY=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
$PY experiments/benchmarks/pufferlib_v100/build.py
$PY experiments/benchmarks/pufferlib_v100/run.py \
  --config experiments/benchmarks/pufferlib_v100/timing_config.json \
  --out concept_mapping/runs/pufferlib_v100_benchmark/timing
```

The runner exclusively reserves the configured GPU through the repository's
shared GPU lock and checks free memory. Each subprocess has a time limit.
Use a new output directory for a new measurement to retain earlier artifacts.
All runs are short fresh training runs. Three repeats measure execution
variability with the same initialization seed, not learning variability.
Checkpoints are checked for finite weights and reloaded for at least 32 headless episodes (parallel batches can overshoot).

## Interpretation

- `native_recipe`: 8192 environments, 64-step rollouts, MinGRU 128 x 4,
  one pass per batch, asynchronous rollout/learner.
- `matched_update_work`: 1024 environments, 64-step rollouts, MinGRU 512 x 1,
  four passes, 8192-sample minibatches, synchronous rollout/learner. This
  matches the JAX pilot's collection batch and number of optimizer updates.
- `similar_parameter_count`: the same batch/update settings with MinGRU 704 x 4.
  Parameter count is read from the saved float32 checkpoint, not assumed.

None is an exact policy/algorithm comparison. PufferLib uses MinGRU, Muon,
and its PPO variant; our model uses a persistent terrain-map encoder, standard
GRU and Adam. PufferLib Classic has a 1345-dimensional local observation,
17 actions, a 10k episode cap, and achievement plus health rewards, without
our extra lava penalty. It has its own C world generator and RNG; full
transition/world-distribution parity with installed JAX Craftax is unverified.
The upstream parity script covers full Craftax, not this Classic port.

Report process-wall-clock throughput (includes initialization/save) and warmed
training SPS separately. Frozen reload evaluation is outside the timed training.
PufferLib emits binned metrics; ignore the first bins when interpreting warmup.
Short runs do not establish learning efficiency or all-22 success.
