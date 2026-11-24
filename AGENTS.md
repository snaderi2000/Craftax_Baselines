# Craftax experiment workflow

For PPO experiment configuration, reward variants, architecture comparisons,
GPU launches, wall-clock limits, checkpoints, or status checks, use the
`craftax-ppo-experiments` skill. Its repository source is
[skills/craftax-ppo-experiments/SKILL.md](skills/craftax-ppo-experiments/SKILL.md).

Prefer a JSON preset under `experiments/ppo_presets/` and the shared
`run_ppo_study.py` entry point over creating another experiment-specific launcher.
Historical launchers and run directories remain useful for reproducibility.
