# Symbolic EfficientZero-style JAX pilot

Reference implementation: [EfficientZero V2](https://github.com/Shengjiewang-Jason/EfficientZeroV2),
commit `12b9e775df1f8afb4a1dc854cf722ea84fbfb61d`, Copyright (c) EVAR Lab,
IIIS, Tsinghua University, GNU GPL version 3. A reference checkout is in
`third_party/EfficientZeroV2`; its original notices and LICENSE are preserved.

`models/symbolic_ez.py`, `symbolic_ez_replay.py`, and `train_symbolic_ez.py`
are modified JAX/Flax pilot implementations under GPL-3.0-only. The reference
training loop (`ez/agents/base.py`), state residual dynamics
(`ez/agents/ez_dmc_state.py`), and cross-entropy/cosine losses
(`ez/utils/loss.py`) informed the translation. The original GPL text is also
included in `models/SYMBOLIC_EZ_GPL_LICENSE.txt`. These files are marked as
adaptations, not the upstream implementation.

Search uses the unmodified `mctx==0.0.5` Python package installed locally in
`third_party/mctx_py`, under its upstream Apache-2.0 license. Its version is
pinned to work with this project's existing JAX environment. No Conda packages
were upgraded or replaced.

This is an EZ-style experiment, not an EZ-V2 reproduction. Differences:

- Symbolic observations and typed local-observation prediction losses.
- GRU history encoder receiving observations, previous actions, and rewards.
- Deterministic latent dynamics with probabilistic decoder/reward/value heads;
  stochastic environment outcomes are approximated, not explicitly branched.
- Immediate reward prediction replaces the upstream reward-prefix LSTM.
- EMA targets and five-step returns with fractional fresh root-search value
  targets replace the complete upstream search-based value-estimation pipeline.
- Periodic replay reanalysis and mixed uniform/prioritized sequence replay.
- JAX batched real interaction and JAX Gumbel search replace PyTorch/Ray/CUDA
  collector infrastructure.

No latent simulator state, achievement flags, crafting rules, resets to saved
states, demonstrations, or simulator transitions inside planning are supplied
to the networks. Achievement flags are used only for reporting evaluation.
