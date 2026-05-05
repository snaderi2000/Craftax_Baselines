# Achievement-Distillation-Style PPO-RNN Schematic

This describes the optional architecture in `ppo_rnn_jax0423.py`:

```text
--arch achdist_strong
```

It is designed to match the main data flow of the PyTorch
`Achievement-Distillation` `PPOGRUStrongModel`.

Assume:

```text
T = num_steps
B = num_envs
R = rnn_hidsize = 256
Z = achdist_hidsize = 1024
P = achdist_impala_outsize = 256
Hpi = layer_size = 1024
Hv = achdist_vf_head_hidsize = 1280
```

Input:

```text
obs:   [T, B, 63, 63, 3]
dones: [T, B]
h0:    [B, 256]
```

## CNN Encoder

```text
obs
[T, B, 63, 63, 3]
        |
        v
AchDist IMPALA Stack 1
channels 64
Conv 3x3, ReLU
MaxPool 3x3 stride 2
GroupNorm(groups=1)
2x residual conv blocks
        |
        v
[T, B, 32, 32, 64]
        |
        v
AchDist IMPALA Stack 2
channels 128
GroupNorm(groups=1)
Conv 3x3, ReLU
MaxPool 3x3 stride 2
GroupNorm(groups=1)
2x residual conv blocks
        |
        v
[T, B, 16, 16, 128]
        |
        v
AchDist IMPALA Stack 3
channels 128
GroupNorm(groups=1)
Conv 3x3, ReLU
MaxPool 3x3 stride 2
GroupNorm(groups=1)
2x residual conv blocks
        |
        v
[T, B, 8, 8, 128]
        |
        v
Flatten
        |
        v
[T, B, 8192]
        |
        v
LayerNorm
Dense 8192 -> 256
ReLU
        |
        v
impala_dense: [T, B, 256]
        |
        v
LayerNorm
Dense 256 -> 1024
ReLU
        |
        v
visual_latent: [T, B, 1024]
```

## GRU Path

```text
visual_latent
[T, B, 1024]
        |
        v
LayerNorm
Dense 1024 -> 256
ReLU
        |
        v
rnn_input: [T, B, 256]
        |
        v
Scanned GRU, reset by dones
input: [T, B, 256]
h0:    [B, 256]
        |
        v
ReLU
        |
        v
memory_latent: [T, B, 256]
```

## Full-GRU Shared Representation

```text
concat([visual_latent, memory_latent])
[T, B, 1024 + 256]
        |
        v
shared: [T, B, 1280]
```

## Masked-GRU Ablation

For:

```text
--no-use_gru --no_gru_memory masked_gru
```

the GRU is still computed to keep the compiled graph close to the full-GRU model,
but its output is zeroed before the heads:

```text
memory_latent = 0 * GRU(rnn_input)
```

Then:

```text
concat([visual_latent, zeros])
[T, B, 1024 + 256]
        |
        v
shared: [T, B, 1280]
```

This tests whether the recurrent signal helps while preserving the same head shape.

## Actor Head

```text
shared
[T, B, 1280]
        |
        v
LayerNorm
Dense 1280 -> 1024
ReLU
        |
        v
Dense 1024 -> action_dim
        |
        v
actor_logits: [T, B, action_dim]
Categorical policy
```

## Critic Head

```text
shared
[T, B, 1280]
        |
        v
LayerNorm
Dense 1280 -> 1280
ReLU
        |
        v
Dense 1280 -> 1
        |
        v
value: [T, B]
```

With:

```text
--normalize_value_targets
```

the value head predicts normalized TD targets during training. Values are
denormalized before GAE/rollout use.

## Why This Differs From The Paper-Style `ppo_rnn` Path

The original `paper` architecture in this repo sends a much larger direct CNN
feature to the actor/critic heads:

```text
paper:
  shared = [z_t 8192, GRU 256] = 8448

achdist_strong:
  shared = [visual_latent 1024, GRU 256] = 1280
```

So in `achdist_strong`, the recurrent part is a much larger fraction of the
decision representation:

```text
paper:          256 / 8448 = about 3%
achdist_strong: 256 / 1280 = 20%
```

