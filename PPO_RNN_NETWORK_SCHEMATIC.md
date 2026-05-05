# PPO-RNN Network Schematic

This describes the network in `ppo_rnn.py` / `ppo_rnn_jax0423.py` for the pixel Craftax environment.

Assume:

```text
T = num_steps
B = num_envs
H = GRU hidden size = 256
L = layer_size, e.g. 2048 in the V100 command
```

Input:

```text
obs:   [T, B, 63, 63, 3]
dones: [T, B]
h0:    [B, 256]
```

## Data Flow

```text
obs
[T, B, 63, 63, 3]
        |
        v
IMPALA Stack 1
Conv 3x3, 64 channels, SAME
GroupNorm
MaxPool 3x3, stride 2
2x residual conv blocks
        |
        v
[T, B, 32, 32, 64]
        |
        v
IMPALA Stack 2
Conv 3x3, 64 channels, SAME
GroupNorm
MaxPool 3x3, stride 2
2x residual conv blocks
        |
        v
[T, B, 16, 16, 64]
        |
        v
IMPALA Stack 3
Conv 3x3, 128 channels, SAME
GroupNorm
MaxPool 3x3, stride 2
2x residual conv blocks
        |
        v
[T, B, 8, 8, 128]
        |
        v
ReLU + flatten spatial/features
        |
        v
z_t: [T, B, 8192]
```

The flattened CNN feature `z_t` goes two places:

```text
z_t [T, B, 8192]
        |
        v
LayerNorm
Dense 8192 -> 256
ReLU
        |
        v
rnn_input_features: [T, B, 256]
        |
        v
Scanned GRU with reset from dones
input:  [T, B, 256]
h0:     [B, 256]
output: [T, B, 256]
        |
        v
y_t: [T, B, 256]
```

Then the CNN feature and recurrent feature are concatenated:

```text
concat([y_t, z_t])
[T, B, 256 + 8192]
        |
        v
shared_input: [T, B, 8448]
```

## Actor Head

```text
shared_input
[T, B, 8448]
        |
        v
LayerNorm
Dense 8448 -> L
ReLU
DenseResBlock L -> L
DenseResBlock L -> L
ReLU
LayerNorm
Dense L -> action_dim
        |
        v
actor_logits: [T, B, action_dim]
Categorical policy pi
```

For Craftax, `action_dim = env.action_space(env_params).n`.

## Critic Head

```text
shared_input
[T, B, 8448]
        |
        v
LayerNorm
Dense 8448 -> L
ReLU
DenseResBlock L -> L
DenseResBlock L -> L
ReLU
LayerNorm
Dense L -> 1
        |
        v
value: [T, B]
```

## One-Step Action Selection

During environment rollout, the code feeds one time step at a time:

```text
last_obs:  [B, 63, 63, 3]
last_done: [B]
hstate:    [B, 256]
```

It adds a time dimension:

```text
ac_in = (
  last_obs[None, ...],   # [1, B, 63, 63, 3]
  last_done[None, ...],  # [1, B]
)
```

The network returns:

```text
new_hstate: [B, 256]
pi:         Categorical over [1, B, action_dim]
value:      [1, B]
```

Then the sampled action is squeezed to:

```text
action: [B]
```

## Parameter-Size Note

The largest parameter cost comes from the actor and critic heads because each starts with:

```text
Dense 8448 -> L
```

and there are separate actor and critic copies. With `L = 2048`, this creates a much larger model than with `L = 512`.

Rough intuition:

```text
L = 2048: large head, about 54.5M params in your run
L = 512:  much smaller head
```

