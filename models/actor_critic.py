import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Optional, Dict, Union, Tuple
from dataclasses import field
import math
import distrax
import jax

class FanInInitReLULayer(nn.Module):
    inchan: int
    outchan: int
    layer_type: str = "conv"
    init_scale: float = 1.0
    kernel_size: Union[int, Sequence[int]] = 3
    strides: Union[int, Sequence[int]] = 1
    padding: Union[str, int, Sequence[int]] = "SAME"

    # normalization flags…
    batch_norm: bool = False
    batch_norm_kwargs: Dict    = field(default_factory=dict)
    group_norm_groups: Optional[int] = None
    layer_norm: bool = False

    # activation & training‐mode
    use_activation: bool = True
    train: bool          = True             # control BatchNorm mode

    @nn.compact
    def __call__(self, x):
        # 1) Normalization (if any)
        if self.batch_norm:
            x = nn.BatchNorm(
                use_running_average=not self.train,
                **self.batch_norm_kwargs
            )(x)
        elif self.group_norm_groups is not None:
            x = nn.GroupNorm(num_groups=self.group_norm_groups)(x)
        elif self.layer_norm:
            x = nn.LayerNorm()(x)

        # 2) Core layer
        if self.layer_type in ("conv", "conv2d", "conv3d"):
            x = nn.Conv(
                features=self.outchan,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                use_bias=(not (self.batch_norm
                           or self.group_norm_groups
                           or self.layer_norm)),
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)

        elif self.layer_type == "linear":
            x = nn.Dense(
                features=self.outchan,
                use_bias=True,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
            )(x)
        else:
            raise ValueError(f"Unsupported layer_type: {self.layer_type}")

        # 3) Activation
        if self.use_activation:
            x = nn.relu(x)

        return x

class CnnBasicBlock(nn.Module):
    inchan: int
    init_scale: float = 1.0
    init_norm_kwargs: Dict = field(default_factory=dict)
    train: bool = True

    @nn.compact
    def __call__(self, x):
        # Paper: each ResNet block:
        # (a) ReLU -> BatchNorm
        # (b) Conv 3x3, stride 1
        residual = x

        # (a) ReLU
        y = nn.relu(x)

        # (b) BatchNorm (if enabled via init_norm_kwargs)
        if self.init_norm_kwargs.get("batch_norm", False):
            bn_kwargs = self.init_norm_kwargs.get("batch_norm_kwargs", {})
            y = nn.BatchNorm(
                use_running_average=not self.train,
                **bn_kwargs,
            )(y)

        # (c) Conv 3x3, stride 1
        y = nn.Conv(
            features=self.inchan,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,  # typical with BN
            kernel_init=orthogonal(math.sqrt(self.init_scale)),
            bias_init=constant(0.0),
        )(y)

        return residual + y

class CnnDownStack(nn.Module):
    inchan: int
    nblock: int
    outchan: int
    init_scale: float = 1.0
    pool: bool = True
    post_pool_groups: Optional[int] = None  # will be ignored now
    init_norm_kwargs: Dict = field(default_factory=dict)
    first_conv_norm: bool = False           # will be ignored too
    train: bool = True

    @nn.compact
    def __call__(self, x):
        # (a) BatchNorm at the start of the stack (if requested)
        if self.init_norm_kwargs.get("batch_norm", False):
            bn_kwargs = self.init_norm_kwargs.get("batch_norm_kwargs", {})
            x = nn.BatchNorm(
                use_running_average=not self.train,
                **bn_kwargs,
            )(x)

        # (b) Conv 3x3, stride 1
        x = nn.Conv(
            features=self.outchan,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)

        # (c) MaxPool 3x3, stride 2
        if self.pool:
            x = nn.max_pool(
                x,
                window_shape=(3, 3),
                strides=(2, 2),
                padding="SAME",
            )

        # (d) Residual blocks (as defined above)
        for _ in range(self.nblock):
            x = CnnBasicBlock(
                inchan=self.outchan,
                init_scale=self.init_scale / math.sqrt(self.nblock),
                init_norm_kwargs=self.init_norm_kwargs,
                train=self.train,
            )(x)

        return x


class ImpalaCNN(nn.Module):
    inshape: Sequence[int]
    chans: Sequence[int]
    outsize: int
    nblock: int
    init_norm_kwargs: Dict = field(default_factory=dict)
    dense_init_norm_kwargs: Dict = field(default_factory=dict)
    first_conv_norm: bool = False
    train: bool = True

    post_pool_groups: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        # x: [B, H, W, C]  (NHWC)
        c, h, w = self.inshape
        # apply down stacks
        for i, outchan in enumerate(self.chans):
            use_first_norm = True if i > 0 else self.first_conv_norm
            x = CnnDownStack(
                inchan=c,
                nblock=self.nblock,
                outchan=outchan,
                init_scale=1.0 / math.sqrt(len(self.chans)),
                pool=True,
                post_pool_groups=self.post_pool_groups,
                init_norm_kwargs=self.init_norm_kwargs,
                first_conv_norm=use_first_norm,
                train=self.train
            )(x)
            # update channel count for next stack
            c = outchan
        # flatten
        x = x.reshape((x.shape[0], -1))
        # final dense
        x = FanInInitReLULayer(
            inchan=x.shape[-1],
            outchan=self.outsize,
            layer_type="linear",
            init_scale=1.4,

            # Explicitly pass only the supported norm kwargs:
            layer_norm=self.dense_init_norm_kwargs.get("layer_norm", False),
            batch_norm=self.dense_init_norm_kwargs.get("batch_norm", False),
            group_norm_groups=self.dense_init_norm_kwargs.get("group_norm_groups", None),

            train=self.train,
        )(x)
        return x






class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"
    train: bool = True

    @nn.compact
    def __call__(self, obs):
        # x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        # x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        # x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        # x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        # x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        # x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        # embedding = x.reshape(x.shape[0], -1)
        print("DEBUG — raw obs.shape:", obs.shape)
        #x = jnp.transpose(obs, (0, 2, 3, 1))
        x = obs
        # 2) Hard‑coded ImpalaCNN
        x = ImpalaCNN(
            inshape=(3, 63, 63),
            chans=(64, 128, 128),
            outsize=256,
            nblock=2,
            init_norm_kwargs={'batch_norm': True, 'batch_norm_kwargs': {'momentum': 0.10}},
            post_pool_groups=None,
            dense_init_norm_kwargs={},
            train=self.train
        )(x)


        # 3) Hard‑coded projection to hidsize=1024 with layer_norm
        x = FanInInitReLULayer(
            inchan=x.shape[-1],
            outchan=1024,
            layer_type='linear',
            init_scale=1.4,
            layer_norm=True,
            train=self.train
        )(x)

        embedding = x
        # 1) Single‐layer policy head → logits → Categorical
        pi_logits = nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        pi = distrax.Categorical(logits=pi_logits)

        # 2) Single‐layer value head → scalar value
        v = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)
        critic = jnp.squeeze(v, axis=-1)

        # 3) Return exactly (pi, critic)
        return pi, critic




#Make the actor critic with a rnn


class ImpalaCNN_RNN(nn.Module):
    inshape: Sequence[int]
    chans: Sequence[int]
    nblock: int
    init_norm_kwargs: Dict = field(default_factory=lambda: {'batch_norm': True, 'batch_norm_kwargs': {'momentum': 0.10}})
    first_conv_norm: bool = False
    train: bool = True

    post_pool_groups: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        # x: [B, H, W, C]  (NHWC)
        c, h, w = self.inshape
       
        # apply down stacks
        for i, outchan in enumerate(self.chans):
            use_first_norm = True if i > 0 else self.first_conv_norm
            x = CnnDownStack(
                inchan=c,
                nblock=self.nblock,
                outchan=outchan,
                init_scale=1.0 / math.sqrt(len(self.chans)),
                pool=True,
                post_pool_groups=self.post_pool_groups,
                init_norm_kwargs=self.init_norm_kwargs,
                first_conv_norm=use_first_norm,
                train=self.train
            )(x)
            # update channel count for next stack
            c = outchan
        x = nn.relu(x)           # <<< before flatten
        # flatten
        x = x.reshape((x.shape[0], -1))
       
        return x


# class ActorCriticConvRNN(nn.Module):
#     action_dim: int
#     cnn_chans: Sequence[int] = (64, 64, 128)

#     # --- GRU controls -------------------------------------------------
#     rnn_hidden: int = 256        # set to 0 for “no-GRU” ablation if you want
#     use_gru: bool   = True       # True = GRU on, False = no-GRU
#     # ------------------------------------------------------------------

#     head_width: int = 2048
#     n_res_blocks: int = 1
#     train: bool = True

#     @nn.compact
#     def encode(self, obs):
#         return ImpalaCNN_RNN(
#             inshape=(3, 63, 63),
#             chans=self.cnn_chans,
#             nblock=2,
#             first_conv_norm=True,
#             post_pool_groups=None,
#             train=self.train,
#         )(obs)

#     @nn.compact
#     def core(self, z, h):
#         do_gru = (self.use_gru and self.rnn_hidden > 0)

#         if do_gru:
#             x = nn.LayerNorm()(z)
#             x = nn.Dense(self.rnn_hidden, kernel_init=orthogonal(np.sqrt(2)),
#                         bias_init=constant(0.0))(x)
#             x = nn.relu(x)

#             if h is None:
#                 h = jnp.zeros((z.shape[0], self.rnn_hidden), z.dtype)

#             h_next, _ = nn.GRUCell(features=self.rnn_hidden)(h, x)
#             y = nn.relu(h_next)
#             shared = jnp.concatenate([z, y], axis=-1)
#         else:
#             h_next = jnp.zeros((z.shape[0], self.rnn_hidden), z.dtype)
#             shared = z



#         a = nn.LayerNorm()(shared)
#         a = nn.Dense(
#             self.head_width,
#             kernel_init=orthogonal(np.sqrt(2)),
#             bias_init=constant(0.0),
#         )(a)
#         a = nn.relu(a)

#         for _ in range(self.n_res_blocks):
#             res = a
#             a = nn.Dense(
#                 self.head_width,
#                 kernel_init=orthogonal(np.sqrt(2)),
#                 bias_init=constant(0.0),
#             )(a)
#             a = nn.relu(a)
#             a = nn.Dense(
#                 self.head_width,
#                 kernel_init=orthogonal(np.sqrt(2)),
#                 bias_init=constant(0.0),
#             )(a)
#             a = nn.relu(a + res)

#         a = nn.LayerNorm()(a)
#         logits = nn.Dense(
#             self.action_dim,
#             kernel_init=orthogonal(0.01),
#             bias_init=constant(0.0),
#         )(a)

#         # --------------------------------------------------------------
#         # 6. Critic head
#         # --------------------------------------------------------------
#         v = nn.LayerNorm()(shared)
#         v = nn.Dense(
#             self.head_width,
#             kernel_init=orthogonal(np.sqrt(2)),
#             bias_init=constant(0.0),
#         )(v)
#         v = nn.relu(v)

#         for _ in range(self.n_res_blocks):
#             res = v
#             v = nn.Dense(
#                 self.head_width,
#                 kernel_init=orthogonal(np.sqrt(2)),
#                 bias_init=constant(0.0),
#             )(v)
#             v = nn.relu(v)
#             v = nn.Dense(
#                 self.head_width,
#                 kernel_init=orthogonal(np.sqrt(2)),
#                 bias_init=constant(0.0),
#             )(v)
#             v = nn.relu(v + res)

#         v = nn.LayerNorm()(v)
#         v = nn.Dense(
#             1,
#             kernel_init=orthogonal(1.0),
#             bias_init=constant(0.0),
#         )(v)
#         value = jnp.squeeze(v, axis=-1)      # (B,)

#         return logits, value, h_next

#     @nn.compact
#     def __call__(
#         self,
#         obs: jnp.ndarray,
#         h: Optional[jnp.ndarray] = None,
#     ) -> Tuple[distrax.Categorical, jnp.ndarray, jnp.ndarray]:
#         z = self.encode(obs)
#         logits, value, h_next = self.core(z, h)
#         pi = distrax.Categorical(logits=logits)
#         return pi, value, h_next

class ResNetBlock(nn.Module):
    channels: int
    train: bool = True

    @nn.compact
    def __call__(self, x):
        res = x
        # (a) ReLU followed by BatchNorm
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not self.train)(x)
        # (b) Conv 3x3, stride 1
        x = nn.Conv(features=self.channels, kernel_size=(3, 3), strides=(1, 1))(x)
        return x + res

class ImpalaStack(nn.Module):
    channels: int
    train: bool = True

    @nn.compact
    def __call__(self, x):
        # (a) Batch Norm
        x = nn.BatchNorm(use_running_average=not self.train)(x)
        # (b) Conv 3x3, stride 1
        x = nn.Conv(features=self.channels, kernel_size=(3, 3), strides=(1, 1))(x)
        # (c) Max Pool 3x3, stride 2
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        # (d) Two ResNet Blocks
        x = ResNetBlock(self.channels, train=self.train)(x)
        x = ResNetBlock(self.channels, train=self.train)(x)
        return x

# --- 2. Actor/Critic Residual Sub-components ---

class DenseResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        res = x
        # Standard Dense Residual Block: Linear -> ReLU -> Linear
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        return nn.relu(x + res)

# --- 3. Full MFRL Agent (Algorithm 2) ---

class ActorCriticConvRNN(nn.Module):
    action_dim: int
    head_width: int = 2048
    rnn_hidden: int = 256
    use_gru: bool = True
    train: bool = True

    @nn.compact
    def encode(self, x): 
        x = x.astype(jnp.float32)
        for channels in [64, 64, 128]:
            x = ImpalaStack(channels=channels, train=self.train)(x)
        x = nn.relu(x)
        return x.reshape((x.shape[0], -1)) # zt (8192)

    @nn.compact
    def core(self, zt, h):
        # 1. RNN Architecture (yt)
        rnn_in = nn.relu(nn.Dense(256)(nn.LayerNorm()(zt)))
        if self.use_gru:
            new_h, yt_raw = nn.GRUCell(self.rnn_hidden)(h, rnn_in)
        else:
            yt_raw, new_h = rnn_in, h
        yt = nn.relu(yt_raw)

        # 2. Shared Concatenation (8448)
        shared_input = jnp.concatenate([zt, yt], axis=-1)

        # --- THE SHARED NECK (The Parameter Fix) ---
        # Instead of each head having its own 17M parameter layer, they share this one.
        shared_neck = nn.LayerNorm()(shared_input)
        shared_neck = nn.Dense(features=self.head_width, name="shared_fc")(shared_neck)
        shared_neck = nn.relu(shared_neck)

        # 3. Actor Head (Branches from shared_neck)
        a = DenseResBlock(self.head_width, name="actor_res1")(shared_neck)
        a = DenseResBlock(self.head_width, name="actor_res2")(a)
        a = nn.LayerNorm()(nn.relu(a))
        actor_logits = nn.Dense(self.action_dim, name="actor_logits")(a)
        pi = distrax.Categorical(logits=actor_logits)

        # 4. Critic Head (Branches from shared_neck)
        c = DenseResBlock(self.head_width, name="critic_res1")(shared_neck)
        c = DenseResBlock(self.head_width, name="critic_res2")(c)
        c = nn.LayerNorm()(nn.relu(c))
        value = nn.Dense(1, name="value_logits")(c)

        return pi, jnp.squeeze(value, axis=-1), new_h

    def __call__(self, x, h):
        zt = self.encode(x)
        return self.core(zt, h)






# class ActorCritic(nn.Module):
#     action_dim: Sequence[int]
#     layer_width: int
#     activation: str = "tanh"

#     @nn.compact
#     def __call__(self, x):
#         if self.activation == "relu":
#             activation = nn.relu
#         else:
#             activation = nn.tanh

#         actor_mean = nn.Dense(
#             self.layer_width,
#             kernel_init=orthogonal(np.sqrt(2)),
#             bias_init=constant(0.0),
#         )(x)
#         actor_mean = activation(actor_mean)

#         actor_mean = nn.Dense(
#             self.layer_width,
#             kernel_init=orthogonal(np.sqrt(2)),
#             bias_init=constant(0.0),
#         )(actor_mean)
#         actor_mean = activation(actor_mean)

#         actor_mean = nn.Dense(
#             self.layer_width,
#             kernel_init=orthogonal(np.sqrt(2)),
#             bias_init=constant(0.0),
#         )(actor_mean)
#         actor_mean = activation(actor_mean)

#         actor_mean = nn.Dense(
#             self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(actor_mean)
#         pi = distrax.Categorical(logits=actor_mean)

#         critic = nn.Dense(
#             self.layer_width,
#             kernel_init=orthogonal(np.sqrt(2)),
#             bias_init=constant(0.0),
#         )(x)
#         critic = activation(critic)

#         critic = nn.Dense(
#             self.layer_width,
#             kernel_init=orthogonal(np.sqrt(2)),
#             bias_init=constant(0.0),
#         )(critic)
#         critic = activation(critic)

#         critic = nn.Dense(
#             self.layer_width,
#             kernel_init=orthogonal(np.sqrt(2)),
#             bias_init=constant(0.0),
#         )(critic)
#         critic = activation(critic)

#         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
#             critic
#         )

#         return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticWithEmbedding(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_emb)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_emb
class ActorCriticConvSymbolicCraftax(nn.Module):
    action_dim: Sequence[int]
    map_obs_shape: Sequence[int]
    layer_width: int

    @nn.compact
    def __call__(self, obs):
        # Split into map and flat obs
        flat_map_obs_shape = (
            self.map_obs_shape[0] * self.map_obs_shape[1] * self.map_obs_shape[2]
        )
        image_obs = obs[:, :flat_map_obs_shape]
        image_dim = self.map_obs_shape
        image_obs = image_obs.reshape((image_obs.shape[0], *image_dim))

        flat_obs = obs[:, flat_map_obs_shape:]

        # Convolutions on map
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_obs)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_embedding)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)
        # image_embedding = jnp.concatenate([image_embedding, obs[:, : CraftaxEnv.get_flat_map_obs_shape()]], axis=-1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)
        embedding = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
