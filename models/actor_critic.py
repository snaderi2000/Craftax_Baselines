import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Optional, Dict, Union
from dataclasses import field
import math
import distrax

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
        conv0 = FanInInitReLULayer(
            inchan=self.inchan,
            outchan=self.inchan,
            layer_type="conv",
            kernel_size=3,
            padding=1,
            init_scale=math.sqrt(self.init_scale),
            **self.init_norm_kwargs,
            train=self.train
        )
        conv1 = FanInInitReLULayer(
            inchan=self.inchan,
            outchan=self.inchan,
            layer_type="conv",
            kernel_size=3,
            padding=1,
            init_scale=math.sqrt(self.init_scale),
            **self.init_norm_kwargs,
            train=self.train
        )
        return x + conv1(conv0(x))

class CnnDownStack(nn.Module):
    inchan: int
    nblock: int
    outchan: int
    init_scale: float = 1.0
    pool: bool = True
    post_pool_groups: Optional[int] = None
    init_norm_kwargs: Dict = field(default_factory=dict)
    first_conv_norm: bool = False
    train: bool = True

    @nn.compact
    def __call__(self, x):
        # first convolution
        first_norm_kwargs = dict(self.init_norm_kwargs)
        if not self.first_conv_norm:
            first_norm_kwargs.update({"batch_norm": False, "group_norm_groups": None})
        x = FanInInitReLULayer(
            inchan=self.inchan,
            outchan=self.outchan,
            layer_type="conv",
            init_scale=1.0,
            kernel_size=(3,3), padding="SAME",
            **first_norm_kwargs,
            train=self.train
        )(x)
        # optional pooling + post-norm
        if self.pool:
            x = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
            if self.post_pool_groups is not None:
                x = nn.GroupNorm(num_groups=self.post_pool_groups)(x)
        # residual blocks
        for _ in range(self.nblock):
            x = CnnBasicBlock(
                inchan=self.outchan,
                init_scale=self.init_scale / math.sqrt(self.nblock),
                init_norm_kwargs=self.init_norm_kwargs,
                train=self.train
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
            init_norm_kwargs={'batch_norm': False, 'group_norm_groups': 1},
            post_pool_groups=1,
            dense_init_norm_kwargs={"layer_norm": True},
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



























################################################################################
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
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

        return pi, jnp.squeeze(critic, axis=-1)


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
