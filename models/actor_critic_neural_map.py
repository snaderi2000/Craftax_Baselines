"""Coordinate-addressed, PPO-trained spatial memory alongside terrain and GRU.

Only visible symbolic observations are written. Map contents have no assigned
semantics. Carry is (GRU, neural cells, seen mask); it is replayed through time
inside PPO, and only the initial carry is kept for each rollout.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import distrax
from flax.linen.initializers import orthogonal

WORLD_SIZE = 64


def initialize_carry(batch, config):
    return (jnp.zeros((batch, config['LAYER_SIZE'])),
            jnp.zeros((batch, WORLD_SIZE, WORLD_SIZE, config['NEURAL_MAP_CHANNELS'])),
            jnp.zeros((batch, WORLD_SIZE, WORLD_SIZE), bool))


def offsets(height, width):
    return jnp.stack(jnp.meshgrid(jnp.arange(height)-height//2,
        jnp.arange(width)-width//2, indexing='ij'), axis=-1).reshape(-1, 2)


def addresses(position, delta):
    coordinates = position[:, None, :] + delta[None, :, :]
    valid = jnp.all((coordinates >= 0) & (coordinates < WORLD_SIZE), axis=-1)
    # Positive out-of-bounds indices are dropped on writes, zero-filled on reads.
    coordinates = jnp.where(valid[..., None], coordinates, WORLD_SIZE)
    batch = jnp.arange(position.shape[0])[:, None]
    return (batch, coordinates[..., 0], coordinates[..., 1]), valid


class NeuralMapCell(nn.Module):
    width: int
    channels: int
    reset_map: bool = False
    reset_gru: bool = False

    @nn.compact
    def __call__(self, carry, inputs):
        h, memory, seen = carry
        embedding, visible, position, done = inputs
        reset = done.astype(bool) | self.reset_map
        memory = jnp.where(reset[:, None, None, None], 0., memory)
        seen = seen & ~reset[:, None, None]
        h = jnp.where((done.astype(bool) | self.reset_gru)[:, None], 0., h)
        indices, valid = addresses(position, offsets(7, 9))
        old = memory.at[indices].get(mode='fill', fill_value=0.)
        context = nn.tanh(nn.Dense(32, name='write_context')(
            jnp.concatenate((embedding, h), axis=-1)))
        features = jnp.concatenate((old, visible,
            jnp.broadcast_to(context[:, None, :], (*old.shape[:-1], 32)),
            jnp.broadcast_to(offsets(7, 9)[None]/4., (*old.shape[:-1], 2))), axis=-1)
        features = nn.tanh(nn.Dense(64, name='write_features')(features))
        candidate = nn.tanh(nn.Dense(self.channels, name='write_candidate')(features))
        gate = nn.sigmoid(nn.Dense(self.channels, bias_init=nn.initializers.constant(-1.),
            name='write_gate')(features))
        memory = memory.at[indices].set(old + gate*(candidate-old), mode='drop')
        seen = seen.at[indices].set(valid, mode='drop')

        # Local read and content/relative-position attention over every seen cell.
        local_indices, _ = addresses(position, offsets(3, 3))
        local = memory.at[local_indices].get(mode='fill', fill_value=0.).reshape(h.shape[0], -1)
        query_input = jnp.concatenate((embedding, h), axis=-1)
        query = nn.Dense(self.channels, name='attention_query')(query_input)
        spatial_query = nn.Dense(2, name='attention_position')(query_input)
        grid = jnp.stack(jnp.meshgrid(jnp.arange(WORLD_SIZE), jnp.arange(WORLD_SIZE), indexing='ij'), -1)
        relative = (grid[None] - position[:, None, None, :])/WORLD_SIZE
        logits = jnp.einsum('bhwc,bc->bhw', memory, query)/(self.channels**.5)
        logits += jnp.einsum('bhwc,bc->bhw', relative, spatial_query)
        weights = nn.softmax(jnp.where(seen, logits, -1.e9).reshape(h.shape[0], -1), axis=-1)
        global_read = jnp.einsum('bn,bnc->bc', weights, memory.reshape(h.shape[0], -1, self.channels))
        location_read = jnp.einsum('bn,bnc->bc', weights, relative.reshape(h.shape[0], -1, 2))
        read = nn.relu(nn.Dense(128, name='read_fusion')(jnp.concatenate((local, global_read, location_read), -1)))
        gru_input = nn.relu(nn.Dense(self.width, name='gru_input')(jnp.concatenate((embedding, read), -1)))
        h, _ = nn.GRUCell(self.width, name='gru')(h, gru_input)
        return (h, memory, seen), jnp.concatenate((h, read), -1)


class NeuralMapActorCritic(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, carry, inputs):
        obs, dones = inputs
        cfg = self.config
        cells = cfg.get('MAP_VIEW_SIZE', 33)**2
        extra = 40 if cfg.get('MAP_SURVIVAL_FEATURES', 'none') != 'none' else 0
        if obs.shape[-1] != 1345 + 2*cells + extra + 2:
            raise ValueError('Neural map requires terrain observation and two position coordinates')
        position = obs[..., -2:].astype(jnp.int32)
        visible = obs[..., :1323].reshape((*obs.shape[:-1], 63, 21))
        terrain = nn.Embed(18, cfg.get('MAP_EMBEDDING_SIZE', 4), name='terrain_embedding')(
            obs[..., 1345:1345+cells].astype(jnp.int32))
        features = jnp.concatenate((obs[..., :1345], terrain.reshape((*obs.shape[:-1], -1)),
            obs[..., 1345+cells:-2]), -1)
        embedding = nn.relu(nn.Dense(cfg['LAYER_SIZE'], kernel_init=orthogonal(2**.5), name='observation')(features))
        # Recompute the cell's intermediates during backprop; preserve temporal
        # gradients through the entire rollout without storing every activation.
        scanned = nn.scan(nn.remat(NeuralMapCell, prevent_cse=False),
            variable_broadcast='params', split_rngs={'params': False}, in_axes=0, out_axes=0)
        carry, recurrent = scanned(cfg['LAYER_SIZE'], cfg['NEURAL_MAP_CHANNELS'],
            cfg.get('NEURAL_MAP_RESET_EVERY_STEP', False), cfg.get('RESET_GRU_EVERY_STEP', False),
            name='memory')(carry, (embedding, visible, position, dones))
        features = jnp.concatenate((embedding, recurrent), -1) if cfg.get('GRU_CONCAT_OBSERVATION', False) else recurrent
        actor, critic = features, features
        for i in range(2):
            actor = nn.relu(nn.Dense(cfg['LAYER_SIZE'], kernel_init=orthogonal(2), name=f'actor_{i}')(actor))
            critic = nn.relu(nn.Dense(cfg['LAYER_SIZE'], kernel_init=orthogonal(2), name=f'critic_{i}')(critic))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(.01), name='policy')(actor)
        value = nn.Dense(1, kernel_init=orthogonal(1.), name='value')(critic)[..., 0]
        return carry, distrax.Categorical(logits=logits), value
