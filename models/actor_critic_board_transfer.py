"""A whole-board attention adapter connecting to the first hidden states of V1."""
import flax.linen as nn
from flax.core import copy
import jax
import jax.numpy as jnp
import optax

from models.actor_critic import ActorCritic
from symbolic_map_memory import BOARD_CELL_FIELDS, BOARD_OBS_SIZE, GLOBAL_MAP_SIZE, PLAYER_BIT, PLAYER_FEATURES, TERRAIN_TYPES


def adapter_only_optimizer(optimizer):
    """Keep gradients, clipping, and Adam moments local to the board reader."""
    def init(params):
        return optimizer.init(params['params']['board_adapter'])

    def update(grads, state, params=None):
        adapter_params = None if params is None else params['params']['board_adapter']
        updates, state = optimizer.update(grads['params']['board_adapter'], state, adapter_params)
        zeros = jax.tree_util.tree_map(jnp.zeros_like, grads)
        modules = copy(zeros['params'], {'board_adapter': updates})
        return copy(zeros, {'params': modules}), state

    return optax.GradientTransformation(init, update)


class BoardAttentionAdapter(nn.Module):
    width: int

    @nn.compact
    def __call__(self, cells, ids, flags, attributes):
        marker = flags[..., -1]
        pos = jnp.argmax(marker, axis=-1)
        address = jnp.arange(GLOBAL_MAP_SIZE**2)
        offsets = jnp.stack((address//GLOBAL_MAP_SIZE - (pos//GLOBAL_MAP_SIZE)[..., None],
                             address%GLOBAL_MAP_SIZE - (pos%GLOBAL_MAP_SIZE)[..., None]), axis=-1)
        relative = jnp.sign(offsets)*jnp.log1p(jnp.abs(offsets))/jnp.log(float(GLOBAL_MAP_SIZE))
        fields = jnp.concatenate((cells, relative), axis=-1)
        # The serialized player scalars belong to the marker cell.
        marked = jnp.take_along_axis(cells, pos[..., None, None], axis=-2)[..., 0, :]
        player = jnp.concatenate((marked, attributes*marker.max(-1, keepdims=True)), axis=-1)
        centers_init = jnp.array([[0.,0.],[-1.,0.],[1.,0.],[0.,-1.],[0.,1.],[0.,0.],[0.,0.],[0.,0.]])
        centers = self.param('centers', lambda k: centers_init)
        log_radius = self.param('log_radius', lambda k: jnp.log(jnp.array([.7,.7,.7,.7,.7,4.,16.,64.])))
        heads = centers.shape[0]
        query = nn.Dense(heads*fields.shape[-1], name='query')(
            nn.tanh(nn.Dense(96, name='player')(player)))
        query = query.reshape((*player.shape[:-1], heads, fields.shape[-1]))
        logits = jnp.einsum('...hc,...nc->...hn', query, fields)/jnp.sqrt(float(fields.shape[-1]))
        # Gaussian position biases. Every head can read every observed cell.
        # The term constant across cells cancels inside each head's softmax.
        distance = (jnp.sum(offsets**2, axis=-1)[..., None, :] -
                    2*jnp.einsum('...nd,hd->...hn', offsets, centers))
        logits -= distance / (2*jnp.exp(2*jnp.clip(log_radius, -3., 6.))[:, None])
        weights = nn.softmax(jnp.where((ids != 0)[..., None, :], logits, -1e9), axis=-1)
        pooled = jnp.einsum('...hn,...nc->...hc', weights, fields)
        summary = jnp.concatenate((player, pooled.reshape((*player.shape[:-1], -1))), axis=-1)
        features = nn.tanh(nn.Dense(self.width, name='features')(summary))
        hidden = nn.tanh(nn.Dense(2*self.width, name='v1_activations')(features))
        return hidden.reshape((*player.shape[:-1], 2, self.width))


class BoardV1ActorCritic(nn.Module):
    action_dim: int
    width: int
    embedding_size: int

    def setup(self):
        self.terrain_embedding = nn.Embed(TERRAIN_TYPES+1, self.embedding_size)
        self.board_adapter = BoardAttentionAdapter(self.width)
        self.actor_critic = ActorCritic(self.action_dim, self.width, 'tanh')

    def with_hidden(self, obs):
        if obs.shape[-1] != BOARD_OBS_SIZE:
            raise ValueError('Frozen V1 adapter requires the dated board-only observation')
        board = obs[..., :-PLAYER_FEATURES].reshape((*obs.shape[:-1], GLOBAL_MAP_SIZE**2, BOARD_CELL_FIELDS))
        ids = board[..., 0].astype(jnp.int32)
        flags = ((board[..., 2:3].astype(jnp.int32) & jnp.array([1,2,4,8,PLAYER_BIT])) != 0).astype(obs.dtype)
        cells = jnp.concatenate((self.terrain_embedding(ids), jax.nn.one_hot(ids, TERRAIN_TYPES+1),
                                 board[..., 1:2], flags), axis=-1)
        hidden = self.board_adapter(cells, ids, flags, obs[..., -PLAYER_FEATURES:])
        pi, value = self.actor_critic(hidden, from_first_hidden=True)
        return pi, value, hidden

    def __call__(self, obs):
        pi, value, _ = self.with_hidden(obs)
        return pi, value
