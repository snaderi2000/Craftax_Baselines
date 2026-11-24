"""Feedforward actor/critic reading explicit terrain memory."""
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
from models.actor_critic import ActorCritic
from symbolic_map_memory import GLOBAL_MAP_FEATURES, GLOBAL_MAP_SIZE, LOCAL_SIZE, LOCAL_SHAPE, TERRAIN_TYPES
from symbolic_map_memory import BOARD_CELL_FIELDS, BOARD_OBS_SIZE, PLAYER_BIT, PLAYER_FEATURES


class GlobalMapReader(nn.Module):
    width: int
    features_only: bool = False
    patch_mlp: bool = False
    spatial_regions: bool = False
    fine_shortcut: bool = False

    @nn.compact
    def __call__(self, memory):
        size = GLOBAL_MAP_SIZE
        cells = size**2
        grid_shape = (*memory.shape[:-1], size, size)
        terrain = memory[..., :cells].astype(jnp.int32).reshape(grid_shape)
        age = memory[..., cells:2*cells].reshape(grid_shape)
        position = jnp.rint(memory[..., -2:] * (size-1)).astype(jnp.int32)
        index = position[..., 0] * size + position[..., 1]
        marker = jax.nn.one_hot(index, cells).reshape(grid_shape)
        tiles = nn.Embed(18, 4, name='terrain')(terrain)
        features = jnp.concatenate((tiles, age[..., None], marker[..., None]), axis=-1)
        if self.spatial_regions:
            # Fixed world regions retain their addresses. Relative coordinates
            # tell the reader where each tile lies with respect to the player.
            axis = jnp.arange(size, dtype=age.dtype) / (size - 1)
            row_offset = axis[:, None] - memory[..., -2, None, None]
            col_offset = axis[None, :] - memory[..., -1, None, None]
            offsets = jnp.stack(jnp.broadcast_arrays(row_offset, col_offset), axis=-1)
            features = jnp.concatenate((features, offsets), axis=-1)
            patches = nn.relu(nn.Conv(8, (4, 4), strides=(4, 4), name='patches')(features))
            # Two non-overlapping reductions: each 32-vector sees exactly 8×8 tiles.
            regions = nn.relu(nn.Conv(32, (2, 2), strides=(2, 2), name='regions')(patches))
            if self.fine_shortcut:
                # Each region gets TL, TR, BL, BR in order, eight channels per patch.
                lead = patches.shape[:-3]
                ordered = patches.reshape((*lead, 8, 2, 8, 2, 8)).swapaxes(-4, -3)
                gate = self.param('fine_gate', nn.initializers.zeros, (32,))
                regions = regions + gate * ordered.reshape(regions.shape)
            return regions
        if self.patch_mlp:
            # Each fixed 8x8 patch has its own dense projection. Keep cell order
            # inside patches and patch order on the board; no sliding filters.
            lead = memory.shape[:-1]
            patches = features.reshape((*lead, 8, 8, 8, 8, 6)).swapaxes(-4, -3)
            patches = patches.reshape((*lead, 64, 384))
            weights = self.param('patch_weights', nn.initializers.normal(384**-0.5), (64, 384, 4))
            bias = self.param('patch_bias', nn.initializers.zeros, (64, 4))
            features = nn.relu(jnp.einsum('...pc,pcf->...pf', patches, weights) + bias)
        else:
            features = nn.relu(nn.Conv(8, (4, 4), strides=(4, 4), name='patches')(features))
            features = nn.relu(nn.Conv(16, (3, 3), strides=(2, 2), name='context')(features))
        features = features.reshape((*memory.shape[:-1], -1))
        features = nn.relu(nn.Dense(64, name='features')(features))
        if self.features_only:
            return features
        offsets = nn.Dense(2*self.width, use_bias=False, kernel_init=nn.initializers.zeros,
                           name='residual')(features)
        return offsets.reshape((*memory.shape[:-1], 2, self.width))


class LandmarkReader(nn.Module):
    width: int

    @nn.compact
    def __call__(self, stats, records):
        from symbolic_landmarks import SLOTS, FIELDS
        slots = records.reshape((*records.shape[:-1],SLOTS,FIELDS))
        # Learned per-resource gates use inventory, needs and relative geometry.
        scores = nn.Dense(SLOTS,name='query')(jnp.concatenate((stats,records),axis=-1))
        available = slots[...,0] > 0
        weights = jax.nn.softmax(jnp.where(available,scores,-1e8),axis=-1)
        selected = (slots*weights[...,None]).reshape(records.shape)
        hidden = nn.tanh(nn.Dense(64,use_bias=False,name='features')(selected))
        out = nn.Dense(2*self.width,use_bias=False,kernel_init=nn.initializers.zeros,name='residual')(hidden)
        return out.reshape((*records.shape[:-1],2,self.width))


class MapMLPActorCritic(nn.Module):
    action_dim:int=17
    width:int=512
    view_size:int=33
    embedding_size:int=4
    global_readout:bool=False
    landmarks:bool=False
    crop_from_world:bool=False

    def from_observed_world(self, obs):
        """Apply crop-plus-world weights to a live view and a single dated world map."""
        if not self.global_readout or self.landmarks or obs.shape[-1] != LOCAL_SIZE + GLOBAL_MAP_FEATURES:
            raise ValueError('Expected a crop-plus-world policy and an observed-only single-map input')
        world = obs[..., LOCAL_SIZE:]
        side = GLOBAL_MAP_SIZE
        radius = self.view_size // 2
        positions = jnp.rint(world[..., -2:] * (side - 1)).astype(jnp.int32).reshape(-1, 2)

        def centered(plane, position, padding):
            return jax.lax.dynamic_slice(jnp.pad(plane, radius, constant_values=padding),
                                         tuple(position), (self.view_size, self.view_size))

        def crop(start, padding):
            planes = world[..., start:start + side**2].reshape(-1, side, side)
            read = jax.vmap(centered, in_axes=(0, 0, None))(planes, positions, padding)
            return read.reshape((*obs.shape[:-1], self.view_size**2))

        unknown_age = jnp.log1p(jnp.asarray(10000)) / jnp.log(10001.)
        return self._read(jnp.concatenate((obs[..., :LOCAL_SIZE], crop(0, 0),
                                           crop(side**2, unknown_age), world), axis=-1))

    def __call__(self,obs):
        return self.from_observed_world(obs) if self.crop_from_world else self._read(obs)

    @nn.compact
    def _read(self,obs):
        cells=self.view_size**2
        local_size = 1345+2*cells
        from symbolic_landmarks import SLOTS, FIELDS
        end = local_size+(GLOBAL_MAP_FEATURES if self.global_readout else 0)
        if obs.shape[-1]!=end+(SLOTS*FIELDS if self.landmarks else 0):
            raise ValueError('Map MLP requires the configured augmented map observation')
        terrain=obs[...,1345:1345+cells].astype(jnp.int32)
        # Categorical embeddings avoid imposing an ordering on tile IDs.
        memory=nn.Embed(18,self.embedding_size,name='terrain_embedding')(terrain)
        features=jnp.concatenate((obs[...,:1345],memory.reshape((*obs.shape[:-1],-1)),
                                  obs[...,1345+cells:local_size]),axis=-1)
        offsets = (GlobalMapReader(self.width, name='global_reader')(obs[..., local_size:end])
                   if self.global_readout else None)
        if self.landmarks:
            offsets = offsets + LandmarkReader(self.width,name='landmark_reader')(obs[...,1323:1345],obs[...,end:])
        return ActorCritic(self.action_dim,self.width,'tanh',name='actor_critic')(
            features, input_offsets=offsets)


class SingleMapActorCritic(nn.Module):
    action_dim: int
    width: int
    patch_mlp: bool = False
    spatial_regions: bool = False
    fine_shortcut: bool = False

    @nn.compact
    def __call__(self, obs):
        if obs.shape[-1] != LOCAL_SIZE + GLOBAL_MAP_FEATURES:
            raise ValueError('Expected the live observation and one dated world map')
        live = obs[..., :LOCAL_SIZE]
        world = GlobalMapReader(self.width, features_only=True, patch_mlp=self.patch_mlp,
                                spatial_regions=self.spatial_regions, fine_shortcut=self.fine_shortcut,
                                name='global_reader')(obs[..., LOCAL_SIZE:])
        if self.spatial_regions:
            world = world.reshape((*obs.shape[:-1], 8 * 8 * 32))
        else:
            live = nn.Dense(128, use_bias=False, name='live_encoder')(live)
        return ActorCritic(self.action_dim, self.width, 'tanh', name='actor_critic')(
            jnp.concatenate((live, world), axis=-1))


class PlayerBoardAttention(nn.Module):
    """One player query over observed cells, with player-relative coordinates."""
    output_width: int

    @nn.compact
    def __call__(self, tokens, tile_ids, marker):
        index = jnp.argmax(marker, axis=-1)
        player = jnp.take_along_axis(tokens, index[..., None, None], axis=-2)[..., 0, :]
        addresses = jnp.arange(GLOBAL_MAP_SIZE**2)
        rows = addresses // GLOBAL_MAP_SIZE - (index // GLOBAL_MAP_SIZE)[..., None]
        cols = addresses % GLOBAL_MAP_SIZE - (index % GLOBAL_MAP_SIZE)[..., None]
        offsets = jnp.stack((rows, cols, abs(rows) + abs(cols)), axis=-1)
        relative = jnp.sign(offsets) * jnp.log1p(jnp.abs(offsets)) / jnp.log(float(GLOBAL_MAP_SIZE))
        cells = jnp.concatenate((tokens, relative), axis=-1)
        # A linear key projection can equivalently be applied to the one query.
        query = nn.tanh(nn.Dense(32, name='query')(player))
        query = nn.Dense(cells.shape[-1], use_bias=False, name='key_query')(query)
        logits = jnp.einsum('...nc,...c->...n', cells, query) / jnp.sqrt(32.)
        weights = nn.softmax(jnp.where(tile_ids != 0, logits, -1e9), axis=-1)
        context = jnp.einsum('...n,...nc->...c', weights, cells)
        # A zero residual preserves both pretrained decision heads on transfer.
        return nn.Dense(self.output_width, use_bias=False, kernel_init=nn.initializers.zeros,
                        name='output')(context)


class FixedBoardActorCritic(nn.Module):
    """Both heads share one world-indexed representation, including the player."""
    action_dim: int
    width: int
    embedding_size: int
    attention: bool = False

    @nn.compact
    def __call__(self, obs):
        if obs.shape[-1] != BOARD_OBS_SIZE:
            raise ValueError('Expected dated world cells and player attributes')
        cells = obs[..., :-PLAYER_FEATURES].reshape((*obs.shape[:-1], GLOBAL_MAP_SIZE**2, BOARD_CELL_FIELDS))
        tile_id = cells[..., 0].astype(jnp.int32)
        terrain = nn.Embed(TERRAIN_TYPES + 1, self.embedding_size, name='terrain')(tile_id)
        flags = cells[..., 2].astype(jnp.int32)[..., None]
        occupancy = ((flags & jnp.array([1, 2, 4, 8, PLAYER_BIT])) != 0).astype(obs.dtype)
        player = occupancy[..., -1:]
        # Store inventory, vitals, facing, daylight and sleep at the player marker.
        attributes = nn.Dense(8, use_bias=False, name='player_attributes')(obs[..., -PLAYER_FEATURES:])
        tokens = jnp.concatenate((terrain, cells[..., 1:2], occupancy,
                                  player * attributes[..., None, :]), axis=-1)
        # Each of the 4,096 fixed positions has its own row of learned weights.
        spatial = nn.Dense(64, name='world_positions')(jnp.swapaxes(tokens, -1, -2))
        features = spatial.reshape((*obs.shape[:-1], -1))
        if self.attention:
            features = features + PlayerBoardAttention(features.shape[-1], name='board_attention')(
                tokens, tile_id, player[..., 0])
        return ActorCritic(self.action_dim, self.width, 'tanh', name='actor_critic')(features)


def map_mlp_network(config,action_dim=17):
    internal_crop = config.get('MAP_INTERNAL_CROP', False)
    spatial_regions = config.get('MAP_SPATIAL_REGIONS', False)
    fine_shortcut = config.get('MAP_FINE_SHORTCUT', False)
    if config.get('MAP_V1_ATTENTION_ADAPTER'):
        if (not config.get('MAP_BOARD_ONLY') or not config.get('MAP_SINGLE_READOUT')
                or not config.get('MAP_GLOBAL_READOUT') or config.get('MAP_FULL_TERRAIN')
                or config.get('MAP_BOARD_ATTENTION') or internal_crop or spatial_regions or fine_shortcut):
            raise ValueError('The V1 attention adapter requires one fixed observed board')
        from models.actor_critic_board_transfer import BoardV1ActorCritic
        return BoardV1ActorCritic(action_dim, int(config.get('LAYER_SIZE', 512)),
                                  int(config.get('MAP_EMBEDDING_SIZE', 4)))
    if config.get('MAP_BOARD_ATTENTION') and not config.get('MAP_BOARD_ONLY'):
        raise ValueError('Player-query attention requires the fixed board-only policy')
    if config.get('MAP_BOARD_ONLY'):
        if (not config.get('MAP_SINGLE_READOUT') or not config.get('MAP_GLOBAL_READOUT') or
                internal_crop or spatial_regions or fine_shortcut or config.get('MAP_PATCH_MLP') or
                config.get('MAP_FULL_TERRAIN') or config.get('MAP_LANDMARKS') or
                config.get('MAP_SURVIVAL_FEATURES', 'none') != 'none'):
            raise ValueError('Board-only policy requires a single observed board without extra readers')
        return FixedBoardActorCritic(action_dim, int(config.get('LAYER_SIZE', 512)),
            int(config.get('MAP_EMBEDDING_SIZE', 4)), config.get('MAP_BOARD_ATTENTION', False))
    if fine_shortcut and not spatial_regions:
        raise ValueError('The fine map shortcut requires spatial regions')
    if spatial_regions and (not config.get('MAP_SINGLE_READOUT') or not config.get('MAP_GLOBAL_READOUT')
                            or internal_crop or config.get('MAP_PATCH_MLP') or config.get('MAP_FULL_TERRAIN')
                            or config.get('MAP_LANDMARKS')):
        raise ValueError('Spatial regions require one observed-only world map without a crop')
    if internal_crop and (not config.get('MAP_SINGLE_READOUT') or not config.get('MAP_GLOBAL_READOUT')
                          or config.get('MAP_PATCH_MLP') or config.get('MAP_FULL_TERRAIN') or config.get('MAP_LANDMARKS')):
        raise ValueError('The internal crop requires one observed world map and the original Map-MLP reader')
    if config.get('MAP_SINGLE_READOUT') and not internal_crop:
        return SingleMapActorCritic(action_dim, int(config.get('LAYER_SIZE', 512)),
                                    config.get('MAP_PATCH_MLP', False), spatial_regions, fine_shortcut)
    return MapMLPActorCritic(action_dim,int(config.get('LAYER_SIZE',512)),
        int(config.get('MAP_VIEW_SIZE',33)),int(config.get('MAP_EMBEDDING_SIZE',4)),
        config.get('MAP_GLOBAL_READOUT',False),bool(config.get('MAP_LANDMARKS')),internal_crop)


def restore_map_shortcut_state(checkpointer, payload, template):
    """Resume V5 or an existing shortcut checkpoint, including Adam's moments."""
    metadata = checkpointer.metadata(str(payload))
    reader = metadata['params']['params']['global_reader']
    if 'fine_gate' in reader:
        return checkpointer.restore(str(payload), item=template)
    if 'regions' not in reader:
        raise ValueError('The fine map shortcut requires a V5 regional checkpoint')

    def is_reader(tree):
        return isinstance(tree, (dict, FrozenDict)) and 'patches' in tree and 'regions' in tree

    def set_gate(tree, add):
        data = dict(tree)
        if add:
            data['fine_gate'] = jnp.zeros(32, dtype=data['regions']['bias'].dtype)
        else:
            del data['fine_gate']
        return freeze(data) if isinstance(tree, FrozenDict) else data

    # The same reader subtree occurs in parameters and both Adam moments.
    def adapt(state, add):
        return jax.tree_util.tree_map(lambda x: set_gate(x, add) if is_reader(x) else x,
                                      state, is_leaf=is_reader)
    old = checkpointer.restore(str(payload), item=adapt(template, False))
    return adapt(old, True)


def transfer_map_mlp_params(target, source):
    """Restore Map + MLP weights, retaining a new reader's zero output."""
    old = unfreeze(source['params'])
    new = unfreeze(target['params'])
    if 'regions' in new.get('global_reader', {}) and 'terrain_embedding' in old:
        # Preserve the teacher's live-input contribution and upper decision layers.
        heads, live_kernels = fold_visible_crop_into_live(old)
        reader = new['global_reader']
        trained_reader = old['global_reader']
        reader['terrain'] = trained_reader['terrain']
        trained_patches = trained_reader['patches']
        reader['patches'] = dict(bias=trained_patches['bias'],
                                 kernel=np.pad(trained_patches['kernel'], ((0,0),(0,0),(0,2),(0,0))))
        for name, live in zip(('Dense_0', 'Dense_4'), live_kernels):
            regions = np.zeros_like(new['actor_critic'][name]['kernel'][LOCAL_SIZE:])
            heads[name]['kernel'] = np.concatenate((live, regions))
        old = dict(global_reader=reader, actor_critic=heads)
    elif 'live_encoder' in new and 'live_encoder' not in old:
        old = compress_map_inputs(old)
        if 'patch_weights' in new['global_reader']:
            # The tile vocabulary transfers. The spatial layers learn from the teacher.
            old['global_reader'] = {**new['global_reader'], 'terrain': old['global_reader']['terrain']}
    if set(old) not in (set(new), set(new)-{'global_reader'}, set(new)-{'landmark_reader'}):
        raise ValueError('Checkpoint does not match the Map + MLP parameter structure')
    expected = {name: new[name] for name in old}
    shapes = lambda tree: jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), tree)
    if shapes(expected) != shapes(old):
        raise ValueError('Checkpoint Map + MLP parameter shapes or dtypes differ')
    transferred = {**target, 'params': {**new, **old}}
    return freeze(transferred) if isinstance(target, FrozenDict) else transferred


def widen_map_mlp_params(target, source, old_size, new_size):
    """Place both first-layer kernels at the same relative coordinates in a wider crop."""
    if old_size < 9 or new_size <= old_size or old_size % 2 != 1 or new_size % 2 != 1:
        raise ValueError('Expected two odd crop sizes with a larger target')
    old = unfreeze(source['params'])
    if set(old) != {'actor_critic', 'terrain_embedding'} or set(target['params']) != set(old):
        raise ValueError('Crop widening requires two plain terrain-memory MLPs')
    embedding_size = old['terrain_embedding']['embedding'].shape[1]
    pad = (new_size - old_size)//2
    for name in ('Dense_0', 'Dense_4'):
        head = old['actor_critic'][name]
        kernel = np.asarray(head['kernel'])
        width = kernel.shape[1]
        if kernel.shape[0] != LOCAL_SIZE + old_size**2*(embedding_size+1):
            raise ValueError('Source checkpoint disagrees with the requested old crop size')
        local, terrain, ages = np.split(kernel, [LOCAL_SIZE, kernel.shape[0]-old_size**2])
        def widen(rows, channels):
            grid = rows.reshape(old_size, old_size, channels, width)
            return np.pad(grid, ((pad,pad),(pad,pad),(0,0),(0,0))).reshape(-1, width)
        head['kernel'] = np.concatenate((local, widen(terrain, embedding_size), widen(ages, 1)), axis=0)
    return transfer_map_mlp_params(target, {'params': old})


def fold_visible_crop_into_live(old):
    """Fold known local tiles into the live one-hot weights; assume unknown elsewhere."""
    if 'global_reader' not in old or 'terrain_embedding' not in old:
        raise ValueError('Single-map transfer requires a trained crop-plus-global Map + MLP')
    heads = dict(old['actor_critic'])
    embedding = np.asarray(old['terrain_embedding']['embedding'])
    live_kernels = []
    head_names = ('Dense_0', 'Dense_4')
    for name in head_names:
        kernel = np.asarray(heads[name]['kernel'])
        live = kernel[:LOCAL_SIZE].copy()
        cells, remainder = divmod(kernel.shape[0]-LOCAL_SIZE, embedding.shape[1]+1)
        side = int(np.sqrt(cells))
        if remainder or side*side != cells or side < 9 or side % 2 == 0:
            raise ValueError('Invalid teacher crop dimensions')
        tile_weights = kernel[LOCAL_SIZE:-cells].reshape(cells, embedding.shape[1], -1)
        age_weights = kernel[-cells:]
        # At reset, cells outside the live window have code 0 and maximum age.
        unknown = embedding[0] @ tile_weights + age_weights
        bias = np.asarray(heads[name]['bias']) + unknown.sum(axis=0)
        for row in range(LOCAL_SHAPE[0]):
            for col in range(LOCAL_SHAPE[1]):
                cell = (side//2 + row-3)*side + side//2 + col-4
                start = (row*9+col)*LOCAL_SHAPE[2]
                live[start:start+TERRAIN_TYPES] += embedding[1:] @ tile_weights[cell]
                bias -= unknown[cell]
        live_kernels.append(live)
        heads[name] = dict(bias=bias)
    return heads, live_kernels


def compress_map_inputs(old):
    """Factor the teacher's live-input weights and retain its global-map and decision layers."""
    heads, live_kernels = fold_visible_crop_into_live(old)
    reader = dict(old['global_reader'])
    residual = np.asarray(reader.pop('residual')['kernel'])
    # Best rank-128 shared linear factorization of the actor and critic live kernels.
    left, singular, right = np.linalg.svd(np.concatenate(live_kernels, axis=1), full_matrices=False)
    rank = min(128, len(singular))
    projection = np.zeros((LOCAL_SIZE, 128), dtype=left.dtype)
    projection[:, :rank] = left[:, :rank]
    output = np.zeros((128, right.shape[1]), dtype=right.dtype)
    output[:rank] = singular[:rank, None] * right[:rank]
    for name, local, world in zip(('Dense_0','Dense_4'), np.hsplit(output, 2), np.hsplit(residual, 2)):
        heads[name]['kernel'] = np.concatenate((local, world), axis=0)
    return dict(live_encoder=dict(kernel=projection), global_reader=reader, actor_critic=heads)
