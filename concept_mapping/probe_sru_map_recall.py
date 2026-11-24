"""Fit the same fresh map readout on each frozen SRU, then score untouched test worlds."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from concept_mapping.sru_reconstruction_data import OUT, SEED, STRIDE, PAIRS, UNIFORM, RESOURCES, UNKNOWN
from models.sru_map_reconstruction import init_decoder, reconstruction_loss
from ppo_rnn_og import ActorCriticRNN


def replay(split, params, cfg):
    net = ActorCriticRNN(17, cfg)
    @jax.jit
    def run(carry, bits, stats, previous, motion):
        local = jnp.unpackbits(bits, axis=-1)[..., :1323]
        obs = jnp.concatenate((local, stats, jax.nn.one_hot(previous, 17), motion), -1)
        h, _, _, sequence = net.apply(params, carry, (obs, jnp.zeros(obs.shape[:2], bool)), return_memory=True)
        return h, sequence[STRIDE-1::STRIDE]

    length, frames = split['bits'].shape[:2]
    output = np.empty((length, frames // STRIDE, cfg['LAYER_SIZE']), np.float32)
    for start in range(0, length, 16):
        end = min(start + 16, length)
        h = jnp.zeros((end-start, cfg['LAYER_SIZE']))
        for t in range(0, frames, 128):
            batch = [np.swapaxes(split[k][start:end, t:t+128], 0, 1) for k in ('bits','stats','previous','motion')]
            h, samples = run(h, *batch)
            output[start:end, t//STRIDE:(t+128)//STRIDE] = np.swapaxes(np.asarray(samples), 0, 1)
    return output


def dataset(name, params, cfg, archive):
    with np.load(archive / (name + '.npz')) as packed:
        data = dict(packed)
    memory = replay(data, params, cfg)
    labels = data['labels']
    known = (labels >= 0) & (labels != UNKNOWN)
    targets = np.where(known, labels, np.maximum(data['terrain'], 0))
    # Offline queries exclude live tiles. Group two and the resource pairs have ages 100–1000.
    age = np.where(known, 1, -1)
    age[..., UNIFORM:2*UNIFORM] = np.where(known[..., UNIFORM:2*UNIFORM], 100, -1)
    age[..., 3*UNIFORM:] = np.where(known[..., 3*UNIFORM:], 100, -1)
    query = np.concatenate((data['offsets'], targets[..., None], age[..., None], (labels >= 0)[..., None]), axis=-1)
    return memory, query.astype(np.int16)


def fit_and_score(args):
    cfg = json.loads(args.config.read_text())
    path = args.checkpoint.resolve()
    payload = path / 'default' if (path / 'default').is_dir() else path
    params = ocp.PyTreeCheckpointer().restore(str(payload))['params']
    memory, queries = dataset('train', params, cfg, args.archive)
    worlds, times, width = memory.shape
    decoder = init_decoder(width, jax.random.PRNGKey(SEED))
    @jax.jit
    def update(state, h, q):
        grad = jax.grad(lambda p: reconstruction_loss(p, h, q)[0])(state.params)
        return state.apply_gradients(grads=grad)
    valid = np.flatnonzero(queries[..., 4].any(-1).ravel())
    memory = memory.reshape(-1, width)
    queries = queries.reshape(-1, *queries.shape[-2:])
    rng = np.random.default_rng(SEED)
    for _ in range(12):
        ix = rng.permutation(valid)
        for start in range(0, len(ix), 128):
            batch = ix[start:start + 128]
            decoder = update(decoder, memory[batch], queries[batch])
    memory, queries = dataset('test', params, cfg, args.archive)
    @jax.jit
    def predict(h, offsets):
        return jax.nn.softmax(decoder.apply_fn(decoder.params, h, offsets)[0])
    probabilities = np.concatenate([np.asarray(predict(h, q[..., :2])) for h, q in
                                   zip(np.array_split(memory, 8), np.array_split(queries, 8))])
    metrics = {}
    per_world = {}
    def record(key, values, mask):
        n = mask.sum(axis=(1, 2))
        total = np.where(mask, values, 0).sum(axis=(1, 2))
        per_world[key] = {'hits':total.tolist(), 'count':n.tolist()}
        metrics[key] = {'rate':float(total.sum()/n.sum()) if n.sum() else None, 'queries':int(n.sum()),
                        'worlds_contributing':int((n>0).sum())}
    for i, name in enumerate(('seen_age1_99','seen_age100_1000','unseen_terrain')):
        group = slice(i*UNIFORM, (i+1)*UNIFORM)
        record(name, probabilities[..., group, :].argmax(-1)==queries[..., group, 2], queries[..., group, 4]>0)
    for i, (name, codes) in enumerate(RESOURCES.items()):
        start = 3*UNIFORM + i*2*PAIRS
        score = probabilities[..., start:start+2*PAIRS, :][..., list(codes)].sum(-1)
        positive, distractor = score[..., 0::2], score[..., 1::2]
        valid = queries[..., start:start+2*PAIRS:2, 4]>0
        record(name+'_location_pair_age100_1000', (positive>distractor) + .5*(positive==distractor), valid)
    report = dict(checkpoint=str(path), archive=str(args.archive), seed=SEED, train_worlds=worlds,
                  test_worlds=len(memory), prefix_steps=times*STRIDE, probe_epochs=12, metrics=metrics,
                  per_test_world=per_world)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2)+'\n')
    tmp.replace(args.out)
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--archive',type=Path,default=OUT)
    fit_and_score(ap.parse_args())
