"""Illustrate real observed-terrain memory updates along a frozen-policy rollout."""
import argparse
import base64
import csv
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image

from concept_mapping.render_ppo_paired_videos import resolve_inputs
from concept_mapping.render_ppo_episodes import _init_network
from concept_mapping.visualize_symbolic_trajectory import render_local_view, _action_name
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import load_all_textures, BlockType
from symbolic_map_memory import with_map_memory
from wrappers import ClassicDeathPenaltyWrapper, ClassicLavaPenaltyWrapper, ClassicAchievementRewardWrapper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--checkpoint', type=int)
    p.add_argument('--seed', type=int, default=18151)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--steps', type=int, default=15)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    args.run_path = None
    args.config_json = None
    path, out = resolve_inputs(args)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg = json.loads(args.config_json.read_text())
    if cfg.get('SYMBOLIC_ENCODER') != 'map_mlp':
        raise ValueError('This walkthrough currently supports map + MLP checkpoints.')
    env = make_craftax_env_from_name(cfg['ENV_NAME'], auto_reset=False)
    if cfg.get('DEATH_PENALTY', 0):
        env = ClassicDeathPenaltyWrapper(env, cfg['DEATH_PENALTY'])
    if cfg.get('LAVA_PENALTY', 0):
        env = ClassicLavaPenaltyWrapper(env, cfg['LAVA_PENALTY'])
    if cfg.get('DIAMOND_REWARD', 1) != 1 or cfg.get('PLANT_REWARD', 1) != 1:
        env = ClassicAchievementRewardWrapper(env, cfg.get('DIAMOND_REWARD', 1), cfg.get('PLANT_REWARD', 1))
    env = with_map_memory(env, cfg)
    ep = env.default_params
    net = _init_network(cfg, env, ep)
    params = ocp.PyTreeCheckpointer().restore(str(path/'default' if (path/'default').is_dir() else path))['params']
    reset, step = jax.jit(env.reset), jax.jit(env.step)

    @jax.jit
    def act(obs, key):
        pi, value = net.apply(params, obs[None])
        key, sample = jax.random.split(key)
        return jax.random.categorical(sample, pi.logits[0]), value[0], key

    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), args.episode)
    key, rk, pk = jax.random.split(key, 3)
    obs, state = reset(rk, ep)
    textures = jax.tree_util.tree_map(np.asarray, load_all_textures(7))
    snapshots, records = [], []
    previous_position = np.asarray(state.env_state.player_position).copy()
    previous_terrain = np.zeros_like(np.asarray(state.terrain))
    action_name, reward = 'RESET', 0.
    for t in range(args.steps+1):
        terrain = np.asarray(state.terrain).copy()
        seen = np.asarray(state.last_seen).copy()
        position = np.asarray(state.env_state.player_position).copy()
        displacement = position-previous_position
        # Validate the actual wrapper writes and its alignment, not a second map implementation.
        assert np.array_equal(np.asarray(state.position), position)
        h, w = terrain.shape
        rr, cc = np.meshgrid(np.arange(position[0]-3, position[0]+4), np.arange(position[1]-4, position[1]+5), indexing='ij')
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        patch = np.asarray(obs[:1323]).reshape(7, 9, 21)[..., :17].argmax(-1)+1
        assert np.array_equal(terrain[rr[valid], cc[valid]], patch[valid])
        visible = np.zeros_like(terrain, dtype=bool)
        visible[rr[valid], cc[valid]] = True
        assert np.all(seen[visible] == t)
        assert np.array_equal(terrain[~visible], previous_terrain[~visible])
        new = (terrain > 0) & (previous_terrain == 0)
        changed = (previous_terrain > 0) & (terrain != previous_terrain)
        record = dict(step=t, action=action_name, x=int(position[1]), y=int(position[0]),
                      dx=int(displacement[1]), dy=int(displacement[0]), reward=reward,
                      new_cells=int(new.sum()), updated_terrain_cells=int(changed.sum()),
                      remembered_cells=int((terrain > 0).sum()))
        records.append(record)
        snapshots.append(dict(terrain=terrain, seen=seen, position=position, new=new,
                              image=np.asarray(render_local_view(np.asarray(obs)[:1345], 7, False, 4, textures=textures))))
        print(json.dumps(record), flush=True)
        previous_terrain, previous_position = terrain, position
        if t == args.steps:
            break
        action, _, pk = act(obs, pk)
        key, sk = jax.random.split(key)
        obs, state, r, done, _ = step(sk, state, action, ep)
        action_name, reward = _action_name(int(action)), float(r)
        if bool(done):
            raise RuntimeError('Episode terminated before the requested walkthrough ended.')

    positions = np.stack([s['position'] for s in snapshots])
    r0, r1 = max(0, int(positions[:, 0].min())-5), min(h, int(positions[:, 0].max())+6)
    c0, c1 = max(0, int(positions[:, 1].min())-6), min(w, int(positions[:, 1].max())+7)
    tiles = np.asarray(textures['block_textures'])[..., :3].astype(np.uint8)
    frames = []
    pdf = PdfPages(out/'map_updates_first_15_steps.pdf')
    for t, (s, rec) in enumerate(zip(snapshots, records)):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), gridspec_kw={'width_ratios': [1, 1.2, 1.2]})
        x, y = rec['x'], rec['y']
        axes[0].imshow(s['image'], extent=(-.5, 8.5, 6.5, -.5), interpolation='nearest')
        axes[0].set(xticks=range(9), yticks=range(7), xlabel='Local tile column', ylabel='Local tile row', title=f'Current view: player at local (4, 3)\nWorld tile (x, y) = ({x}, {y})')
        codes = s['terrain'][r0:r1, c0:c1]
        mosaic = np.full((codes.shape[0]*7, codes.shape[1]*7, 3), 225, np.uint8)
        for i, j in np.ndindex(codes.shape):
            if codes[i, j]:
                mosaic[i*7:(i+1)*7, j*7:(j+1)*7] = tiles[codes[i, j]-1]
        extent = (c0-.5, c1-.5, r1-.5, r0-.5)
        axes[1].imshow(mosaic, extent=extent, interpolation='nearest')
        ages = np.ma.masked_where(codes == 0, t-s['seen'][r0:r1, c0:c1])
        cmap = plt.get_cmap('YlOrRd').copy()
        cmap.set_bad('#e1e1e1')
        heat = axes[2].imshow(ages, extent=extent, interpolation='nearest', cmap=cmap, vmin=0, vmax=max(1, args.steps))
        fig.colorbar(heat, ax=axes[2], fraction=.045, label='Steps since last observed (before scaling)')
        axes[1].set_title(f'Remembered terrain: {rec["remembered_cells"]} cells\n{rec["new_cells"]} newly seen; gray = unknown')
        axes[2].set_title('Age at the same map locations\nVisible cells reset to age 0')
        for ax in axes[1:]:
            ax.plot(positions[:t+1, 1], positions[:t+1, 0], color='#0067ce', lw=1.6)
            ax.scatter([x], [y], c='#0067ce', edgecolors='white', s=65, zorder=4)
            ax.add_patch(Rectangle((x-4.5, y-3.5), 9, 7, fill=False, edgecolor='#00bfd0', lw=2))
            ax.set(xlabel='World tile x (column)', ylabel='World tile y (row)', xticks=np.arange(c0, c1, 2), yticks=np.arange(r0, r1, 2))
            ax.set_xlim(c0-.5, c1-.5)
            ax.set_ylim(r1-.5, r0-.5)
        fig.suptitle(f'Step {t}: {rec["action"]}  |  measured displacement (dx, dy) = ({rec["dx"]}, {rec["dy"]})', fontsize=16, fontweight='bold')
        fig.text(.5, .02, 'Cyan rectangle = current 7×9 view. Blue dot/path = player. Terrain and age share coordinates. Maps shown in fixed world coordinates for explanation.', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .05, 1, .91))
        frame = out/f'step_{t:02d}.png'
        fig.savefig(frame, dpi=130)
        pdf.savefig(fig)
        plt.close(fig)
        frames.append(frame)
    pdf.close()
    with (out/'trajectory.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(out/'map_snapshots.npz', terrain=np.stack([s['terrain'] for s in snapshots]), last_seen=np.stack([s['seen'] for s in snapshots]), positions_yx=positions)
    images = [Image.open(f).convert('RGB') for f in frames]
    images[0].save(out/'map_updates.gif', save_all=True, append_images=images[1:], duration=900, loop=0)
    embedded = ['data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode() for f in frames]
    html = '''<!doctype html><meta charset="utf-8"><title>How the Craftax map updates</title>
<style>body{font-family:system-ui;margin:24px;background:#fafafa;color:#181818}img{width:100%;max-width:1600px}button,input{font-size:18px}pre{background:#eee;padding:16px;overflow:auto}p{max-width:1100px}</style>
<h1>First 15 actions: a real map + MLP episode</h1>
<p>The left panel renders pixels for you. The wrapper actually reads symbolic terrain labels. The middle and right panels share world tile coordinates; the policy instead receives a 33×33 crop centered on itself. No unseen terrain is copied from the simulator.</p>
<button onclick="move(-1)">Previous</button> <button onclick="move(1)">Next</button> <button onclick="playing=!playing">Play / pause</button>
<input id="slider" type="range" min="0" max="15" value="0" oninput="show(+this.value)"> <b id="step"></b><br><img id="frame"><pre id="data"></pre>
<h2>What the code does</h2><pre>displacement = simulator_position_now - simulator_position_before
map_position += displacement
for each visible local tile (row, col):
    world_row = map_position.row + row - 3
    world_col = map_position.col + col - 4
    terrain[world_row, world_col] = observed_terrain_type
    last_seen[world_row, world_col] = current_step
age = current_step - last_seen
policy_input = original_observation + centered_terrain_crop + scaled_age_crop</pre>
<p>A local tile is 28×28 rendered pixels here (7 texture pixels enlarged 4×). The player's tile is local column 4, row 3, centered at pixel (126, 98). Tile coordinates identify world locations; rendered pixel coordinates only locate things inside an image. Positive x means right; positive y means down. Turning, mining, or a blocked move can leave displacement at (0, 0).</p>
<p>The age plot shows raw steps for readability. The network receives log(1 + min(age,10000)) / log(10001). Unknown terrain uses a separate code and age 1. Memory is updated deterministically; the MLP learns how to use it.</p>
<script>const images=IMAGES, rows=ROWS;let current=0,playing=false;document.getElementById('slider').max=images.length-1;function show(i){current=i;document.getElementById('frame').src=images[i];document.getElementById('slider').value=i;document.getElementById('step').textContent='Step '+i;document.getElementById('data').textContent=JSON.stringify(rows[i],null,2)}function move(d){show(Math.max(0,Math.min(images.length-1,current+d)))}setInterval(()=>{if(playing)show((current+1)%images.length)},1000);show(0)</script>'''
    (out/'index.html').write_text(html.replace('IMAGES', json.dumps(embedded)).replace('ROWS', json.dumps(records)))
    (out/'metadata.json').write_text(json.dumps(dict(study=str(args.study), checkpoint=args.checkpoint, seed=args.seed, episode=args.episode, steps=args.steps, validation='Passed: visible terrain writes, position alignment, timestamps, unchanged terrain outside current view.', coordinates='World x=column, y=row; displacement uses actual simulator position differences.', terrain_codes={int(b.value)+1:b.name for b in BlockType if b.value < 17}), indent=2)+'\n')
    print(f'Finished: {out}', flush=True)


if __name__ == '__main__':
    main()
