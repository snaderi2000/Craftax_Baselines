"""Build the GRU atlas's route plate and extract its four verified whole-world replays."""
import argparse
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
STUDY = HERE / 'runs/lava_only_gru_concat_dp2_10b_seed50'
OUT = HERE / 'gru_atlas/dist/media'
TEXTURES = ['debug_tile', 'debug_tile', 'grass', 'water', 'stone', 'tree', 'wood', 'path',
            'coal', 'iron', 'diamond', 'table', 'furnace', 'sand', 'lava', 'plant_on_grass', 'ripe_plant_on_grass']
# Color-graded game sprites retain their source texture within each categorical tile.
COLORS = [(32,38,41), (23,31,30), (37,76,53), (22,57,76), (73,79,73), (31,62,45),
          (71,78,49), (56,69,55), (60,70,67), (89,91,69), (143,161,102), (77,77,56),
          (87,89,84), (102,103,70), (159,88,61), (46,81,51), (56,88,55)]


def route_plate(asset_folder):
    tile = 16
    margin = 32
    with np.load(STUDY / 'world_transitions_1000/episodes/episode_0040.npz') as ep:
        grid = ep['initial_map']
        obs_bits = np.unpackbits(ep['local_bits'], axis=-1, count=1323)
        first_sighting = np.flatnonzero(obs_bits.reshape(-1, 7, 9, 21)[:, :, :, 10].any(axis=(1,2)))[0]
        positions = ep['player_position'][:first_sighting + 1]
    textures = []
    for name, color in zip(TEXTURES, COLORS):
        image = Image.open(asset_folder / (name + '.png')).convert('L')
        img = np.asarray(image, dtype=float)
        contrast = (img - img.mean()) / max(img.std(), 20)
        textures.append(np.clip(np.asarray(color) * (1 + 0.09 * contrast[..., None]), 0, 255).astype(np.uint8))
    sheet = np.asarray(textures)[grid].transpose(0, 2, 1, 3, 4).reshape(64 * tile, 64 * tile, 3)
    extent = sheet.shape[0]
    im = Image.new('RGB', (extent + 2 * margin,) * 2, '#10211e')
    im.paste(Image.fromarray(sheet), (margin, margin))
    draw = ImageDraw.Draw(im)
    draw.rectangle((margin-1, margin-1, margin+extent, margin+extent), outline='#537567')
    for x in range(0, 65, 4):
        p = margin + x * tile
        tick = 9 if x % 16 == 0 else 4
        for ends in [(p,margin-1,p,margin-tick),(p,margin+extent,p,margin+extent+tick),
                     (margin-tick,p,margin-1,p),(margin+extent,p,margin+extent+tick,p)]:
            draw.line(ends,fill='#627e70')
    # Unique consecutive coordinates make long rests cheap to draw.
    xy = margin + tile * positions[:, ::-1].astype(int) + tile//2
    xy = xy[np.r_[True, np.any(np.diff(xy, axis=0) != 0,axis=1)]]
    points = list(map(tuple, xy))
    route = Image.new('L', im.size, 0)
    d = ImageDraw.Draw(route)
    d.line(points, fill=255, width=3, joint='curve')
    d.line(points[-27:], fill=255, width=5, joint='curve')
    im = Image.composite(Image.new('RGB', im.size, '#359b82'), im, route.filter(ImageFilter.GaussianBlur(4)))
    im = Image.composite(Image.new('RGB', im.size, '#94ddba'), im, route.point(lambda x: int(x*0.85)))
    d = ImageDraw.Draw(im)
    def marker(center, shape, color, radius):
        x,y = map(int,center)
        def path(r):
            return [(x,y-r),(x+r,y),(x,y+r),(x-r,y),(x,y-r)]
        if shape == 'diamond':
            d.polygon(path(radius+4), fill='#152c2b')
            d.polygon(path(radius), fill=color)
            d.polygon(path(radius-5), fill='#473f20')
            d.polygon(path(5), fill=color)
        else:
            draw_shape = d.ellipse if shape == 'circle' else d.rectangle
            for r,fill in [(radius+4,'#18342d'),(radius,color),(radius-4,'#18342d'),(4,color)]:
                draw_shape((x-r,y-r,x+r,y+r),fill=fill)
    marker(points[0], 'circle', '#f8f5da', 13)
    marker(points[-1], 'square', '#66f6d9', 13)
    for row,col in np.argwhere(grid == 10):
        marker((margin+col*tile+tile//2, margin+row*tile+tile//2), 'diamond', '#ffd377', 18)
    im.save(OUT / 'hero.webp', lossless=True, method=6)
    print(f'Hero plate through pre-action step {first_sighting}; {np.sum(grid==10)} initial diamond tile(s).')


def replay_assets():
    exe = shutil.which('ffmpeg')
    if not exe:
        raise RuntimeError('ffmpeg is required')
    # Sample frames with daylight and the local HUD visible. All four copies preserve the original H.264 stream.
    for episode, poster_at in [(15, 12), (25, 36), (27, 50), (40, 96.5)]:
        source, = STUDY.glob(f'diamond_miss*/*_{episode}_world.mp4')
        base = OUT / f'ep{episode}'
        subprocess.run([exe,'-v','error','-y','-i',str(source),'-map','0:v:0','-c','copy',
                        '-movflags','+faststart',str(base.with_suffix('.mp4'))],check=True)
        subprocess.run([exe,'-v','error','-y','-ss',str(poster_at),'-i',str(source),'-frames:v','1',
                        '-threads','1','-q:v','3',str(base.with_suffix('.jpg'))], check=True)


def success_excerpt():
    """Encode states 1300–1580 from the archive with the standard paired renderer."""
    sys.path.insert(0, str(HERE.parent))
    from concept_mapping.render_rnn_ppo_episodes import _MP4Stream, _label_frame
    from concept_mapping.visualize_symbolic_trajectory import render_world_frame
    from concept_mapping.world_episode import observation, spectator_state
    from craftax.craftax_classic.constants import load_all_textures
    import jax

    with np.load(STUDY / 'world_transitions_1000/episodes/episode_0000.npz') as source:
        ep = dict(source)
    textures, panel_textures = [jax.tree_util.tree_map(np.asarray,load_all_textures(k)) for k in (7,28)]
    with _MP4Stream(str(OUT / 'ep0.mp4'), 20, shutil.which('ffmpeg')) as stream:
        for t in range(1300,1581,4):
            obs = observation(ep,t)
            panel = _label_frame(obs,int(ep['actions'][t]),float(ep['rewards'][t]),bool(ep['dones'][t]),
                                 float(ep['values'][t]),t,0,7,4,textures,panel_textures)
            frame = render_world_frame(spectator_state(ep,t),obs,panel,textures)
            stream.append(frame)
            if t == 1480:
                frame.save(OUT / 'ep0.jpg', quality=90)
        assert stream.frames == 71


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,help='Directory of Craftax Classic PNG sprites')
    parser.add_argument('--success-only',action='store_true',help='Only regenerate episode 0')
    args = parser.parse_args()
    if args.assets is None:
        pkg = find_spec('craftax')
        args.assets = Path(next(iter(pkg.submodule_search_locations))) / 'craftax_classic/assets'
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.success_only:
        route_plate(args.assets)
        replay_assets()
    success_excerpt()
