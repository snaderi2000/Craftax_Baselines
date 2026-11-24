"""Check whether the frozen symbolic death-cost critic can observe arrow direction.

Synthetic paired probes, not natural attack states. Only arrow_directions
differs within each pair. NOOP transitions validate hit versus pass-by.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image, ImageDraw, ImageFont
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Action, BlockType
from craftax.craftax_classic.renderer import render_craftax_symbolic, render_craftax_pixels
from models.actor_critic import ActorCritic
from wrappers import ClassicDeathPenaltyWrapper


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'wandb/run-20261006_194739-uj5n3za2'
    checkpoint = 9999941632
    out = root / 'concept_mapping/runs/arrow_direction_death_penalty2_10b'
    out.mkdir(parents=True, exist_ok=True)
    network = ActorCritic(17, 512)
    saved = ocp.PyTreeCheckpointer().restore(str(run / 'files/policies' / str(checkpoint) / 'default'))
    params = saved['params']
    value = jax.jit(lambda obs: network.apply(params, obs)[1])
    base = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
    env = ClassicDeathPenaltyWrapper(base, 2.)
    step = jax.jit(env.step)
    reset = jax.jit(base.reset)
    render = jax.jit(lambda s: render_craftax_pixels(s, 7))
    rows, observations = [], []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    for seed in range(9100, 9105):
        _, natural = reset(jax.random.PRNGKey(seed), base.default_params)
        y, x = map(int, natural.player_position)
        # Clear the adjacent arrow cell and both destinations identically.
        common = natural.replace(
            map=natural.map.at[y-1:y+2, x-1:x+3].set(BlockType.PATH.value),
            mob_map=jnp.zeros_like(natural.mob_map),
            player_food=9, player_drink=9, player_energy=9,
            player_recover=0., is_sleeping=False,
            zombies=natural.zombies.replace(mask=jnp.zeros_like(natural.zombies.mask)),
            skeletons=natural.skeletons.replace(mask=jnp.zeros_like(natural.skeletons.mask)),
            cows=natural.cows.replace(mask=jnp.zeros_like(natural.cows.mask)),
            arrows=natural.arrows.replace(
                mask=jnp.zeros_like(natural.arrows.mask).at[0].set(True),
                position=natural.arrows.position.at[0].set(jnp.array([y,x+1]))))
        for health in (1, 2):
            hit = common.replace(player_health=health,
                arrow_directions=common.arrow_directions.at[0].set(jnp.array([0,-1])))
            miss = hit.replace(arrow_directions=hit.arrow_directions.at[0].set(jnp.array([-1,0])))
            obs = np.stack([render_craftax_symbolic(hit), render_craftax_symbolic(miss)])
            assert np.array_equal(obs[0], obs[1])
            values = np.asarray(value(jnp.asarray(obs))).reshape(-1)
            assert values[0] == values[1]
            outcomes = [step(jax.random.PRNGKey(seed+100), s, int(Action.NOOP.value), base.default_params)
                        for s in (hit, miss)]
            assert bool(outcomes[0][3]) and not bool(outcomes[1][3])
            row = dict(seed=seed, health=health, observations_identical=True,
                       V_hit=float(values[0]), V_miss=float(values[1]),
                       miss_minus_hit=float(values[1]-values[0]),
                       hit_next_health=float(outcomes[0][1].player_health),
                       miss_next_health=float(outcomes[1][1].player_health),
                       hit_reward=float(outcomes[0][2]), miss_reward=float(outcomes[1][2]),
                       hit_terminal=bool(outcomes[0][3]), miss_terminal=bool(outcomes[1][3]))
            rows.append(row)
            observations.append(obs)
            if seed == 9100:
                panels = []
                for state, name, val in zip((hit,miss), ('Toward agent', 'Upward: passes beside'), values):
                    pixels = np.asarray(render(state)).clip(0,255).astype(np.uint8)
                    img = Image.fromarray(pixels).resize((pixels.shape[1]*4,pixels.shape[0]*4), Image.Resampling.NEAREST)
                    panel = Image.new('RGB',(img.width,img.height+65),'white')
                    panel.paste(img,(0,0))
                    ImageDraw.Draw(panel).text((4,img.height+5),f'{name}\nvalue: {val:.4f}',font=font,fill='black')
                    panels.append(panel)
                sheet = Image.new('RGB',(panels[0].width*2+16,panels[0].height),'white')
                for i,panel in enumerate(panels): sheet.paste(panel,(i*(panel.width+16),0))
                sheet.save(out/f'example_health{health}.png')
    with (out/'per_state_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/'observations.npz',observations=np.stack(observations))
    summary=dict(checkpoint=str(run/'files/policies'/str(checkpoint)),
        pairs=len(rows), health_levels=[1,2], death_cost=-2,
        synthetic=True, manipulation='Arrow one tile right: leftward versus upward; direction only within pair.',
        common_edits='Clear local corridor, remove other mobs, full resources, zero recover accumulator.',
        identical_observation_pairs=sum(r['observations_identical'] for r in rows),
        mean_miss_minus_hit=float(np.mean([r['miss_minus_hit'] for r in rows])),
        fraction_miss_higher=float(np.mean([r['miss_minus_hit']>0 for r in rows])),
        conclusion='Symbolic observations omit arrow direction. Feedforward orientation discrimination is impossible with these inputs.',
        transition_validation='Same RNG and NOOP: all toward-agent branches terminate; all upward branches survive.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
