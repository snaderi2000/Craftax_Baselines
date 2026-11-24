"""Derive the GRU field atlas from the archived local observations and world states."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'runs/lava_only_gru_concat_dp2_10b_seed50/world_transitions_1000'
OUT = HERE / 'gru_atlas/dist'
WIDTH = 64
DIAMOND_TILE = 10  # craftax.craftax_classic.constants.BlockType.DIAMOND
OBS_SHAPE = (7, 9)
DIAMOND_BITS = np.arange(63) * 21 + DIAMOND_TILE
CELL_DISTANCE = (abs(np.arange(7)[:, None] - 3) + abs(np.arange(9) - 4)).ravel()
CASES = {0, 15, 25, 27, 40}


def first(mask):
    return int(np.argmax(mask)) if mask.any() else None


def build(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with (source / 'episodes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    meta = json.loads((source / 'dataset_metadata.json').read_text())
    known = json.loads((source / 'summary.json').read_text())
    names = list(rows[0])[9:31]
    assert names[19] == 'collect_diamond' and names[-1] == 'make_iron_sword'
    aggregate = {k: np.zeros((64, 64), np.int64) for k in
                 ('starts','occupancy','diamonds','failures','deaths','successOccupancy','failureOccupancy')}
    off = dict(failures=0, deaths=0)
    cases, stats = [], []
    final_flags = []
    missed_counts = np.zeros(len(names), dtype=int)
    for row in rows:
        eid, n = int(row['episode']), int(row['length'])
        timeout = row['timeout'] == 'True'
        with np.load(source / row['file'], allow_pickle=False) as ep:
            xy = ep['player_position']
            ach = ep['achievements']
            actual = ach[-1]
            assert all(bool(int(row[k])) == bool(v) for k,v in zip(names, actual))
            final_flags.append(dict(zip(names, actual.astype(int).tolist())))
            missed_counts += ~actual
            complete = bool(actual.all())
            assert complete == (row['all_22'] == 'True')
            assert len(xy) == n+1 and ((xy[:-1] >= 0) & (xy[:-1] < WIDTH)).all()
            end_on_map = bool(((xy[-1] >= 0) & (xy[-1] < WIDTH)).all())
            hits = np.bincount(xy[:-1, 0] * WIDTH + xy[:-1, 1], minlength=WIDTH**2).reshape(WIDTH, WIDTH)
            aggregate['starts'][tuple(xy[0])] += 1
            aggregate['occupancy'] += hits
            aggregate['successOccupancy' if complete else 'failureOccupancy'] += hits
            for key, include in (('failures', not complete), ('deaths', not timeout)):
                if include:
                    if end_on_map:
                        aggregate[key][tuple(xy[-1])] += 1
                    else:
                        off[key] += 1
            terrain = ep['initial_map']
            diamond_at_reset = terrain == DIAMOND_TILE
            aggregate['diamonds'] += diamond_at_reset
            diamond_positions = np.argwhere(diamond_at_reset)
            nearest = int(abs(diamond_positions - xy[0]).sum(axis=1).min()) if len(diamond_positions) else None
            # Lossless projection of the terrain channel in each exact pre-action observation.
            bits = ep['local_bits'][:-1]
            flags = ((bits[:, DIAMOND_BITS // 8] >> (7 - DIAMOND_BITS % 8)) & 1).astype(bool)
            visible = flags.any(axis=1)
            iron = ep['local_stats'][:-1, 8] > 0
            adjacent = flags[:, CELL_DISTANCE == 1].any(axis=1)
            first_seen = first(visible)
            first_diamond = first(ach[:,19])
            first_all = first(ach.all(axis=1))
            item = dict(episode=eid, steps=n, startRow=int(xy[0,0]), startColumn=int(xy[0,1]),
                        endRow=int(xy[-1,0]), endColumn=int(xy[-1,1]), timeout=timeout, all22=complete,
                        diamond=bool(actual[19]), missingAchievements=int((~actual).sum()),
                        initialDiamondTiles=len(diamond_positions), nearestInitialDiamond=nearest,
                        distinctCells=int((hits>0).sum()), firstVisible=first_seen,
                        visibleSteps=int(visible.sum()), visibleWithPickaxeSteps=int((visible & iron).sum()),
                        adjacentWithPickaxeSteps=int((adjacent & iron).sum()), firstDiamond=first_diamond,
                        firstAll22=first_all)
            stats.append(item)
            if eid in CASES:
                cases.append(dict(**item, initialMap=terrain.ravel().tolist(), path=xy.tolist()))
    misses = [x for x in stats if not x['diamond']]
    never = [x for x in misses if x['firstVisible'] is None]
    with_pickaxe = sum(x['visibleWithPickaxeSteps'] > 0 for x in misses)
    adjacent_with_pickaxe = sum(x['adjacentWithPickaxeSteps'] > 0 for x in misses)
    by_achievement = sorted((dict(name=name, misses=int(count)) for name,count in zip(names,missed_counts)),
                            key=lambda x:-x['misses'])
    N=len(stats)
    summary = dict(episodes=N, transitions=int(aggregate['occupancy'].sum()), all22=sum(x['all22'] for x in stats),
                   diamond=N-len(misses), timeout=sum(x['timeout'] for x in stats), death=sum(not x['timeout'] for x in stats),
                   successTimeout=sum(x['timeout'] and x['all22'] for x in stats),
                   misses=len(misses), missNeverSeen=len(never), missSeen=len(misses)-len(never),
                   missSeenWithPickaxe=with_pickaxe, missAdjacentWithPickaxe=adjacent_with_pickaxe,
                   offMapFailures=off['failures'], offMapDeaths=off['deaths'], initialDiamondTiles=int(aggregate['diamonds'].sum()),
                   mapsWithoutDiamonds=sum(x['initialDiamondTiles']==0 for x in stats),
                   totalFailure=N-sum(x['all22'] for x in stats),
                   diamondOnly=sum(x['missingAchievements']==1 and not x['diamond'] for x in stats),
                   plantOnly=sum(x['missingAchievements']==1 and not flags['eat_plant'] for x,flags in zip(stats, final_flags)),
                   missBothPlantAndDiamond=sum(not f['eat_plant'] and not f['collect_diamond'] for f in final_flags),
                   missPlantOrDiamond=sum(not f['eat_plant'] or not f['collect_diamond'] for f in final_flags),
                   timeoutDiamondMiss=sum(x['timeout'] for x in misses),
                   timeoutDiamondNeverSeen=sum(x['timeout'] for x in never),
                   deathBefore1000=sum(not x['timeout'] and x['steps']<1000 for x in stats),
                   earlyFailureBefore1000=sum(not x['all22'] and x['steps']<1000 for x in stats),
                   medianUniqueCells=float(np.median([x['distinctCells'] for x in stats])),
                   medianNearestDiamond=float(np.median([x['nearestInitialDiamond'] for x in stats if x['nearestInitialDiamond'] is not None])))
    for key in ('episodes','transitions','all22','diamond','timeout'):
        assert summary[key]==known['all_22' if key=='all22' else key]
    assert aggregate['failures'].sum()+off['failures']==summary['totalFailure']
    assert aggregate['deaths'].sum()+off['deaths']==summary['death']
    assert aggregate['starts'].sum()==N
    timeline = [dict(step=t, alive=sum(x['timeout'] or x['steps']>t for x in stats),
                     diamond=sum(x['firstDiamond'] is not None and x['firstDiamond']<=t for x in stats),
                     all22=sum(x['firstAll22'] is not None and x['firstAll22']<=t for x in stats))
                for t in range(0,10001,100)]
    counts = dict(checkpoint=meta['training_steps'], temperature=meta['temperature'], seed=meta['evaluation_seed'],
                  summary=summary, maps={k:v.ravel().tolist() for k,v in aggregate.items()},
                  achievements=by_achievement,timeline=timeline,cases=cases)
    (dest/'data.js').write_text('window.ATLAS='+json.dumps(counts,separators=(',',':'))+';\n')
    with (dest/'episode_metrics.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=[*stats[0], *names])
        writer.writeheader()
        writer.writerows({**episode, **flags} for episode,flags in zip(stats,final_flags))
    print(json.dumps(summary,indent=2))
    print('Spawn coordinates (row, col), counts:', [(v.tolist(),int(aggregate['starts'][tuple(v)])) for v in np.argwhere(aggregate['starts'])][:12])
    print('Most frequently missed:',by_achievement[:8])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ARCHIVE)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    build(args.source,args.out)
