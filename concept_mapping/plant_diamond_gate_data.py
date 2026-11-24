"""Build causal plant/diamond probes from the archived split used in the transport pilot."""
import argparse
import json
from pathlib import Path

import numpy as np

from concept_mapping.spatial_transport_queries import BANDS, BOARD, LOCAL, STRIDE
from concept_mapping.sru_reconstruction_data import ARCHIVE

SOURCE = Path(__file__).resolve().parent / "runs/sru_spatial_transport_offline"
OUT = SOURCE / "plant_diamond_write_gate"
CODES = (16, 10)  # ripe plant and diamond
REPLACEMENT = (15, 7)  # eaten plant becomes unripe, mined diamond becomes path


def world_queries(tiles, position, rng, diagnostics):
    length = len(position)
    nblocks = len(tiles) // STRIDE
    query = np.zeros((nblocks, len(BANDS), 2, 2, 2), np.int8)
    mask = np.zeros(query.shape[:-2], bool)
    last = np.full(4096, -1)
    terrain = np.full(4096, -1)
    first = np.full((2, 4096), length)
    before = [{}, {}]
    revised = [set(), set()]
    events = []
    coords = position[:, None] + LOCAL
    inside = ((coords >= 0) & (coords < 64)).all(-1)
    cells = coords[..., 0]*64 + coords[..., 1]
    for block in range(length // STRIDE):
        start = block * STRIDE
        end = start + STRIDE - 1
        sl = slice(start,end+1)
        valid = inside[sl]
        ids = cells[sl][valid]
        frames = np.flatnonzero(valid)
        latest = np.full(4096, -1)
        np.maximum.at(latest, ids, frames)
        updated = latest >= 0
        view = tiles[sl].ravel()
        terrain[updated] = view[latest[updated]]
        last[updated] = start + latest[updated] // 63
        offsets = BOARD - position[end]
        distance = abs(offsets).sum(-1)
        age = end - last
        off = (last >= 0) & (abs(offsets) > (3,4)).any(-1)
        def pair(p, neg):
            gap = abs(distance[neg]-distance[p])
            n = rng.choice(neg[gap==gap.min()])
            return offsets[[p,n]]
        for r,code in enumerate(CODES):
            if diagnostics:
                hit = view[frames] == code
                np.minimum.at(first[r], ids[hit], start + frames[hit] // 63)
                for cell in np.flatnonzero((terrain==code) & (age==0)):
                    before[r][int(cell)] = (block, offsets[cell].tolist())
                    revised[r].discard(int(cell))
                for cell in np.flatnonzero((terrain==REPLACEMENT[r]) & off):
                    if cell not in before[r] or cell in revised[r]:
                        continue
                    old_block, old_offset = before[r][cell]
                    events.append(dict(task="revision", r=r, old=old_block, new=block,
                                       xy_old=old_offset, xy_new=offsets[cell].tolist()))
                    revised[r].add(int(cell))
            for b,(lo,hi) in enumerate(BANDS):
                eligible = off & (age >= lo) & (age <= hi)
                pos = np.flatnonzero(eligible & (terrain == code))
                neg = np.flatnonzero(eligible & (terrain != code))
                if pos.size and neg.size:
                    query[block,b,r] = pair(rng.choice(pos), neg)
                    mask[block,b,r] = True
        if not diagnostics:
            continue
        for r,code in enumerate(CODES):
            other = 1-r
            old_ids = np.flatnonzero(off & (terrain==code))
            new_ids = np.flatnonzero(off & (terrain==CODES[other]))
            if not len(old_ids) or not len(new_ids):
                continue
            candidates = np.argwhere(last[old_ids,None] < first[other,new_ids][None,:])
            if not len(candidates):
                continue
            i,j = candidates[rng.integers(len(candidates))]
            locations = [old_ids[i],new_ids[j]]
            probes = []
            for p,landmark in zip(locations,(code,CODES[other])):
                neg = np.flatnonzero(off & (terrain != landmark) & (BOARD != BOARD[locations[0]]).any(-1)
                                     & (BOARD != BOARD[locations[1]]).any(-1))
                if not len(neg): break
                near = abs(age[neg]-age[p])
                probes.append(pair(p,neg[near==near.min()]).tolist())
            if len(probes)==2:
                events.append(dict(task="both", r=r, new=block, probes=probes))
    return query,mask,events


def prepare(base=SOURCE, out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    split=json.loads((base/"worlds.json").read_text())
    for mode,rows in split.items():
        with np.load(base/f"{mode}.npz") as archive:
            tiles, motion, length = [archive[k] for k in ("tiles","motion","length")]
        queries, masks, events = [], [], []
        for i,row in enumerate(rows):
            ep = int(row["episode"])
            with np.load(ARCHIVE/"episodes"/f"episode_{ep:04d}.npz") as a:
                position = a["player_position"][:length[i]].astype(int)
            np.testing.assert_array_equal(np.diff(position,axis=0),motion[i,1:length[i]])
            q,m,diag = world_queries(tiles[i], position, np.random.default_rng([71259,ep]),mode=="test")
            queries.append(q)
            masks.append(m)
            events.extend(dict(world=i,**e) for e in diag)
        np.savez(out/f"{mode}.npz", tiles=tiles, motion=motion, length=length,query=np.stack(queries),valid=np.stack(masks))
        if mode=="test":
            (out/"events.json").write_text(json.dumps(events))
        print(mode, np.stack(masks).sum((0,1)),len(events),flush=True)


if __name__=="__main__":
    prepare()
