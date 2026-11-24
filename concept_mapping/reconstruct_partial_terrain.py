"""Fit and score a terrain completion model using recorded Craftax local observations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F

from concept_mapping.world_episode import terrain


N_TILES = 17
UNKNOWN = N_TILES
SIDE = 64
STEP = 512
SEED = 27153
RUNS = Path(__file__).resolve().parent / "runs"
ARCHIVE = RUNS / "lava_only_gru_concat_dp2_10b_seed50/world_transitions_1000/episodes"
COLORS = ["#353535", "#454545", "#83ad65", "#517fa4", "#878786", "#305c41",
          "#a67e51", "#bba687", "#42424c", "#be826b", "#47efed", "#ce995c",
          "#716566", "#e4cd8e", "#db6739", "#a1c047", "#e1bf45", "#eceee9"]


def snapshot(path):
    with np.load(path) as ep:
        positions = ep["player_position"]
        t = min(STEP, len(positions) - 2)  # Always use a living pre-action frame.
        positions = positions[:t + 1]
        bits = np.unpackbits(ep["local_bits"][:t + 1], axis=-1, count=1323)
        patches = bits.reshape(-1, 7, 9, 21)[..., :N_TILES].argmax(axis=-1)
        rows = positions[:, 0, None, None] + np.arange(-3, 4)[None, :, None]
        cols = positions[:, 1, None, None] + np.arange(-4, 5)[None, None, :]
        valid = (rows >= 0) & (rows < SIDE) & (cols >= 0) & (cols < SIDE)
        cells = (rows * SIDE + cols)[valid]
        last = np.full(SIDE * SIDE, -1)
        np.maximum.at(last, cells, np.flatnonzero(valid))
        last = last.reshape(SIDE, SIDE)
        seen = last >= 0
        memory = np.full((SIDE, SIDE), UNKNOWN, dtype=np.uint8)
        memory[seen] = patches.ravel()[last[seen]]
        age = np.ones((SIDE, SIDE), dtype=np.float32)
        age[seen] = np.log1p(t - last[seen] // 63) / np.log1p(STEP)
        truth = terrain(ep, t)
        np.testing.assert_array_equal(memory[age == 0], truth[age == 0])
        return memory, age, truth, t


class TerrainCompletion(nn.Module):
    def __init__(self):
        super().__init__()
        def block(n_in, n_out):
            return nn.Sequential(nn.Conv2d(n_in, n_out, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(n_out, n_out, 3, padding=1), nn.ReLU())
        axis = torch.linspace(-1., 1., SIDE)
        self.register_buffer("coords", torch.stack(torch.meshgrid(axis, axis, indexing="ij")))
        self.fine = block(N_TILES + 4, 12)  # Tile or unknown, age, row, column.
        self.mid = block(12, 20)
        self.coarse = block(20, 32)
        self.up_mid = block(32 + 20, 20)
        self.up_fine = block(20 + 12, 12)
        self.out = nn.Conv2d(12, N_TILES, 1)

    def forward(self, memory, age):
        tiles = F.one_hot(memory.long(), UNKNOWN + 1).permute(0, 3, 1, 2).float()
        inputs = torch.cat((tiles, age[:, None], self.coords.expand(len(tiles), -1, -1, -1)), 1)
        fine = self.fine(inputs)
        mid = self.mid(F.avg_pool2d(fine, 2))
        coarse = self.coarse(F.avg_pool2d(mid, 2))
        up = self.up_mid(torch.cat((mid, F.interpolate(coarse, scale_factor=2)), 1))
        return self.out(self.up_fine(torch.cat((fine, F.interpolate(up, scale_factor=2)), 1)))


def plot(path, memory, truth, prob, episode, step):
    seen = memory != UNKNOWN
    prediction = np.where(seen, memory, prob.argmax(0))
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 4.55))
    fig.subplots_adjust(left=.025, right=.98, top=.78, bottom=.25, wspace=.12)
    fig.text(.025, .92, "Reconstructing a Craftax world from local views", size=19, weight="bold")
    fig.text(.025, .853, f"Held-out episode {episode}  ·  step {step}  ·  {seen.mean():.0%} of terrain observed", size=11)
    titles = ["Last seen by Crafty", "Predicted full terrain", "True terrain at this step", "Confidence on unseen tiles"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, size=11, loc="left", pad=6)
        ax.set_axis_off()
    for ax, grid in zip(axes[:3], (memory, prediction, truth)):
        ax.imshow(grid, cmap=ListedColormap(COLORS), vmin=0, vmax=UNKNOWN, interpolation="nearest")
    for ax in axes[1:3]:
        ax.contour(seen, levels=[.5], colors="white", linewidths=.75)
    confidence = np.ma.masked_where(seen, prob.max(0))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eceee9")
    im = axes[3].imshow(confidence, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    pos = axes[3].get_position()
    cax = fig.add_axes([pos.x0, .2, pos.width, .018])
    fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    patches = [Patch(color=COLORS[i], label=name) for i, name in
               [(UNKNOWN, "Unseen"), (2, "Grass"), (3, "Water"), (4, "Stone"),
                (5, "Tree"), (7, "Path"), (13, "Sand"), (14, "Lava"), (10, "Diamond")]]
    fig.legend(handles=patches, loc="lower left", bbox_to_anchor=(.018, .128),
               ncol=9, frameon=False, fontsize=9, handlelength=1.05, columnspacing=1.15)
    fig.text(.025, .067, "White outlines mark explored terrain. Confidence is the model’s probability for its most likely terrain type.", size=10)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def expected_top_decile_hits(scores, diamond):
    """Count ties fractionally so a uniform forecast gets chance performance."""
    k = max(1, int(np.ceil(.1 * len(scores))))
    cutoff = np.partition(scores, -k)[-k]
    above = scores > cutoff
    tied = scores == cutoff
    return diamond[above].sum() + (k - above.sum()) * diamond[tied].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=RUNS / "terrain_reconstruction_512")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--evaluate", action="store_true", help="Score the checkpoint previously fitted in --output.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    paths = sorted(args.archive.glob("episode_*.npz"))
    if len(paths) < 10:
        raise ValueError("Expected an archive of at least 10 worlds")
    order = np.random.default_rng(SEED).permutation(len(paths))
    snapshots = [snapshot(paths[i]) for i in order]
    memory, age, truth, step = map(np.stack, zip(*snapshots))
    del snapshots
    n = 4 * len(paths) // 5
    net = TerrainCompletion()
    mem, ages, labels = map(torch.from_numpy, (memory, age, truth))
    checkpoint = args.output / "model.pt"
    if args.evaluate:
        net.load_state_dict(torch.load(checkpoint, weights_only=True))
        epochs = json.loads((args.output / "metrics.json").read_text())["epochs"]
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=.002)
        epochs = args.epochs
        for epoch in range(epochs):
            for ids in torch.randperm(n).split(16):
                logits = net(mem[ids], ages[ids])
                target = labels[ids].long().masked_fill(mem[ids] != UNKNOWN, -100)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
        torch.save(net.state_dict(), checkpoint)

    # Learn the coordinate-only baseline on the same masked training targets.
    counts = np.stack([((truth[:n] == tile) & (memory[:n] == UNKNOWN)).sum(0)
                       for tile in range(N_TILES)]).astype(float)
    counts = gaussian_filter(counts, sigma=(0, 1.5, 1.5))
    prior = counts.argmax(0)
    prior_diamond = counts[10] / np.maximum(1e-10, counts.sum(0))
    near, far = 0, 1
    total = np.zeros(2, dtype=np.int64)
    correct = {key: total.copy() for key in ("completion", "coordinate_prior", "nearest_seen")}
    diamond_hits = {"completion": 0., "coordinate_prior": 0.}
    diamonds = 0
    heldout = np.arange(n, len(paths))
    # Choose the example by median observed coverage without inspecting prediction quality.
    coverage = (memory[n:] != UNKNOWN).mean(axis=(1, 2))
    example = n + np.abs(coverage - np.median(coverage)).argmin()
    net.eval()
    with torch.inference_mode():
        for batch in np.array_split(heldout, max(1, (len(heldout) + 15) // 16)):
            probs = net(mem[batch], ages[batch]).softmax(1).numpy()
            for i, prob in zip(batch, probs):
                unseen = memory[i] == UNKNOWN
                distance, index = distance_transform_edt(unseen, return_indices=True)
                preds = dict(completion=prob.argmax(0), coordinate_prior=prior,
                             nearest_seen=memory[i][tuple(index)])
                actual_diamonds = truth[i][unseen] == 10
                diamonds += int(actual_diamonds.sum())
                for key, score in [("completion", prob[10]), ("coordinate_prior", prior_diamond)]:
                    diamond_hits[key] += expected_top_decile_hits(score[unseen], actual_diamonds)
                for j, mask in [(near, unseen & (distance <= 4)), (far, distance > 4)]:
                    total[j] += mask.sum()
                    for name, pred in preds.items():
                        correct[name][j] += ((pred == truth[i]) & mask).sum()
                if i == example:
                    plot(args.output / "reconstruction.png", memory[i], truth[i], prob,
                         int(paths[order[i]].stem.split("_")[-1]), step[i])
    scores = {key: dict(all_unseen=values.sum() / total.sum(), within_4=values[near] / total[near],
                        beyond_4=values[far] / total[far]) for key, values in correct.items()}
    summary = dict(train_worlds=n, test_worlds=len(heldout), max_step=STEP, seed=SEED, epochs=epochs,
                   unseen_tiles=total.tolist(), example=paths[order[example]].name, scores=scores,
                   unseen_diamonds=diamonds,
                   diamond_recall_at_top_decile={k: h / diamonds if diamonds else None
                                                 for k, h in diamond_hits.items()})
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
