"""Read a stitched Craftax map and estimate hidden-diamond probability in each 8×8 region."""

import json

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

from concept_mapping.benchmark_terrain_completion import CHECKPOINT, REGION, load_snapshots
from concept_mapping.reconstruct_partial_terrain import COLORS, RUNS, SEED, STEP, UNKNOWN, TerrainCompletion


OUT = RUNS / "regional_diamond_map"


def average_region(grid):
    return F.avg_pool2d(grid[:, None], REGION, REGION)[:, 0]


class RegionalDiamondMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.terrain_model = TerrainCompletion()
        self.region_head = nn.Linear(9, 1)

    def features(self, memory, age):
        prob = self.terrain_model(memory, age).softmax(1)
        hidden = (memory == UNKNOWN).float()
        fraction = average_region(hidden)
        rock = prob[:, [4, 8, 9, 10]].sum(1)
        mass = average_region(hidden * rock)
        def hidden_average(grid):
            return average_region(hidden * grid) / fraction.clamp_min(1 / (REGION ** 2))
        coord = self.terrain_model.coords
        radius = torch.sqrt(coord.square().sum(0))
        inputs = torch.stack((fraction, mass, hidden_average(rock), hidden_average(prob[:, 7]),
                              hidden_average(prob[:, 14]), hidden_average(prob[:, 10]),
                              hidden_average(prob.max(1).values), hidden_average(radius),
                              average_region(((memory == 4) | (memory == 8) | (memory == 9) |
                                              (memory == 10)).float())), -1)
        return prob, inputs, mass

    def forward(self, terrain, last_seen, elapsed):
        """Batched terrain (0 unknown, tile+1 otherwise), timestamps and step from MapMemoryState."""
        memory = torch.where(last_seen >= 0, terrain.long() - 1, UNKNOWN)
        age = (elapsed[:, None, None] - last_seen).clamp(0, STEP).float().log1p() / np.log1p(STEP)
        age = torch.where(last_seen >= 0, age, 1.)
        prob, inputs, mass = self.features(memory, age)
        regional = self.region_head(inputs).squeeze(-1).sigmoid()
        regional = torch.where(inputs[..., 0] > 0, regional, 0.)
        return prob, regional, mass


def fit(model):
    files, memory, ages, truth = load_snapshots()
    n = len(files) * 4 // 5
    features, labels = [], []
    with torch.inference_mode():
        for start in range(0, n, 16):
            s = slice(start, min(start + 16, n))
            mem = torch.from_numpy(memory[s])
            _, x, _ = model.features(mem, torch.from_numpy(ages[s]))
            targets = average_region(((mem == UNKNOWN) & (torch.from_numpy(truth[s]) == 10)).float()) > 0
            eligible = x[..., 0] > 0
            features.append(x[eligible].numpy())
            labels.append(targets[eligible].numpy())
    x = np.concatenate(features).astype(float)
    y = np.concatenate(labels)
    center, scale = x.mean(0), x.std(0).clip(1e-9)
    standardized = (x - center) / scale
    def logistic_loss(coefficients):
        weights, intercept = coefficients[:-1], coefficients[-1]
        logits = standardized @ weights + intercept
        residual = expit(logits) - y
        loss = (np.logaddexp(0, logits) - y * logits).sum() + .5 * (weights @ weights)
        return loss, np.r_[standardized.T @ residual + weights, residual.sum()]
    result = minimize(logistic_loss, np.zeros(x.shape[1]+1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    weight = result.x[:-1] / scale
    bias = result.x[-1] - np.dot(weight, center)
    with torch.no_grad():
        model.region_head.weight.copy_(torch.from_numpy(weight[None]))
        model.region_head.bias.fill_(bias)
    return n


def plot(data, idx, regional, target):
    times = data["elapsed"][idx]
    fig, axes = plt.subplots(3, len(times), figsize=(10, 7.6), layout="constrained")
    fig.suptitle("One Craftax world as the observed map grows", weight="bold", fontsize=17)
    cmap = ListedColormap(COLORS)
    ceiling = max(.05, float(regional.max()))
    for j, step in enumerate(times):
        cell = data["terrain"][idx, j]
        known = cell > 0
        mem = np.where(known, cell.astype(int) - 1, UNKNOWN)
        truth = data["truth"][idx, j]
        row, col = data["position"][idx, j]
        axes[0, j].set_title(f"Step {step}")
        for ax, grid in [(axes[0, j], mem), (axes[2, j], truth)]:
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=UNKNOWN, interpolation="nearest")
            ax.add_patch(Rectangle((col-4.5, row-3.5), 9, 7, fill=False, edgecolor="#c31a63", lw=1.4))
        axes[1, j].imshow(regional[j], cmap="magma", vmin=0, vmax=ceiling, interpolation="nearest",
                          extent=(-.5, 63.5, 63.5, -.5))
        for ax in (axes[1, j], axes[2, j]):
            rr, cc = np.where((truth == 10) & ~known)
            ax.scatter(cc, rr, s=42, marker="x", c="#39eef0", lw=1.2)
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(axes[:, 0], ["Observed terrain", "Hidden diamond\nprobability", "Actual terrain"]):
        ax.set_ylabel(title)
    bar = fig.colorbar(axes[1, -1].images[0], ax=axes[1, :], shrink=.78)
    bar.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    fig.supxlabel("Pink box shows the 7×9 view. Cyan crosses reveal hidden diamonds for comparison.", fontsize=10)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def evaluate(model, data):
    summary = {}
    steps = data["steps"]
    count = len(data["elapsed"])
    coverage = (data["terrain"][:, -1] > 0).mean((1, 2))
    example = int(np.abs(coverage - np.median(coverage)).argmin())
    example_maps = []
    for j, step in enumerate(steps):
        terrain = data["terrain"][:, j]
        truth = data["truth"][:, j]
        last_seen = data["last_seen"][:, j]
        seen = terrain > 0
        assert np.array_equal(seen, last_seen >= 0)
        probs, scores, rock_scores = [], [], []
        for start in range(0, count, 16):
            s = slice(start, start+16)
            prob, reg, rock = model(torch.from_numpy(terrain[s]), torch.from_numpy(last_seen[s]),
                                    torch.from_numpy(data["elapsed"][s, j]))
            probs.append(prob.argmax(1).numpy())
            scores.append(reg.numpy())
            rock_scores.append(rock.numpy())
        pred, regional, rock = map(np.concatenate, (probs, scores, rock_scores))
        example_maps.append(regional[example])
        diamonds = (truth == 10) & ~seen
        target = diamonds.reshape(count, 8, 8, 8, 8).any((2, 4)).reshape(count, 64)
        candidates = (~seen).reshape(count, 8, 8, 8, 8).any((2, 4)).reshape(count, 64)
        eligible = target.any(1)
        def hits(score):
            score = np.where(candidates, score.reshape(count, -1), -np.inf)
            top = np.argsort(-score, axis=1, kind="stable")[:, :3]
            return np.take_along_axis(target, top, axis=1).any(1)[eligible]
        a, b = hits(regional), hits(rock)
        delta = a.astype(int) - b.astype(int)
        boot = np.random.default_rng(SEED).choice(delta, (5000, len(delta)), replace=True).mean(1)
        distance = np.stack([distance_transform_edt(~mask) for mask in seen])
        near, far = (~seen) & (distance <= 4), distance > 4
        mismatch = (terrain.astype(int) - 1 != truth) & seen
        visible = last_seen == data["elapsed"][:, j, None, None]
        assert not mismatch[visible].any()
        summary[int(step)] = dict(worlds=count, at_requested_step=int((data["elapsed"][:,j] == step).sum()),
                                  observed_cells=int(seen.sum()), stale_cells=int(mismatch.sum()),
                                  visible_cells=int(visible.sum()), visible_errors=int(mismatch[visible].sum()),
                                  unseen_near_accuracy=float((pred == truth)[near].mean()),
                                  unseen_far_accuracy=float((pred == truth)[far].mean()),
                                  worlds_with_hidden_diamond=int(eligible.sum()),
                                  regional_top3_hits=int(a.sum()), rock_top3_hits=int(b.sum()),
                                  paired_delta_95=np.quantile(boot, [.025, .975]).tolist(),
                                  regional_brier=float(((regional.reshape(count,64)[candidates]
                                                       - target[candidates])**2).mean()))
    plot(data, example, np.stack(example_maps), OUT / "map_growth.png")
    return summary


def main():
    torch.set_num_threads(4)
    model = RegionalDiamondMap().eval()
    model.terrain_model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    n = fit(model)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / "regional_map.pt"
    torch.save(model.state_dict(), checkpoint)
    reader = RegionalDiamondMap().eval()
    reader.load_state_dict(torch.load(checkpoint, weights_only=True))
    with np.load(OUT / "fresh_worlds.npz") as archive:
        data = dict(archive)
    with torch.inference_mode():
        scores = evaluate(reader, data)
    report = dict(training_worlds=n, collection_policy=data["checkpoint"].item(),
                  map_or_timestamp_errors=int(data["errors"][:, -1].sum()),
                  audited_local_views=int((data["elapsed"][:, -1] + 1).sum()), scores=scores)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
