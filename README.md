<p align="center">
 <img width="80%" src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax_Baselines/main/images/logo.png" />
</p>

# Craftax Baselines

This repository contains the code for running the baselines from the [Craftax paper](https://arxiv.org/abs/2402.16801).
For packaging reasons, this is separate to the [main repository](https://github.com/MichaelTMatthews/Craftax/).

# Installation
```commandline
git clone https://github.com/MichaelTMatthews/Craftax_Baselines.git
cd Craftax_Baselines
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

## V100 Installation
For V100 machines, use the separate pinned requirements file:

```commandline
conda create -n craftax-v100 python=3.10 -y
conda activate craftax-v100
pip install -U pip

pip install -r requirements_v100.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify the install:

```commandline
python - <<'PY'
import jax, numpy as np, scipy
print("jax", jax.__version__)
print("numpy", np.__version__)
print("scipy", scipy.__version__)
print(jax.devices())
print(jax.lib.xla_bridge.get_backend().platform_version)
PY
```

# Run Experiments

### PPO
```commandline
python ppo.py
```

### PPO-RNN
```commandline
python ppo_rnn.py
```

### ICM
```commandline
python ppo.py --train_icm
```

### E3B
```commandline
python ppo.py --train_icm --use_e3b --icm_reward_coeff 0
```

### RND
```commandline
python ppo_rnd.py
```

# Visualisation
You can save trained policies with the `--save_policy` flag.  These can then be viewed with the `view_ppo_agent` script (pass in the path up to the `files` directory).

## Value-function counterfactual experiments

The `concept_mapping/` utilities include reproducible probes of whether a
frozen symbolic PPO critic represents conditional resource value.  These probes
edit the symbolic observation only and evaluate the critic; they do not alter
the live environment transition state.  Every run saves its raw base states,
counterfactual observations, critic values, metadata, and a manifest with the
checkpoint and collection settings.

### Diamond collectability

Base states are collected five steps before a real diamond collection.  A clean
state must contain a visible diamond and iron pickaxe, must not already have a
diamond, and must not permit immediately remaking the removed iron pickaxe.
For each base state we evaluate:

```text
A: no diamond, no iron pickaxe
B: no diamond, iron pickaxe
C: diamond, no iron pickaxe
D: diamond, iron pickaxe
```

The main affordance comparison is `V(D) > V(C)`.  The same data also supports
the tantalization-like diagnostic `V(A) - V(C)`: a positive value means a
visible but uncollectable diamond is assigned lower value than a matched state
without a visible diamond.

For a fresh single-worker collection:

```commandline
python -u concept_mapping/diamond_affordance_clean_filter.py \
  --run_path wandb/run-20260630_214658-m0mw4end \
  --out_dir concept_mapping/runs/diamond_affordance_clean_5000_fresh_m0mw4end \
  --timestep 9999941632 --target_states 5000 --fresh
```

`run_diamond_affordance_4gpu.sh` shards the same collection across GPUs 1--4,
then deduplicates and merges the results.

### Water value

Water base states are sampled five steps before a real increase in the `drink`
intrinsic.  The base observation must contain visible water.  Each raw base is
used to form a 2x2 probe:

```text
empty drink, no water      empty drink, water present
full drink,  no water      full drink,  water present
```

Empty and full drink are set to `0/10` and `9/10`, respectively.  The no-water
counterfactual replaces visible water tiles with grass, leaving the remaining
map and inventory unchanged.  The main tests are:

```text
V(empty, water) > V(empty, no water)
[V(empty, water) - V(empty, no water)]
  > [V(full, water) - V(full, no water)]
```

### Sword ranking

Sword base states are sampled when a zombie or skeleton is visible.  Three
controlled sword counterfactuals clear all sword slots and then give the agent
exactly one wood, stone, or iron sword.  The primary ordering is:

```text
V(iron sword only) > V(stone sword only) > V(wood sword only)
```

The factual state and a no-sword state are also saved for visual inspection,
but are not part of the primary ranking statistic.

### Joint water/sword collection

`collect_water_sword_counterfactuals.py` collects both tasks from one frozen
policy rollout.  An episode can contribute up to three well-separated water
states and independently up to three well-separated sword states.  The default
within-task separation is 256 environment steps; episode IDs are retained for
clustered analyses.

```commandline
python -u concept_mapping/collect_water_sword_counterfactuals.py \
  --run_path wandb/run-20260630_214658-m0mw4end \
  --out_dir concept_mapping/runs/water_sword_500_m0mw4end \
  --timestep 9999941632 \
  --water_target 500 --sword_target 500 \
  --max_per_episode_per_task 3 --min_spacing 256
```

### Progression-ordered inventory heuristic

This probe tests whether the frozen PPO critic uses a progression-consistent
inventory as a heuristic for how far the agent has advanced through an
episode.  It is a within-state counterfactual experiment rather than an
observational correlation: for each sampled rollout observation, the map,
visible mobs, player position, health, food, drink, energy, direction, light,
and all other non-inventory features are fixed exactly.  The factual inventory
is replaced by an empty inventory and then progression components are added
one at a time.

The progression hierarchy is:

```text
wood
-> {wood pickaxe, wood sword}
-> {stone, coal}
-> {stone pickaxe, stone sword}
-> iron
-> {iron pickaxe, iron sword}
-> diamond
```

Each component is represented by presence only: resources are set to one unit
and tools to one copy.  Saplings are deliberately excluded.  Items within a
braced pair have no specified order.  We therefore evaluate all 16 valid
tie-break orderings (two choices for each of four pairs), then average the
critic prediction at every inventory level.  This prevents the result from
depending on an arbitrary choice such as whether a wood sword is added before
a wood pickaxe.

For base state $i$ and inventory level $k \in \{0,\ldots,11\}$, the critic
prediction is $V_{ik}$.  The primary analysis is the fixed-effects model:

```text
V_ik = alpha_i + beta * k + epsilon_ik
```

where the state-specific intercept `alpha_i` absorbs every unchanged feature
of the sampled external state.  The hypothesis is `beta < 0`: adding more
progression-ordered inventory components lowers predicted value even though
the external environment is unchanged.  Standard errors are clustered by base
state.

The completed 10B-checkpoint run used 1,500 generic frozen-policy rollout
states, 16 valid orderings per state, and 12 inventory levels (including empty
inventory), for 288,000 raw critic evaluations.  Its fixed-effects slope was
`-0.3656` critic-value units per added component (95% CI `[-0.3804, -0.3509]`,
`p < 10^-300`).  In 78.2% of consecutive inventory additions, critic value
decreased.  The held-out within-state $R^2$ was 0.465.

Run the three-GPU collection and evaluation with:

```commandline
bash run_inventory_progress_3gpu.sh
```

The launcher uses GPUs 2--4, saves one independent 500-state shard per GPU,
and merges the results automatically.  The merged directory contains
`base_states.npz`, raw `counterfactual_values.csv`, `regression_summary.json`,
`mean_values_by_level.csv`, and the publication figure
`inventory_progress_value.png`/`.pdf`.

#### Empirically ordered sensitivity analysis

To avoid relying exclusively on the hand-specified crafting hierarchy, we also
estimated the order from 50 fresh frozen-policy episodes. For each of the 12
inventory components, including sapling, we recorded its first-acquisition step
and ordered components by their median first-acquisition step among episodes in
which they appeared. The resulting single chain was:

```text
sapling -> wood -> wood pickaxe -> stone -> wood sword -> stone pickaxe
-> coal -> iron -> stone sword -> iron sword -> iron pickaxe -> diamond
```

We then reran precisely the same within-state counterfactual analysis on the
same 1,500 base states, but along this empirical chain. The critic again
declined strongly as components were added: fixed-effects slope `-0.3182`
(95% CI `[-0.3335, -0.3029]`, `p < 10^-300`), with 72.6% of consecutive
additions lowering value and held-out within-state $R^2 = 0.385$. Thus, the
inventory-progress result does not depend on the initial hand-written ordering.

```commandline
bash run_inventory_progress_empirical_order_3gpu.sh
```

This launcher reuses the saved 1,500 base observations, evaluates one
empirically ordered path per state on GPUs 2--4, and writes separate results to
`concept_mapping/runs/inventory_progress_empirical_order_3gpu_m0mw4end/`.
