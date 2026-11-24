"""Pure-Python candidate generation and one-field mutation, with provenance."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random

TASKS = ('inverse', 'observation_delta', 'stats_delta', 'event')
HORIZONS = (1, 4, 16)
WEIGHT_MIN, WEIGHT_MAX = .001, 1.


@dataclass(frozen=True)
class Recipe:
    task: str
    horizon: int
    weight: float

    def __post_init__(self):
        if self.task == 'none':
            if self.horizon != 1 or self.weight != 0:
                raise ValueError('Baseline is none / horizon 1 / weight 0')
        elif (self.task not in TASKS or self.horizon not in HORIZONS
              or (self.task == 'inverse' and self.horizon != 1)
              or not math.isfinite(self.weight) or not WEIGHT_MIN <= self.weight <= WEIGHT_MAX):
            raise ValueError('Invalid auxiliary recipe')

    @property
    def id(self):
        raw = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self):
        return asdict(self)


BASELINE = Recipe('none', 1, 0.)


def random_recipe(rng, task=None):
    task = task or rng.choice(TASKS)
    horizon = 1 if task == 'inverse' else rng.choice(HORIZONS)
    weight = round(10 ** rng.uniform(math.log10(WEIGHT_MIN), math.log10(WEIGHT_MAX)), 8)
    return Recipe(task, horizon, weight)


def mutate(parent, rng, field=None, jitter=False):
    """Change exactly one field. No checkpoint/optimizer state is inherited."""
    if parent.task == 'none':
        raise ValueError('PPO baseline is a control, never a mutation parent')
    # An inverse task can switch targets at h=1. A longer-horizon task can
    # switch to the other forward targets without also changing its horizon.
    tasks = [t for t in TASKS if t != parent.task and (t != 'inverse' or parent.horizon == 1)]
    fields = ['task', 'weight'] + (['horizon'] if parent.task != 'inverse' else [])
    field = field or rng.choices(fields, weights=[1 if f != 'weight' else 2 for f in fields])[0]
    if field not in fields:
        raise ValueError('Mutation field is incompatible with parent')
    if field == 'task':
        child = replace(parent, task=rng.choice(tasks))
    elif field == 'horizon':
        ix = HORIZONS.index(parent.horizon)
        neighbors = [HORIZONS[j] for j in (ix - 1, ix + 1) if 0 <= j < len(HORIZONS)]
        child = replace(parent, horizon=rng.choice(neighbors))
    else:
        if jitter:
            low, high = max(WEIGHT_MIN, parent.weight * .5), min(WEIGHT_MAX, parent.weight * 2.)
            weight = parent.weight
            while weight == parent.weight:
                weight = round(10 ** rng.uniform(math.log10(low), math.log10(high)), 8)
            child = replace(parent, weight=weight)
        else:
            choices = [round(parent.weight * factor, 8) for factor in (.5, 2.)
                       if WEIGHT_MIN <= round(parent.weight * factor, 8) <= WEIGHT_MAX]
            child = replace(parent, weight=rng.choice(choices))
    change = dict(field=field, before=getattr(parent, field), after=getattr(child, field))
    if field == 'weight':
        change.update(operator='log_uniform_jitter' if jitter else 'double_or_halve', factor=child.weight / parent.weight)
    return child, change


def tournament(population, fitness, rng):
    contestants = rng.sample(population, min(3, len(population)))
    return max(contestants, key=lambda r: fitness[r.id])


def initial_population(seed, size=8):
    if size < len(TASKS):
        raise ValueError('Initial population must cover all task families')
    rng = random.Random(seed)
    recipes = [random_recipe(rng, task) for task in TASKS]
    while len(recipes) < size:
        candidate = random_recipe(rng)
        if candidate.id not in {r.id for r in recipes}:
            recipes.append(candidate)
    return recipes
