from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp


@dataclass(frozen=True)
class SymbolicObsSpec:
    """Layout metadata for Craftax symbolic observations."""

    obs_dim: int
    view_shape: Optional[Tuple[int, int]] = None
    block_dim: int = 37
    item_dim: int = 5
    creature_dim: int = 36
    light_dim: int = 1
    map_dim: Optional[int] = None

    @property
    def cell_dim(self) -> int:
        return self.block_dim + self.item_dim + self.creature_dim + self.light_dim

    @property
    def inferred_map_dim(self) -> int:
        if self.map_dim is not None:
            return self.map_dim
        if self.view_shape is None:
            return 0
        return self.view_shape[0] * self.view_shape[1] * self.cell_dim

    @property
    def stats_dim(self) -> int:
        return self.obs_dim - self.inferred_map_dim

    def validate(self) -> None:
        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {self.obs_dim}")
        if self.inferred_map_dim < 0:
            raise ValueError(f"map_dim must be nonnegative, got {self.inferred_map_dim}")
        if self.stats_dim < 0:
            raise ValueError(
                f"layout exceeds obs_dim: map_dim={self.inferred_map_dim}, "
                f"obs_dim={self.obs_dim}"
            )


def classic_symbolic_spec(obs_dim: int = 1345) -> SymbolicObsSpec:
    """Default flat-only spec for Craftax-Classic symbolic observations.

    The paper describes 7x9 local map structure, but the documented full cell
    width does not multiply to the observed 1345 classic vector. Until we verify
    the exact packed local schema from the installed Craftax version, the safe
    default is to let the encoder consume the flat vector.
    """

    spec = SymbolicObsSpec(obs_dim=obs_dim, view_shape=None, map_dim=0)
    spec.validate()
    return spec


def split_obs(obs: jnp.ndarray, spec: SymbolicObsSpec) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Split observations into optional flat map data and stats data."""

    spec.validate()
    map_dim = spec.inferred_map_dim
    if obs.shape[-1] != spec.obs_dim:
        raise ValueError(f"expected obs last dim {spec.obs_dim}, got {obs.shape[-1]}")
    if map_dim == 0:
        return None, obs
    return obs[..., :map_dim], obs[..., map_dim:]

