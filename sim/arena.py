"""Arena geometry: geofence bounds, distance to bounds, out-of-bounds check.

Backend-agnostic: every array function takes `xp` (numpy or jax.numpy). This
module must not import jax or crazyflow at module level, because the safety
filter (task 5) and the WS5 flight node share it.

Semantics (contract section 10, arena.yaml):
- bounds are [lo, hi] per axis in the world frame and limit the drone center.
- A drone is out of bounds if its center is strictly outside bounds on any axis,
  or its position is non-finite. On the boundary counts as inside.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "arena.yaml"
_AXES = ("x", "y", "z")
_SYM_TOL = 1e-9


@dataclass(frozen=True)
class Arena:
    """Arena geometry loaded from arena.yaml. Use as a closure constant under jit."""

    lo: np.ndarray  # (3,) geofence lower corner, m
    hi: np.ndarray  # (3,) geofence upper corner, m
    geofence_margin: float
    min_separation: float
    min_vertical_offset: float

    @property
    def inner_lo(self) -> np.ndarray:
        """Lower corner of the box the safety filter keeps drones in."""
        return self.lo + self.geofence_margin

    @property
    def inner_hi(self) -> np.ndarray:
        """Upper corner of the box the safety filter keeps drones in."""
        return self.hi - self.geofence_margin


def arena_from_dict(cfg: dict[str, Any]) -> Arena:
    """Build and validate an Arena from a parsed arena.yaml dict."""
    bounds = cfg["bounds"]
    lo, hi = [], []
    for ax in _AXES:
        pair = bounds[ax]
        if len(pair) != 2:
            raise ValueError(f"bounds.{ax} must be [lo, hi], got {pair}")
        lo.append(float(pair[0]))
        hi.append(float(pair[1]))
    arena = Arena(
        lo=np.asarray(lo, dtype=np.float64),
        hi=np.asarray(hi, dtype=np.float64),
        geofence_margin=float(cfg["geofence_margin"]),
        min_separation=float(cfg["min_separation"]),
        min_vertical_offset=float(cfg["min_vertical_offset"]),
    )
    _validate(arena)
    return arena


def load_arena(path: str | Path = DEFAULT_PATH) -> Arena:
    """Load and validate config/arena.yaml."""
    with open(path) as f:
        return arena_from_dict(yaml.safe_load(f))


def _validate(a: Arena) -> None:
    for i, ax in enumerate(_AXES):
        if not a.lo[i] < a.hi[i]:
            raise ValueError(f"bounds.{ax}: lo must be < hi")
    for i, ax in enumerate(_AXES[:2]):
        if abs(a.lo[i] + a.hi[i]) > _SYM_TOL:
            raise ValueError(f"bounds.{ax} must be symmetric about 0 (origin = arena center)")
    if not a.lo[2] > 0.0:
        raise ValueError("bounds.z lo must be > 0 (floor clearance)")
    if a.geofence_margin < 0.0:
        raise ValueError("geofence_margin must be >= 0")
    if np.any(a.inner_hi <= a.inner_lo):
        raise ValueError("geofence_margin leaves an empty inner box")
    if not a.min_separation > 0.0:
        raise ValueError("min_separation must be > 0")
    if a.min_vertical_offset < 0.0:
        raise ValueError("min_vertical_offset must be >= 0")
    # Mirrored spawns need |xy| >= min_separation / 2 inside the inner box.
    if a.min_separation / 2.0 > float(np.min(a.inner_hi[:2])):
        raise ValueError("inner box too small for two drones at min_separation")


def dist_to_bounds(pos: Any, arena: Arena, xp: ModuleType = np) -> Any:
    """Signed distance from drone center to each geofence face.

    pos: (..., 3) world position. Returns (..., 6) in contract obs order
    x-, x+, y-, y+, z-, z+. Positive inside, negative outside, m.
    """
    d_lo = pos - xp.asarray(arena.lo, dtype=pos.dtype)
    d_hi = xp.asarray(arena.hi, dtype=pos.dtype) - pos
    return xp.stack(
        [d_lo[..., 0], d_hi[..., 0], d_lo[..., 1], d_hi[..., 1], d_lo[..., 2], d_hi[..., 2]],
        axis=-1,
    )


def out_of_bounds(pos: Any, arena: Arena, xp: ModuleType = np) -> Any:
    """Geofence violation per drone. pos: (..., 3). Returns (...,) bool.

    True if the center is strictly outside bounds or the position is non-finite.
    """
    outside = xp.any(dist_to_bounds(pos, arena, xp) < 0.0, axis=-1)
    nonfinite = ~xp.all(xp.isfinite(pos), axis=-1)
    return outside | nonfinite
