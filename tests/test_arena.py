"""Tests for sim.arena and sim.spawn."""

from __future__ import annotations

import copy
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from sim.arena import Arena, arena_from_dict, dist_to_bounds, load_arena, out_of_bounds
from sim.spawn import sample_spawn

CFG: dict[str, Any] = {
    "contract_version": "0.2",
    "bounds": {"x": [-2.0, 2.0], "y": [-1.5, 1.5], "z": [0.3, 2.0]},
    "geofence_margin": 0.3,
    "min_separation": 0.5,
    "min_vertical_offset": 0.5,
}
XPS = [np, jnp]


@pytest.fixture
def arena() -> Arena:
    return arena_from_dict(CFG)


def _cfg(**overrides: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(CFG)
    for k, v in overrides.items():
        if k in ("x", "y", "z"):
            cfg["bounds"][k] = v
        else:
            cfg[k] = v
    return cfg


# ---- loader -----------------------------------------------------------------


def test_repo_config_loads() -> None:
    a = load_arena()
    assert a.lo.shape == (3,) and a.hi.shape == (3,)


def test_load_from_file(tmp_path: Path) -> None:
    p = tmp_path / "arena.yaml"
    p.write_text(yaml.safe_dump(CFG))
    a = load_arena(p)
    np.testing.assert_allclose(a.lo, [-2.0, -1.5, 0.3])
    np.testing.assert_allclose(a.inner_hi, [1.7, 1.2, 1.7])


@pytest.mark.parametrize(
    "overrides",
    [
        {"x": [2.0, -2.0]},  # lo > hi
        {"x": [-2.0, 2.5]},  # x not symmetric
        {"y": [-1.0, 1.5]},  # y not symmetric
        {"z": [0.0, 2.0]},  # z lo not above floor
        {"x": [-2.0]},  # not a pair
        {"geofence_margin": -0.1},
        {"geofence_margin": 0.9},  # empty inner box in z
        {"min_separation": 0.0},
        {"min_separation": 3.0},  # can't fit mirrored spawns
        {"min_vertical_offset": -0.1},
    ],
)
def test_invalid_configs_raise(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        arena_from_dict(_cfg(**overrides))


# ---- dist_to_bounds ---------------------------------------------------------


@pytest.mark.parametrize("xp", XPS)
def test_dist_at_known_point(arena: Arena, xp: ModuleType) -> None:
    pos = xp.asarray([0.5, -1.0, 1.0], dtype=xp.float32)
    d = np.asarray(dist_to_bounds(pos, arena, xp))
    # order x-, x+, y-, y+, z-, z+
    np.testing.assert_allclose(d, [2.5, 1.5, 0.5, 2.5, 0.7, 1.0], atol=1e-6)


@pytest.mark.parametrize("xp", XPS)
def test_dist_zero_on_each_face(arena: Arena, xp: ModuleType) -> None:
    center = np.array([0.0, 0.0, 1.0])
    for face in range(6):
        ax, upper = divmod(face, 2)
        p = center.copy()
        p[ax] = arena.hi[ax] if upper else arena.lo[ax]
        d = np.asarray(dist_to_bounds(xp.asarray(p, dtype=xp.float32), arena, xp))
        assert abs(d[face]) < 1e-6
        assert np.all(np.delete(d, face) > 0)


@pytest.mark.parametrize("xp", XPS)
def test_dist_negative_outside(arena: Arena, xp: ModuleType) -> None:
    pos = xp.asarray([2.1, 0.0, 1.0], dtype=xp.float32)
    d = np.asarray(dist_to_bounds(pos, arena, xp))
    assert d[1] == pytest.approx(-0.1, abs=1e-6)


# ---- out_of_bounds ----------------------------------------------------------


@pytest.mark.parametrize("xp", XPS)
def test_oob_each_axis(arena: Arena, xp: ModuleType) -> None:
    eps = 1e-3
    for ax in range(3):
        for upper in (False, True):
            p = np.array([0.0, 0.0, 1.0])
            edge = arena.hi[ax] if upper else arena.lo[ax]
            step = eps if upper else -eps
            p[ax] = edge
            assert not bool(out_of_bounds(xp.asarray(p, dtype=xp.float32), arena, xp))
            p[ax] = edge + step
            assert bool(out_of_bounds(xp.asarray(p, dtype=xp.float32), arena, xp))


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_oob_nonfinite(arena: Arena, xp: ModuleType, bad: float) -> None:
    pos = xp.asarray([0.0, bad, 1.0], dtype=xp.float32)
    assert bool(out_of_bounds(pos, arena, xp))


@pytest.mark.parametrize("xp", XPS)
def test_oob_batch_shape(arena: Arena, xp: ModuleType) -> None:
    pos = np.zeros((5, 2, 3))
    pos[..., 2] = 1.0
    pos[3, 1, 0] = 3.0  # world 3, drone 1 out
    oob = np.asarray(out_of_bounds(xp.asarray(pos, dtype=xp.float32), arena, xp))
    assert oob.shape == (5, 2)
    assert oob.sum() == 1 and oob[3, 1]


def test_numpy_jax_match(arena: Arena) -> None:
    rng = np.random.default_rng(0)
    pos = rng.uniform(-3.0, 3.0, size=(64, 2, 3)).astype(np.float32)
    np.testing.assert_allclose(
        dist_to_bounds(pos, arena, np), np.asarray(dist_to_bounds(jnp.asarray(pos), arena, jnp))
    )
    np.testing.assert_array_equal(
        out_of_bounds(pos, arena, np), np.asarray(out_of_bounds(jnp.asarray(pos), arena, jnp))
    )


def test_jit(arena: Arena) -> None:
    f = jax.jit(lambda p: (dist_to_bounds(p, arena, jnp), out_of_bounds(p, arena, jnp)))
    d, oob = f(jnp.zeros((4, 2, 3)).at[..., 2].set(1.0))
    assert d.shape == (4, 2, 6) and oob.shape == (4, 2) and not bool(oob.any())


# ---- spawn ------------------------------------------------------------------


def test_spawn_valid(arena: Arena) -> None:
    n = 4096
    pos, yaw = sample_spawn(jax.random.PRNGKey(0), arena, n)
    pos, yaw = np.asarray(pos), np.asarray(yaw)
    assert pos.shape == (n, 2, 3) and yaw.shape == (n, 2)
    assert np.all(pos >= arena.inner_lo - 1e-6) and np.all(pos <= arena.inner_hi + 1e-6)
    sep = np.linalg.norm(pos[:, 0, :2] - pos[:, 1, :2], axis=-1)
    assert np.all(sep >= arena.min_separation - 1e-5)
    np.testing.assert_allclose(pos[:, 1, :2], -pos[:, 0, :2], atol=1e-6)
    np.testing.assert_allclose(pos[:, 1, 2], pos[:, 0, 2])
    assert np.all(np.abs(yaw) <= np.pi)
    assert not np.any(out_of_bounds(pos, arena, np))


def test_spawn_deterministic(arena: Arena) -> None:
    f = jax.jit(lambda k: sample_spawn(k, arena, 16))
    a, b = f(jax.random.PRNGKey(7)), f(jax.random.PRNGKey(7))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
