"""Integration: geofence termination and sampled spawns in the Crazyflow sim (WS1 task 2)."""

from __future__ import annotations

from typing import Any

import crazyflow  # noqa: F401  # must be imported before scipy (SCIPY_ARRAY_API)
import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim import Sim

from sim import onboard
from sim.arena import dist_to_bounds, load_arena, out_of_bounds
from sim.factory import PHYSICS_STEPS_PER_POLICY_STEP, POLICY_HZ, make_sim, spawn_hovering
from sim.spawn import sample_spawn

DT = 1.0 / POLICY_HZ
V_CMD = 0.5  # m/s


def _rollout_pos(sim: Sim, data: Any, cmds: jax.Array) -> np.ndarray:
    """Hold each (N, M, 4) setpoint in cmds (K, N, M, 4) for one policy step.

    Returns positions after each policy step, (K, N, M, 3).
    """
    step_fn = sim.build_step_fn()

    def body(d: Any, c: jax.Array) -> tuple[Any, jax.Array]:
        d = onboard.set_velocity_cmd(d, c)
        d = step_fn(d, PHYSICS_STEPS_PER_POLICY_STEP)
        return d, d.states.pos

    @jax.jit
    def rollout(d: Any, cs: jax.Array) -> jax.Array:
        return jax.lax.scan(body, d, cs)[1]

    return np.asarray(rollout(data, cmds))


def test_geofence_trips_on_first_step_past_wall() -> None:
    arena = load_arena()
    lo, hi = arena.lo, arena.hi
    x0 = 0.5 * float(hi[0])
    z0 = 0.5 * float(lo[2] + hi[2])
    # World 0: drone 0 (yaw 0) flies forward to x+, drone 1 (yaw pi) flies forward to x-.
    # World 1: drone 0 descends to z-, drone 1 hovers.
    pos = jnp.array([[[x0, 0.0, z0], [-x0, 0.0, z0]]] * 2)
    yaw = jnp.array([[0.0, jnp.pi]] * 2)
    cmd = jnp.array(
        [
            [[V_CMD, 0.0, 0.0, 0.0], [V_CMD, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, -V_CMD, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    # (world, drone, face index in x-, x+, y-, y+, z-, z+ order, distance to that face)
    cases: list[tuple[int, int, int, float]] = [
        (0, 0, 1, float(hi[0]) - x0),
        (0, 1, 0, -x0 - float(lo[0])),
        (1, 0, 4, z0 - float(lo[2])),
    ]
    t_max = max(c[3] for c in cases) / (0.8 * V_CMD) + 1.0
    n_steps = int((t_max + 0.5) * POLICY_HZ)

    sim = make_sim(2)
    data = spawn_hovering(sim.data, pos, yaw)
    traj = _rollout_pos(sim, data, jnp.broadcast_to(cmd, (n_steps, *cmd.shape)))
    oob = out_of_bounds(traj, arena, np)  # (K, N, M)
    dist = dist_to_bounds(traj, arena, np)  # (K, N, M, 6)

    assert not oob[:, 1, 1].any(), "hovering drone left bounds"
    for w, m, face, d0 in cases:
        hits = np.flatnonzero(oob[:, w, m])
        assert hits.size, f"world {w} drone {m} never left bounds"
        k = int(hits[0])
        assert k > 0
        t = (k + 1) * DT  # positions are recorded after each step
        assert d0 / (1.1 * V_CMD) <= t <= d0 / (0.8 * V_CMD) + 1.0, f"exit time {t:.2f} s"
        assert dist[k - 1, w, m, face] >= 0.0
        assert -1.1 * V_CMD * DT <= dist[k, w, m, face] < 0.0, "overshoot > one step of travel"
        assert np.all(np.delete(dist[k, w, m], face) > 0.0), "left through the wrong face"
        assert np.all(oob[k : k + 10, w, m]), "came back inside while flying outward"


def test_sampled_spawns_hover_inside_bounds() -> None:
    arena = load_arena()
    n = 64
    sim = make_sim(n)
    pos, yaw = sample_spawn(jax.random.PRNGKey(0), arena, n)
    data = spawn_hovering(sim.data, pos, yaw)
    traj = _rollout_pos(sim, data, jnp.zeros((POLICY_HZ, n, 2, 4)))  # 1 s of zero setpoints
    assert not out_of_bounds(traj, arena, np).any()
    drift = np.linalg.norm(traj - np.asarray(pos)[None], axis=-1).max()
    assert drift < 0.05, f"hover drift {drift:.3f} m"
