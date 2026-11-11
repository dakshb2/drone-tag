"""WS1 task 1 demo: 2 drones x N worlds flown by velocity + yaw-rate setpoints.

Drone 0 circles (constant forward speed + yaw rate, radius v / w). Drone 1 strafes and bobs.
Each world gets a different phase so the worlds visibly differ.

    python -m sim.demo_velocity --worlds 4 --render      # watch world 0 in the MuJoCo viewer
    python -m sim.demo_velocity --bench                  # throughput of a fully jitted rollout
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from sim import onboard
from sim.factory import (
    PHYSICS_STEPS_PER_POLICY_STEP,
    POLICY_HZ,
    make_sim,
    spawn_hovering,
)


def scripted_cmd(t: Array, n_worlds: int) -> Array:
    """(N, 2, 4) velocity setpoints [vx_b, vy_b, vz, yaw_rate] at time t (s)."""
    phase = jnp.linspace(0.0, jnp.pi, n_worlds)
    zero = jnp.zeros(n_worlds)
    circle = jnp.stack([jnp.full(n_worlds, 0.5), zero, zero, jnp.full(n_worlds, 0.8)], -1)
    strafe = jnp.stack(
        [
            zero,
            0.4 * jnp.sin(0.8 * t + phase),
            0.2 * jnp.sin(1.3 * t + phase),
            zero,
        ],
        -1,
    )
    return jnp.stack([circle, strafe], 1)


def start_poses(n_worlds: int) -> tuple[Array, Array]:
    """Drones 2 m apart at 1 m height, facing each other."""
    pos = jnp.broadcast_to(jnp.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]), (n_worlds, 2, 3))
    yaw = jnp.broadcast_to(jnp.array([0.0, jnp.pi]), (n_worlds, 2))
    return pos, yaw


def run(n_worlds: int, seconds: float, render: bool) -> None:
    sim = make_sim(n_worlds)
    sim.data = spawn_hovering(sim.data, *start_poses(n_worlds))
    n_steps = int(seconds * POLICY_HZ)
    traj = np.zeros((n_steps, n_worlds, 2, 3))
    for k in range(n_steps):
        t0 = time.perf_counter()
        cmd = scripted_cmd(jnp.asarray(k / POLICY_HZ), n_worlds)
        sim.data = onboard.set_velocity_cmd(sim.data, cmd)
        sim.step(PHYSICS_STEPS_PER_POLICY_STEP)
        traj[k] = np.asarray(sim.data.states.pos)
        if render:
            sim.render()
            time.sleep(max(0.0, 1.0 / POLICY_HZ - (time.perf_counter() - t0)))  # real time
    if render:
        sim.close()

    v = np.diff(traj, axis=0) * POLICY_HZ
    print(f"{n_worlds} worlds x 2 drones, {seconds:.1f} s at {POLICY_HZ} Hz policy rate")
    print(f"  final positions, world 0: d0 {traj[-1, 0, 0].round(2)}  d1 {traj[-1, 0, 1].round(2)}")
    speed = np.linalg.norm(v, axis=-1).max(axis=(0, 1))
    print(f"  max speed: d0 {speed[0]:.2f} m/s, d1 {speed[1]:.2f} m/s")
    print(f"  height range: {traj[..., 2].min():.2f} to {traj[..., 2].max():.2f} m")
    print(f"  worlds differ: {not np.allclose(traj[:, 0], traj[:, -1])}")


def make_rollout(step_fn: Callable[..., Any]) -> Callable[[Any, Array], Any]:
    """Jitted rollout: hold each (N, M, 4) setpoint in ``cmds`` (K, N, M, 4) for one policy step."""

    def body(d: Any, c: Array) -> tuple[Any, None]:
        d = onboard.set_velocity_cmd(d, c)
        return step_fn(d, PHYSICS_STEPS_PER_POLICY_STEP), None

    @jax.jit
    def rollout(data: Any, cmds: Array) -> Any:
        return jax.lax.scan(body, data, cmds)[0]

    return rollout


def bench(world_counts: list[int], policy_steps: int) -> None:
    """Throughput of a jitted lax.scan rollout, excluding compile time."""
    print(f"backend: {jax.default_backend()}, {policy_steps} policy steps per rollout")
    for n in world_counts:
        sim = make_sim(n)
        rollout = make_rollout(sim.build_step_fn())
        data0 = spawn_hovering(sim.data, *start_poses(n))
        ts = jnp.arange(policy_steps) / POLICY_HZ
        cmds = jax.vmap(partial(scripted_cmd, n_worlds=n))(ts)  # (K, N, 2, 4)
        jax.block_until_ready(rollout(data0, cmds).states.pos)  # compile
        t0 = time.perf_counter()
        out = jax.block_until_ready(rollout(data0, cmds).states.pos)
        dt = time.perf_counter() - t0
        assert bool(jnp.all(jnp.isfinite(out)))
        sps = n * policy_steps / dt
        print(f"  N={n:5d}: {sps:12,.0f} world-steps/s  ({sps / POLICY_HZ:9,.0f}x real time)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--render", action="store_true", help="show world 0 in the MuJoCo viewer")
    p.add_argument("--bench", action="store_true", help="measure jitted rollout throughput")
    p.add_argument("--bench-worlds", type=int, nargs="+", default=[1, 64, 512, 2048])
    p.add_argument("--bench-steps", type=int, default=500)
    args = p.parse_args()
    if args.bench:
        bench(args.bench_worlds, args.bench_steps)
    else:
        run(args.worlds, args.seconds, args.render)


if __name__ == "__main__":
    main()
