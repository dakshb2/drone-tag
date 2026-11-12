"""Episode spawn positions inside the arena (sim only, JAX).

Point-symmetric spawns: drone 1 is drone 0 mirrored through the arena center
in xy, at the same height, so neither drone starts with a positional advantage.
Spawns lie inside the safety-filter inner box with horizontal separation
>= min_separation. Yaws are independent and uniform.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sim.arena import Arena

_EPS = 1e-9


def sample_spawn(key: jax.Array, arena: Arena, n_worlds: int) -> tuple[jax.Array, jax.Array]:
    """Sample spawn poses for 2 drones per world.

    Returns pos (n_worlds, 2, 3) and yaw (n_worlds, 2), for spawn_hovering.
    Drone 0 is placed at angle theta, radius r in [min_separation / 2, r_max(theta)],
    where r_max is the distance to the inner box edge along theta.
    """
    k_th, k_r, k_z, k_yaw = jax.random.split(key, 4)
    inner_lo = jnp.asarray(arena.inner_lo, dtype=jnp.float32)
    inner_hi = jnp.asarray(arena.inner_hi, dtype=jnp.float32)
    hx, hy = inner_hi[0], inner_hi[1]  # inner box is centered at 0 in xy

    theta = jax.random.uniform(k_th, (n_worlds,), minval=-jnp.pi, maxval=jnp.pi)
    c, s = jnp.cos(theta), jnp.sin(theta)
    r_max = jnp.minimum(hx / jnp.maximum(jnp.abs(c), _EPS), hy / jnp.maximum(jnp.abs(s), _EPS))
    r_min = arena.min_separation / 2.0
    r = r_min + jax.random.uniform(k_r, (n_worlds,)) * (r_max - r_min)
    z = jax.random.uniform(k_z, (n_worlds,), minval=inner_lo[2], maxval=inner_hi[2])

    p0 = jnp.stack([r * c, r * s, z], axis=-1)
    p1 = jnp.stack([-r * c, -r * s, z], axis=-1)
    pos = jnp.clip(jnp.stack([p0, p1], axis=1), inner_lo, inner_hi)
    yaw = jax.random.uniform(k_yaw, (n_worlds, 2), minval=-jnp.pi, maxval=jnp.pi)
    return pos, yaw
