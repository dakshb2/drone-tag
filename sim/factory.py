"""Build the WS1 Crazyflow sim from config/ and spawn drones in hover.

Rates are contract constants (docs/contracts.md section 2), not configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import crazyflow  # noqa: F401  # must be imported before scipy (SCIPY_ARRAY_API)
import jax.numpy as jnp
import yaml
from crazyflow.sim import Sim
from crazyflow.utils import leaf_replace
from jax import Array

from sim import onboard

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
PHYSICS_HZ = 500
POLICY_HZ = 50
PHYSICS_STEPS_PER_POLICY_STEP = PHYSICS_HZ // POLICY_HZ


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load((config_dir / name).read_text())
    return cfg


def make_sim(n_worlds: int, n_drones: int = 2, config_dir: Path = CONFIG_DIR) -> Sim:
    """Crazyflow sim with the onboard velocity controller installed, reset to defaults."""
    drone = load_config("drone.yaml", config_dir)
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        drone=drone["crazyflow_drone_model"],
        dynamics="first_principles",
        control="attitude",
        freq=PHYSICS_HZ,
        attitude_freq=PHYSICS_HZ,
    )
    onboard.install(sim, drone["onboard_controller"])
    sim.reset()
    return sim


def yaw_quat(yaw: Array) -> Array:
    """xyzw quaternions for pure yaw rotations, shape (..., 4)."""
    half = 0.5 * yaw
    zeros = jnp.zeros_like(half)
    return jnp.stack([zeros, zeros, jnp.sin(half), jnp.cos(half)], -1)


def spawn_hovering(data: Any, pos: Array, yaw: Array, mask: Array | None = None) -> Any:
    """Place drones at rest in hover equilibrium and resync the onboard controller.

    Args:
        data: SimData from a sim built by make_sim.
        pos: (N, M, 3) world positions.
        yaw: (N, M) headings, rad.
        mask: (N,) worlds to change; None means all.
    """
    st = data.states
    wmask = None if mask is None else mask[:, None]
    st = leaf_replace(
        st,
        wmask,
        pos=jnp.asarray(pos, dtype=st.pos.dtype),
        quat=yaw_quat(jnp.asarray(yaw, dtype=st.quat.dtype)),
        vel=jnp.zeros_like(st.vel),
        ang_vel=jnp.zeros_like(st.ang_vel),
        rotor_vel=onboard.hover_rotor_vel(data),
    )
    return onboard.sync(data.replace(states=st), mask)
