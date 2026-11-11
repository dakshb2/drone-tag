"""WS1 tests for sim/factory.py and the velocity demo."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sim import onboard
from sim.demo_velocity import scripted_cmd, start_poses
from sim.factory import PHYSICS_STEPS_PER_POLICY_STEP, make_sim, spawn_hovering


def test_spawn_hovering_sets_pose_and_controller() -> None:
    sim = make_sim(2)
    pos = jnp.array([[[-1.0, 0.5, 1.2], [1.0, -0.5, 0.8]], [[0.0, 0.0, 1.0], [0.5, 0.5, 1.5]]])
    yaw = jnp.array([[0.3, -2.0], [3.0, 1.0]])
    sim.data = spawn_hovering(sim.data, pos, yaw)
    np.testing.assert_allclose(sim.data.states.pos, pos, atol=1e-6)
    np.testing.assert_allclose(onboard.quat_yaw(sim.data.states.quat), yaw, atol=1e-6)
    ob = sim.data.plugins[onboard.PLUGIN_KEY]
    np.testing.assert_allclose(ob.yaw_sp, yaw, atol=1e-6)
    sim.step(50 * PHYSICS_STEPS_PER_POLICY_STEP)  # 1 s of zero command: stays put
    np.testing.assert_allclose(sim.data.states.pos, pos, atol=0.02)


def test_spawn_mask_only_moves_selected_worlds() -> None:
    sim = make_sim(2)
    sim.data = spawn_hovering(sim.data, *start_poses(2))
    before = sim.data.states.pos
    moved = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.5]), (2, 2, 3))
    sim.data = spawn_hovering(sim.data, moved, jnp.zeros((2, 2)), mask=jnp.array([False, True]))
    np.testing.assert_allclose(sim.data.states.pos[0], before[0])
    np.testing.assert_allclose(sim.data.states.pos[1], moved[1], atol=1e-6)


def test_scripted_cmd_shape_and_limits() -> None:
    cmd = scripted_cmd(jnp.asarray(1.0), 3)
    assert cmd.shape == (3, 2, 4)
    assert bool(jnp.all(jnp.abs(cmd) <= 1.0))
