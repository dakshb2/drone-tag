"""WS1 tests for sim/onboard.py, the ported firmware PID velocity controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import crazyflow  # noqa: F401  # must be imported before scipy (SCIPY_ARRAY_API)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from crazyflow.sim import Sim

from sim import onboard

ROOT = Path(__file__).resolve().parents[1]
FREQ = 500
CTRL_STEPS = FREQ // 50  # physics steps per 50 Hz policy step


def drone_cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "config" / "drone.yaml").read_text())


def make_sim(n_worlds: int = 1, n_drones: int = 2) -> Sim:
    cfg = drone_cfg()
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        drone=cfg["crazyflow_drone_model"],
        dynamics="first_principles",
        control="attitude",
        freq=FREQ,
        attitude_freq=FREQ,
    )
    onboard.install(sim, cfg["onboard_controller"])
    sim.reset()
    return sim


def place(sim: Sim, z: float = 1.0, yaw: float = 0.0) -> None:
    """Put every drone at height z with the given yaw, hovering, and resync the controller."""
    st = sim.data.states
    half = yaw / 2.0
    quat = jnp.zeros_like(st.quat).at[..., 2].set(np.sin(half)).at[..., 3].set(np.cos(half))
    st = st.replace(
        pos=st.pos.at[..., 2].set(z),
        quat=quat,
        vel=jnp.zeros_like(st.vel),
        ang_vel=jnp.zeros_like(st.ang_vel),
        rotor_vel=onboard.hover_rotor_vel(sim.data),
    )
    sim.data = onboard.sync(sim.data.replace(states=st))


def fly(sim: Sim, cmd: list[float], seconds: float) -> np.ndarray:
    """Hold a velocity command, return world positions at every 50 Hz step, (T, N, M, 3)."""
    full = jnp.broadcast_to(jnp.asarray(cmd), sim.data.plugins[onboard.PLUGIN_KEY].cmd.shape)
    sim.data = onboard.set_velocity_cmd(sim.data, full)
    out = []
    for _ in range(int(seconds * 50)):
        sim.step(CTRL_STEPS)
        out.append(np.asarray(sim.data.states.pos))
    return np.stack(out)


def yaw_of(sim: Sim) -> np.ndarray:
    return np.asarray(onboard.quat_yaw(sim.data.states.quat))


@pytest.fixture(scope="module")
def sim() -> Sim:
    return make_sim()


def warm_up(sim: Sim, yaw: float = 0.0) -> None:
    """Settled-integral start plus 0.5 s for rotor spin-up, as an episode start."""
    place(sim, yaw=yaw)
    fly(sim, [0.0, 0.0, 0.0, 0.0], 0.5)


def test_hover_holds_height(sim: Sim) -> None:
    warm_up(sim)
    pos = fly(sim, [0.0, 0.0, 0.0, 0.0], 2.0)
    assert np.all(np.abs(pos[..., 2] - pos[0, ..., 2]) < 0.03)
    assert np.all(np.abs(pos[-1, ..., :2] - pos[0, ..., :2]) < 0.03)


@pytest.mark.parametrize(
    ("yaw", "cmd", "axis", "sign"),
    [
        (0.0, [0.5, 0.0, 0.0, 0.0], 0, 1.0),  # +vx body -> +x world
        (0.0, [0.0, 0.5, 0.0, 0.0], 1, 1.0),  # +vy body -> +y world
        (np.pi / 2, [0.5, 0.0, 0.0, 0.0], 1, 1.0),  # facing +y: +vx body -> +y world
        (np.pi / 2, [0.0, 0.5, 0.0, 0.0], 0, -1.0),  # facing +y: +vy body -> -x world
        (0.0, [0.0, 0.0, 0.3, 0.0], 2, 1.0),  # +vz -> climbs
    ],
)
def test_velocity_tracks_in_contract_frames(
    sim: Sim, yaw: float, cmd: list[float], axis: int, sign: float
) -> None:
    warm_up(sim, yaw)
    pos = fly(sim, cmd, 3.0)
    speed = float(np.linalg.norm(cmd[:3]))
    v = np.diff(pos, axis=0) * 50.0  # (T-1, N, M, 3)
    v_end = v[-25:].mean(axis=0)  # last 0.5 s
    # Firmware xy integral is tiny (ki/kp = 1/25 per s), so drag leaves ~9% lag after 3 s.
    # vz has a strong integral and settles exactly.
    lo = 0.97 if axis == 2 else 0.85
    assert np.all((sign * v_end[..., axis] > lo * speed) & (sign * v_end[..., axis] < 1.03 * speed))
    others = [a for a in range(3) if a != axis]
    assert np.all(np.abs(v_end[..., others]) < 0.05)


def test_yaw_rate_tracks_through_wrap(sim: Sim) -> None:
    warm_up(sim, yaw=2.5)
    rate = 1.0
    yaws = []
    sim.data = onboard.set_velocity_cmd(
        sim.data, jnp.broadcast_to(jnp.asarray([0.0, 0.0, 0.0, rate]), (1, 2, 4))
    )
    for _ in range(4 * 50):  # 4 s, crosses +pi after ~0.6 s
        sim.step(CTRL_STEPS)
        yaws.append(yaw_of(sim))
    yaw_rate = np.diff(np.unwrap(np.stack(yaws), axis=0), axis=0) * 50.0
    steady = yaw_rate[50:]  # after 1 s
    assert np.all(np.abs(steady - rate) < 0.1)
    assert np.max(np.abs(yaw_rate)) < 1.5 * rate  # no spin at the wrap


def test_shapes_and_pure_step_jit() -> None:
    n_worlds, n_drones = 3, 2
    s = make_sim(n_worlds, n_drones)
    ob = s.data.plugins[onboard.PLUGIN_KEY]
    assert ob.cmd.shape == (n_worlds, n_drones, 4)
    data = onboard.set_velocity_cmd(s.data, jnp.zeros((n_worlds, n_drones, 4)))
    data = jax.block_until_ready(s._step(data, CTRL_STEPS))
    assert data.controls.attitude.cmd.shape == (n_worlds, n_drones, 4)
    assert bool(jnp.all(jnp.isfinite(data.states.pos)))


def test_reset_mask_only_touches_selected_worlds() -> None:
    s = make_sim(2, 2)
    place(s)
    s.data = onboard.set_velocity_cmd(s.data, jnp.full((2, 2, 4), 0.2))
    s.step(50)
    before = s.data.plugins[onboard.PLUGIN_KEY].integ
    s.reset(mask=jnp.array([True, False]))
    after = s.data.plugins[onboard.PLUGIN_KEY].integ
    hover = onboard.hover_integral(s.data)
    np.testing.assert_allclose(after[0, :, 2], hover[0], rtol=1e-6)
    assert bool(jnp.all(after[0, :, :2] == 0.0))
    assert bool(jnp.all(after[1] == before[1]))
    assert not bool(jnp.all(before[0] == after[0]))


def test_unsettled_start_sinks_then_recovers() -> None:
    """Firmware-like zero integral: drone sags, the integral recovers it, as on hardware."""
    s = make_sim(1, 1)
    place(s)
    s.data = onboard.sync(s.data, settled=False)
    pos = fly(s, [0.0, 0.0, 0.0, 0.0], 8.0)
    assert pos[:, 0, 0, 2].min() < 0.95  # visible sag with this model's hover PWM
    v_end = (pos[-1, 0, 0, 2] - pos[-26, 0, 0, 2]) * 2.0
    assert abs(v_end) < 0.01


def test_deterministic() -> None:
    def run() -> np.ndarray:
        s = make_sim(1, 2)
        place(s)
        return fly(s, [0.3, -0.2, 0.1, 0.5], 1.0)

    np.testing.assert_array_equal(run(), run())


def test_nonzero_kd_rejected() -> None:
    oc = drone_cfg()["onboard_controller"]
    oc["vel_xy"]["kd"] = 0.1
    with pytest.raises(ValueError, match="kd"):
        onboard.OnboardParams.from_config(oc, pwm2newton=1e-5)
