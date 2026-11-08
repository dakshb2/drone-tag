"""WS0 tests: environment, configs vs contracts, naming, dummy job, Crazyflow smoke."""

from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from train import dummy
from train.naming import checkpoint_path, run_name

ROOT = Path(__file__).resolve().parents[1]

# Required top-level keys per docs/contracts.md section 10.
REQUIRED_KEYS = {
    "arena.yaml": {"bounds", "geofence_margin", "min_separation", "min_vertical_offset"},
    "drone.yaml": {"limits", "mass", "crazyflow_drone_model"},
    "hit_model.yaml": {
        "cone_half_angle",
        "max_range",
        "dwell_time",
        "hit_cooldown",
        "receivers",
        "p_detect",
    },
    "randomization.yaml": {
        "mass_scale",
        "thrust_scale",
        "latency_steps",
        "obs_noise_std",
        "battery_sag",
    },
    "game.yaml": {"hp", "max_episode_time", "reward"},
}


def test_jax_runs():
    assert float(jnp.sum(jnp.arange(4.0))) == 6.0


@pytest.mark.parametrize("name", sorted(REQUIRED_KEYS))
def test_config_matches_contract(name):
    cfg = yaml.safe_load((ROOT / "config" / name).read_text())
    assert cfg["contract_version"] == "0.1"
    missing = REQUIRED_KEYS[name] - set(cfg)
    assert not missing, f"{name} missing keys: {missing}"


def test_run_name_format():
    n = run_name("C", "selfplay_v1", 2, datetime(2026, 11, 8, 14, 30))
    assert n == "C-selfplay_v1-s2-20261108-1430"
    assert checkpoint_path(Path("checkpoints"), n, 1200).name == "step_000001200.npz"
    with pytest.raises(ValueError):
        run_name("E", "x", 0)
    with pytest.raises(ValueError):
        run_name("A", "Bad-Name", 0)


def test_dummy_job_converges():
    cfg = dummy.load_config(ROOT / "config/train/dummy.yaml")
    assert dummy.run(cfg, wandb_mode="disabled") < 1e-3


@pytest.mark.slow
def test_crazyflow_two_drones_hover():
    from crazyflow import Sim

    n_worlds, n_drones = 4, 2
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, control="state")
    sim.reset()
    p0 = np.asarray(sim.data.states.pos)
    assert p0.shape == (n_worlds, n_drones, 3)
    cmd = np.zeros((n_worlds, n_drones, 16))  # state command: pos, vel, acc, quat, body rates
    cmd[..., :3] = p0
    cmd[..., 2] += 0.5
    cmd[..., 12] = 1.0  # quaternion qw
    sim.state_control(cmd)
    sim.step(1000)  # 2 s at 500 Hz
    z = np.asarray(sim.data.states.pos)[..., 2]
    assert np.all(np.abs(z - 0.5) < 0.1)
