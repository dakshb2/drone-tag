"""Port of the Crazyflie firmware PID velocity controller as a Crazyflow pipeline stage.

The contract action (docs/contracts.md section 3) is a velocity setpoint to the onboard
controller. Crazyflow 0.4.0 has no velocity mode, so this module adds one on top of its
attitude mode, mirroring the firmware's PID controller in velocity mode:

- yaw setpoint integrated from the yaw-rate command at ``attitude_rate`` (controller_pid.c)
- body-yaw velocity PIDs -> roll/pitch setpoints, vz PID -> thrust in PWM, at
  ``position_rate`` (position_controller_pid.c, velocityController)
- PWM thrust converted to newtons with Crazyflow's linear map, scaled per drone by
  ``thrust_scale`` (domain randomization hook)

The result is written to Crazyflow's staged attitude command [roll, pitch, yaw, thrust], so the
stage must run before ``attitude_controller``. Supported configuration: attitude control with
first_principles dynamics (so_rpy has a yaw bias and spins at the +-pi wrap, see
docs/decisions.md 2026-11-11).

Sign conventions: the firmware's pitch is positive nose-up, Crazyflow's xyz Euler pitch is
positive nose-down. Hence pitch = +PID(vx) here where the firmware has pitch = -PID(vx); roll
keeps the firmware sign, roll = -PID(vy).

Not ported: derivative terms and their filters (all firmware velocity kd are 0; nonzero kd is
rejected), yaw setpoint modes other than velocity, thrust == 0 reinitialization (thrust_min > 0).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass as std_dataclass
from typing import Any

import jax.numpy as jnp
from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.pipeline import append_fn, insert_fn_before
from crazyflow.utils import CORE_NDIM_KEY, leaf_replace
from flax.struct import dataclass, field
from jax import Array

PLUGIN_KEY = "onboard"
ANCHOR_STAGE = "attitude_controller"


@std_dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    i_limit: float
    output_limit: float


@std_dataclass(frozen=True)
class OnboardParams:
    """Static controller parameters. Baked into the jitted step as constants."""

    position_rate: int
    attitude_rate: int
    vel_xy: PIDGains
    vel_z: PIDGains
    roll_pitch_limit: float
    thrust_base_pwm: float
    thrust_min_pwm: float
    thrust_max_pwm: float
    yaw_max_delta: float
    pwm2newton: float  # collective thrust in N per PWM unit, from the Crazyflow drone model

    @staticmethod
    def from_config(oc: dict[str, Any], pwm2newton: float) -> OnboardParams:
        """Build from the drone.yaml onboard_controller block."""

        def gains(axis: str) -> PIDGains:
            g = oc[axis]
            if float(g["kd"]) != 0.0:
                raise ValueError(f"onboard_controller.{axis}.kd != 0: derivative term not ported")
            return PIDGains(
                kp=float(g["kp"]),
                ki=float(g["ki"]),
                i_limit=float(g["i_limit"]),
                output_limit=float(g["output_limit"]),
            )

        return OnboardParams(
            position_rate=int(oc["position_rate"]),
            attitude_rate=int(oc["attitude_rate"]),
            vel_xy=gains("vel_xy"),
            vel_z=gains("vel_z"),
            roll_pitch_limit=float(oc["roll_pitch_limit"]),
            thrust_base_pwm=float(oc["thrust_base_pwm"]),
            thrust_min_pwm=float(oc["thrust_min_pwm"]),
            thrust_max_pwm=float(oc["thrust_max_pwm"]),
            yaw_max_delta=float(oc["yaw_max_delta"]),
            pwm2newton=pwm2newton,
        )


@dataclass
class OnboardData:
    """Per-world controller state, stored in SimData.plugins[PLUGIN_KEY]."""

    cmd: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 4) [vx_b, vy_b, vz, yaw_rate], SI
    yaw_sp: Array = field(metadata={CORE_NDIM_KEY: 0})  # (N, M) rad
    integ: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 3) PID integrals vx_b, vy_b, vz
    rp_sp: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 2) latched roll, pitch, rad
    thrust_pwm: Array = field(metadata={CORE_NDIM_KEY: 0})  # (N, M) latched thrust, PWM
    thrust_scale: Array = field(metadata={CORE_NDIM_KEY: 0})  # (N, M) motor strength factor
    vel_steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1) last velocity-loop tick
    att_steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1) last yaw-integration tick
    params: OnboardParams = field(pytree_node=False)

    @staticmethod
    def create(n_worlds: int, n_drones: int, params: OnboardParams) -> OnboardData:
        zeros = jnp.zeros((n_worlds, n_drones))
        never = -jnp.ones((n_worlds, 1), dtype=jnp.int32)
        return OnboardData(
            cmd=jnp.zeros((n_worlds, n_drones, 4)),
            yaw_sp=zeros,
            integ=jnp.zeros((n_worlds, n_drones, 3)),
            rp_sp=jnp.zeros((n_worlds, n_drones, 2)),
            thrust_pwm=jnp.full((n_worlds, n_drones), params.thrust_base_pwm),
            thrust_scale=jnp.ones((n_worlds, n_drones)),
            vel_steps=never,
            att_steps=never.copy(),
            params=params,
        )


def wrap_angle(a: Array) -> Array:
    """Wrap to (-pi, pi]."""
    return jnp.pi - jnp.mod(jnp.pi - a, 2 * jnp.pi)


def quat_yaw(quat: Array) -> Array:
    """Yaw (rad) of xyzw quaternions, matching scipy's xyz Euler convention."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _due(step: Array, sim_freq: int, last: Array, rate: int) -> Array:
    """(N, 1) mask of worlds whose loop at ``rate`` ticks this step."""
    return ((step - last) >= (sim_freq // rate)) | (last == -1)


def _pid(error: Array, integ: Array, g: PIDGains, dt: float, mask: Array) -> tuple[Array, Array]:
    """Firmware runPid without D/FF terms. Integral includes the current error."""
    new_integ = jnp.clip(integ + error * dt, -g.i_limit, g.i_limit)
    out = jnp.clip(g.kp * error + g.ki * new_integ, -g.output_limit, g.output_limit)
    return out, jnp.where(mask, new_integ, integ)


def onboard_control(data: Any) -> Any:
    """Pipeline stage: velocity setpoint -> staged attitude command. Pure, jittable."""
    ob: OnboardData = data.plugins[PLUGIN_KEY]
    params = ob.params
    step, sim_freq = data.core.steps, data.core.freq
    yaw = quat_yaw(data.states.quat)  # (N, M)

    # 1. Yaw setpoint integration (controller_pid.c, ATTITUDE_RATE block)
    att_mask = _due(step, sim_freq, ob.att_steps, params.attitude_rate)
    yaw_sp = wrap_angle(ob.yaw_sp + ob.cmd[..., 3] / params.attitude_rate)
    if params.yaw_max_delta != 0.0:
        delta = wrap_angle(yaw_sp - yaw)
        yaw_sp = wrap_angle(yaw + jnp.clip(delta, -params.yaw_max_delta, params.yaw_max_delta))
    ob = leaf_replace(ob, att_mask, yaw_sp=yaw_sp, att_steps=step)

    # 2. Velocity loop (position_controller_pid.c, velocityController)
    vel_mask = _due(step, sim_freq, ob.vel_steps, params.position_rate)
    m3 = vel_mask[..., None]  # (N, 1, 1) for (N, M, k) leaves
    dt = 1.0 / params.position_rate
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    v = data.states.vel  # world frame, true state
    v_body = jnp.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)
    err_xy = ob.cmd[..., :2] - v_body
    err_z = ob.cmd[..., 2:3] - v[..., 2:3]
    out_xy, integ_xy = _pid(err_xy, ob.integ[..., :2], params.vel_xy, dt, m3)
    out_z, integ_z = _pid(err_z, ob.integ[..., 2:], params.vel_z, dt, m3)
    lim = params.roll_pitch_limit
    roll = jnp.clip(-out_xy[..., 1], -lim, lim)
    pitch = jnp.clip(out_xy[..., 0], -lim, lim)
    thrust_pwm = jnp.clip(
        out_z[..., 0] + params.thrust_base_pwm, params.thrust_min_pwm, params.thrust_max_pwm
    )
    ob = leaf_replace(
        ob,
        vel_mask,
        rp_sp=jnp.stack([roll, pitch], -1),
        thrust_pwm=thrust_pwm,
        vel_steps=step,
    )
    ob = dataclasses.replace(ob, integ=jnp.concatenate([integ_xy, integ_z], -1))

    # 3. Stage the attitude command for Crazyflow's attitude controller
    thrust_n = ob.thrust_pwm * params.pwm2newton * ob.thrust_scale
    rpyt = jnp.stack([ob.rp_sp[..., 0], ob.rp_sp[..., 1], ob.yaw_sp, thrust_n], -1)
    attitude = data.controls.attitude.replace(staged_cmd=rpyt)
    return data.replace(
        controls=data.controls.replace(attitude=attitude),
        plugins={**data.plugins, PLUGIN_KEY: ob},
    )


def hover_integral(data: Any) -> Array:
    """(N, M) vz integral at which the z loop outputs hover thrust with zero velocity error.

    This is the state the firmware integrator settles to after hovering long enough. With
    vz error 0: thrust_pwm = thrust_base_pwm + ki * integ, and hover needs m * g newtons.
    """
    ob: OnboardData = data.plugins[PLUGIN_KEY]
    p = ob.params
    mass = jnp.broadcast_to(data.params.mass[..., 0], ob.thrust_scale.shape)
    g = -data.params.gravity_vec[..., 2]
    hover_pwm = mass * g / (p.pwm2newton * ob.thrust_scale)
    return jnp.clip((hover_pwm - p.thrust_base_pwm) / p.vel_z.ki, -p.vel_z.i_limit, p.vel_z.i_limit)


def hover_rotor_vel(data: Any) -> Array:
    """(N, M, 4) rotor speed in RPM at which first_principles thrust equals m * g.

    Solves k2 w^2 + k1 w + k0 = m g / 4 with Crazyflow's shared rpm2thrust curve. Use it when
    spawning drones in the air, so the rotors are not spinning up from rest.
    """
    ob: OnboardData = data.plugins[PLUGIN_KEY]
    k = data.params.rpm2thrust[0]
    k0, k1, k2 = k[0], k[1], k[2]
    g = -data.params.gravity_vec[..., 2]
    f = data.params.mass[..., 0] * g / 4.0
    w = (-k1 + jnp.sqrt(k1 * k1 - 4.0 * k2 * (k0 - f))) / (2.0 * k2)
    return jnp.broadcast_to(w, (*ob.thrust_scale.shape, 4))


def sync(data: Any, mask: Array | None = None, settled: bool = True) -> Any:
    """Re-initialize the controller to the current drone state.

    Sets the yaw setpoint to the current yaw and the xy integrals to zero. The vz integral is set
    to its settled hover value if ``settled`` (a drone that has hovered on zero velocity
    setpoints before the policy takes over), else zero (firmware controllerPidReinitialize, as
    on a setpoint mode change). Call after changing drone poses directly (e.g. random spawns).
    ``mask`` is (N,) over worlds; None means all.
    """
    ob: OnboardData = data.plugins[PLUGIN_KEY]
    wmask = None if mask is None else mask[:, None]
    integ_z = hover_integral(data) if settled else jnp.zeros_like(ob.thrust_scale)
    integ = jnp.zeros_like(ob.integ).at[..., 2].set(integ_z)
    ob = leaf_replace(
        ob,
        wmask,
        yaw_sp=quat_yaw(data.states.quat),
        integ=integ,
        vel_steps=-jnp.ones_like(ob.vel_steps),
        att_steps=-jnp.ones_like(ob.att_steps),
    )
    return data.replace(plugins={**data.plugins, PLUGIN_KEY: ob})


def _reset_stage(data: Any, default_data: Any, mask: Array | None = None) -> Any:
    return sync(data, mask)


def set_velocity_cmd(data: Any, cmd: Array) -> Any:
    """Set the velocity setpoint [vx_b, vy_b, vz, yaw_rate] (SI), shape (N, M, 4).

    Body-yaw frame for vx, vy; world for vz and yaw rate (contract section 3). Scaling from
    [-1, 1] and the safety filter happen before this call, not here.
    """
    ob: OnboardData = data.plugins[PLUGIN_KEY]
    cmd = jnp.asarray(cmd, dtype=ob.cmd.dtype)
    assert cmd.shape == ob.cmd.shape, f"cmd shape {cmd.shape} != {ob.cmd.shape}"
    return data.replace(plugins={**data.plugins, PLUGIN_KEY: dataclasses.replace(ob, cmd=cmd)})


def install(sim: Sim, oc: dict[str, Any]) -> OnboardParams:
    """Add the onboard velocity controller to a Sim in place.

    Args:
        sim: Sim built with control=attitude and dynamics=first_principles.
        oc: the drone.yaml onboard_controller block.

    Returns:
        The parameters baked into the step function.
    """
    if sim.control != Control.attitude or sim.dynamics != Dynamics.first_principles:
        raise ValueError("onboard controller needs control=attitude, dynamics=first_principles")
    att = sim.data.controls.attitude
    for rate in (int(oc["position_rate"]), int(oc["attitude_rate"])):
        if sim.freq % rate != 0:
            raise ValueError(f"sim freq {sim.freq} Hz not divisible by controller rate {rate}")
    if att.freq < int(oc["attitude_rate"]):
        raise ValueError(f"Crazyflow attitude_freq {att.freq} < attitude_rate")
    # Same PWM -> N map as Crazyflow's Mellinger: collective = pwm / pwm_max * 4 * thrust_max
    pwm2newton = 4.0 * float(att.params["thrust_max"]) / float(att.params["pwm_max"])
    params = OnboardParams.from_config(oc, pwm2newton)

    ob = OnboardData.create(sim.n_worlds, sim.n_drones, params)
    sim.data = sim.data.replace(plugins={**sim.data.plugins, PLUGIN_KEY: ob})
    sim.data = sync(sim.data)
    insert_fn_before(sim.step_pipeline, ANCHOR_STAGE, onboard_control, name="onboard_controller")
    append_fn(sim.reset_pipeline, _reset_stage, name="onboard_reset")
    sim.build_default_data()
    sim.build_step_fn()
    sim.build_reset_fn()
    return params
