# Contracts

Version: 0.2
Status: agreed 2026-11-11 by Daksh
Rule: never change silently. Propose a CONTRACT CHANGE PROPOSAL (what, why, affected workstreams), agree as a team, bump the version, log it in docs/decisions.md.

## 1. Frames and units
- World frame: right-handed, z up, origin at the arena center on the floor, x and y aligned with the arena edges.
- Body-yaw frame: origin at the drone, z up, x along the drone heading projected onto the horizontal plane. Rotates with yaw only.
- Units: SI (m, s, rad). Angles wrapped to (-pi, pi]. Exception: onboard controller thrust values are in firmware PWM units (section 10).
- Yaw: about +z, counterclockwise positive, 0 along world +x.

## 2. Rates
- Physics: 500 Hz (sim only).
- Policy, control and hit checking: 50 Hz (dt = 0.02 s), in sim and on hardware.
- Actions are held constant between policy steps.

## 3. Action (per drone, 4 floats in [-1, 1])
| idx | name     | frame    | scaled by (drone.yaml) |
|-----|----------|----------|------------------------|
| 0   | vx       | body-yaw | limits.v_xy            |
| 1   | vy       | body-yaw | limits.v_xy            |
| 2   | vz       | world    | limits.v_z             |
| 3   | yaw_rate | world z  | limits.yaw_rate        |

- The action is a velocity setpoint to the Crazyflie onboard controller. The sim reproduces the setpoint-to-motion response identified by WS5, not ideal velocity tracking.
- Onboard controller: the Crazyflie firmware PID controller (stabilizer.controller = 1). The firmware Mellinger controller ignores horizontal velocity setpoints and must not be used. Gains and limits are in drone.yaml onboard_controller; the sim runs a port of this controller.
- Hardware transport: no firmware setpoint packet takes body-yaw vx, vy together with vz. The flight node rotates (vx, vy) into the world frame using the current yaw and sends a velocityWorld setpoint (yaw rate in deg/s on the wire).
- Every action passes through the safety filter (section 8) before reaching the controller, in sim and on hardware. The policy never commands the drone directly.

## 4. Observation (per drone, 28 floats, this exact order)
| idx   | value                                         | frame    |
|-------|-----------------------------------------------|----------|
| 0-2   | own position relative to arena center         | world    |
| 3-5   | own velocity                                  | body-yaw |
| 6-7   | sin, cos of own yaw                           | world    |
| 8     | own yaw rate                                  |          |
| 9-11  | opponent relative position                    | body-yaw |
| 12-14 | opponent relative velocity                    | body-yaw |
| 15-16 | sin, cos of opponent yaw minus own yaw        |          |
| 17    | own hp / hp_max                               |          |
| 18    | opponent hp / hp_max                          |          |
| 19    | own cooldown remaining / hit_cooldown         |          |
| 20    | opponent cooldown remaining / hit_cooldown    |          |
| 21    | time remaining / max_episode_time             |          |
| 22-27 | distance to bounds: x-, x+, y-, y+, z-, z+    |          |

- Velocities are estimated by the same method in sim and real. WS5 defines the method from the real setup (e.g. differenced mocap poses plus a filter); WS1 implements it identically in sim.
- No opponent history. Memory comes only from the network (GRU in variants B, C, D). Variant A sees only the current observation.
- Any further normalization (running mean and std) lives in the policy and is exported with it.
- Excluded: opponent actions, opponent internal state, anything the real system cannot provide.

## 5. Opponent prediction target (variants C and D)
- Opponent relative position at steps t+1 .. t+H, expressed in own body-yaw frame frozen at time t. Shape (H, 3). H set in the train config (default 10).
- Loss: mean squared error.

## 6. Hit model
- Emitter axis: body +x rotated by the full attitude (roll, pitch, yaw).
- off_axis_angle: angle between the shooter's emitter axis and the line from shooter to target.
- incidence_angle: angle between the line from target to shooter and a receiver's axis. Computed per receiver; the receiver with the highest p_detect is used.
- Geometric condition, checked every 50 Hz step: off_axis_angle <= cone_half_angle, distance <= max_range, line of sight clear.
- Per-step detection: if the geometric condition holds, a detection occurs with probability p_detect(distance, off_axis_angle, incidence_angle).
- Hit: n_dwell = round(dwell_time / 0.02) consecutive steps with a detection. A step without detection resets the count.
- After a hit, that shooter cannot score for hit_cooldown seconds, and its dwell count resets.
- Firmware (WS4) uses the same rule: a 20 ms window counts as a detection if at least one valid packet with the shooter's ID was decoded in it.

## 7. Hit event (same schema in sim and real logs)
| field           | type  | unit                                        |
|-----------------|-------|---------------------------------------------|
| t               | float | s (episode time in sim, ROS time on hardware) |
| shooter_id      | int   |                                             |
| target_id       | int   |                                             |
| receiver_index  | int   | -1 if unknown                               |
| distance        | float | m                                           |
| off_axis_angle  | float | rad                                         |
| incidence_angle | float | rad                                         |

On hardware, geometry fields are filled from mocap poses at time t by WS3 tools.

## 8. Safety filter
- One implementation, written so it runs under both numpy and jax, used by the sim (WS1) and the flight node (WS5). Parameters from arena.yaml and drone.yaml.
- Applies, in order: velocity limits, geofence margin, minimum horizontal separation, no vertical stacking (downwash).
- Hardware adds a command timeout (hover then land), battery cutoff and a kill switch. These are not simulated.

## 9. Policy export
- Weights as .npz plus a JSON metadata file with: architecture, observation normalization statistics, contract version, training config hash.
- Reference numpy inference function in train/export.py, with a test that it matches JAX inference.

## 10. Config files and required keys
- arena.yaml (WS1): contract_version, bounds {x, y, z}, geofence_margin, min_separation, min_vertical_offset
- drone.yaml (WS5): contract_version, limits {v_xy, v_z, yaw_rate}, mass, crazyflow_drone_model, onboard_controller {position_rate, attitude_rate, vel_xy {kp, ki, kd, i_limit, output_limit}, vel_z {kp, ki, kd, i_limit, output_limit}, roll_pitch_limit, thrust_base_pwm, thrust_min_pwm, thrust_max_pwm, yaw_max_delta}. onboard_controller mirrors the flashed firmware's PID controller defaults: vel_xy in rad and m, vel_z and thrust_* in PWM (vel_z gains include the firmware thrustScale of 1000).
- hit_model.yaml (WS4 values, WS1 code): contract_version, cone_half_angle, max_range, dwell_time, hit_cooldown, receivers (axis unit vectors in body frame), p_detect {distance, off_axis, incidence, table}. Table interpolated trilinearly; 0 outside the table.
- randomization.yaml (WS1, latency from WS5): contract_version, ranges for mass, thrust, latency, observation noise, battery sag.
- game.yaml (WS1): contract_version, hp, max_episode_time, reward {hit_dealt, hit_taken, crash, time_penalty}

Schema changes follow the contract change process.

## Changelog
- 0.2 (2026-11-11): drone.yaml gains onboard_controller; section 3 names the firmware PID controller and the velocityWorld transport; PWM units exception in section 1.
- 0.1 (2026-11-08): initial.
