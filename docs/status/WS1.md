# WS1: Sim Environment
Owner: Daksh
Last updated: 2026-11-11 by Daksh
Contract version used: 0.2

## Current state (one paragraph)
Task 1 (Crazyflow familiarization) is done. Crazyflow 0.4.0 runs 2 drones per world across N worlds with the contract's velocity + yaw-rate action, implemented as a port of the Crazyflie firmware PID velocity controller inserted into Crazyflow's step pipeline (attitude mode, first_principles dynamics, Crazyflie 2.1 Brushless model). Drones spawn in hover equilibrium. A demo flies scripted commands, renders in the MuJoCo viewer and benchmarks a fully jitted rollout. No arena, hit model, game logic, safety filter or observations yet.

## Done
- Crazyflow 0.4.0 API survey and probes (2026-11-11), findings under Key numbers and Known issues
- contracts v0.2: drone.yaml onboard_controller block, PID controller and velocityWorld notes (2026-11-11), docs/contracts.md, config/drone.yaml
- Firmware PID velocity controller as a pipeline stage (2026-11-11), sim/onboard.py, tests/test_onboard.py
- Sim factory and hover spawn (2026-11-11), sim/factory.py, tests/test_factory.py
- Velocity demo, viewer and throughput bench (2026-11-11), sim/demo_velocity.py

## In progress
- None

## Next up
- Task 2: arena bounds, geofence and termination from config/arena.yaml
- Task 3: hit model (pure JAX) with placeholder p_detect
- Task 4: game logic, reward, episode timer, 28-float observations per contract section 4
- Task 5: shared numpy/jax safety filter

## Blockers / needs from other workstreams
- Needs from WS5: fly with the PID controller (stabilizer.controller = 1); Mellinger ignores horizontal velocity setpoints (status: in contract v0.2)
- Needs from WS5: send velocityWorld setpoints, rotating body-yaw vx, vy to world with current yaw; yaw rate in deg/s on the wire (status: in contract v0.2)
- Needs from WS5: after takeoff, hover on zero velocityWorld setpoints >= 5 s before the policy takes over; firmware zeroes PID integrators on setpoint mode change (status: requested, decisions.md 2026-11-11)
- Needs from WS5: firmware release to be flashed; onboard_controller values come from firmware master 2026-11-11 (status: requested)
- Needs from WS5: vx, vz and yaw-rate step responses for G5, incl. real hover PWM vs thrust_base_pwm and horizontal drag (status: requested)
- Needs from WS5: velocity estimation method for observations, contract section 4 (status: requested)
- Needs from WS4: frozen payload mass for drone.yaml mass (status: requested)

## Outputs other workstreams can use
- sim.factory.make_sim(n_worlds), spawn_hovering(data, pos, yaw, mask): sim built from config/, contract rates
- sim.onboard.set_velocity_cmd(data, cmd): SI setpoint [vx_b, vy_b, vz, yaw_rate], shape (N, M, 4)
- sim.onboard.sync(data, mask, settled): controller re-init; settled=False mimics a firmware mode-change reset
- sim.demo_velocity.make_rollout(step_fn): jitted lax.scan rollout pattern (for WS2)

## Key numbers
- Throughput, Mac Apple Silicon CPU, jitted rollout, 2 drones, 50 Hz policy, 10 physics substeps: 2.4 to 3.1e5 world-steps/s (~5,000x real time), flat for N = 1 to 2048. GPU not measured yet.
- Velocity response (sim, firmware gains): vx 0.5 m/s step t90 = 0.46 s, 0.456 m/s after 3 s (~9% steady lag, tiny xy ki vs drag); vz step t90 = 0.24 s, no steady error; yaw rate 1.0 rad/s tracked at 1.001 through +-pi
- 1 m/s xy step: 15 mm altitude dip
- Hover PWM in the Crazyflow Brushless model: 34,880 vs firmware thrust_base 30,000; vz integral time constant ~1.5 s
- Brushless model: 43.4 g bare, thrust-to-weight 1.88 bare
- so_rpy yaw (rejected): cmd 2.0 rad settles at 1.838; 36 rad/s spin crossing +-pi

## Decisions made this period
- See docs/decisions.md entries dated 2026-11-11

## Known issues / gotchas
- import crazyflow before anything that imports scipy, else SCIPY_ARRAY_API error
- "Attaching a spec with an empty prefix" and "Failed to import warp" warnings are harmless; MUJOCO_LOG.TXT is gitignored
- Crazyflow spawns drones on the floor at (+-0.5, -0.5, 0); use spawn_hovering
- so_rpy dynamics are unusable for this game (yaw bias, wrap spin); onboard.install rejects them
- onboard.install rebuilds Crazyflow's default data and step/reset functions; get step functions from sim.build_step_fn() after installing
- The onboard controller uses true velocity, not an onboard estimate
- thrust_scale (per-drone motor strength) exists in the controller state but is not randomized yet (task 6)
- Sim inner attitude loop is Crazyflow's Mellinger; hardware runs firmware PID attitude. Revisit at G5 if step responses differ
- PWM to newton map is Crazyflow's linear one; the main expected sim-to-real thrust gap
