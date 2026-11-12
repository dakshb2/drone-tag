# Decision log
Format: YYYY-MM-DD | WS<n> | decision | reason | alternatives considered

2026-11-08 | WS0 | conda/mamba for Python only, all packages via pip with exact pins | Crazyflow reports conda-installed MuJoCo failing (MUJOCO_PATH); conda doesn't lock pip deps | uv, pixi
2026-11-08 | WS0 | Python 3.12 | inside Crazyflow 0.4.0's supported range (3.11 to 3.13) | 3.11, 3.13
2026-11-08 | WS0 | environment.yml uses conda-forge + nodefaults | avoids Anaconda commercial channel and per-user channel config leaking in | default channels
2026-11-08 | WS0 | GPU via jax[cuda12] pip extra | bundles CUDA; GPU machine needs only an NVIDIA driver | conda cudatoolkit, cuda13
2026-11-08 | WS0 | Weights & Biases for experiment tracking | shared remote dashboards for the team | TensorBoard
2026-11-08 | WS0 | checkpoints (*.npz) not committed to git | large binaries; store on W&B or shared storage | git LFS
2026-11-08 | WS0 | constraints.txt from pip freeze locks transitive deps | pyproject pins only direct deps; transitive drift breaks reproducibility | conda-lock, uv lock
2026-11-08 | WS0 | contracts v0.1: no opponent history in observations | variant A must have no memory for a fair A vs B comparison | keep history for all, add separate A0 variant
2026-11-08 | WS0 | contracts v0.1: vx, vy actions in body-yaw frame; vz and yaw_rate in world | matches Crazyflie hover/velocity setpoint semantics; easier to learn with ego-centric obs | world-frame actions
2026-11-08 | WS0 | contracts v0.1: hit checked at 50 Hz, per-step detection, n consecutive detections for dwell | firmware can implement the same rule exactly | single p_detect draw after dwell
2026-11-08 | WS0 | contracts v0.1: p_detect(distance, off_axis, incidence); emitter axis uses full attitude | WS4 measures all three; forward flight pitches the beam | 2-argument p_detect, yaw-only cone
2026-11-08 | WS0 | contracts v0.1: action is a firmware velocity setpoint; sim reproduces identified response | Crazyflow 0.4.0 has no velocity mode; sim must match real controller | ideal velocity tracking in sim
2026-11-08 | WS0 | contracts v0.1: prediction target in body-yaw frame frozen at t; added game.yaml | targets must not depend on future own yaw; game rules need a config | world-frame targets, rules in hit_model.yaml
2026-11-08 | WS0 | placeholder configs carry exactly the contract v0.1 keys; tests enforce keys and version | catches schema drift between workstreams early | no configs until owners fill them
2026-11-08 | WS0 | run name <variant>-<exp>-s<seed>-<YYYYMMDD-HHMM>; W&B entity drone-tag-uiuc, project drone-tag | one searchable convention for runs and checkpoints | free-form names
2026-11-08 | WS0 | CI on GitHub Actions, ubuntu-latest only, setup-python 3.12 + constraints.txt | Mac covered by local runs; macOS minutes cost ~10x; conda only supplies Python | macOS runners, conda in CI
2026-11-08 | WS0 | ruff isort pins wandb as third-party | local wandb/ run folder made ruff misclassify it | rename W&B dir via WANDB_DIR
2026-11-08 | WS0 | generate constraints.txt with pip list --format=freeze, excluding pip/setuptools/wheel | pip freeze writes conda-installed packages as build-server file paths, which broke CI | pip freeze
2026-11-11 | WS1 | sim drone model cf21B_500 (Crazyflie 2.1 Brushless) | stock CF2.1 has ~15 g spare payload; with the IR deck, thrust-to-weight drops to ~1.1-1.2, too little for dodging; Brushless ~1.5 | cf2x_L250 (standard CF2.1)
2026-11-11 | WS1 | velocity action via ported firmware PID velocity loop (100 Hz) + yaw-rate integration (500 Hz) as a Crazyflow pipeline stage, attitude mode, first_principles dynamics | mirrors the hardware's onboard controller; so_rpy has ~8% yaw bias and spins at the +-pi wrap (measured) | so_rpy + yaw patch, state mode with moving position target, ideal velocity tracking
2026-11-11 | WS1 | episodes start airborne after a 1 s zero-velocity warm-up excluded from the episode | matches hardware handoff after takeoff with the integrator already settled; takeoff/landing are WS5's job | spawn on floor and learn takeoff, spawn airborne with zero integrator
2026-11-11 | WS1 | contracts v0.2: drone.yaml onboard_controller block mirroring the firmware PID controller; hardware must use PID controller and velocityWorld transport | velocity loop needs firmware gains without hardcoding; firmware Mellinger ignores horizontal velocity setpoints | gains hardcoded in sim code, separate WS1-owned config
2026-11-11 | WS1 | onboard_controller thrust values kept in firmware PWM units, exception to SI | PWM is what the firmware computes; PWM-to-newton conversion is the uncertain sim-to-real step and belongs in sim code where thrust randomization acts | convert to newtons in config
2026-11-11 | WS1 | config tests read the contract version from docs/contracts.md | one place to bump the version | hardcoded version string in tests
2026-11-11 | WS1 | episode start: drones spawned hovering (hover rotor RPM, vz integral pre-loaded to its settled hover value) plus 0.5 s settle; supersedes the 1 s warm-up line | with firmware gains the vz integral settles with a ~1.5 s time constant, so a 1 s warm-up still sags; pre-loading is exact and costs no sim time | 5 s warm-up per reset, zero integral
2026-11-11 | WS1 | hardware handoff requirement for WS5: after takeoff, hover on zero velocityWorld setpoints for >= 5 s before the policy takes over | the firmware PID controller resets its integrators on every setpoint mode change, so takeoff to velocity mode starts at zero integral; the sim assumes a settled start | hand over immediately and train with sync(settled=False)
2026-11-12 | WS1 | arena bounds limit the drone center; frame radius and net clearance are built into arena.yaml values | no extra config key; same box drives sim termination and the hardware geofence | drone_radius key
2026-11-12 | WS1 | geofence checked once per 50 Hz policy step; non-finite state counts as a violation | same rate as hit and game logic; overshoot between checks is at most v_max*dt | check every physics substep
2026-11-12 | WS1 | sim/arena.py takes xp (numpy or jax.numpy) and imports no jax at module level | the safety filter and the WS5 flight node share it without pulling in JAX or Crazyflow | separate numpy and jax copies
2026-11-12 | WS1 | point-symmetric spawns (drone 1 = drone 0 mirrored through center in xy, same z), independent random yaw | no positional first-mover advantage in a symmetric game | independent uniform spawns with rejection
2026-11-12 | WS1 | arena.yaml z lo raised from 0.0 to 0.3 m | keeps drones out of ground effect and terminates before floor contact | z lo 0.0 with floor-contact crash detection
2026-11-12 | WS1 | crash termination (drone-drone collision, tilt) deferred to task 4 | needs keys not in the contract; batch with game.yaml changes into one contract bump | add keys now in task 2
