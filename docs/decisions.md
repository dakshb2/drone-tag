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
