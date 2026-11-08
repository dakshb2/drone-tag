# WS0: Repo & Infrastructure
Owner: Daksh
Last updated: 2026-11-08 by Daksh
Contract version used: none yet (v0.1 in progress)

## Current state (one paragraph)
Repo skeleton per PLAN.md section 4 exists. Conda env (Python 3.12) plus pip-pinned deps install on macOS Apple Silicon; JAX 0.12.0.1 runs on CPU and Crazyflow 0.4.0 imports. No tests, CI, contracts or tracking yet.

## Done
- Repo skeleton (2026-11-08), repo root
- environment.yml + pyproject.toml with pinned deps (2026-11-08), repo root
- README setup instructions (2026-11-08), README.md

## In progress
- docs/contracts.md v0.1: seed from PLAN.md section 6, add open proposals from the plan review

## Next up
- Infra tests in tests/ and pytest passing
- CI: ruff, mypy, pytest on push (GitHub Actions, macOS + Ubuntu)
- W&B team + dummy training job that logs remotely
- Run/checkpoint naming convention
- GPU compute options (lab, Illinois campus cluster, cloud) and how to launch a job
- Branch protection on main once CI exists

## Blockers / needs from other workstreams
- None

## Outputs other workstreams can use
- environment.yml, pyproject.toml: install with `mamba env create -f environment.yml && pip install -e ".[dev]"`

## Key numbers
- Pinned: crazyflow 0.4.0, jax 0.12.0.1, mujoco 4.0.0, wandb 0.31.0

## Decisions made this period
- See docs/decisions.md entries dated 2026-11-08

## Known issues / gotchas
- On some Macs mamba and conda use different env locations: use `mamba activate drone-tag`, not `conda activate`.
- "Failed to import warp" on `import crazyflow` is harmless (optional NVIDIA backend).
- For WS1: Crazyflow 0.4.0 has no velocity + yaw-rate control mode. Modes are state (16 values), attitude, body_rate, force_torque, rotor_vel. The contract's velocity action needs a wrapper that matches the Crazyflie firmware's velocity handling.
