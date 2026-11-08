# WS0: Repo & Infrastructure
Owner: Daksh
Last updated: 2026-11-08 by Daksh
Contract version used: 0.1

## Current state (one paragraph)
Repo, environment, contracts, configs, tests, CI and experiment tracking all work. A fresh clone installs on macOS Apple Silicon (local) and Ubuntu (CI); ruff, mypy and 9 tests pass in CI on every push. A dummy jitted JAX job logs to W&B team drone-tag-uiuc. Not yet done: GPU compute and branch protection.

## Done
- Repo skeleton per PLAN.md section 4 (2026-11-08), repo root
- Conda env (Python only) + pip deps pinned in pyproject.toml, transitive deps locked in constraints.txt (2026-11-08)
- docs/contracts.md v0.1 (2026-11-08)
- Placeholder configs with contract v0.1 keys, incl. new game.yaml (2026-11-08), config/
- Run and checkpoint naming convention (2026-11-08), train/naming.py
- Dummy W&B job (2026-11-08), train/dummy.py, run with `python -m train.dummy`
- Infra tests (2026-11-08), tests/test_infra.py
- CI: ruff, mypy, pytest on ubuntu-latest (2026-11-08), .github/workflows/ci.yml
- W&B team drone-tag-uiuc, academic plan applied for (complimentary access meanwhile)

## In progress
- GPU compute: target Illinois Campus Cluster Grainger GPU nodes (free for engineering students). Next: submit user form, confirm eligibility.

## Next up
- Install + dummy job on the GPU machine; document how to launch a job there
- Branch protection on main (needs GitHub Student Pack; checking)
- Neelesh onboarding: GitHub + W&B invites sent

## Blockers / needs from other workstreams
- None

## Outputs other workstreams can use
- Setup: README.md
- contracts v0.1: docs/contracts.md
- train.naming: run_name(), checkpoint_path(), WANDB_ENTITY, WANDB_PROJECT
- config/*.yaml placeholders with contract keys; tests enforce keys and version

## Key numbers
- Pinned: crazyflow 0.4.0, jax 0.12.0.1, mujoco 4.0.0, wandb 0.31.0, Python 3.12
- CI time: about 2 to 3 minutes per push

## Decisions made this period
- See docs/decisions.md entries dated 2026-11-08

## Known issues / gotchas
- Use `mamba activate drone-tag`, not `conda activate` (different env locations on some Macs).
- Regenerate constraints.txt only with the command in README.md; plain `pip freeze` breaks CI.
- "Failed to import warp" on `import crazyflow` is harmless.
- WS1: Crazyflow 0.4.0 has no velocity control mode (modes: state with 16 values, attitude, body_rate, force_torque, rotor_vel). The contract's velocity action needs a wrapper matching the Crazyflie firmware's velocity setpoint behavior.
- WS4/WS5 (from plan review): do a paper mass budget in week 1; a Crazyflie 2.1 has roughly 15 g usable payload, so decide Brushless early. Bench-test whether the 940 nm emitter shows up as a ghost marker in mocap or disturbs Lighthouse sensors. Check reflections off walls, floor and net; firmware must ignore its own ID. Measure radio latency with both drones at 50 Hz plus logging, not one drone.
- WS6 (from plan review): the full hardware eval matrix is infeasible; pre-register a small subset with episode counts from sim variance.
