# Decision log
Format: YYYY-MM-DD | WS<n> | decision | reason | alternatives considered

2026-11-08 | WS0 | conda/mamba for Python only, all packages via pip with exact pins | Crazyflow reports conda-installed MuJoCo failing (MUJOCO_PATH); conda doesn't lock pip deps | uv, pixi
2026-11-08 | WS0 | Python 3.12 | inside Crazyflow 0.4.0's supported range (3.11 to 3.13) | 3.11, 3.13
2026-11-08 | WS0 | environment.yml uses conda-forge + nodefaults | avoids Anaconda commercial channel and per-user channel config leaking in | default channels
2026-11-08 | WS0 | GPU via jax[cuda12] pip extra | bundles CUDA; GPU machine needs only an NVIDIA driver | conda cudatoolkit, cuda13
2026-11-08 | WS0 | Weights & Biases for experiment tracking | shared remote dashboards for the team | TensorBoard
2026-11-08 | WS0 | checkpoints (*.npz) not committed to git | large binaries; store on W&B or shared storage | git LFS
