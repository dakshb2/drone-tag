# drone-tag

Two Crazyflies play IR laser tag. Policies are trained with self-play RL in Crazyflow and transferred to real drones. Full plan: [docs/PLAN.md](docs/PLAN.md).

## Setup (macOS or Linux)

Requires conda or mamba (Miniforge recommended: `brew install miniforge`).

```bash
git clone https://github.com/dakshb2/drone-tag.git
cd drone-tag
mamba env create -f environment.yml
mamba activate drone-tag
pip install -e ".[dev]"
```

GPU machine (Linux + NVIDIA): also run `pip install -e ".[gpu]"`.

Check: `python -c "import jax, crazyflow; print(jax.__version__, jax.default_backend())"`
