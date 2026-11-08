"""WS0 smoke job: a jitted toy regression that logs to W&B.

Proves JAX runs on this machine and metrics reach the team dashboard.

    python -m train.dummy                       # logs to W&B (needs `wandb login`)
    python -m train.dummy --wandb-mode offline  # no network
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import wandb
import yaml

from train.naming import WANDB_ENTITY, WANDB_PROJECT, run_name

Params = dict[str, jax.Array]
StepFn = Callable[[Params, jax.Array], tuple[Params, jax.Array]]
WandbMode = Literal["online", "offline", "disabled"]


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def make_step(lr: float, batch: int, dim: int = 8) -> StepFn:
    true_w = jnp.arange(1.0, dim + 1.0)

    def loss_fn(params: Params, x: jax.Array) -> jax.Array:
        return jnp.mean((x @ params["w"] + params["b"] - x @ true_w) ** 2)

    @jax.jit
    def step(params: Params, key: jax.Array) -> tuple[Params, jax.Array]:
        x = jax.random.normal(key, (batch, dim))
        loss, grads = jax.value_and_grad(loss_fn)(params, x)
        return jax.tree.map(lambda p, g: p - lr * g, params, grads), loss

    return step


def run(cfg: dict[str, Any], wandb_mode: WandbMode = "online") -> float:
    wb = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name("dummy", "smoke", int(cfg["seed"])),
        group="smoke",
        job_type="dummy",
        config={**cfg, "jax_backend": jax.default_backend()},
        mode=wandb_mode,
    )
    step = make_step(float(cfg["lr"]), int(cfg["batch"]))
    params: Params = {"w": jnp.zeros(8), "b": jnp.zeros(())}
    key = jax.random.key(int(cfg["seed"]))
    loss = jnp.array(jnp.inf)
    for i in range(int(cfg["steps"])):
        key, sub = jax.random.split(key)
        params, loss = step(params, sub)
        if i % int(cfg["log_every"]) == 0:
            wb.log({"loss": float(loss)}, step=i)
    wb.summary["final_loss"] = float(loss)
    wb.finish()
    return float(loss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train/dummy.yaml"))
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"final loss: {run(load_config(args.config), args.wandb_mode):.2e}")


if __name__ == "__main__":
    main()
