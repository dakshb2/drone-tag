"""Run and checkpoint naming. Every training script uses these helpers.

Run name:        <variant>-<exp>-s<seed>-<YYYYMMDD-HHMM>   e.g. C-selfplay_v1-s2-20261108-1430
W&B:             entity drone-tag-uiuc, project drone-tag, group = <exp>, job_type = <variant>
Checkpoint path: checkpoints/<run name>/step_<9-digit step>.npz  (+ .json metadata)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

WANDB_ENTITY = "drone-tag-uiuc"
WANDB_PROJECT = "drone-tag"
VARIANTS = ("A", "B", "C", "D", "BR", "dummy")  # BR = best-response agent (exploitability)
_EXP_RE = re.compile(r"^[a-z0-9_]+$")


def run_name(variant: str, exp: str, seed: int, when: datetime | None = None) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if not _EXP_RE.match(exp):
        raise ValueError(f"exp must match [a-z0-9_]+, got {exp!r}")
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M")
    return f"{variant}-{exp}-s{seed}-{stamp}"


def checkpoint_path(root: Path, name: str, step: int) -> Path:
    return root / name / f"step_{step:09d}.npz"
