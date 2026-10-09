"""Run a deterministic, dependency-light smoke dataset through all stages."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

from run_stage import STAGES, run_stage


def configured_seed() -> int:
    """Read the integer seed from the shared YAML configuration."""
    config_path = Path("configs/base.yaml")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("seed:"):
            return int(line.split(":", maxsplit=1)[1].strip())
    raise ValueError(f"Missing seed in {config_path}")


def main() -> None:
    """Generate 200 rows and execute every scaffold stage."""
    rng = random.Random(configured_seed())
    labels = ("Urgency", "Scarcity", "Social Proof", "Normal")
    rows = [
        {"text": f"smoke example {index}", "label": rng.choice(labels)}
        for index in range(200)
    ]
    assert len(rows) == 200
    for stage in STAGES:
        run_stage(stage)
    for module in (
        "models.cnn",
        "models.bilstm",
        "models.small_transformer",
        "models.finetune",
    ):
        command = [sys.executable, "-m", module, "--smoke"]
        if module == "models.finetune":
            command.extend(["--model", "all"])
        subprocess.run(command, check=True)
    print("[DarkPatternGuard] smoke: processed 200 deterministic rows")


if __name__ == "__main__":
    main()
