"""Run deterministic Stage 1 pipeline commands."""

import argparse
from pathlib import Path

STAGES = ("data", "baselines", "neural", "llm", "ensemble", "report", "app")


def run_stage(stage: str) -> None:
    """Run a scaffold stage without creating model artifacts."""
    config_path = Path("configs/base.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Required configuration is missing: {config_path}")
    print(f"[DarkPatternGuard] {stage}: configuration verified")


def main() -> None:
    """Parse the stage name and execute it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
