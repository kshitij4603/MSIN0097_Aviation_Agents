from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

PIPELINE_STEPS = [
    "preprocess_aviation_data.py",
    "eda_delay_drivers.py",
    "task2_eda.py",
    "task2_basic_eda.py",
    "task2_advanced_stats.py",
    "task3_baseline_model.py",
    "task4_optimized_model.py",
    "task5_model_audit.py",
]


def run_step(script_name: str) -> None:
    script_path = SCRIPT_DIR / script_name
    print(f"[run_pipeline] Running {script_name}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Codex aviation pipeline in chronological order.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution order without running any scripts.",
    )
    args = parser.parse_args()

    if args.dry_run:
        for step in PIPELINE_STEPS:
            print(step)
        return

    for step in PIPELINE_STEPS:
        run_step(step)

    print("[run_pipeline] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
