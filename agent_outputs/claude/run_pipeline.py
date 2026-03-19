"""run_pipeline.py
Claude (Apple Silicon) — Task 6: Cross-Platform Pipeline Runner

Executes all Claude pipeline scripts in strict chronological order.
Uses sys.executable to guarantee the active virtual environment's Python
is used on any OS (Mac, Linux, Windows) without hardcoding a path.

Usage:
    python agent_outputs/claude/run_pipeline.py
    python agent_outputs/claude/run_pipeline.py --dry-run   # print order only
    python agent_outputs/claude/run_pipeline.py --step task3_claude_baseline.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# Ordered pipeline — each entry is relative to SCRIPT_DIR
PIPELINE_STEPS: list[str] = [
    "task2_claude_master_eda.py",
    "task3_claude_baseline.py",
    "task4_claude_optimized.py",
    "task5_claude_audit.py",
]


def run_step(script_name: str) -> None:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    print(f"\n{'='*60}")
    print(f"[run_pipeline] Running: {script_name}")
    print(f"{'='*60}")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),  # repo root so relative data/ paths resolve
        check=True,          # raises CalledProcessError on non-zero exit
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Claude aviation pipeline in chronological order."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution order without running any scripts.",
    )
    parser.add_argument(
        "--step",
        metavar="SCRIPT",
        help="Run a single named step instead of the full pipeline (e.g. task3_claude_baseline.py).",
    )
    args = parser.parse_args()

    steps = [args.step] if args.step else PIPELINE_STEPS

    if args.dry_run:
        print("Dry-run execution order:")
        for i, s in enumerate(steps, 1):
            status = "EXISTS" if (SCRIPT_DIR / s).exists() else "MISSING"
            print(f"  {i}. {s}  [{status}]")
        return

    for step in steps:
        run_step(step)

    print(f"\n{'='*60}")
    print("[run_pipeline] All steps completed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
