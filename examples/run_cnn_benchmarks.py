"""Runs the CNN benchmark scripts as a single suite.

Command:
    python examples/run_cnn_benchmarks.py --quick
    python examples/run_cnn_benchmarks.py --full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _run(script: str, *extra: str) -> int:
    cmd = [sys.executable, str(EXAMPLES / script), *extra]
    print(f"\n>>> {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="1 epoch smoke tests")
    p.add_argument("--full", action="store_true", help="Full benchmark epochs")
    args = p.parse_args()

    epochs = "15" if args.full else "2"
    trials = "2" if args.full else "1"

    scripts = [
        ("cifar10_resnet_benchmark.py", ["--epochs", epochs, "--trials", trials]),
        ("cifar10_lt_benchmark.py", ["--mode", "transfer", "--epochs", "3" if not args.full else "10"]),
        ("imagenet_lt_benchmark.py", ["--backend", "cifar100", "--epochs", epochs, "--trials", trials]),
        ("mlperf_resnet_integration.py", ["--epochs", "3" if not args.full else "10"]),
    ]

    failed = 0
    for script, extra in scripts:
        rc = _run(script, *extra)
        if rc != 0:
            failed += 1
            print(f"FAILED: {script} (exit {rc})")

    if failed:
        print(f"\n{failed} benchmark(s) failed.")
        sys.exit(1)
    print("\nAll CNN benchmarks completed successfully.")


if __name__ == "__main__":
    main()
