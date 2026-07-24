#!/usr/bin/env python3
"""Create parity reports for Heat Predictor baseline replay output."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .hp_replay_analysis import analyze_replay_run, write_replay_outputs
except ImportError:
    from hp_replay_analysis import analyze_replay_run, write_replay_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    analysis = analyze_replay_run(args.run_root, args.replay_dir)
    write_replay_outputs(analysis, args.output_dir)
    print(
        f"Analyzed {len(analysis.summaries)} replay file(s); "
        f"gate={'PASS' if analysis.gate_passed else 'FAIL'}"
    )
    return 0 if analysis.gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
