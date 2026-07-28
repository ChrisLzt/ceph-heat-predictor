#!/usr/bin/env python3
"""Generate comparison tables for Heat Predictor feature ablation replay."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .hp_ablation_analysis import analyze_ablation_run, write_ablation_outputs
except ImportError:
    from hp_ablation_analysis import analyze_ablation_run, write_ablation_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--baseline-profile", default="A0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    analysis = analyze_ablation_run(
        args.run_root,
        args.replay_dir,
        args.profiles,
        baseline_profile=args.baseline_profile,
    )
    write_ablation_outputs(analysis, args.output_dir)
    accepted = [
        name for name, decision in analysis.profile_decisions.items()
        if name != analysis.baseline_profile and decision.accepted
    ]
    print(
        f"Analyzed {len(analysis.profiles)} profiles; "
        f"accepted={','.join(sorted(accepted)) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
