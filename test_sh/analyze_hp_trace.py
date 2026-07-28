#!/usr/bin/env python3
"""Analyze a multi-workload Heat Predictor Trace run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .hp_trace_analysis import analyze_run_root
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hp_trace_analysis import analyze_run_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze_run_root(args.run_root, args.output_dir)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"analyzed {result.analyzer.record_count} evaluated records from "
        f"{result.trace_file_count} OSD Trace files into {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
