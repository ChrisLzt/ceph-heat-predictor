#!/usr/bin/env python3
"""Convert a Heat Predictor binary trace session to CSV and metadata JSON."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Iterator


MAGIC = b"HPTRACE1"
SCHEMA_VERSION = 2
MAX_FEATURES = 8
HEADER = struct.Struct("<8sIIIIiIQQQQ64s64s")
RECORD = struct.Struct("<QQQQQ8d6dQQQQQBBbbI")
OUTCOMES = {
    0: "evaluated",
    1: "evaluation_capacity_drop",
    2: "prediction_error",
}


def decode_text(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def read_header(stream: BinaryIO) -> dict[str, object]:
    raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError("trace file is shorter than its fixed header")
    values = HEADER.unpack(raw)
    metadata = {
        "magic": values[0].decode("ascii", errors="replace"),
        "schema_version": values[1],
        "header_size": values[2],
        "record_size": values[3],
        "feature_count": values[4],
        "osd_id": values[5],
        "session_id": values[7],
        "start_wall_time_ns": values[8],
        "start_monotonic_time_ns": values[9],
        "config_hash": f"{values[10]:016x}",
        "git_commit": decode_text(values[11]),
        "phase": decode_text(values[12]),
    }
    if values[0] != MAGIC:
        raise ValueError(f"unexpected trace magic: {values[0]!r}")
    if values[1] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported trace schema {values[1]}, expected {SCHEMA_VERSION}"
        )
    if values[2] != HEADER.size:
        raise ValueError(
            f"header size {values[2]} does not match converter size {HEADER.size}"
        )
    if values[3] != RECORD.size:
        raise ValueError(
            f"record size {values[3]} does not match converter size {RECORD.size}"
        )
    if not 0 <= values[4] <= MAX_FEATURES:
        raise ValueError(f"invalid feature count: {values[4]}")
    return metadata


def record_columns(feature_count: int) -> list[str]:
    return [
        "io_sequence",
        "object_key_hash",
        "prediction_time_ns",
        "label_deadline_ns",
        "label_completion_time_ns",
        *[f"feature_{index}" for index in range(feature_count)],
        "heat_after_current_access",
        "heat_label_threshold_at_prediction",
        "predicted_hot_probability",
        "hot_predict_threshold",
        "label_heat",
        "label_heat_threshold",
        "tracked_access_count",
        "time_since_previous_access_ns",
        "long_window_access_count",
        "short_window_access_count",
        "future_window_access_count",
        "outcome",
        "cold_start_fallback",
        "evaluation_capacity_drop",
        "predicted_label",
        "actual_label",
    ]


def decode_record(raw: bytes, feature_count: int) -> list[object]:
    values = RECORD.unpack(raw)
    features = values[5:13]
    outcome = values[24]
    flags = values[25]
    return [
        *values[0:5],
        *features[:feature_count],
        *values[13:24],
        OUTCOMES.get(outcome, f"unknown_{outcome}"),
        bool(flags & 0x01),
        bool(flags & 0x02),
        values[26],
        values[27],
    ]


def iter_records(stream: BinaryIO, feature_count: int) -> Iterator[list[object]]:
    while True:
        raw = stream.read(RECORD.size)
        if not raw:
            return
        if len(raw) != RECORD.size:
            raise ValueError("trace ends with a partial record")
        yield decode_record(raw, feature_count)


def convert(input_path: Path, output_path: Path) -> tuple[dict[str, object], int]:
    with input_path.open("rb") as source:
        metadata = read_header(source)
        feature_count = int(metadata["feature_count"])
        count = 0
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(record_columns(feature_count))
            for record in iter_records(source, feature_count):
                writer.writerow(record)
                count += 1
    metadata["record_count"] = count
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata, count


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "trace.bin"
        output = Path(directory) / "trace.csv"
        header = HEADER.pack(
            MAGIC,
            SCHEMA_VERSION,
            HEADER.size,
            RECORD.size,
            2,
            3,
            0,
            9,
            100,
            200,
            0x1234,
            b"commit\0".ljust(64, b"\0"),
            b"phase\0".ljust(64, b"\0"),
        )
        record = RECORD.pack(
            1,
            2,
            3,
            4,
            5,
            *([1.0, 2.0] + [0.0] * 6),
            *([3.0] * 6),
            6,
            7,
            8,
            9,
            10,
            0,
            1,
            1,
            1,
            0,
        )
        source.write_bytes(header + record)
        metadata, count = convert(source, output)
        if count != 1 or metadata["osd_id"] != 3:
            raise AssertionError("trace converter self-test lost metadata or records")
        rows = list(csv.reader(output.open(encoding="utf-8")))
        if len(rows) != 2 or rows[0][5:7] != ["feature_0", "feature_1"]:
            raise AssertionError("trace converter self-test produced invalid CSV")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("PASS: hp trace converter")
        return 0
    if args.input is None:
        parser.error("input is required unless --self-test is used")
    output = args.output or args.input.with_suffix(".csv")
    try:
        metadata, count = convert(args.input, output)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"converted {count} records from osd.{metadata['osd_id']} "
        f"session {metadata['session_id']} to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
