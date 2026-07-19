#!/usr/bin/env python3
"""Project a schema-v1 four-feature Trace CSV to schema-v2 C4 binary input."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


MAGIC = b"HPTRACE1"
HEADER = struct.Struct("<8sIIIIiIQQQQ64s64s")
RECORD = struct.Struct("<QQQQQ8d6dQQQQQBBbbI")


def encode_text(value: object, size: int = 64) -> bytes:
    encoded = str(value).encode("utf-8")[: size - 1]
    return encoded.ljust(size, b"\0")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def metadata_path_for(source: Path) -> Path:
    return source.with_suffix(source.suffix + ".metadata.json")


def validate_metadata(metadata: dict[str, object]) -> None:
    expected = {
        "magic": "HPTRACE1",
        "schema_version": 1,
        "header_size": 192,
        "record_size": 208,
        "feature_count": 4,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"schema-v1 projection requires {field}={value!r}, "
                f"got {metadata.get(field)!r}"
            )


def project_row(row: dict[str, str]) -> bytes:
    if row.get("outcome") != "evaluated":
        raise ValueError("schema-v2 replay projection accepts evaluated rows only")
    features = [float(row[f"feature_{index}"]) for index in range(3)]
    features.extend([0.0] * 5)
    flags = 0
    if parse_bool(row["cold_start_fallback"]):
        flags |= 0x01
    if parse_bool(row["evaluation_capacity_drop"]):
        flags |= 0x02
    return RECORD.pack(
        int(row["io_sequence"]),
        int(row["object_key_hash"]),
        int(row["prediction_time_ns"]),
        int(row["label_deadline_ns"]),
        int(row["label_completion_time_ns"]),
        *features,
        float(row["heat_after_current_access"]),
        float(row["heat_label_threshold_at_prediction"]),
        float(row["predicted_hot_probability"]),
        float(row["hot_predict_threshold"]),
        float(row["label_heat"]),
        float(row["label_heat_threshold"]),
        int(row["tracked_access_count"]),
        int(row["time_since_previous_access_ns"]),
        int(row["long_window_access_count"]),
        int(row["short_window_access_count"]),
        int(row["future_window_access_count"]),
        0,
        flags,
        int(row["predicted_label"]),
        int(row["actual_label"]),
        0,
    )


def project_trace(source: Path, output: Path, config_hash: int) -> int:
    metadata = json.loads(metadata_path_for(source).read_text(encoding="utf-8"))
    validate_metadata(metadata)
    header = HEADER.pack(
        MAGIC,
        2,
        HEADER.size,
        RECORD.size,
        3,
        int(metadata["osd_id"]),
        0,
        int(metadata["session_id"]),
        int(metadata["start_wall_time_ns"]),
        int(metadata["start_monotonic_time_ns"]),
        config_hash,
        encode_text(metadata.get("git_commit", "")),
        encode_text(metadata.get("phase", "")),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(newline="", encoding="utf-8") as input_stream:
        reader = csv.DictReader(input_stream)
        with output.open("wb") as output_stream:
            output_stream.write(header)
            for row in reader:
                output_stream.write(project_row(row))
                count += 1
    expected_count = int(metadata["record_count"])
    if count != expected_count:
        output.unlink(missing_ok=True)
        raise ValueError(
            f"CSV record count {count} does not match metadata {expected_count}"
        )
    return count


def parse_config_hash(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not 0 <= parsed <= 0xFFFFFFFFFFFFFFFF:
        raise argparse.ArgumentTypeError("config hash must fit uint64")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config-hash", required=True, type=parse_config_hash)
    args = parser.parse_args()
    try:
        count = project_trace(args.source, args.output, args.config_hash)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"projected {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
