#!/usr/bin/env python3
"""Check actual OSD adapter outputs without a running cluster."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

with tempfile.TemporaryDirectory(prefix="hp-osd-module-") as directory:
    result = subprocess.run(
        [os.environ["HP_OSD_MODULE_PROBE"], directory],
        check=False, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    rows = dict(line.split("\t", 1) for line in result.stdout.splitlines())
    data = {key: json.loads(value) for key, value in rows.items()}
    assert "object_hp_status" not in data["before_init"]
    assert data["after_init"]["object_hp_status"]["hp_io_count"] == 0
    assert data["initial"]["enabled"] is False
    assert data["initial"]["hp_io_count"] == 0
    assert data["filtered"]["enabled"] is True
    assert data["filtered"]["hp_io_count"] == 0
    assert data["observed"]["hp_io_count"] == 6
    assert data["observed"]["hp_pending_io_count"] == 6
    counters = data["observed_perf"]["object_hp_status"]
    for op in ("read", "sync_read", "sparse_read", "write", "writefull", "writesame"):
        assert counters[f"hp_op_{op}_count"] == 1
    assert data["other_instance"]["enabled"] is False
    assert data["other_instance"]["hp_io_count"] == 0
    assert data["reset"]["discarded_pending_io"] == 6
    assert data["reset"]["enabled"] is True
    assert data["reset"]["hp_heat_state_count"] == 0
    assert data["trace_start"]["ok"] is True
    assert data["trace_start"]["phase"] == "refactor-probe"
    assert data["trace_reset"]["discarded_pending_io"] == 1
    trace = data["trace_rotated"]["trace"]
    assert trace["enabled"] is True
    assert trace["session_id"] == data["trace_start"]["session_id"] + 1
    assert trace["path"] != data["trace_start"]["path"]
    assert data["trace_stop"]["enabled"] is False
    assert data["trace_stop"]["write_error_count"] == 0
    assert data["disabled"]["enabled"] is False
    assert data["disabled_observed"]["hp_io_count"] == 0
    assert data["disabled_observed"]["hp_predict_error_count"] == 0
    assert data["disabled_observed"]["hp_background_error_count"] == 0
    assert data["trace_at_shutdown"]["enabled"] is True
    assert "object_hp_status" not in data["after_shutdown"]
    assert "object_hp_status" not in data["after_destructor"]
    assert data["replacement"]["enabled"] is False
    assert data["replacement"]["hp_io_count"] == 0
    # This short probe has no expired labels: each drained file has its header.
    paths = [data["trace_start"]["path"], trace["path"], data["trace_at_shutdown"]["path"]]
    assert len(set(paths)) == 3
    for path in paths:
        assert Path(path).is_file() and Path(path).stat().st_size == 192
print("PASS: OSD commands, filtering, isolation, concurrent control, Trace and teardown")
