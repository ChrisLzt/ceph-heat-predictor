# Agent Instructions

This repository contains a Ceph OSD object-level heat predictor prototype and
notes for its ICFS/IDFS port.

Before making changes, read the relevant CODEX document:

- `CODEX_CEPH.md`: Ceph object-layer heat predictor implementation, hook location, model features, and `object_hp_status` fields.
- `CODEX_ICFS.md`: ICFS/IDFS porting notes, OSS/MergePG hook points, ICFS-specific op coverage, and migration risks.
- `CEPH_OPERATIONS_MANUAL.md`: single-node Ceph deployment, OSD/MGR operation, and basic heat predictor checks.

Test workloads and reports are maintained outside this repository under
`/home/chris/ceph-test/new_workload/`.

Do not rely on old KernelDevice/BlockDevice heat-predictor assumptions unless the user explicitly asks to inspect historical code.

The current implementation is object-layer only. Keep Ceph-specific object op
adaptation in `src/osd/ObjectHeatPredictor.*`; keep the reusable algorithm in
`src/heatpredictor/`.
