# CloudLab Cache Study, 2026-09-19

This directory preserves the scripts used for the completed five-workload
CloudLab study. It is an experimental branch addition, not a Ceph production
change, a teuthology suite, or a general-purpose cluster installer.

## Pinned Inputs

- Ceph OSD/MGR: `fbfd7114508d14b7e582263bd8a4fbc3883fa39f`.
- Workload repository: <https://github.com/ChrisLzt/ceph-test>, commit
  `5406ff849346908e7029fa6a883ff0ed97b9540a`.
- Actual workload models/configuration: the supplied
  `cache-analysis-materials-20260919.tar.gz`, including
  `01-current-joint-test/configs/*/rendered`. The model hashes, read-only
  operations, distributions and full 658.17 GiB layout were retained; only
  data-root paths changed for CloudLab.
- SES: user-supplied 1.2.0, build `1.2.0.25061615`. All 40 pinned source files
  matched; see `ses-preflight.json`. SES source and its installer are not vendored.
- Vdbench base: `versity/scoutfs-vdbench` at
  `d904d81d565c5882d0aabf2434eef46851954767`. The colleague's modified jar was
  unavailable. The independently identified replacement is documented in
  `fractional-runtime-identity.json`, not claimed to be byte-identical.

## Read Before Running

**Do not run every script in this directory.** The files retain the tested lab's
host names, users, mount paths, container names and initial-state assumptions.
Read `REPORT.md`, `PROTOCOL.md` and `RUNBOOK.md` first. No local Docker is needed.

`prepare_build.py`, `build_integrated.sh`, `prepare_ssd_file.py`,
`export_osds.py`, `init_new_osd.py`, `upgrade_runtime.py` and
`configure_study.py` are one-time remote provisioning/configuration records.
They may allocate hundreds of GiB, create OSD identities, initialize an empty
block device, replace running binaries, or change cluster settings. They are
**not** repeat-test entrypoints. Never reinitialize existing OSDs or repurpose
devices without first validating ownership, mount layout and current contents.
`prepare_remaining.py` was a one-off companion to the already-running first
dataset preparer; do not launch it alongside a fresh serial preparer.

`prepare_workloads.py`, `run_ses_case.py`, `study_agent.py` and the SES tests
also require the pinned workload repository's `workload_common` Python package.
On the tested client it resides at
`/mnt/ceph-lab/cache-study-20260919/workload_common`. Frozen configuration,
external runtimes and prepared READY markers must already exist at the paths
used by the scripts. This Git directory alone is not a fresh-cluster bootstrap.

## Offline Checks

From this directory, Python 3 with the standard library is sufficient for:

```sh
python3 -m unittest test_analyze_cloudlab test_audit_vdbench test_export_results -q
```

These 20 tests exercise zero denominators, strict thresholds, query weighting,
switch boundaries, missing/reset counters, telemetry gaps, independent optional
diagnostics, histogram counts, and short-window exports. They do not start
containers, contact nodes, or generate storage I/O.

With the external pinned `workload_common` package on `PYTHONPATH`, run the
separate SES adapter tests using:

```sh
python3 -m unittest test_ses_cloudlab.CloudLabSESTest -q
```

Real SES execution additionally needs the source distribution and dependencies
recorded in `ses-preflight.json`; framework source hashes are checked before
loading it. Plotting requires matplotlib. The three Java source files are the
fractional-probability sampler patch and test, used remotely with Java 17 by
`build_vdbench_runtime.py` and `verify_runtime_methods.py`. The unused earlier
source-replacement experiment and all third-party jars are deliberately omitted.

## Repeat and Recompute

Use `RUNBOOK.md` only after checking the existing lab is idle. The controller
samples three OSDs, starts each case serially, requests LRU-to-S3FIFO/HP switching
at 180 seconds, records acknowledgements, drains HP labels, and restores baseline
settings. Dataset preparation is separate. Concurrent controllers are unsupported.

For offline recomputation, restore the separately delivered evidence package and
point the scripts to its completed run, not to `reference-results`:

```sh
python3 analyze_cloudlab.py /path/to/evidence/cloudlab-runs/fixed8g-ses-postprepare-001
python3 export_results.py /path/to/evidence/cloudlab-runs/fixed8g-ses-postprepare-001
python3 plot_cloudlab.py /path/to/evidence/cloudlab-runs/fixed8g-ses-postprepare-001
```

`reference-results/` contains compact, completed-run CSV/JSON summaries and the
plot. It does not replace raw samples. Raw counter streams, workload HTML logs,
full Vdbench disassembly, data files, source distributions, compiled binaries and credential material are
not committed. The external package hash is recorded in
`reference-results/DELIVERY-QA.json`; `package_results.py` requires that full
external evidence layout and verifies archive and per-file hashes.

## Interpretation

All five S3FIFO phase Onode lookup ratios exceeded 96% in this one disclosed
CloudLab run. LRU also exceeded 99%; this does not establish S3FIFO improvement.
The FUSE client path, fixed cache budget, warm preparation state, mixed daemon
versions, single replica, SSD-backed loop devices, telemetry gap and actual
workload-distribution deviations are documented in `REPORT.md`.

Onode lookups are metadata events, not application data-hit requests. MDS,
extent-map, buffer-byte and hot/cold metrics remain separate. No-query windows
are unavailable, not 100%. These are adapted research workloads, not an
unmodified CCF certification run or complete control-plane acceptance test.
