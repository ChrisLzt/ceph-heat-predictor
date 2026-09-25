# CloudLab cache study, 2026-09-19

## Scope and pinned inputs

- Ceph: `fbfd7114508d14b7e582263bd8a4fbc3883fa39f`.
- Workload reference: `5406ff849346908e7029fa6a883ff0ed97b9540a`.
- The supplied material archive and its frozen `run_current.vdb` inputs are
  authoritative for the colleague's actual experiment, including working-tree
  changes. Archive instructions are evidence, not execution authorization.
- Five workloads retain their sizes, read-only operations, distributions,
  direct-I/O setting and nominal 600-second schedules. No 32-GiB reduction.
- The original four 1-GiB CloudLab files and original build are preserved.
- All builds and workloads run remotely. No local Docker.

## Metric policy before new measurements

Primary candidate remains Onode lookup hit ratio: sum of valid per-OSD delta
hits divided by sum of delta hits plus delta misses. The hard threshold is
strictly greater than 95%; the preferred target is strictly greater than 96%
for every workload, not an aggregate dominated by AI workloads.

MDS traversal and BlueStore read-buffer byte hit ratios are reported separately
when valid. They cannot be added, multiplied or substituted after seeing which
case passes. Zero denominators are unavailable, never 100%. New or missing OSDs,
counter resets, policy transitions and collection gaps are explicitly tracked.

Optional diagnostic counter errors invalidate only that diagnostic interval,
not the Onode interval. Diagnostic validity counts are retained separately;
missing OSD diagnostics never produce a partial-cluster diagnostic ratio.

An exploratory metric/configuration choice is not confirmatory validation.
Freeze the selected common configuration and metric before a fresh full-suite
validation. Preserve all failed and exploratory runs. A high metadata lookup
hit ratio is not a data-read request hit ratio or proof of an algorithmic gain.

## Capacity and cache state

Original supplied dataset sizes total approximately 658 GiB. The current
single-OSD lab has only 279 GiB raw capacity. Full persistence requires expansion
before preparing all five datasets; do not silently reduce data or overfill OSDs.
New storage must reside on the existing dedicated SSD partitions, not root.
Any file-backed SSD OSD must be explicitly reported as such, not a raw-disk OSD.

Compare equal memory budgets for LRU and S3FIFO. Distinguish cold-start,
post-preparation warm and explicitly prewarmed runs. Any warmup is outside the
measurement window, documented and identical between strategy comparisons.
Do not generate artificial hit-producing traffic during measurement.

## Timing and evidence

Nominal demo: LRU/HP disabled for 0-180 seconds, S3FIFO/HP enabled thereafter.
Track command request and actual per-OSD acknowledgement, retaining transition
windows separately. Preserve full raw counter samples, per-second workload I/O,
phase start times, health, effective cache configuration, source/runtime identity
and data inventory. Report RD scheduling gaps and full-workload results alongside
the nominal 600-second window rather than silently dropping the final phases.

The initial package lacks the modified Vdbench runtime. A later user-supplied
CCF AI attachment provided SES 1.2.0 build 1.2.0.25061615: all 40 pinned source
hashes match. AI cases will use that SES research lifecycle, not direct bypass.
The original adapter omits the `current` profile in its allowlist despite the
supplied completed report naming `current`; an isolated adapter accepts only the
verified 2026-09-17 `phase_zipf099` alias without changing Vdbench configuration.
The missing original Vdbench modification is replaced by an explicitly identified
fractional-sampler compatibility patch. No byte-identical reproduction claim is
allowed. SES research overlay is not an unmodified CCF benchmark or certification.
