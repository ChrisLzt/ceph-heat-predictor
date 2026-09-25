# Onode Prefetch Pressure Recovery

Cache-side revision of `056694358003bd060c24c4703e236be219edb092`.
No changes to Heat Predictor, workload data, object formats or demand hit/miss
accounting. This is experimental, not a claim of >96% for all workloads/windows
or a validated 128-node deployment.

## Problem

The original worker only admitted metadata below 80% of the dynamic metadata
byte budget and below 90% of each shard's entry quota. A continuously occupied
cache could leave the worker active with queued PGs but no scans or new loads.
Normal trimming to the full entry quota did not ensure spare capacity for this
more conservative admission gate. Simply deleting unused speculative Onodes
did not necessarily release enough extent/blob/other metadata bytes.

Candidate-level transient rejection also advanced the PG cursor past that
candidate. Retrying the PG later did not revisit those skipped candidates.

## Changes

- Keep the 80% byte and 90% entry admission guards. Do not raise the OSD memory
  target, metadata allocation, shard quotas or buffer-cache allocation.
- Before an admission blocked by pressure, attempt bounded S3FIFO reclamation.
  Use the existing eviction/pin handling. Prefer an unused speculative entry
  when it is at the small queue's eviction end; otherwise follow S3FIFO order.
  This is not a global scan for unused entries.
- Recheck actual MetaCache bytes after each successful eviction; destruction
  may release attributes, extents and blobs, not just the Onode structure.
  Do not estimate freed bytes by multiplying entries by `sizeof(Onode)`.
- Limit each recovery pass to 128 queue operations, at most one eviction per
  four-operation shard visit, and at most one pass per 100 ms per worker. Try
  shard locks rather than waiting. These are work-count bounds, not hard wall
  time bounds on destructors or allocator operations.
- Back off recovery for one second if no entry was evicted, or if byte pressure
  required relief but the pass produced no net reduction in observed bytes.
  Validate queued collections before reclaiming; stale/deleted PG tasks cannot
  keep driving eviction.
- Round-robin shards for byte pressure; target the blocked shard for entry
  pressure. Aim for 75% bytes / approximately 85% entries where the bounded
  pass can reach them; admission still uses 80% / 90%. No persistent reservation
  is taken away from demand I/O.
- If quota, memory or collection-lock pressure temporarily rejects a candidate,
  retain its inclusive cursor and requeue. Each queue still stores only one PG
  cursor; no growing per-object retry list. Retries are rate-limited, not silently
  discarded after an arbitrary retry count. Other queued PGs get their turns.
- If decoding a candidate consumes the remaining headroom, account that live
  candidate during recovery. It is released before a deferred retry.
- LRU/cancellation stops recovery through the existing activation generation
  and policy checks. Existing resident Onodes are not overwritten by a decode.

The worker can reclaim clean, evictable demand metadata to admit speculation.
It may therefore reduce useful residency or hurt latency. Evaluate usefulness,
unused removals, eviction cost and demand IOPS/latency alongside hit rate.
Pinned/unreclaimable metadata or a budget too small for the candidate can still
prevent progress. Zero budget never triggers reclamation. The 4 GiB OSD target
is not a hard RSS limit; neither are these admission guards.

## Configuration and Control

Use the deployment's existing hosts, service manager, sockets and client mount.
There are no hard-coded OSD paths or hostnames in this procedure.

```ini
[osd]
bluestore_onode_prefetch = true
bluestore_onode_prefetch_reclaim = true
```

Both are startup-only. Install the matching OSD binary and perform the site's
rolling restart once before the experiment. `prefetch_reclaim` defaults true
but has no effect unless prefetch is configured and S3FIFO is active. For a
same-binary opportunistic-only control, set it false at startup; keep prefetch
itself true. Leave memory/autotuning, shard redistribution, HP and dataset
settings identical between groups.

No restart is required at the 180-second online switch:

```sh
# ASOK is the actual socket for each participating OSD.
ceph --admin-daemon "$ASOK" onode_cache policy lru
ceph --admin-daemon "$ASOK" object_hp disable

# At second 180 of the 600-second workload:
ceph --admin-daemon "$ASOK" onode_cache policy s3fifo
ceph --admin-daemon "$ASOK" object_hp enable

ceph --admin-daemon "$ASOK" onode_cache status
ceph --admin-daemon "$ASOK" dump_mempools
ceph --admin-daemon "$ASOK" object_hp status
```

Confirm all OSDs and shards, not only command acknowledgements. Do not reset HP
repeatedly within a measurement stage. Return to LRU after a case; do not clear
cache or restart between cases in the continuous-run test.

## Diagnostics

`onode_cache status` / `prefetch` retains all original fields and adds:

| Field | Meaning |
|---|---|
| `reclaim_enabled` | Startup recovery option, not proof of current activity |
| `meta_used_bytes` | Actual sum of the same eight MetaCache mempools |
| `shard_pressure_pauses` | Candidate checks rejected by shard quota |
| `resident_skips` | Candidate checks finding an already-resident Onode |
| `lock_retries` | Collection try-lock failures retained for retry |
| `candidate_retries` | Candidate cursor retained after transient rejection |
| `reclaim_passes` | Rate-limited recovery passes attempted |
| `reclaim_examined` | S3FIFO queue operations, including promotions/pin skips |
| `reclaim_evicted` | Onodes actually removed by the recovery path |
| `reclaim_no_progress` | Passes with no eviction, or no net byte relief when needed |
| `reclaim_lock_skips` | Recovery shard try-lock failures |

`pressure_pauses` counts failed byte checks, including checks after decode;
it is not elapsed time. `reclaim_no_progress` does not prove every object is
pinned: lock contention, policy cancellation and budgets moving can also
prevent eviction. Byte snapshots are not atomic with other RPCs/foreground
allocation. `scanned` includes retry attempts and is not a unique-object count.
No new field enters the demand hit-rate numerator or denominator.

## Verification and Handoff

Correctness tests are in `src/test/objectstore/test_bluestore_onode_cache.cc`:
bounded eviction, pin preservation, LRU gating, unchanged quotas/counters,
unused-prefetch accounting, a real filesystem-backed metadata-pressure recovery
test, and an opportunistic-only control that resumes when budget becomes
available. A zero-quota/recovery test verifies that the rejected candidate is
retried without a new generation. Existing tests exercise updates, deletes,
remount and policy races.

The pressure fixtures use a separate worker with a controlled byte budget and
pause shard resizing while exercising a specific gate. They verify mechanisms
and data integrity, not autotuning behavior or a representative workload result.

```sh
cmake --build "$BUILD_DIR" --parallel 8 --target \
  ceph-osd unittest_bluestore_onode_cache unittest_bluestore_types
ctest --test-dir "$BUILD_DIR" -R '^unittest_bluestore_onode_cache$' \
  --output-on-failure --no-tests=error
```

The filesystem-backed tests create/remove their own test store. Use the CTest
working directory or a dedicated empty test directory, never a live OSD path.

Before claiming a workload improvement:

1. First repeat the colleague's continuous Baleen -> GraphChi -> WRF sequence
   on two OSDs, 4 GiB each, same kernel CephFS mount, HP and persistent data.
   Restart only before the batch, not between cases. Compare recovery false
   versus true with the same candidate binary and initial-state procedure.
2. Capture all OSD status, mempools and HP snapshots throughout cases and gaps.
   Verify same cache instances, expected activation generations, pressure,
   recovery, loads and first demand uses. Then repeat all five workloads.
3. Use demand-count-weighted intervals. Report transition/invalid windows,
   minimum short-window hit rate, all-window pass counts, HP accuracy, client
   IOPS/latency and memory. Freeze the window definition across groups.
4. Keep ramp-up visible. Pressure recovery does not fix every first-access miss
   or guarantee an immediate >96% rate after the switch. Do not lower baseline
   by clearing cache, change fixed data/HP, or add background loads to hits.

No existing CloudLab workload result from the parent revision is evidence for
this revision's performance. Remote build/test evidence is recorded separately
from any future workload/acceptance report.

## Verified on 2026-09-21

The final candidate was built and tested in a remote CloudLab Linux builder.
No local Docker was started and no running cluster OSD was replaced/restarted.

| Gate | Result |
|---|---|
| Build `ceph-osd` and both affected test targets | Passed |
| Actual CTest Onode cache entrypoint | Passed, 14.48 seconds |
| 27 non-filesystem cache/budget/policy tests, 50 repetitions | 1350 executions passed |
| 3 filesystem-backed pressure tests, 5 repetitions | 15 executions passed |
| 27 enabled BlueStore type tests | Passed |
| Four changed cache/config/test source hashes | Match the built candidate |
| 27 frozen Heat Predictor source hashes | Unchanged |

The type-test filter retains the parent experiment's exclusions:
`sb_info_space_efficient_map_t.size` and `bluestore_blob_t.csum_bench`.
This is not an assertion that every Ceph test or sanitizer suite was run.

The pressure tests demonstrate same-generation recovery from occupied metadata,
no background increments to demand hits/misses, preserved reads/attributes,
updates/deletion/remount correctness, and cursor retry after shard capacity
becomes available. Controlled test budgets are not a workload benchmark.

Candidate `ceph-osd` SHA-256:
`bbb04165840ef235a6e5e8176f30c0449cc24e264686905f6a5b9a2e06385de1`.
The remote build tree retains its older base version string; use this binary
hash and the four source hashes, not that version string, to identify it.
The manifest is scoped to those files and the frozen HP files, not a claim
that the entire remote checkout is a clean Git tree of the eventual commit.

[Verification receipt and raw-log hashes](evidence/verification.json) and
[CTest summary](evidence/ctest.txt) are included. Full logs remain in the
experiment workspace; no workload trace or colleague's diagnostic archive was
uploaded as part of this patch.

The two-OSD continuous three-workload comparison, five-workload performance
rerun, short-window >96% target and 128-node acceptance remain unverified for
this revision. Deploy/benchmark it as a candidate, not an acceptance result.
