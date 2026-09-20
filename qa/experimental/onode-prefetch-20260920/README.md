# Bounded Onode Prefetch

Experimental cache-side metadata prefetch. Default off. This is not a claim
that the 128-node acceptance target has been met. Predictor source and fixed
`SINGLE_workload` data are unchanged.

The pressure-recovery revision is documented in
[Onode prefetch pressure recovery](../onode-prefetch-pressure-20260921/README.md).
The observed benchmark results below describe the original implementation,
not a benchmark of that revision.

## Behavior

- Start with `bluestore_onode_prefetch=true`. This is a startup option.
- Native LRU keeps prefetch inactive. The existing `onode_cache policy s3fifo`
  command enables it after all local shards switch; `policy lru` cancels it.
- Only PGs touched by real Onode demand queries are scheduled. Each activation
  makes one cursor pass over each touched PG, not a scan of future workload
  files, labels, or test traces. New objects created after the pass may be missed.
- One worker per OSD examines at most 32 candidates per batch. The default rate
  is 1024 candidates/second and the queue holds at most 256 PG cursors plus one
  in-flight cursor. It uses try-locks for foreground enqueue and collection access.
- Read Onode metadata from the existing KV database; do not read object data or
  fault external extent shards. Reject records over 1 MiB before decoding.
- Admit only below 90% of the shard entry quota and below 80% of the existing
  tracked metadata budget. The original implementation only used spare space;
  the pressure-recovery revision can reclaim evictable S3FIFO entries with
  bounded work when `bluestore_onode_prefetch_reclaim=true` (default true).
  These are admission guards, not hard process RSS limits. Reading one oversized
  DB value and concurrent demand allocations can temporarily exceed them.
- Existing/dirty resident Onodes are never replaced. Collection locks, shard
  locks and activation generations protect deletion, splits, policy changes
  and shutdown. No persistent object or data format changes.

Admission never overwrites an existing object with a speculative decode.
Pressure reclamation may evict clean, unpinned demand entries through the cache's
normal eviction path; it does not guarantee that prefetch improves every workload.

Startup tunables: `bluestore_onode_prefetch_rate` (1..16384),
`bluestore_onode_prefetch_max_queued` (1..4096), and
`bluestore_onode_prefetch_max_record` (1 KiB..16 MiB).

This first implementation is active-PG metadata read-ahead, not prediction-guided
prefetch. It runs concurrently with the unchanged Heat Predictor. It can scan
cold metadata in an active PG; useful/unused counts and latency must be evaluated
before deploying it widely. Re-enabling starts a new pass and is not an excuse
to omit the associated work from a report.

## Accounting

The original `onode_hits` and `onode_misses` remain demand-only. Prefetch does
not call the demand lookup method, touch frequency, or reset these counters.
A later real demand query finding a prefetched resident Onode is a real hit.
Demand queries that miss before a racing background admission remain misses.

`onode_cache status` adds `prefetch`:

- `configured`, `active`, `generation`, `queued_pgs`, `rate`, `meta_budget_bytes`;
- `scanned`, `db_reads`, `encoded_bytes`, `errors`, `oversized`, `queue_full`,
  `pressure_pauses`;
- `loaded`, `used` (first subsequent demand hit only), `unused_removed`.

Cache shards expose the three residency counters separately. A PG split can
move an entry between shards, so use cluster/OSD sums for usefulness accounting.
Loaded minus used minus unused removed includes still-resident speculative
entries; it is not an accuracy measure. DB reads and encoded bytes include
race/oversize rejections. KV iterator I/O is additional work and not included in
the explicit metadata-get byte counter.

## Control-Plane Integration

Install the matching experimental OSD binary first. Set the following options
in the OSD's startup configuration, then restart it using the site's existing
service manager and rolling-restart procedure:

```ini
[osd]
bluestore_onode_prefetch = true
bluestore_onode_prefetch_rate = 1024
bluestore_onode_prefetch_max_queued = 256
bluestore_onode_prefetch_max_record = 1048576
```

These options are startup-only. Do not claim that changing the config database
alone enables prefetch in an already-running daemon. Keep the site's existing
memory target, autotuner, client mount, predictor configuration and dataset.
The experiments also keep `bluestore_cache_s3fifo_rebalance_shards=false`.

For each participating OSD, set `ASOK` to its actual admin-socket path, accessible
in the same host/container namespace as the `ceph` client. The existing commands
are unchanged:

```sh
# Before starting the 600-second workload: native cache, predictor disabled.
ceph --admin-daemon "$ASOK" onode_cache policy lru
ceph --admin-daemon "$ASOK" object_hp disable

# At 180 seconds: no restart, cache drop or counter reset.
ceph --admin-daemon "$ASOK" onode_cache policy s3fifo
ceph --admin-daemon "$ASOK" object_hp enable

# Read back every OSD, not just the request acknowledgement.
ceph --admin-daemon "$ASOK" onode_cache status
ceph --admin-daemon "$ASOK" object_hp status
```

Confirm `effective_policy=s3fifo` on every shard, `prefetch.configured=true`,
`prefetch.active=true`, and predictor `enabled=true`. Record request and final
confirmation timestamps; OSDs do not switch atomically across the cluster.
An empty prefetch queue after completing a pass is normal. Do not repeatedly
re-enable the predictor during sampling: its enable command resets its state.

Compute demand query hit rate from counter differences over the same valid
interval: `sum(delta(onode_hits)) / sum(delta(onode_hits + onode_misses))`.
Never add prefetch loads to the numerator. Reject counter resets and sampling
gaps, display transition windows separately, and show prefetched loads, first
demand uses, encoded bytes and memory next to the rate.

Return to LRU to stop queued/in-flight admissions. Existing resident metadata
is retained; this is not a cold-cache reset. Permanently disabling this
experimental worker requires setting the startup option false and restarting.

## Verification Protocol

1. Build and run cache tests plus a real filesystem-backed BlueStore test in
   the remote builder. Check LRU gating, activation cancellation, counter
   invariance, quota rejection, duplicate admission, concurrent policy changes,
   unused eviction, metadata reads, updates, deletes and shutdown/remount.
2. Compare prefetch off/on with the same candidate binary, client, workload,
   initial-state procedure and OSD memory target. Keep shard redistribution off.
3. Run full WRF schedules on CloudLab first: native LRU/HP off for 180 seconds,
   then S3FIFO/HP on for 420 nominal seconds. Do not drop caches at the switch.
4. Report demand-count-weighted hit rate, minimum valid short-window rate,
   invalid sampling gaps, HP confusion counts, IOPS, memory and prefetch work.
   Keep transition and ramp-up visible. A stage average does not prove every
   short window exceeds 96%.
5. Preserve the existing datasets and restore the pre-experiment runtime.

The three-node lab is not the colleague's server or the formal 128-node system.
Repeated trials and representative 128-node topology/memory/concurrency must
still be validated before any formal acceptance claim. Single-pair results for
all five workloads are recorded below.

## Observed WRF Result

On 2026-09-20, the same candidate binary was tested with prefetch off/on on
three CloudLab OSDs, kernel CephFS, a 4 GiB per-OSD memory target and autotuning.
Both trials used the same restart/remount procedure, paused background scrub
and balancer, unchanged predictor source, and existing persistent 112 GiB WRF
data. This is one paired observation, not a statistical performance study.

| Measurement | Prefetch off | Prefetch on |
|---|---:|---:|
| LRU baseline demand query hit rate | 91.3269% | 91.5830% |
| S3FIFO/HP-on demand query hit rate | 89.6823% | 98.0711% |
| Minimum valid approximately 10-second window | 75.6927% | 85.2075% |
| Windows strictly above 96% / valid windows | 9/40 | 36/41 |
| Hot/cold classification accuracy | 89.0738% | 89.7068% |
| OSD read IOPS during valid S3FIFO intervals | 360.29 | 345.28 |

The prefetch-on run stayed above 96% for all 36 consecutive valid windows from
nominal second 240 to 600, minimum 97.35%. This started about 58 seconds after
all OSDs confirmed the switch. The ramp-up period is not removed from the stage
average. Neither trial proves all 42 target windows exceeded 96%: transition
windows are invalid, and the off trial's final endpoint fell outside second 600.

The on trial admitted 173229 prefetched Onodes, 16054 subsequently used at least
once by demand, with 84348022 encoded metadata-get bytes. It raised metadata
residency rather than inventing hits; speculative work is excluded from demand
counts. Its observed OSD read IOPS was 4.16% lower and later Vdbench RD response
times were higher. Do not describe this as a proven end-to-end speedup.

Candidate SHA-256:
`bba9de7f58f3249dae894cbb60a2e56111934f805a626e0a41400915f861e605`.
Workload commit: `303e43e2e1c98cb74ec156af75ce2546faaa72eb`.
The measured binary was built before this patch was committed. Its version
string still names the integrated base, not this branch. Identify that artifact
by its SHA-256 and the recorded source hashes, not by its version string.
Full raw samples and a Chinese report remain in the experiment workspace at
`outputs/onode-prefetch-20260920/`. Compact evidence is included below.
No 128-node result is claimed.

## Additional Four Workloads

Eight more complete runs on 2026-09-20 used the same candidate binary, frozen
SINGLE workload commit, predictor, kernel client, 4 GiB memory target and
autotuning. Every run restarted the OSDs and remounted the client beforehand;
there was no restart or cache drop at second 180. No persistent dataset was
regenerated. Each workload was measured once with prefetch off, then once on.

| Workload | Off hit rate | On hit rate | On hot/cold accuracy | On minimum valid 10s window |
|---|---:|---:|---:|---:|
| Baleen | 95.9602% | 99.6337% | 96.9633% | 96.0251% |
| GraphChi | 91.5783% | 96.7145% | 91.1527% | 88.0448% |
| AI training | 98.8340% | 99.8196% | 94.1142% | 97.4812% |
| AI inference | 99.8211% | 99.9754% | 93.2005% | 99.7164% |

All four on-stage demand-count-weighted averages strictly exceed 96%, but
GraphChi passes only 28 of 40 valid approximately 10-second windows. The other
three pass all their valid windows (40/40, 41/41, 40/40 respectively); boundary
and transition windows are not counted as passes. No run proves all 42 target
windows exceed 96%. AI inference's LRU baseline in the on trial was already
98.2997%, so this does not establish a universal low-before/high-after demo.

Observed on/off OSD read-IOPS differences were +0.61%, +0.50%, -1.84%, +0.24%
respectively. These are single-pair observations, not significant speedups.
Prefetch admitted 159605-176942 Onodes per run; first-demand-use proportions
were 3.02%-5.77%, with 73.47-81.70 MiB of explicit encoded metadata-get bytes
through the restore-to-LRU acknowledgement. This cost scope includes drain
and post-run checks, whereas hit-rate statistics use the nominal600 interval.

All eight runs had unchanged persistent-file inventories, no OSD restart mixed
into measurement, and no HP drop/error or final accounting mismatch. Candidate
binary hashes were verified on all OSDs for every deployment. The old runtime,
LRU/HP-off state and background-maintenance settings were restored afterwards.

Full evidence, separate cache/heat/cost/client CSVs and the report are retained
in the experiment workspace at
`outputs/onode-prefetch-20260920/prefetch-other-20260920/`.
The complete remote raw-output archive is 18560736 bytes, SHA-256
`abf2937dd369b7c93df1974f5302615b788d1bf236b86b0d5a239f9b3bd937c5`.
The raw archive and sampled JSONL are external evidence, not embedded in this
repository summary. No cache implementation or frozen predictor/workload was
changed for these additional tests.

## Verification Evidence

- [Binary and source identity](evidence/binary-identity.json): original build-time
  record, including 27 frozen predictor source hashes. Its historical
  "uncommitted" note describes the measured build, not the publication status.
- [Onode tests](evidence/onode.xml): 25 remote tests passed, including a real
  filesystem-backed BlueStore prefetch/read/update/delete/remount test.
- [BlueStore type tests](evidence/types.xml): 27 selected tests passed; size
  stress and the checksum benchmark were excluded.
- [CTest](evidence/ctest.log): cache suite passed in its isolated working
  directory. Publication does not represent a new build or benchmark run.
- [WRF paired summary](evidence/wrf-summary.json) and
  [additional-run audit](evidence/other-validation.json).
- Separate [cache](evidence/cache-hit-rates.csv),
  [hot/cold](evidence/hot-cold-accuracy.csv),
  [prefetch cost](evidence/prefetch-cost.csv), and
  [client performance](evidence/client-performance.csv) results for the other
  four workloads. These are compact exports, not the complete raw archive.

`summarize_remaining.py <evidence-root>` regenerates the additional workload
summaries from a collected evidence directory containing `plan.json`, completed
per-run samples, analyses, short-window results, and restore acknowledgements.
It does not connect to a cluster or generate test data. The input is the
collected experiment directory, not just the remote raw-output archive.

This branch also preserves the earlier, default-off shard-quota experiment
and its [failed WRF comparison](../onode-pressure-aware-20260920/README.md).
It was present in the tested binary but disabled throughout the prefetch
comparisons. Do not enable it on the assumption that these results validate it.

All hostnames and runtime adapters under these experimental directories belong
to the historical CloudLab lab. Use the generic control-plane commands above
with the actual OSD sockets and existing site service manager on other servers.
