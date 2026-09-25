# Fixed-workload Onode cache experiment

Status: experimental, default off, not an acceptance pass. The 128-node
deployment has not been tested. The three-node WRF pair completed and failed
the 96% target with either quota mode.

## Measured Result

| WRF metric | Uniform quotas | Rebalanced quotas |
| --- | ---: | ---: |
| LRU / HP off | 91.993617% | 91.724946% |
| S3FIFO / HP on | 89.700311% | 89.712795% |
| Minimum valid short window | 75.775452% | 74.196477% |
| Valid short windows above 96% | 10/41 | 11/41 |
| HP accuracy after label drain | 88.053712% | 88.859830% |

This single pair shows no meaningful improvement for this case, not a general
comparison of cache algorithms. Enabled-stage misses exactly matched net
resident growth in both runs (19,521 and 19,367), and sampled ghost queues
were empty. The busiest uniformly budgeted shard reached only about 63.1%
of its target. These observations suggest compulsory misses dominate here;
they are not a substitute for an exact eviction/unique-object trace.

All five datasets passed preflight, but only WRF was benchmarked in this pair.
Do not merge or recommend this option as a solution to the 96% acceptance goal.

## Frozen Inputs

- Integrated source baseline: `fbfd7114508d14b7e582263bd8a4fbc3883fa39f`.
- Previous real-hit accounting fix: `da10bdb168750e4428b0b98790c88b5d7a7f34f6`.
- Local parent commit: `3ebdf5936c4a48b7ab38cf8475463c1c1d788e93`.
- Workload: `ChrisLzt/ceph-test`, commit
  `303e43e2e1c98cb74ec156af75ce2546faaa72eb`, `SINGLE_workload`.
- Stock Vdbench JAR SHA256:
  `8d53b728baf4e3eb28b538765b81de606ce2b1dfca39c66822d02704b462304a`.
- All five existing persistent datasets passed layout and allocated-file checks.
  Only configuration paths were adapted; no model or data was rewritten.
- All 15 prepare/baseline/current configurations passed parser-only checks.
  The measured profile is `current`, not the alternate `baseline` profile.
- The 27 Heat Predictor source files are unchanged and SHA256-checked before
  compiling. This experiment does not consume predictions or change training.

This stock-Vdbench suite is not the earlier fractional-Vdbench/SES runtime.
Its results must not be presented as an exact reproduction of that runtime.

## Cache-Only Candidate

`bluestore_cache_s3fifo_rebalance_shards` defaults to false. When enabled and
all Onode shards are S3FIFO, underused shards donate unused quotas to busy
shards. A small admission reserve allows the working set to move. Mixed
policies and LRU retain equal quotas. Turning on the option does not increase
the total entry quota calculated by the existing metadata allocator and does
not take memory from the buffer cache.

`onode_cache status` additionally exposes the option and per-shard
`target_onodes`. Hits and misses keep the original real-lookup definition.
Pinned entries can exceed targets; entry quotas are not hard RSS limits.
The existing average-bytes-per-Onode calculation remains in use.

This is a capacity-utilization hypothesis, not a guarantee of 96% hits.
It cannot eliminate a compulsory miss on an object that was never resident.
It is concurrent with Heat Predictor, not prediction-guided eviction.

## Lab Protocol

These adapters are explicitly for the disposable CloudLab testbed. They are
not deployment scripts for a colleague's server or the 128-node system.

- Three OSDs, kernel CephFS client, same existing persistent files.
- Per OSD: 4 GiB `osd_memory_target`, cache autotuning on, metadata/KV fallback
  ratios 0.45/0.45. This is a memory target, not a fixed 4 GiB cache allocation.
- Lab pools have size 1. This is not a production durability configuration.
- Same candidate binary with option false (uniform) and true (borrowed).
- Restart existing OSD processes and remount the same client before each run.
  Do not format disks, recreate pools, regenerate data, or flush at t=180.
- For each full WRF schedule: native LRU + HP off for 180 seconds, then online
  S3FIFO + HP on for 420 nominal seconds. Record request and confirmation times.
- Keep the balancer off during the pair; restore its prior on state afterwards.
- Restore the previous fixed OSD binary after the pair, preserving all data.
- Seven WRF RDs sum to 600 seconds; report both nominal 600 seconds and the
  full RD schedule, because RD startup adds wall-clock overhead.

The before/after phases use different segments of a changing workload. Their
difference alone is not causal evidence of a policy benefit. Compare the
matching enabled stages in the uniform/borrowed pair, and repeat before
claiming significance.

## Measurement

Main metric is the query-weighted cluster Onode rate:

```
100 * sum_osd(delta(onode_hits)) /
      sum_osd(delta(onode_hits) + delta(onode_misses))
```

Never average OSD percentages, mix MDS/buffer counters into this denominator,
include preparation lookups, or manufacture low baseline hit rates.

`analyze_windows.py` checks nominal 10-second bins using actual sample
endpoints within 3 seconds. It reports the actual bounds, rejects transitions,
counter resets, missing OSDs and collection gaps, and does not interpolate.
These are sampled short windows, not exact synchronized 10-second counters.
Unavailable boundary windows are not passes. The stage average is reported
separately from the minimum short-window rate and the number above 96%.

HP confusion counts, accuracy, drops and queue accounting remain separate.
IOPS and cache memory diagnostics must accompany any performance claim.

## Verification

- Remote build: `ceph-osd`, Onode cache tests, BlueStore type tests.
- 18 Onode/cache-budget tests pass; 50 repeated runs pass.
- 27 BlueStore type tests pass (size stress and checksum benchmark excluded).
- 25 existing analysis/runtime-audit tests and 3 short-window tests pass locally.
- The old SES wrapper test needs its separate historical workload package and
  was not part of the new stock workload validation.

## Next Decision

Classify the remaining misses before adding more cache policy changes:

1. Capacity/eviction misses: consider shared quotas, admission and retaining
   useful Onodes while reclaiming expensive extent/blob state safely.
2. Compulsory misses: evaluate bounded, asynchronous metadata prefetch based
   only on information available at request time, not the future test trace.
3. Memory competition with HP: observe actual allocator budgets and queues;
   do not silently give the enabled phase a larger memory allowance.

For a prefetch prototype, count background work separately, retain the demand
lookup denominator, track useful/unused prefetches and pollution, and enforce
memory/IO/concurrency limits. Do not count a demand read waiting on a metadata
load as an existing resident hit. Predictive hints require an existing public
output or an explicitly approved integration hook; the frozen predictor's
OSD-facing API currently does not export per-object cache hints.

Before scaling, freeze the dataset sharing model, OSD count, replica/EC layout,
RAM budget, client mount type/concurrency, initial cache state and exact
acceptance window. 128 x 1.8 TB SSD is not enough information to infer a hit rate.
