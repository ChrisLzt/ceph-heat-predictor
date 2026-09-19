# CloudLab cache study runbook

## Locations

- Server/MON/MGR/MDS/OSD0: `wzp@hp117.utah.cloudlab.us`.
- Workload client/OSD1: `wzp@hp118.utah.cloudlab.us`.
- OSD2: `wzp@hp081.utah.cloudlab.us`.
- Persistent CephFS data: `/mnt/ceph-lab/cephfs/cache-study-20260919/`.
- Study tools, prepared manifests and workload reports on hp118:
  `/mnt/ceph-lab/cache-study-20260919/`.
- New source/build on hp117: `/mnt/ceph-lab/integrated-20260919/`.
- Measured run: `fixed8g-ses-postprepare-001`.

The experiment uses one replica and existing CloudLab allocations. It is not
durable against a node loss or allocation expiration. OSD1/2 use fully allocated
SSD-backed loop files, not dedicated raw SSDs. Never reinitialize those files or
rerun first-time OSD initialization against the existing cluster.

## Read-only health check

```sh
ssh wzp@hp117.utah.cloudlab.us \
  'sudo docker exec ceph-lab-server ceph -s'
```

Expected at rest: three OSDs up/in, 137 PGs active+clean, with the explicitly
accepted `POOL_NO_REDUNDANCY` warning. Other warnings require investigation.

## Repeat after checking the cluster is idle

The controller does not prepare or delete data. It checks all READY markers and
file identity, samples all three OSDs, runs cases serially, switches policy after
180 seconds, drains HP labels, and restores LRU/HP-disabled when leaving.

From this experiment directory on the local workstation, use a new, unused run id:

```sh
python3 run_cloudlab_study.py \
  --run-id fixed8g-ses-repeat-002
python3 analyze_cloudlab.py cloudlab-runs/fixed8g-ses-repeat-002
```

This is a warm follow-on repetition, not a cold-start trial. A cold-start protocol
requires a separately documented cache/client-state reset and new measurements.
Do not run two controllers concurrently or modify the cache budget mid-run.

## Evidence and boundaries

`COMPLETE.json` means the controller finished its execution/inventory checks;
it is not a third-party certification. Inspect `analysis.json`, sampling gaps,
counter resets, HP accounting, `workload-skew-audit.json`, and all original
workload logs. The Vdbench skew report has an independently documented integer
denominator bug; use histogram integer counts for the corrected distribution.

The modified colleague Vdbench binary was not supplied. The replacement runtime
identity and fractional-sampling tests are in `fractional-runtime-identity.json`.
The provided SES source hashes match, but the isolated `current`-profile adapter
is a disclosed compatibility overlay, not standard SES certification.

MDS, extent-map, data-buffer, and object-context ratios are diagnostic metrics;
they must not be combined to inflate the accepted Onode lookup hit ratio.

## Delivery bundle

`cache-study-results-20260919.tar.gz` contains the report, separate CSV result
families, raw counter samples, timing/control evidence, study scripts, tests,
and a nested archive of original Vdbench/SES workload reports. `MANIFEST.json`
records each included file's size and SHA-256. Run `package_results.py` to verify
the downloaded remote archive and rebuild the package. `DELIVERY-QA.json`
outside the archive additionally records the completed package's hash.

The bundle does not contain the 658 GiB dataset, full Ceph source/builds,
container images, credentials, or the complete supplied SES distribution.
Those inputs remain at their disclosed remote or supplied attachment locations.
