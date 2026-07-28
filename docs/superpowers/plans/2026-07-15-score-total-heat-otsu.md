# Score Total-Heat Otsu Implementation Plan

> Execute in the existing `main` worktree. Preserve unrelated changes in
> `codex_docs/CODEX_CEPH_TODO.md`.

## Goal

Make D2 the default score-normalized total-heat algorithm: current total heat
updates a per-object Otsu vote, features use current total heat, and the label
compares deadline total heat with the threshold evaluated at that deadline.

## Step 1: Lock the contract with failing probes

- Extend `test_sh/hp_algorithm_probe.cc` to require D2 by default.
- Add probes for 800 bins, width `0.01`, lower heat `10`, and upper heat near
  `10 * exp(8)`.
- Add score round-trip and time-decay probes.
- Add latest-vote replacement, moving-lower-bound, capacity eviction, reset,
  and deadline-total-heat label probes.
- Build and run the probe against current code; record the expected failure.

## Step 2: Implement the score histogram and D2 wiring

- Add `src/heatpredictor/hp_score_otsu_histogram.h` with a fixed 800-bin array,
  moving absolute-score origin, physical lower clamp, and Otsu statistics.
- Change D2 defaults and configuration constants in `hp_config.h` and
  `src/osd/CMakeLists.txt`.
- In `hp_evaluation_queue.h`, maintain one score vote per object on access,
  enforce the threshold-window capacity, update the threshold in score space,
  publish it as a stable heat value, and version published thresholds for
  prediction/deadline lookup.
- Keep D0/D1 completed-added-heat paths intact.
- Label D2 samples from deadline total heat; retain future-added heat only as a
  diagnostic.
- Ensure reset clears both Otsu implementations and all score-threshold state.

## Step 3: Local verification

- Build and run `hp_algorithm_probe` for default D2.
- Build and run the probe with D0 and D1 compile definitions where supported.
- Build the affected Ceph targets and inspect warnings/errors.

## Step 4: Install and live-cluster verification

- Run `sudo ninja -j64`, `sudo ninja install`, and `sudo ldconfig`.
- Restart Ceph services that load the changed code.
- Wait for all PGs to become `active+clean`.
- Verify `ceph osd hp reset`, MGR `ceph osd hp status`, and per-OSD perf output.

## Step 5: Five formal Vdbench workloads

- Use the five formal Vdbench workloads under `ceph-test/new_workload`.
- Reset Heat Predictor before each workload and verify the reset result.
- Run each workload exactly once.
- Capture MGR `hp status` every 30 seconds during each test.
- Save only final `hp_status.json` plus Vdbench output and timestamped logs.
- Use a Luna subagent only for passive process/log monitoring; the main agent
  validates raw output and final status.

## Step 6: Report and documentation

- Write a Chinese report under `ceph-test/new_workload/hp_runs/reports`.
- Include accuracy, balanced accuracy, precision, recall, predicted/actual hot
  percentages, threshold/Otsu state, Vdbench rate and MB/s, and elapsed time.
- Update `codex_docs/CODEX_CEPH.md` for the implemented D2 mechanism and remove
  completed items from TODO without disturbing unrelated TODO edits.
