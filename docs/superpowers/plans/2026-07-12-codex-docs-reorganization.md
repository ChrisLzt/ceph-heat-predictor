# CODEX Documentation Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize active CODEX documents, split the Ceph TODO by responsibility, and define a four-level verification policy with one-run workload validation by default.

**Architecture:** `codex_docs/` owns active agent context. Its root contains stable context documents and a TODO index; focused TODO files under `codex_docs/todo/` own experiment protocol, prediction threshold, Otsu threshold, and final validation work. Historical plans under `docs/superpowers/plans/` remain immutable references.

**Tech Stack:** Markdown, Git, ripgrep.

## Global Constraints

- Preserve all existing TODO findings and completion states.
- Default formal workload validation to one run per workload.
- Repeat only anomalous, borderline, or publication-grade results.
- Do not modify historical plan references under `docs/superpowers/plans/`.
- Do not alter existing source-code changes in the dirty worktree.

---

### Task 1: Create the CODEX documentation hierarchy

**Files:**
- Create: `codex_docs/README.md`
- Move: `CODEX_CEPH.md` to `codex_docs/CODEX_CEPH.md`
- Move: `CODEX_ICFS.md` to `codex_docs/CODEX_ICFS.md`
- Move: `CODEX_AUTO.md` to `codex_docs/CODEX_AUTO.md`

- [x] Move the active context documents without changing their technical content.
- [x] Add an index explaining each document's ownership boundary.

### Task 2: Split the Ceph TODO by responsibility

**Files:**
- Replace: `CODEX_CEPH_TODO.md` with `codex_docs/CODEX_CEPH_TODO.md`
- Create: `codex_docs/todo/EXPERIMENT_PROTOCOL.md`
- Create: `codex_docs/todo/PREDICTION_THRESHOLD.md`
- Create: `codex_docs/todo/OTSU_THRESHOLD.md`
- Create: `codex_docs/todo/FINAL_VALIDATION.md`

- [x] Keep global constraints and known findings in the TODO index.
- [x] Move Tasks 1 and 6 to experiment protocol.
- [x] Move Tasks 2 and 3 to prediction threshold.
- [x] Move Tasks 4 and 5 to Otsu threshold.
- [x] Move Task 7 to final validation.
- [x] Replace the old execution order with links and dependency order.

### Task 3: Define verification levels and run-count policy

**Files:**
- Modify: `AGENTS.md`
- Modify: `codex_docs/CODEX_CEPH_TODO.md`
- Modify: `codex_docs/todo/EXPERIMENT_PROTOCOL.md`
- Modify: `codex_docs/todo/FINAL_VALIDATION.md`

- [x] Define L0 documentation, L1 local algorithm, L2 Ceph integration, and L3 workload validation.
- [x] Make one workload run the default at L3.
- [x] Require targeted repeats only for differences below 0.5 percentage points, anomalous results, historical conflicts, or publication-grade statistics.

### Task 4: Repair and verify references

**Files:**
- Modify: `AGENTS.md`
- Modify: active Markdown files under `codex_docs/`

- [x] Update active references to use `codex_docs/` paths.
- [x] Run `git diff --check` and verify no active Markdown file references a removed root-level CODEX path.
- [x] Verify the root contains no `CODEX*.md` files and every TODO index link resolves.

### Task 5: Centralize the operations manual and remove index duplication

**Files:**
- Move: `CEPH_OPERATIONS_MANUAL.md` to `codex_docs/CEPH_OPERATIONS_MANUAL.md`
- Modify: `AGENTS.md`
- Modify: `codex_docs/README.md`

- [x] Move the operations manual into `codex_docs/` and repair active references.
- [x] Keep mandatory agent behavior in `AGENTS.md` and make it point to one document index.
- [x] Keep the document catalog only in `codex_docs/README.md`.
