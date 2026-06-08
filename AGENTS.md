# Agent Instructions

This repository contains a Ceph OSD object-level heat predictor prototype.

Before making changes, read `CODEX_CONTEXT.md` and use it as the source of truth for:

- current hook location
- model parameters
- heat predictor features
- test scripts
- expected `object_hp_status` output

Do not rely on old KernelDevice/BlockDevice heat-predictor assumptions unless the user explicitly asks to inspect historical code.
