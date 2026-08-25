# fix-phase0-gate1-dgpu-bar-leak

> **2026-08-25 归档说明**: 此 change **不再需要实施**。Gate 1 leak 已被 [`commit 09786635`](https://github.com/chisuhua/PTX-EMU/commit/09786635) (4-phase refactor Phase 3) **物理消除**。详见 [`docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md`](../../../../../docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md)。

原 1-line 描述: Phase 0 byte-identical gate 1 failure: 10 `tlm::gpu::DGpuBar` symbols leaked into `libcudart.so` via `--whole-archive cpptlm_core`. Minimal fix: remove the `--whole-archive` flag (preserving ABI consumers), regenerate baseline with audit log.