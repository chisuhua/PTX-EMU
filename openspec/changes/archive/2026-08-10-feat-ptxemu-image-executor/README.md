# 2026-08-10-feat-ptxemu-image-executor (Archived)

> **⚠️ Archive metadata only** — 原始 change 归档时缺 `design.md`，仅保留 `proposal.md` + `tasks.md`。本 README 由 `docs-index-fix-pre-existing-orphans` 补齐。

## Purpose

PTX-EMU Image Executor (per ADR-0029)：通过 `libptxemu_device.so` + `cpptlm_module.h` 提供 5 个 `extern "C"` ABI 函数 (`ptxemu_image_load` / `ptxemu_image_execute` / `ptxemu_image_unload` / `ptxemu_image_kernel_name` / `ptxemu_image_module_version`)，允许外部 caller (UsrLinuxEmu HAL、TaskRunner shim、其他 future consumers) 以 opaque handle 方式加载 PTXIR bytes 并执行 kernel，无需 ANTLR 解析路径。替代 `__cudaRegisterFatBinary` 单 LD_PRELOAD front door 限制，支持嵌入式 binary 部署和非 NVIDIA 硬件真机部署。

## Implementation

- **Proposal**: `proposal.md` (Why/What/Capabilities/Impact 4 段，描述 PTX-EMU in-memory image executor 设计)
- **Tasks**: `tasks.md` (TDD 三阶段 + ship gate)
- **Implementation commit**: `3501ae64` — `Merge feat/ptxemu-image-executor: PTX-EMU Image Executor ship (v0.1.0)` (verify: `git show 3501ae64 --stat`)
- **Artifact commit**: `4c6305f6` — `docs(openspec): feat-ptxemu-image-executor proposal (artifacts FIRST)`
- **Archive commit**: `fe06c88f` — `chore(openspec): archive feat-ptxemu-image-executor`

## Related

- **ADR**: ADR-0029 (`docs/adr/`) — PTX-EMU Image Executor
- **Cross-repo**: UsrLinuxEmu ADR-076 — HAL backend integration (PTX-EMU 端抽象接口)
- **5 ABI entry points**: `ptxemu_image_load` / `ptxemu_image_execute` / `ptxemu_image_unload` / `ptxemu_image_kernel_name` / `ptxemu_image_module_version`
- **HSK-6 chain**: `libptxemu_device.so` 与 CppTLM 桥接解耦 (commit `25e36f60`)

---

**Status**: ✅ RESOLVED (by `3501ae64`, 2026-08-10)
**Added by**: `docs-index-fix-pre-existing-orphans` (2026-08-23)