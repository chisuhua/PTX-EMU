# D-PTX Debt Registry ERRATA (2026-08-12)

**Source change:** `fix-path-coverage-gaps` (2026-08-12)
**Status:** Active

This ERRATA extends the D-PTX-N debt numbering defined in
[ADR-0021 §10](../adr/ADR-0021-cpptlm-d1-full-integration.md) (which defines D-PTX-1 through D-PTX-6).
Future D-PTX-N debts MUST be registered here first to avoid numbering conflicts.

---

## D-PTX-7: PTXIR fat-binary 端到端未验证

**Description:** PTXIR-Embedded CUBIN 路径（cudart SimModule，Path 1B）的 e2e 测试仅验证
格式兼容性（NVIDIA cuobjdump 容忍尾部 PTXIR），未验证 PTX-EMU 真的能从 `/proc/self/exe` 加载
并 dispatch PTXIR 到 `g_ptx_interpreter`。

**Source:** [ADR-0024 Risk 1](../adr/ADR-0024-ptxir-embedded-cubin.md) + archive change
[implement-ptxir-cubin-embed-extension](../../openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md)
silent descoping.

**Closure:** `fix-path-coverage-gaps` Phase 1 — `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`.

---

## D-PTX-8: Driver API 真实成功 kernel 执行未验证

**Description:** CUDA Driver API 路径（cuModule* 系列，Path 1C）的 e2e 测试仅验证
load/get_function/unload 调用成功，未验证 `cuLaunchKernel` 真的 dispatch 到 PTX-EMU 并
产出正确 output buffer。

**Source:** Driver API coverage gap identified during `fix-path-coverage-gaps` design analysis
(2026-08-12).

**Closure:** `fix-path-coverage-gaps` Phase 2 — `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp`
(uses `cuModuleLoadData` with real PTXIR blob, NOT the `cuModuleLoad` stub at `cudart_sim.cpp:510`).

---

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2026-08-12 | fix-path-coverage-gaps | Initial registration of D-PTX-7 + D-PTX-8 |
