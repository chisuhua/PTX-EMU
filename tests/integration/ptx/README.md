# PTX Integration Tests

> **测试类型**: 类型二 — 指令序列集成测试（通过 `execute_warp_instruction` 驱动）
> **父目录规范**: [../../../AGENTS.md § 测试分类规范](../../../AGENTS.md)

## 测试覆盖

本目录包含 24 个测试文件，覆盖 PTX 指令的集成序列验证。

## 历史决策：PTX 单元测试迁移（Oracle A2, 2026-06）

2026-06 Oracle A2 审查决定：以下 7 个 `unit_ptx_*` 测试使用 `<<<1,1>>>` 在真实 GPU 上执行，验证 NVIDIA PTX 语义而非 PTX-EMU 模拟器行为。它们被移至 `tests/reference/ptx_builtin/` 作为历史"语义参考"。

等价覆盖已迁移至本目录：

| 原 unit 测试 | 等价 integration 测试 |
|---|---|
| unit_ptx_integer | test_integer_arith.cpp |
| unit_ptx_float | test_float_arith.cpp |
| unit_ptx_extended | test_extended_prec.cpp |
| unit_ptx_bitwise | test_bitwise_shift.cpp |
| unit_ptx_cvt | test_cvt_arith.cpp |
| unit_ptx_ld_st | test_ld_st.cpp |
| unit_ptx_cvta | test_cvta.cpp |

对应的 7 个 `add_catch_test` 注释块已于 2026-07-18 从 `tests/unit/CMakeLists.txt` 中清理。