# fix-path-coverage-gaps-followup

## Why

The original `fix-path-coverage-gaps` (archived `2026-08-12`) closed 4-path cudart coverage (AC-M1) and added D-PTX-7/D-PTX-8 debt registry, but GAP-4 (ABI baseline regression test) was marked as "future scope" rather than implemented. The existing `integration_libptxemu_abi_baseline` test has been failing in regression suite because:

1. **Addresses leak into baseline**: The test compares full `nm -D` output including load addresses which change on every build (Linker re-randomizes, debug symbols shift)
2. **No regeneration mechanism**: When ABI legitimately changes (new symbol added/removed/renamed), there's no documented regeneration path

Result: AC-M3 (output-correctness 4/4 → 4/4) and AC-M4 (`ctest -L path_1X` single-path regression) are met, but AC-N2 (LABELS contain e2e + ctest-filter stable) is undermined by an always-failing pre-existing test.

## What Changes

- **修复 `integration_libptxemu_abi_baseline`**: 剥离地址，只比较符号名 + 类型（`T`/`U`/`W` 等）
- **重新生成 `libptxemu_abi_baseline.txt`**: 不含地址的格式，仅 5 个 ptxemu_ 符号
- **添加重新生成文档**: `baselines/README.md` 说明如何用 `nm -D libptxemu_device.so | awk '{print $2, $3}' | grep ptxemu_ | sort -u` 重新生成
- **不修改生产代码**: 仅 `tests/integration/cpptlm/` 子树
- **不重命名任何 ABI 符号**: 保持 `ptxemu_image_load/execute/unload/kernel_name/module_version` 5 个 ABI 不变

## Capabilities

### Modified Capabilities

- `integration_libptxemu_abi_baseline` 测试契约: 从"全行字节级相同（含地址）"修改为"剥离地址后的符号列表相同"

## Impact

| 组件 | 影响类型 | 说明 |
|------|----------|------|
| `tests/integration/cpptlm/test_libptxemu_abi_baseline.cpp` | 修改 | 剥离地址 + 排序去重 |
| `tests/integration/cpptlm/baselines/libptxemu_abi_baseline.txt` | 重新生成 | 不含地址 |
| `tests/integration/cpptlm/baselines/README.md` | 新建 | 重新生成说明 |
| 生产代码 | 不动 | `libptxemu_device.so` ABI 不变 |

## In Scope

- 重写 `run_nm()` 使用 `awk '{print $2, $3}'` 剥离地址
- 重新生成 baseline 文件（5 个 ptxemu_ 符号）
- 添加 `baselines/README.md`
- 验证 ctest 100% pass

## Out of Scope

- 修改 `libptxemu_device.so` ABI（不动生产代码）
- 重命名或添加 ptxemu_ 符号（per ADR-0029 D3 5-symbol contract）
- 修改 baseline 文件位置（仍 `tests/integration/cpptlm/baselines/`）