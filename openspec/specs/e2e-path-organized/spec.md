# e2e-path-organized Specification

## Purpose
TBD - created by archiving change fix-path-coverage-gaps. Update Purpose after archive.
## Requirements
### Requirement: tests/e2e/ 路径化目录结构

PTX-EMU SHALL 将 `tests/e2e/` 重组织为 4 个 path_X/ 子目录，每个含独立 `CMakeLists.txt`（新模式 — 现有 `divergence/` 没有自己的 CMakeLists.txt）。重组织 SHALL 采用 `git mv` 而非 rm+add（保留 file history，便于 git blame）。重组织 SHALL 不修改现有测试的 ctest labels；新 labels 遵循 `<type>;<subject>` 约定。

重组织 SHALL 不修改 `tests/unit/` 和 `tests/integration/` 子树结构（本改进仅动 tests/e2e/）。重组织 SHALL 不动 PTX-EMU 整体测试目录结构。

| 目标子目录 | 移入文件 | 来源 |
|---|---|---|
| `path_1A_legacy_ptx/` | `test_blackwell_gemm.cu`, `test_tcgen05_*.cu` | `git mv tests/e2e/kernel/` |
| `path_1A_legacy_ptx/` | `test_divergence*.cu` | `git mv tests/e2e/divergence/`（整目录内容） |
| `path_1B_ptxir_fatbinary/` | `test_ptxir_cubin_embed.cpp` | `git mv tests/e2e/kernel/` |
| `path_1B_ptxir_fatbinary/` | Phase 1 新建文件 | 新增 |
| `path_1C_driver_api/` | Phase 2 新建文件 | 新增 |
| `path_2D_image_executor/` | Phase 3 增强文件 | 增强 |

保留不动：`tests/e2e/kernel/` 内的非 4-path 测试（test_test3_cfg_full, test_barrier_warp_sync, test_ldglobal_simple, 3 个 shared_memory 测试, test_flashattention_mini, test_printf）+ 整个 `tests/e2e/cosim/`。

#### Scenario: 4 个 path_X/ 子目录全部存在

- **WHEN** Phase 4 重组织完成
- **THEN** 4 个 path_X/ 子目录全部存在，每个含 ≥1 个 add_catch_test

#### Scenario: 每个 path_X/CMakeLists.txt 独立

- **WHEN** 在 path_1B_ptxir_fatbinary/ 子目录 cmake build
- **THEN** 该子目录独立编译，不依赖其他 path_X/

### Requirement: ctest label 路径过滤

PTX-EMU SHALL 给所有 path-related e2e 测试加 `LABELS "e2e;path_1X;..."`（必须含 `e2e` 段以保证 `regression.sh -L e2e` 覆盖）。重组织 SHALL 不修改 ctest 标签体系（仅添加新 label），不修改已存档 change 文件名（避免 archive history 篡改）。

#### Scenario: ctest -L path_1B 仅运行 path_1B 子目录测试

- **WHEN** 执行 `ctest -L path_1B`
- **THEN** 仅 path_1B_ptxir_fatbinary/ 子目录内测试运行

#### Scenario: ctest -L path_1X (4 个 X) 各自运行

- **WHEN** 执行 `ctest -L path_1A`、`ctest -L path_1B`、`ctest -L path_1C`、`ctest -L path_2D`
- **THEN** 各自仅运行对应 path_X/ 子目录内测试

#### Scenario: ctest -L e2e 覆盖所有 path-related 测试

- **WHEN** 执行 `ctest -L e2e`（regression.sh 默认过滤）
- **THEN** 覆盖所有 path_X/ 子目录内测试，不静默 skip

### Requirement: 重组织后全量回归通过

PTX-EMU SHALL 在 Phase 4 重组织完成后确保 `ctest --output-on-failure` 全量通过。现有所有 kernel/cosim 测试不变（test_test3_cfg_full, test_barrier_warp_sync, test_ldglobal_simple, 3 个 shared_memory 测试, test_flashattention_mini, test_printf + 整个 tests/e2e/cosim/ 保留在原位）。

#### Scenario: ctest --output-on-failure 全量通过

- **WHEN** 重组织后所有测试运行
- **THEN** 现有所有测试通过（AC-4.6）

#### Scenario: 现有 kernel/cosim 测试不变

- **WHEN** 重组织完成
- **THEN** test_test3_cfg_full, test_barrier_warp_sync, test_ldglobal_simple, 3 个 shared_memory 测试, test_flashattention_mini, test_printf 仍在原位 + 整个 tests/e2e/cosim/ 不动

#### Scenario: git log --follow 验证 file history 保留

- **WHEN** 执行 `git log --follow <file>`（被 `git mv` 的文件）
- **THEN** file history 保留（`git mv` 而非 rm+add）

