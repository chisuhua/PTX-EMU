# e2e-path-organized Specification

## MODIFIED Requirements

### Requirement: tests/e2e/ 路径化目录结构

PTX-EMU SHALL keep `tests/e2e/` organized into four `path_X/` subdirectories, each with its own `CMakeLists.txt`. Existing test files SHALL be moved using `git mv` so that file history is preserved. The re-organization SHALL NOT change the existing `e2e` ctest label and SHALL keep ctest names with the `e2e_` prefix.

The re-organization SHALL apply only to `tests/e2e/`. The `tests/unit/` and `tests/integration/` subtrees SHALL NOT be reorganized. The PTX-EMU overall test directory layout SHALL remain otherwise unchanged.

#### Scenario: 4 个 path_X/ 子目录全部存在

- **WHEN** the image-executor fix is complete
- **THEN** 4 个 path_X/ 子目录全部存在，每个含 ≥1 个 add_catch_test

#### Scenario: 每个 path_X/CMakeLists.txt 独立

- **WHEN** 在 path_1B_ptxir_fatbinary/ 子目录 cmake build
- **THEN** 该子目录独立编译，不依赖其他 path_X/

### Requirement: ctest label 路径过滤

PTX-EMU SHALL keep all path-related E2E tests labeled with `LABELS "e2e;path_1X;..."`, including a new `e2e;path_2D;cuda_samples` label for the third-party CUDA Samples harness. The label system SHALL NOT replace the existing labels and SHALL NOT rename archived test files.

#### Scenario: ctest -L path_1B 仅运行 path_1B 子目录测试

- **WHEN** 执行 `ctest -L path_1B`
- **THEN** 仅 path_1B_ptxir_fatbinary/ 子目录内测试运行

#### Scenario: ctest -L path_1X (4 个 X) 各自运行

- **WHEN** 执行 `ctest -L path_1A`、`ctest -L path_1B`、`ctest -L path_1C`、`ctest -L path_2D`
- **THEN** 各自仅运行对应 path_X/ 子目录内测试

#### Scenario: ctest -L e2e 覆盖所有 path-related 测试

- **WHEN** 执行 `ctest -L e2e`
- **THEN** 覆盖所有 path_X/ 子目录内测试，包括新增的 CUDA Samples path_2D harness，不静默 skip

### Requirement: 重组织后全量回归通过

PTX-EMU SHALL keep the existing kernel/cosim tests in place and SHALL ensure `ctest --output-on-failure` passes after the image-executor fix and the new CUDA Samples harness are added.

#### Scenario: ctest --output-on-failure 全量通过

- **WHEN** the new harness and the path_2D fixes are merged
- **THEN** 现有所有测试通过，包括新增的 CUDA Samples 路径

#### Scenario: 现有 kernel/cosim 测试不变

- **WHEN** the harness is added
- **THEN** test_test3_cfg_full, test_barrier_warp_sync, test_ldglobal_simple, 3 个 shared_memory 测试, test_flashattention_mini, test_printf 仍在原位 + 整个 tests/e2e/cosim/ 不动

#### Scenario: git log --follow 验证 file history 保留

- **WHEN** 执行 `git log --follow <file>`（被 `git mv` 的文件）
- **THEN** file history 保留（`git mv` 而非 rm+add）