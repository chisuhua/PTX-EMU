# pc-api Specification

## Purpose
TBD - created by archiving change dead-code-cleanup. Update Purpose after archive.
## Requirements
### Requirement: Zero-Production-Refs

`ThreadContext::force_set_pc`、`WarpContext::get_pc`、`WarpContext::set_pc`、
`WarpContext::pc` 在**生产代码** (`src/` + `include/`，**不包含** `tests/`)
中零引用。

#### Scenario: verify-zero-refs-production

- **WHEN** 运行
  ```bash
  grep -rnE "force_set_pc|WarpContext::(get_pc|set_pc)|warp\.(get_pc|set_pc)\s*\(" \
    --include="*.cpp" --include="*.h" src/ include/ \
    | grep -v "build/" | grep -v "antlr4_generated_src/"
  ```
- **THEN** 输出为空（仅可能命中 `warp_context.h` 注释占位 "Removed 2026-07-XX"）

### Requirement: Clean-Build-No-Warnings

删除 deprecated API 后，**clean build** 不得产生 `[[deprecated]]` 调用的
warning 或编译错误。

#### Scenario: verify-clean-build

- **WHEN** 运行
  ```bash
  rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc) 2>&1 | tee /tmp/build.log
  grep -iE "deprecated|warning" /tmp/build.log \
    | grep -v "bench/" | grep -v "antlr4_generated_src/" \
    | grep -v "Removed 2026-07"
  ```
- **THEN** 输出为空

### Requirement: No-Regression

所有现有测试（unit + integration + e2e + PTX 语法）无新增 FAIL。

#### Scenario: verify-no-regression

- **WHEN** 运行
  ```bash
  cd build && ctest --output-on-failure
  ./scripts/sanity.sh --quick
  ./tests/ptx/test_all_ptx.sh
  ```
- **THEN** 全部 PASS，且与 baseline (`/tmp/sanity-baseline.txt`) 对比无新增 FAIL

### Requirement: Doc-Sync-Resolved

ADR-0003、ADR-0008、`src/ptxsim/core/AGENTS.md`、
`src/ptxsim/instructions/AGENTS.md` 不再推荐 `force_set_pc`。

#### Scenario: verify-doc-sync

- **WHEN** 运行
  ```bash
  grep -rn "force_set_pc" docs/adr/ADR-0003-commit-pc-pattern.md \
    docs/adr/ADR-0008-barrier-semantics.md \
    src/ptxsim/core/AGENTS.md \
    src/ptxsim/instructions/AGENTS.md
  ```
- **THEN** 仅命中 "Removed 2026-07-XX" 或 "历史实现" 字样，**无** "use
  `force_set_pc`" 推荐

