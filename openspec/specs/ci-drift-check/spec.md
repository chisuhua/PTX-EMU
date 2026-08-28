# ci-drift-check Specification

## Purpose
TBD - created by archiving change ptxemu-public-device-api. Update Purpose after archive.
## Requirements
### Requirement: `.github/workflows/drift_check.yml` MUST 存在

新增 workflow 文件 MUST 验证 PTX-EMU 内部公共头布局与仓内 hash 一致:

```yaml
name: drift-check
on:
  pull_request:
    paths:
      - 'include/ptxemu/**'
      - 'include/ptx_ir/**'
  workflow_dispatch:
jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Header hash drift check
        run: |
          # 验证 device_api.h PUBLIC 字段稳定
          grep -q "PTXEMU_API_VERSION 1" include/ptxemu/device_api.h
          # 验证 IPtxEmuDevice 接口签名冻结 (count 虚方法)
          EXPECTED_METHODS=12  # S1 facade 12 callsites 1:1
          ACTUAL=$(grep -c "virtual.*=.*0\|virtual.*override" include/ptxemu/device_api.h)
          test "$ACTUAL" -ge "$EXPECTED_METHODS"
```

#### Scenario: Phase 2 PR 修改 device_api.h 触发 drift_check
- **WHEN** GitHub PR 修改 `include/ptxemu/device_api.h`
- **THEN** drift_check workflow 自动 run, 验证 `PTXEMU_API_VERSION=1` 守卫宏保留 + 虚方法数量 >= 12

### Requirement: `consumer_smoke` MUST 不在 Phase 2 PR 范围

`consumer_smoke` (验证 PTX-EMU 端能在 CMake 链式构建 CppTLM) MUST 不进 Phase 2 PR, 延后至 HSK-9 准入 (Decision 2 答复)。

#### Scenario: Phase 2 PR 不包含 consumer_smoke
- **WHEN** 读 Phase 2 PR diff
- **THEN** 0 引用 `tests/build_cpptlm_consume/` 路径

### Requirement: `drift_check` 与 `build-and-test` 解耦

`drift_check` MUST 作为独立 workflow, 不与 `build-and-test` 串行依赖。允许仅跑 `drift_check` (PR 阶段) 而不跑 `build-and-test` (release 阶段) 单独验证。

#### Scenario: PR 阶段仅跑 drift_check
- **WHEN** PR 标题含 `[skip-build]` 或 path filter 仅含 `include/ptxemu/**`
- **THEN** `drift_check` 跑通即视为合规, `build-and-test` 可跳过

### Requirement: drift_check workflow verifies no empty-body IPtxEmuDevice method stubs in `device_api_impl.cc`

The drift_check workflow (`.github/workflows/drift_check.yml`) MUST extend its `paths` trigger filter to include `src/ptxemu/**` (alongside existing `include/ptxemu/**`), and add Invariant 6: after Phase 2.2/2.3 implementation, no IPtxEmuDevice override method in `src/ptxemu/device_api_impl.cc` may contain an **empty body** that unconditionally returns a constant default value (`return false`, `return -1`, `return ThreadState::kIdle`, default-constructed `WarpStatus s{}; return s;`, or empty void no-op). Legitimate error-path guards (`if (!sm) return false;` followed by real delegation) MUST NOT trigger the invariant — only stub patterns (body containing a single constant return with no logic) MUST fail.

> **Invariant 6 (NEW)**: This is added as the 6th invariant in drift_check workflow, alongside the existing 5 invariants (PTXEMU_API_VERSION==1, IPtxEmuDevice ≥ 12 pure virtuals, C++17 compat, 4 symbols present, ptxemu_core STATIC target name).

#### Scenario: Phase 2.2/2.3 commit triggers drift_check on src/ptxemu changes

- **WHEN** a commit modifying `src/ptxemu/device_api_impl.cc` is pushed to any branch
- **AND** the file contains no empty-body stubs (per regex pattern below)
- **THEN** drift_check Invariant 6 PASSES
- **AND** the overall drift_check workflow exits 0

#### Scenario: Regression commit reintroducing empty-body stubs fails Invariant 6

- **WHEN** a future commit reintroduces empty-body stubs in `src/ptxemu/device_api_impl.cc`
- **THEN** drift_check Invariant 6 FAILS
- **AND** the CI pipeline blocks merge to main
- **AND** the regression is detected before reaching production (analogous to BUG-RETHANG prevention)

#### Scenario: Legitimate error-path returns do NOT trigger failure

- **WHEN** a delegation method contains error guards like `if (!sm) return false;` followed by real delegation logic
- **THEN** drift_check Invariant 6 PASSES (the `return false` is part of valid control flow, not a stub)
- **AND** only single-statement constant returns trigger failure

#### Scenario: Implementation pattern enforcement via drift_check

- **WHEN** contributors add new methods to `IPtxEmuDevice` (would require HSK-9)
- **AND** add corresponding empty-body stubs to `device_api_impl.cc`
- **THEN** drift_check Invariant 6 immediately flags the new stubs
- **AND** the contributor MUST implement the delegation before merging (no silent no-op stubs allowed)

#### Scenario: Invariant 6 regex (suggested)

- **MATCH**: `^\s*return\s+(false|nullptr|-1|ThreadState::kIdle|true);?\s*$` (single constant return, with optional semicolon)
- **EXCLUDE**: methods with >1 statement (delegation logic + error guards)
- **EXCLUDE**: `attach_timing` (void return type — stub pattern is no statements at all, but tracked by separate "empty void body" pattern)
- **IMPLEMENTATION**: bash + grep -E "^\s+return (false|nullptr|-1|ThreadState::kIdle);" -- context 5 (must be only return in method body); OR Python AST parse

#### Scenario: Deferred stub methods (per design Non-Goal 5) are explicitly exempted

- **WHEN** a method is in the **deferred stubs list**:
  - `warp_exe_once` (`src/ptxemu/device_api_impl.cc` L85-88, returns `-1`)
  - `get_thread_state` (L99-102, returns `ThreadState::kIdle`)
  - `get_warp_status` (L121-126, returns default-constructed `WarpStatus s{}`)
- **THEN** drift_check Invariant 6 EXEMPTS these 3 methods from empty-body detection
- **AND** this exemption is documented in `openspec/changes/device-api-delegation/design.md` Non-Goal 5 (deferred to Phase 2.2.1/2.3.1 follow-up change)
- **AND** Invariant 6 implementation MUST encode this exemption via explicit method-name whitelist
- **WHEN** the Phase 2.2.1/2.3.1 follow-up change implements these 3 methods
- **THEN** the exemption MUST be removed from Invariant 6 simultaneously
- **AND** the follow-up change commit message MUST reference this spec scenario as the exemption removal trigger

> **Rationale** (per design.md Non-Goal 5 + Metis MR-Oracle inventory): these 3 methods were `nlohmann::`-style documentation-only stubs in the HSK-8 Phase 2 PR (commit `d281a21e`). Phase 2.2/2.3 R7-constrained minimum scope implements only 4 in-scope methods (set_scoreboard / set_active_mask / set_next_pc / attach_timing). The 3 deferred stubs are tracked as follow-up work and SHOULD remain stub bodies until the Phase 2.2.1/2.3.1 change lands.

### Requirement: drift_check workflow Invariant 7 verifies CMake vendored path correctness

The drift_check workflow MUST extend its `paths` trigger filter to include `CMakeLists.txt` (alongside existing `src/ptxemu/**` and `include/ptxemu/**`), and add Invariant 7: after this change, `CMakeLists.txt` MUST NOT contain `${CMAKE_SOURCE_DIR}/antlr4` hardcoded paths (per `cmake-antlr4-relative-paths/spec.md` requirement).

> **Invariant 7 (NEW)**: This is added as the 7th invariant in drift_check workflow, alongside the existing 6 invariants.

#### Scenario: CMakeLists.txt change triggers drift_check on path-correctness

- **WHEN** a commit modifying `CMakeLists.txt` is pushed to any branch
- **AND** the file uses `${PROJECT_SOURCE_DIR}/antlr4` (or `${CMAKE_CURRENT_SOURCE_DIR}/antlr4`) instead of `${CMAKE_SOURCE_DIR}/antlr4`
- **THEN** drift_check Invariant 7 PASSES
- **AND** the overall drift_check workflow exits 0

#### Scenario: Regression to `${CMAKE_SOURCE_DIR}/antlr4` hardcoding fails Invariant 7

- **WHEN** a future commit modifies `CMakeLists.txt` to re-introduce `${CMAKE_SOURCE_DIR}/antlr4` hardcoded path
- **THEN** drift_check Invariant 7 FAILS
- **AND** the CI pipeline blocks merge to main
- **AND** the regression is detected before reaching CppTLM-side chained builds (analogous to BUG-RETHANG prevention)

#### Scenario: Invariant 7 implementation

- **MATCH**: `${CMAKE_SOURCE_DIR}/antlr4` (any occurrence in `CMakeLists.txt`)
- **EXCLUDE**: `${CMAKE_CURRENT_SOURCE_DIR}/antlr4` (acceptable for subdirectory-relative references)
- **EXCLUDE**: `${PROJECT_SOURCE_DIR}/antlr4` (the correct fix)
- **IMPLEMENTATION**: bash + grep -nE "CMAKE_SOURCE_DIR.*antlr4|antlr4.*CMAKE_SOURCE_DIR" CMakeLists.txt (returns 0 lines = PASS)

### Requirement: drift_check MUST enforce no bare IR type names outside `include/ptx_ir/` shim (Invariant 8)

The `.github/workflows/drift_check.yml` workflow MUST include an 8th invariant that scans caller source files for unqualified IR type names. It MUST scan `src/`, non-canonical `include/ptxsim/`, `include/ptxemu/`, `include/cudart/`, `include/ptx_parser/`, `include/register/`, `include/utils/`, non-shim `include/ptx_ir/`, `include/ptxir/`, and `tests/`. It MUST exclude the three forwarding shims and canonical definition headers under `include/ptxemu/ir/`. The token set MUST include at least `StatementType`, `OperandType`, `InstructionState`, `Qualifier`, `OperandContext`, `InstrVariant`, `Tcgen05Instr`, `Tcgen05OpKind`, and `Tcgen05Dtype`. The implementation MUST exclude `ptxemu::ir::`-qualified occurrences using a token-aware filter, not a bare `\\bType\\b` grep. This guards against regressions where new caller code uses bare IR names instead of the canonical qualified form.

#### Scenario: Invariant 8 grep configuration
- **WHEN** reading `.github/workflows/drift_check.yml` Invariant 8 step
- **THEN** the step uses a token-aware Python scanner across the caller roots
- **AND** the step excludes the three forwarding shim headers and all canonical definition headers under `include/ptxemu/ir/`
- **AND** the scanner handles `//`/`/* */` comments, char literals, ordinary strings, and C++ raw strings without reporting tokens inside them
- **AND** the scanner ignores qualified tokens and bare names lexically inside `namespace ptxemu::ir` canonical definition blocks

#### Scenario: Invariant 8 fails on bare type
- **WHEN** a new file `src/ptxsim/instructions/foo.cpp` contains the line `Qualifier q = Qualifier::Q_F32;` (no `ptxemu::ir::` prefix)
- **AND** drift_check workflow runs on a PR that introduces this file
- **THEN** Invariant 8 grep matches the unqualified `Qualifier` token
- **AND** the workflow exits with non-zero status
- **AND** the PR cannot be merged (per HSK-8 §5 hard-fail on workflow)

#### Scenario: Invariant 8 passes on qualified type
- **WHEN** a file `src/ptxsim/instructions/foo.cpp` contains the line `ptxemu::ir::Qualifier q = ptxemu::ir::Qualifier::Q_F32;`
- **AND** drift_check workflow runs
- **THEN** the token-aware scanner does not match the `Qualifier` token because it is preceded by the `ptxemu::ir::` qualification
- **AND** the workflow exits with zero status for this step

#### Scenario: Invariant 8 exempts shim and canonical definition headers
- **WHEN** `include/ptx_ir/ptx_types.h` contains `using ::ptxemu::ir::Qualifier;` and `include/ptxemu/ir/statement.h` contains `Qualifier dataType;`
- **AND** drift_check workflow runs
- **THEN** the scanner excludes both the forwarding shim and canonical definition header, while still scanning non-shim `include/ptx_ir/` headers and all `include/ptxir/` headers
- **AND** the workflow exits with zero status for this step

