# tcgen05-handler-test-coverage

> **架构依据**: ADR-0016 (Blackwell-only tcgen05)
> **前置 change**: `implement-tcgen05-handlers-core` (archived @ df6dde7)
> **关键修正(Metis 修订)**:
> 1. 路径 `tests/integration/parser/` → `tests/integration/ptx/`
> 2. 路径 `tests/ptx/reference/` → `tests/reference/ptx_tcgen05/`
> 3. 新增 Requirement: `tcgen05.h` handler 头文件
> 4. 所有 golden value 场景强化为可执行断言(非 grep-only)
> 5. 新增 Requirement: dead-code coverage 显式标注

## ADDED Requirements

### Requirement: Handler header file SHALL exist with forward declarations

The file `include/ptxsim/instructions/tcgen05.h` SHALL exist and declare
all 5 `processTcgen05*` handler functions in `namespace ptxsim`, allowing
test code to call them directly without linker errors.

The header file SHALL contain a top-of-file comment explaining that the
5 handler functions are currently **dead code** (not registered in the
dispatch table — `S_TCGEN05_*` is explicitly excluded from the
`ptx_op.def` X-Macro loop per `ptx_op.def:129-136`), and that the header
exists solely to support dead-code coverage tests until the dispatch
issue is resolved in a separate `fix-tcgen05-handler-dispatch` change.

#### Scenario: header file exists with 5 declarations
- **WHEN** `ls include/ptxsim/instructions/tcgen05.h` is run
- **THEN** the file exists
- **AND** it contains declarations for `processTcgen05Mma`, `processTcgen05Ld`, `processTcgen05St`, `processTcgen05Commit`, `processTcgen05Wait`
- **AND** it contains a top-of-file comment marking the handlers as dead code

#### Scenario: header comment references design D4
- **WHEN** `grep -E "DEAD-CODE-NOTICE|design\\.md D4" include/ptxsim/instructions/tcgen05.h` is run
- **THEN** at least one reference to `design.md D4` is found, documenting the rationale

---

### Requirement: 5 integration parse tests SHALL exist for 5 core handlers

The `tests/integration/ptx/` directory SHALL contain 5 test files
covering parse → IR validation for each of the 5 core tcgen05 handlers
(mma / ld / st / commit / wait), as delivered in `implement-tcgen05-handlers-core`
(commit `df6dde7`).

Each test MUST:
1. Construct a PTX string for the target instruction
2. Parse via ANTLR directly (NOT via `ptxsim::testing::step_warp`, since
   those helpers do not exist for tcgen05)
3. Verify `std::get<Tcgen05Instr>(stmt.data).op_kind` matches the expected `Tcgen05OpKind`
4. Verify qualifiers and operands count match the expected values
5. Be registered with ctest labels `integration;ptx;tcgen05;parse;<op>`

#### Scenario: mma parse test passes with specific assertions
- **WHEN** `ctest -R integration_ptx_tcgen05_mma_parse -V` is run
- **THEN** test passes
- **AND** `Tcgen05Instr.op_kind == Tcgen05OpKind::MMA`
- **AND** qualifiers contain `KIND::F16` and `CTA_GROUP::1`
- **AND** operands count == 4

#### Scenario: ld parse test passes with specific assertions
- **WHEN** `ctest -R integration_ptx_tcgen05_ld_parse -V` is run
- **THEN** test passes
- **AND** `Tcgen05Instr.op_kind == Tcgen05OpKind::LD`
- **AND** qualifiers contain SYNC + ALIGNED + SHAPE_32x32b + SHARED_CTA
- **AND** operands count == 2

#### Scenario: st parse test passes (symmetric to ld)
- **WHEN** `ctest -R integration_ptx_tcgen05_st_parse -V` is run
- **THEN** test passes
- **AND** `Tcgen05Instr.op_kind == Tcgen05OpKind::ST`
- **AND** qualifiers mirror ld
- **AND** operands count == 2

#### Scenario: commit parse test passes (zero-operand variant)
- **WHEN** `ctest -R integration_ptx_tcgen05_commit_parse -V` is run
- **THEN** test passes
- **AND** `Tcgen05Instr.op_kind == Tcgen05OpKind::COMMIT`
- **AND** qualifiers contain `CTA_GROUP::1`
- **AND** operands count == 0

#### Scenario: wait parse test passes (load + store variants)
- **WHEN** `ctest -R integration_ptx_tcgen05_wait_parse -V` is run
- **THEN** test passes
- **AND** `Tcgen05Instr.op_kind == Tcgen05OpKind::WAIT`
- **AND** qualifiers contain `LOAD` or `STORE` plus `CTA_GROUP::1`
- **AND** operands count == 0

---

### Requirement: Golden value SHALL exist with verifiable arithmetic

The file `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h` SHALL
contain pre-computed f16 × f16 → f32 fragment multiply values
(8 rows × 4 cols = 32 elements) for tcgen05.mma, hand-computed from
PTX ISA §9.7.16.

#### Scenario: golden value file contains correct constexpr array
- **WHEN** `grep "GOLDEN_MMA_F16_F16_F32" tests/reference/ptx_tcgen05/tcgen05_mma_golden.h` is run
- **THEN** file contains `constexpr std::array<float, 32>`
- **AND** `GOLDEN_MMA_F16_F16_F32.size() == 32`

#### Scenario: golden value contains specific verifiable values
- **WHEN** a reviewer inspects the file
- **THEN** `GOLDEN_MMA_F16_F16_F32[0] == 1.0f` (i=0, j=0)
- **AND** `GOLDEN_MMA_F16_F16_F32[3] == 4.0f` (i=0, j=3)
- **AND** `GOLDEN_MMA_F16_F16_F32[4] == 2.0f` (i=1, j=0)
- **AND** `GOLDEN_MMA_F16_F16_F32[28] == 8.0f` (i=7, j=0)
- **AND** `GOLDEN_MMA_F16_F16_F32[31] == 32.0f` (i=7, j=3)
- **AND** each value `GOLDEN_MMA_F16_F16_F32[i]` corresponds to `A[i/4] * B[i%4]` where A=[1..8], B=[1..4]
- **AND** a `// UNVERIFIED-AGAINST-HARDWARE` annotation is present

---

### Requirement: Dead-code coverage unit test SHALL verify processTcgen05Mma output

The file `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp` SHALL exist
and contain a unit test that:

1. Includes `ptxsim/instructions/tcgen05.h` (the handler header)
2. Includes the golden value header
3. Constructs a minimal `ThreadContext` + TMEM context
4. **Directly calls** `ptxsim::processTcgen05Mma(...)` (bypassing the
   dispatch table — since `S_TCGEN05_MMA` is not yet routed)
5. Verifies the output matches `GOLDEN_MMA_F16_F16_F32` within 1e-4 tolerance

The test file MUST contain a top-of-file comment block labeled
"DEAD-CODE COVERAGE TEST" explaining that the dispatch path is broken
(see `design.md D4`) and that the test calls the handler directly.

The test MUST be labeled `unit;ptx_ir;tcgen05;mma;golden` in ctest.

#### Scenario: dead-code coverage test passes
- **WHEN** `ctest -R unit_ptx_ir_tcgen05_mma_golden -V` is run
- **THEN** test passes with all 32 elements matching golden value within 1e-4

#### Scenario: dead-code coverage test has explicit dead-code notice
- **WHEN** `head -10 tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp` is run
- **THEN** it contains a comment block explaining DEAD-CODE COVERAGE TEST
- **AND** it references `design.md D4`

---

### Requirement: E2E kernel test SHALL exist for tcgen05.mma GEMM

The file `tests/e2e/kernel/test_tcgen05_mma_gemm.cu` SHALL contain a
CUDA kernel that exercises the full tcgen05 execution path including
mma / ld / st / commit / wait.

The kernel MUST be one of:
- (Priority 1) Real Blackwell GEMM PTX extracted via `cuobjdump -xptx`
  from Cutlass 3.x or equivalent
- (Priority 2) Manually constructed `tcgen05.mma` + supporting
  instructions, **only if** Phase 3.0 verification (`nvcc -ptx` on a
  minimal sample) confirms f16 tcgen05 PTX generation succeeds
- (Priority 3, deep fallback) Float-precision GEMM pattern from
  `tests/e2e/kernel/test_blackwell_gemm.cu:11` (which itself documents
  the f16 → f32 workaround required by ANTLR grammar limits)

The choice MUST be documented in the file header.

The test MUST be labeled `e2e;kernel;tcgen05;gemm;sm100` in ctest.

#### Scenario: E2E kernel compiles and passes
- **WHEN** `ctest -R e2e_tcgen05_mma_gemm -V` is run
- **THEN** test compiles and passes
- **AND** the kernel header documents which priority level was used (1, 2, or 3)

#### Scenario: E2E kernel exercises all 5 core instructions
- **WHEN** `grep -c "tcgen05\\.\\(mma\\|ld\\|st\\|commit\\|wait\\)" tests/e2e/kernel/test_tcgen05_mma_gemm.cu` is run
- **THEN** count >= 5 (one for each handler)

---

### Requirement: Existing baseline tests SHALL continue to pass

No regression in existing test suite (170+/170+ at baseline).

#### Scenario: full ctest passes with zero regression
- **WHEN** `cd build && ctest --output-on-failure` is run after all phases
- **THEN** existing tests still pass
- **AND** new tests pass: 5 integration parse + 1 unit golden + 1 E2E = **7 new tests, 177+/177+ total**
- **AND** PTX syntax baseline holds (`./tests/ptx/test_all_ptx.sh` still PASSes for 12 tcgen05 fixtures)

#### Scenario: pre-existing PTX syntax tests still pass
- **WHEN** `./tests/ptx/test_all_ptx.sh` is run
- **THEN** 12 tcgen05_*.ptx fixtures still parse correctly through ANTLR

---

## MODIFIED Requirements

(None — this delta does not modify existing capabilities.)
