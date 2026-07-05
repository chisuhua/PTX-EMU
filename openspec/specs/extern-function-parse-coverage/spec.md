# extern-function-parse-coverage Specification

## Purpose
TBD - created by archiving change add-extern-function-declaration. Update Purpose after archive.
## Requirements
### Requirement: Extern-Function-Simple-Form-Parsed MUST

The system SHALL parse `.extern .func funcName` simple form and populate
`ptxContext.externFuncs` with the function name.

`ptx_parser.cpp:exitExternFuncStatement` (line 996) extracts
`ctx->ID()->getText()` and stores in `ExternFuncDecl::name`.

#### Scenario: simple-extern-func-added-to-externFuncs
- **WHEN** PTX source contains `.extern .func simple_func`
- **THEN** `ptxContext.externFuncs.size() == 1`
- **AND** `ptxContext.externFuncs[0].name == "simple_func"`
- **AND** `ptxContext.externFuncs[0].params.size() == 0`

#### Scenario: extern-distinct-from-entry-kernel
- **WHEN** PTX source contains both `.extern .func helper` and `.entry kernel_func`
- **THEN** `ptxContext.externFuncs.size() == 1`
- **AND** `ptxContext.externFuncs[0].name == "helper"`
- **AND** `ptxContext.ptxKernels` contains kernel_func（entry function not in externFuncs）

### Requirement: Extern-Function-With-Params-Parsed MUST

The system SHALL parse `.extern .func (.param ...) funcName` form with
parameters and populate `ExternFuncDecl::params` correctly.

`ptx_parser.cpp:exitExternFuncStatement` copies `tempExternFuncParams`
to `decl.params`. `tempExternFuncParams` is populated via
`enterParamDecl` listener handler.

#### Scenario: extern-func-with-single-param
- **WHEN** PTX source contains `.extern .func (.param .b32 x) func_with_param`
- **THEN** `ptxContext.externFuncs.size() == 1`
- **AND** `ptxContext.externFuncs[0].name == "func_with_param"`
- **AND** `ptxContext.externFuncs[0].params.size() == 1`
- **AND** `ptxContext.externFuncs[0].params[0].paramName == "x"`
- **AND** `ptxContext.externFuncs[0].params[0].byteSize == 4`（`.b32` = 4 bytes）

#### Scenario: extern-func-with-multiple-params
- **WHEN** PTX source contains `.extern .func (.param .b32 x, .param .b64 y, .param .f32 z) multi_param`
- **THEN** `ptxContext.externFuncs.size() == 1`
- **AND** `ptxContext.externFuncs[0].params.size() == 3`
- **AND** params[0].byteSize == 4, params[1].byteSize == 8, params[2].byteSize == 4

### Requirement: Extern-Function-VisitFunctionDecl-Path MUST

The system SHALL also extract extern function name via `PtxVisitor::visitFunctionDecl`
(line 486) when traversing the AST manually.

`visitFunctionDecl` handles both `functionHeader` and `extern function form`
(`.extern .func (.param ...) funcName` via `ctx->ID()` fallback).

#### Scenario: visit-function-decl-handles-extern-form
- **WHEN** PTX source contains `.extern .func visited_func`
- **AND** `PtxVisitor::visitFunctionDecl` is invoked
- **THEN** `currentKernel->kernelName == "visited_func"`
- **AND** `currentKernel->ifVisibleKernel == false`（extern marker）

#### Scenario: visit-function-decl-distinguishes-entry-vs-extern
- **WHEN** PTX source contains both `.extern .func helper` and `.entry kernel_func`
- **AND** `PtxVisitor::visitFunctionDecl` is invoked for both
- **THEN** For extern: `kernelName="helper"`, `ifEntryKernel=false`
- **AND** For entry: `kernelName="kernel_func"`, `ifEntryKernel=true`

### Requirement: No-Regression-Extern-Function MUST

The system SHALL NOT regress existing PTX parsing functionality when
modifying extern function handling.

#### Scenario: ctest-full-pass-after-fix
- **WHEN** Running `ctest --output-on-failure` after Phase 1 Fix #1
- **THEN** 100% PASS
- **AND** no new FAIL introduced by TODO deletion or test addition

#### Scenario: ptx-syntax-test-pass
- **WHEN** Running `./tests/ptx/test_all_ptx.sh` after Phase 1
- **THEN** 100% PASS
- **AND** no regression in PTX syntax tests

### Requirement: Stale-TODO-Comment-Removed MUST

The system SHALL remove the stale TODO comment at
`src/ptx_parser/ptx_visitor.cpp:350` to avoid misleading future readers.

The TODO claims "Add extern function declaration handling" in
`visitDeclaration`, but function declarations are NOT processed in the
DeclarationContext (they are siblings in PtxFileContext processed by
`visitFunctionDecl`).

#### Scenario: stale-todo-not-in-source
- **WHEN** `grep -n "TODO.*extern function declaration" src/ptx_parser/ptx_visitor.cpp`
- **THEN** Output is empty（0 匹配）

#### Scenario: explanation-comment-instead
- **WHEN** Reading `src/ptx_parser/ptx_visitor.cpp:350`
- **THEN** Comment explains "function decl processed in visitFunctionDecl, not here"
- **AND** No misleading TODO remains

### Requirement: AGENTSMD-Extern-Function-Doc-Sync MUST

The system SHALL sync documentation reflecting actual extern function
handling state.

#### Scenario: root-AGENTSMD-known-limitations-sync
- **WHEN** Reading root `AGENTS.md` "已知限制" table
- **THEN** Extern function entry describes actual state（"已支持双路径"）
- **AND** No "未处理" / "not handled" claim

#### Scenario: parser-AGENTSMD-structure-sync
- **WHEN** Reading `src/ptx_parser/AGENTS.md` STRUCTURE section
- **THEN** Documents `PtxListener::exitExternFuncStatement` (line 996) + `PtxVisitor::visitFunctionDecl` (line 486) as extern function processing paths

### Requirement: Oracle-Test-Exists MUST

The system SHALL provide unit test coverage for extern function parsing
via `tests/unit/parser/test_extern_function.cpp`.

#### Scenario: test-exists-and-registered
- **WHEN** `tests/unit/parser/test_extern_function.cpp` exists
- **AND** `tests/unit/CMakeLists.txt` contains `add_catch_test(unit_extern_function ...)`
- **THEN** `ctest -N -R unit_extern_function` lists 1 test
- **AND** Test has labels `unit;parser;extern`

#### Scenario: test-three-scenarios
- **WHEN** Running `unit_extern_function`
- **THEN** 至少 3 个 TEST_CASE 覆盖：
  - (1) 简单 `.extern .func name` 形式 → name 提取
  - (2) 带参数 `.extern .func (.param .b32 x) name` → params 提取
  - (3) extern vs entry 区分 → ifEntryKernel / ifVisibleKernel

#### Scenario: test-must-not-trigger-pre-existing-parser-bugs
- **WHEN** Running `unit_extern_function` with minimal PTX input
- **THEN** No regression in pre-existing parser issues
- **AND** Test input uses ONLY extern function syntax（避免触发 LSP 错误区域）

