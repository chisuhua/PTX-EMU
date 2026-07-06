## ADDED Requirements

### Requirement: ANTLR Lexer MUST Recognize tcgen05 Keywords

The system SHALL provide lexer tokens in `src/grammar/ptxLexer.g4` that match all 12 Blackwell `tcgen05.*` instruction families per NVIDIA PTX ISA 8.6 §9.7.16. The lexer MUST accept the following tokens without conflicts with existing grammar:

- 1 主指令 token: `TCGEN05: 'tcgen05'`
- 11 sub-op tokens: `MMA_ / LD_ / ST_ / CP_ / ALLOC_ / DEALLOC_ / RELINQUISH_ / COMMIT_ / WAIT_ / FENCE_ / AR_`
- 3 mma 变体 tokens: `SP / WS / BLOCK_SCALE_`
- 2 scale vec size tokens: `SCALE_VEC_SIZE_2X_ / SCALE_VEC_SIZE_4X_`
- 2 wait 变体 tokens: `LOAD_ / STORE_`
- 2 fence time tokens: `BEFORE_THREAD_SYNC_ / AFTER_THREAD_SYNC_`
- 1 mbarrier arrive token: `MBARRIER_ARRIVE_`
- 1 multicast token: `MULTICAST_`
- 1 cta_group token: `CTA_GROUP_`
- 1 sync token: `SYNC_`
- 1 aligned token: `ALIGNED_`
- 1 shared::cta token: `SHARED_CTA_`
- 1 shared::cluster token: `SHARED_CLUSTER_`
- 1 sem token: `SEM_`
- 1 pack token: `PACK_`
- 10 dtype tokens: `F16_ / BF16_ / TF32_ / F8_ / F4_ / MXF4_ / MXF8_ / I8_ / MXF4NVF4_ / F8F6F4_`

Total: ~36 new tokens. Each token name uses `_` suffix to avoid conflicts with existing tokens (e.g., `F16` is already used for `.f16` data type in arithmetic instructions).

#### Scenario: lexer-recognizes-tcgen05-prefix
- **WHEN** the PTX source contains `tcgen05.mma`
- **THEN** the lexer tokenizes it as `TCGEN05` followed by `MMA_`
- **AND** no lexer error is raised

#### Scenario: lexer-recognizes-all-12-sub-ops
- **WHEN** the PTX source contains any of `tcgen05.alloc / tcgen05.dealloc / tcgen05.relinquish_alloc_permit / tcgen05.ld / tcgen05.st / tcgen05.cp / tcgen05.mma / tcgen05.commit / tcgen05.wait / tcgen05.fence / tcgen05.commit.arrive`
- **THEN** the lexer tokenizes each sub-op correctly without conflicts

#### Scenario: lexer-recognizes-mma-variants
- **WHEN** the PTX source contains `tcgen05.mma.sp` or `tcgen05.mma.ws` or `tcgen05.mma.block_scale`
- **THEN** the lexer tokenizes the variant suffixes correctly

#### Scenario: lexer-recognizes-dtype-qualifiers
- **WHEN** the PTX source contains any of `.kind::f16 / .kind::bf16 / .kind::tf32 / .kind::f8 / .kind::f4 / .kind::mxf4 / .kind::mxf8 / .kind::i8`
- **THEN** the lexer tokenizes the dtype qualifiers correctly

#### Scenario: lexer-no-conflict-with-existing-tokens
- **WHEN** all existing tests/ptx/*.ptx are re-parsed after token addition
- **THEN** no lexer conflicts occur
- **AND** all existing tests still pass (per `./tests/ptx/test_all_ptx.sh`)

### Requirement: ANTLR Parser MUST Accept tcgen05 Instruction Syntax

The system SHALL provide parser rules in `src/grammar/ptxInstructions.g4` that match the PTX ISA 8.6 grammar for all 12 `tcgen05.*` instruction families. The parser MUST:

- DELETE existing `matrixInst: wmmaInst;` and `tcgenInst: stBulkInst;` rules
- DELETE existing `wmmaInst / wmmaOp / wmmaLayout / wmmaShape / wmmaKind` rules
- ADD `tcgen05Inst` rule with complete syntax for all 12 sub-ops
- ADD `tcgen05Op / tcgen05Kind / tcgen05Layout / tcgen05Shape / tcgen05BlockScale / tcgen05WsMask / tcgen05CpShape / tcgen05Qualifier` sub-rules
- REPLACE `matrixInst` to point to `tcgen05Inst`

#### Scenario: parser-accepts-tcgen05-mma-f16
- **WHEN** the PTX source contains `tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc;`
- **THEN** the parser builds a valid AST
- **AND** no parse error is raised

#### Scenario: parser-accepts-tcgen05-ld-32x32b
- **WHEN** the PTX source contains `tcgen05.ld.sync.aligned.32x32b.x4.b32 {r0, r1, r2, r3}, [tmem];`
- **THEN** the parser builds a valid AST
- **AND** the `32x32b` shape and `x4` are recognized

#### Scenario: parser-accepts-tcgen05-cp-multicast
- **WHEN** the PTX source contains `tcgen05.cp.cta_group::1.128x256b.multicast::cluster [tmem], sdesc, mcast_mask;`
- **THEN** the parser builds a valid AST
- **AND** the `multicast::cluster` qualifier is recognized

#### Scenario: parser-accepts-tcgen05-commit-mbarrier
- **WHEN** the PTX source contains `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];`
- **THEN** the parser builds a valid AST

#### Scenario: parser-accepts-tcgen05-fence
- **WHEN** the PTX source contains `tcgen05.fence::before_thread_sync;` or `tcgen05.fence::after_thread_sync;`
- **THEN** the parser builds a valid AST

#### Scenario: parser-rejects-pre-blackwell-wmma
- **WHEN** the PTX source contains pre-Blackwell `wmma.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 ...`
- **THEN** the parser MUST raise an error (per ADR-0016: pre-Blackwell NOT supported)
- **AND** the error message references `wmma.*` as legacy

#### Scenario: parser-rejects-malformed-tcgen05
- **WHEN** the PTX source contains malformed tcgen05 (e.g., `tcgen05.invalid_op`)
- **THEN** the parser raises a clear error message identifying the unknown sub-op

### Requirement: ANTLR Grammar Modifications MUST Pass Baseline Tests

The system SHALL ensure that any modifications to ANTLR grammar files do NOT regress the existing PTX syntax test suite. All existing tests in `tests/ptx/*.ptx` MUST continue to parse successfully after grammar changes.

#### Scenario: test-all-ptx-baseline-passes
- **WHEN** `./tests/ptx/test_all_ptx.sh` is run after grammar changes
- **THEN** 100% of existing tests pass (no regression)
- **AND** new tcgen05 fixtures are added and also pass

#### Scenario: generated-parser-compiles
- **WHEN** `cmake --build build --target GenerateParser` is run after grammar changes
- **THEN** the ANTLR generator produces C++ source files without errors
- **AND** the parser compiles successfully in the next `cmake --build build` run

### Requirement: Grammar File Documentation MUST Be Updated

The system SHALL update `src/grammar/AGENTS.md` to reflect the new tcgen05 grammar structure, removing references to wmma and adding the new tcgen05 rules.

#### Scenario: agents-md-updated
- **WHEN** the grammar changes are complete
- **THEN** `src/grammar/AGENTS.md` mentions `tcgen05*` in the instruction grammar section
- **AND** removes references to deleted `wmma*` rules
- **AND** documents the new `tcgen05Inst` rule structure
