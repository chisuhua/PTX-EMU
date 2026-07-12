## MODIFIED Requirements

### Requirement: IR MUST Provide Tcgen05Instr Struct Independent of WmmaInstr

The system SHALL provide a new `Tcgen05Instr` struct in `include/ptx_ir/statement_context.h` that is structurally independent of any pre-Blackwell `WmmaInstr`. The struct MUST contain:

- `Tcgen05OpKind op_kind`: enum identifying the instruction family (one of 11 values)
- `std::vector<Qualifier> qualifiers`: full qualifier list from PTX source
- `std::vector<OperandContext> operands`: parsed operands (including the new tmem_slot operand for LD/ST/CP families)
- `std::string instructionText`: original PTX text (for logging)
- `uint32_t cta_group = 1`: parsed `.cta_group` value (will be populated by FU-1/C3 follow-up)
- `Tcgen05Dtype dtype = Tcgen05Dtype::F16`: parsed `.kind` value
- `uint32_t num_regs = 0`: parsed `x1/x2/x4` for ld/st
- `bool has_block_scale = false`: parsed `.block_scale` flag
- **`uint32_t tmem_slot = 0`** *(NEW per Oracle C2 fix in `fix-tcgen05-ld-st-slot-routing`)*: parsed TMEM slot operand for LD/ST/CP instructions; default `0` preserves backward compatibility with the previous hardcoded handler behavior.

The system MUST also provide:
- `enum class Tcgen05OpKind { ALLOC, DEALLOC, RELINQUISH, LD, ST, CP, MMA, MMA_WS, COMMIT, WAIT, FENCE }`
- `enum class Tcgen05Dtype { F16, BF16, TF32, F8, F4, MXF4, MXF8, I8, MXF4NVF4, INVALID }`

#### Scenario: tcgen05-instr-struct-has-all-fields
- **WHEN** `Tcgen05Instr` is instantiated with all fields set
- **THEN** all **9** fields are accessible and read/write correctly (the original 8 plus `tmem_slot`)
- **AND** the new `tmem_slot` field defaults to `0` when `Tcgen05Instr{}` is value-initialized

#### Scenario: tcgen05-instr-tmem_slot-default-is-zero
- **WHEN** `Tcgen05Instr` is default-constructed via `Tcgen05Instr{}` or aggregate initialization `Tcgen05Instr{.op_kind=LD, .cta_group=1}`
- **THEN** `instr.tmem_slot == 0` (backward compatible with pre-C2-fix hardcoded slot 0 in handlers)
- **AND** no test fixture breakage from the new field addition

#### Scenario: tcgen05-instr-tmem_slot-stored-after-parse
- **WHEN** a `tcgen05.ld`, `.st`, or `.cp` instruction with operand `tmem_slot=32` is parsed by the grammar + visitor + factory
- **THEN** the resulting `StatementContext`'s `Tcgen05Instr` SHALL have `tmem_slot == 32`
- **AND** the existing `test_tcgen05_ld_parse.cpp` test SHALL be extended with a new scenario that verifies this (per `fix-tcgen05-ld-st-slot-routing/tasks.md` Phase 1 step 1.4)

#### Scenario: tcgen05-opkind-enum-has-11-values
- **WHEN** `Tcgen05OpKind` is enumerated
- **THEN** exactly 11 distinct enum values exist (ALLOC through FENCE)

#### Scenario: tcgen05-dtype-enum-has-10-values
- **WHEN** `Tcgen05Dtype` is enumerated
- **THEN** exactly 10 distinct enum values exist (F16 through INVALID)

#### Scenario: tcgen05-instr-no-wmma-coupling
- **WHEN** the Tcgen05Instr definition is read
- **THEN** zero references to `WmmaInstr` or `wmmaType` exist
- **AND** Tcgen05Instr can be compiled without `wmma.h` being included

### Requirement: IR MUST Provide StatementFactory Function for Tcgen05Instr

The system SHALL provide a `makeTcgen05Instr(...)` factory function in `include/ptx_ir/statement_factory.h` that creates a `StatementContext` with type `S_TCGEN05_*` and data `Tcgen05Instr`. The factory MUST accept a new optional `tmem_slot` parameter (per Oracle C2 fix in `fix-tcgen05-ld-st-slot-routing`) with default value `0` to preserve backward compatibility with all existing call sites. The system MUST DELETE the existing `makeWmmaInstr(...)` factory function (per wmma cleanup).

#### Scenario: make-tcgen05-instr-creates-statement-with-tmem_slot
- **WHEN** `makeTcgen05Instr(Tcgen05OpKind::LD, quals, operands, "tcgen05.ld ...", /*tmem_slot=*/32)` is called
- **THEN** the returned `StatementContext` has `type = S_TCGEN05_LD`
- **AND** the data variant contains a `Tcgen05Instr` with `tmem_slot == 32`
- **AND** all other fields match the input
- **AND** `instructionText` matches the input string

#### Scenario: make-tcgen05-instr-default-tmem_slot-is-zero
- **WHEN** `makeTcgen05Instr(Tcgen05OpKind::MMA, quals, operands, "tcgen05.mma ...")` is called WITHOUT the `tmem_slot` argument
- **THEN** the returned `Tcgen05Instr` has `tmem_slot == 0` (default parameter behavior)
- **AND** all existing call sites that don't specify `tmem_slot` continue to compile and behave identically

#### Scenario: make-wmma-instr-removed
- **WHEN** `makeWmmaInstr(...)` is called
- **THEN** the code does not compile (function does not exist)
- **AND** all references are removed from src/ include/ tests/
