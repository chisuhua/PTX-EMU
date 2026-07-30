# ptxir-roundtrip-testing Specification

## Purpose
TBD - created by archiving change ptxir-format-compliance. Update Purpose after archive.
## Requirements
### Requirement: All V1 instruction types MUST roundtrip correctly
The PTXIR serializer and deserializer SHALL preserve all fields of every V1-supported instruction type (24 types: BranchInstr, LabelInstr, VoidInstr, BarrierInstr, GenericInstr for S_MOV/S_ADD/S_SUB/S_MUL/S_LD/S_ST/S_SETP/S_CVT, DeclarationInstr, BarWarpSyncInstr, PragmaInstr, DollarNameInstr, MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, VoteInstr, ShflInstr, AtomInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective) through a complete serialize-deserialize cycle.

#### Scenario: BranchInstr roundtrip
- **WHEN** a `vector<StatementContext>` containing a `BranchInstr` with target="L1", predicate="%p0", predicate_negated=false, reconvergence_pc=42 is serialized to a `.ptxir` file and then deserialized
- **THEN** the deserialized `StatementContext` MUST have type=S_BRA and a `BranchInstr` with identical target, predicate, predicate_negated, and reconvergence_pc values

#### Scenario: GenericInstr roundtrip
- **WHEN** a `vector<StatementContext>` containing a `GenericInstr` of type S_ADD with qualifiers=[.f32], operands=[RegOperand("%r0"), RegOperand("%r1"), ImmOperand("1")] is serialized and deserialized
- **THEN** the deserialized `GenericInstr` MUST have type=S_ADD, identical qualifiers, and operands with matching kind and value (RegOperand/ImmOperand)

#### Scenario: DeclarationInstr roundtrip
- **WHEN** a `vector<StatementContext>` containing a `DeclarationInstr` of kind=REG, dataType=.u32, name="my_var", array_size=16 is serialized and deserialized
- **THEN** the deserialized `DeclarationInstr` MUST have identical kind, dataType, name, and array_size fields

#### Scenario: Mixed instruction types roundtrip
- **WHEN** a `vector<StatementContext>` containing 100+ statements of mixed types (BranchInstr, GenericInstr, DeclarationInstr, BarrierInstr, etc.) is serialized and deserialized
- **THEN** the deserialized vector MUST have identical length, identical statement types in identical order, and identical data fields per statement

### Requirement: Error paths MUST throw rather than silently fail
The PTXIR reader SHALL throw a `std::runtime_error` with a descriptive message when encountering invalid data, rather than silently skipping or returning default-constructed values.

#### Scenario: Invalid magic number
- **WHEN** a `.ptxir` file has magic bytes that are not "PTXIR"
- **THEN** `deserialize_statements()` MUST throw `std::runtime_error` with message containing "Invalid PTXIR magic"

#### Scenario: Unsupported version
- **WHEN** a `.ptxir` file has `version` field other than `PTXIR_VERSION` (1)
- **THEN** `deserialize_statements()` MUST throw `std::runtime_error` with message containing "Unsupported PTXIR version"

#### Scenario: Unknown section type
- **WHEN** a TOC entry has a `type` field that is not a valid `PtxirSectionType` enum value
- **THEN** `PtxirReader::read()` MUST throw `std::runtime_error` with message identifying the unknown type

#### Scenario: String table index out of range
- **WHEN** an instruction operand references a string table index that exceeds the string table size
- **THEN** `PtxirReader::read()` MUST throw `std::runtime_error` with the invalid index value

### Requirement: Cross-architecture byte order MUST be deterministic
The PTXIR format MUST use little-endian byte order for all multi-byte fields, regardless of the host system's native byte order, to ensure files written on one architecture can be read on another.

#### Scenario: Endianness independence
- **WHEN** a `.ptxir` file written on a little-endian system (e.g., x86_64) is read on a big-endian system (if available)
- **THEN** the deserialized statements MUST be byte-identical to the original (verified by `memcmp` of serialized data, then deserialize produces same field values)

#### Scenario: Native endianness conversion
- **WHEN** the host system is big-endian
- **THEN** the writer MUST use `__builtin_bswap16` / `__builtin_bswap32` to convert to little-endian before writing, and the reader MUST use the same conversion when reading

### Requirement: Test independence from ANTLR runtime
The PTXIR unit test file `tests/unit/test_ptxir_serialization.cpp` MUST construct test data using `statement_factory` (manual StatementContext construction) and MUST NOT depend on ANTLR runtime libraries, to enable testing in resource-constrained environments (2-core systems) where ANTLR compilation OOMs.

#### Scenario: Unit test compiles without ANTLR
- **WHEN** `cmake --build build --target unit_ptxir_serialization` is invoked on a 2-core system
- **THEN** the build MUST succeed (no OOM) and the test target MUST be linkable against `ptxir_writer` + `ptxir_reader` + `ptxir` static libraries only

