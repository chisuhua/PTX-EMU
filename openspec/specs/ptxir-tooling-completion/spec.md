# ptxir-tooling-completion Specification

## Purpose
TBD - created by archiving change ptxir-format-compliance. Update Purpose after archive.
## Requirements
### Requirement: generate_ptxir() function MUST exist with specified signature
The `ptxir_serialization.h` header SHALL declare a `generate_ptxir()` function with the following signature: `bool generate_ptxir(const std::string& ptx_path, const std::string& ptxir_path, const std::string& kernel_name = "")`, and the implementation in `ptxir_serialization.cpp` SHALL convert a PTX text file to a `.ptxir` file via ANTLR parsing followed by serialization. The function MUST compute and embed `reconvergence_pc` for all `S_BRA` and `S_BAR` instructions via CFGBuilder before serialization.
target: ptxir-tooling-completion

#### Scenario: Successful PTX to PTXIR conversion with embedded reconvergence_pc
- **WHEN** `generate_ptxir("input.ptx", "output.ptxir")` is called with a valid PTX file
- **THEN** the function MUST return `true`
- **AND** "output.ptxir" MUST exist as a valid file readable by `deserialize_statements()`
- **AND** the deserialized statements MUST have `reconvergence_pc` populated for all `S_BRA` and `S_BAR` instructions
- **AND** the values MUST match what `CFGBuilder::build()` + `CFGBuilder::computePostDominators()` would produce

#### Scenario: Empty kernel_name defaults to first kernel
- **WHEN** `generate_ptxir("input.ptx", "output.ptxir")` is called with empty `kernel_name`
- **THEN** the function MUST serialize the first kernel found in the PTX file
- **AND** `reconvergence_pc` MUST be embedded for that kernel

#### Scenario: Specific kernel_name selection
- **WHEN** `generate_ptxir("input.ptx", "output.ptxir", "my_kernel")` is called
- **THEN** the function MUST serialize only the kernel named "my_kernel" from the PTX file
- **AND** if "my_kernel" does not exist, the function MUST return `false`

#### Scenario: ANTLR parse failure returns false
- **WHEN** `generate_ptxir("invalid.ptx", "output.ptxir")` is called and the PTX file fails to parse
- **THEN** the function MUST return `false`
- **AND** no `.ptxir` file MUST be created (or, if created, MUST be a 0-byte file)

### Requirement: load_ptxir() apply_cfg parameter MUST integrate CFGBuilder
The `load_ptxir()` function in `ptxir_serialization.h` SHALL support a `bool apply_cfg = false` parameter. When `apply_cfg = true`, the function MUST call `CFGBuilder::build()` on the deserialized statements before returning them, matching the behavior of `load_ptx_statements(ptx_path, "", true)`. When `apply_cfg = false` and the PTXIR file is v3+, the function MUST use embedded `reconvergence_pc` values from the binary.
target: ptxir-tooling-completion

#### Scenario: apply_cfg=false with v3 file uses embedded reconvergence_pc
- **WHEN** `load_ptxir("v3_output.ptxir", false)` is called on a v3 PTXIR file
- **THEN** the returned `vector<StatementContext>` MUST have `reconvergence_pc` populated from the embedded binary values
- **AND** the function MUST NOT call `CFGBuilder::build()` or `CFGBuilder::computePostDominators()`

#### Scenario: apply_cfg=true populates reconvergence_pc (v2 compatibility)
- **WHEN** `load_ptxir("v2_input.ptxir", true)` is called on a v2 PTXIR file
- **THEN** the returned `vector<StatementContext>` MUST have `S_BRA` and `S_BAR` instructions with `reconvergence_pc` field set to a valid PC value (post-CFGBuilder analysis)
- **AND** the behavior MUST be equivalent to `load_ptx_statements(ptx_path, "", true)` when the same PTX source is used to generate the .ptxir

#### Scenario: Default apply_cfg is false
- **WHEN** `load_ptxir("input.ptxir")` is called without specifying `apply_cfg`
- **THEN** the function MUST default to `apply_cfg = false`

### Requirement: Tooling functions MUST be in ptxir_serialization.cpp
Both `generate_ptxir()` and `load_ptxir(apply_cfg)` MUST be implemented in `src/ptxir/ptxir_serialization.cpp` (not in separate files), consistent with `serialize_statements()` and `deserialize_statements()` placement.
target: ptxir-tooling-completion

#### Scenario: Single translation unit
- **WHEN** `src/ptxir/CMakeLists.txt` is inspected
- **THEN** the `add_library(ptxir STATIC ...)` source list MUST include `ptxir_serialization.cpp` (and only that file, not new files for tooling)

#### Scenario: Header signature stable
- **WHEN** the public API in `include/ptxir/ptxir_serialization.h` is inspected
- **THEN** the 4 function declarations (`serialize_to_string`, `deserialize_from_string`, `serialize_statements`, `deserialize_statements`) MUST remain unchanged
- **AND** 2 new declarations MUST be added: `generate_ptxir()` and `load_ptxir()`

