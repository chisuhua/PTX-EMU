# PTXIR Serialization Skill

## Overview

PTXIR (PTX Intermediate Representation) is a binary serialization format for PTX-EMU's `StatementContext` IR. It enables fast loading of pre-parsed PTX kernels, bypassing ANTLR parsing overhead (~200ms → ~5ms).

## Format Specification

### File Structure

```
┌─────────────────────────────────┐
│         PTXIRHeader (24B)       │
├─────────────────────────────────┤
│      Section TOC (6B * N)       │
├─────────────────────────────────┤
│         REGDECL Section         │
├─────────────────────────────────┤
│        KERNEL Section           │
├─────────────────────────────────┤
│      STRING_TABLE Section       │
└─────────────────────────────────┘
```

### Header (24 bytes)

```cpp
struct PtxirHeader {
    char magic[4];           // "PTXIR"
    uint16_t version;        // 1
    uint16_t flags;          // reserved
    uint16_t section_count;  // number of sections
    uint32_t string_offset;  // string table offset
    uint32_t string_size;    // string table size
};
```

### Section Types

| Type | Value | Description |
|------|-------|-------------|
| REGDECL | 1 | Register declarations (operand table) |
| KERNEL | 2 | Kernel statements |
| STRING_TABLE | 3 | String pool for labels/identifiers |

## API Reference

### test_helpers.hpp Functions

```cpp
// Serialize StatementContext vector to .ptxir file
bool serialize_statements(const std::vector<StatementContext>& stmts,
                          const std::string& path);

// Deserialize .ptxir file to StatementContext vector
std::vector<StatementContext> deserialize_statements(const std::string& path);

// Generate .ptxir from PTX source (PTX → ANTLR → serialize)
bool generate_ptxir(const std::string& ptx_path,
                   const std::string& ptxir_path,
                   const std::string& kernel_name = "");

// Load .ptxir with optional CFG builder application
std::vector<StatementContext> load_ptxir(const std::string& ptxir_path,
                                         bool apply_cfg = false);
```

### Core Classes

#### PtxirWriter

```cpp
class PtxirWriter {
public:
    explicit PtxirWriter(std::ostream& out);
    void write(const std::vector<StatementContext>& stmts);
};
```

**Process:**
1. Pre-pass: enumerate all `RegOperand` → assign compact u32 IDs
2. Write 24-byte header
3. Write sections: REGDECL, KERNEL, STRING_TABLE

#### PtxirReader

```cpp
class PtxirReader {
public:
    explicit PtxirReader(std::istream& in);
    std::vector<StatementContext> read();
};
```

**Process:**
1. Read and validate header (magic "PTXIR", version 1)
2. Read string table
3. Read kernel statements

## Workflow

### Mode 4: Fast Load (Binary)

```
.ptxir file → deserialize → StatementContext[] → execute
     ↓
  ~5ms load time
```

### Mode 2 → Mode 4 Conversion

```bash
# Generate .ptxir from existing PTX file
python3 docs/skills/three-mode-testing/generate_tests.py \
    --benchmark test_divergence_sync_standalone \
    --ptxir
```

### Roundtrip Test

```cpp
// Serialize → Deserialize → Compare
auto stmts_ref = load_ptx_statements(ptx_path, "", false);
serialize_statements(stmts_ref, "test.ptxir");
auto stmts_loaded = deserialize_statements("test.ptxir");
CHECK(stmts_loaded.size() == stmts_ref.size());
```

## Supported Statement Types

| StatementType | Encoding | Notes |
|---------------|----------|-------|
| S_BRA | BranchInstr | target, predicate, reconvergence_pc |
| S_LABEL | LabelInstr | labelName |
| S_EXIT, S_RET | VoidInstr | - |
| S_BAR | BarrierInstr | barId, qualifiers |
| S_MOV, S_ADD, S_SUB, S_MUL | GenericInstr | operands, qualifiers |
| S_LD, S_ST | GenericInstr | operands |
| S_SETP | GenericInstr | operands |
| S_PRAGMA | PragmaInstr | content |
| S_DOLLOR | DollarNameInstr | name |
| S_REG, S_CONST, S_SHARED, S_LOCAL, S_GLOBAL, S_PARAM | DeclarationInstr | kind, dataType, name, array_size |

## Limitations

- **ANTLR runtime required** for `load_ptx_statements()` and `generate_ptxir()`
- **CFG builder not serialized** — call `apply_cfg_builder()` after loading if needed
- **String table at end** — requires two-pass for offset resolution

## Build Requirements

```cmake
# ptxir_writer and ptxir_reader are built as shared libraries
find_library(ptxir_writer REQUIRED)
find_library(ptxir_reader REQUIRED)
target_link_libraries(test_ptxir_serialization PRIVATE ptxir_writer ptxir_reader)
```

## Directory Structure

```
src/ptx_ir/
├── ptxir_format.h      # Binary format definitions
├── ptxir_writer.h/cpp  # Serialization
├── ptxir_reader.h/cpp # Deserialization

tests/three_mode_testing/
├── test_ptxir_serialization.cpp  # Mode 4 tests
├── test_helpers.hpp              # serialize/deserialize helpers

tests/ptxir/                     # Pre-generated .ptxir files
```

## See Also

- [PTX-TO-STATEMENTS-IMPLEMENTATION.md](../../developer-guide/PTX-TO-STATEMENTS-IMPLEMENTATION.md)
- [THREE-MODE-TESTING-GUIDE.md](../../developer-guide/THREE-MODE-TESTING-GUIDE.md)
- `openspec/changes/ptxir-serialization-architecture/` — Full design doc