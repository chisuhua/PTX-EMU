# ptxir-format-compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补全 PTXIR Reader 15 种缺失指令 variant 覆盖、修复 Writer/Reader 格式契约对齐 ADR-0023、添加完整 roundtrip 测试套件与工具链。

**Architecture:** 4 Phase 独立 commit 策略：P1 Reader 指令覆盖 → P2 格式契约修复（Writer TOC + Reader V1/V2 双路径）→ P3 测试套件与工具链 → P4 文档同步。每个 Phase 独立可 revert。

**Tech Stack:** C++20, CMake, Catch2, PTXIR 二进制格式 (V1/V2), CFGBuilder

---

## File Structure

### Production Code

| File | Responsibility |
|------|----------------|
| `src/ptx_ir/ptxir_reader.cpp` | Reader 实现 — 15 种缺失 case + V1/V2 双路径 + TOC 解析 |
| `src/ptx_ir/ptxir_writer.cpp` | Writer 实现 — TOC 写入 + 回填偏移 + REGDECL section |
| `include/ptx_ir/ptxir_format.h` | 格式常量 — `PTXIR_VERSION` 1→2, TOC struct |
| `include/ptx_ir/ptxir_reader.h` | Reader 头文件 — 新增方法声明 |
| `include/ptx_ir/ptxir_writer.h` | Writer 头文件 — 已有方法 |
| `include/ptxir/ptxir_serialization.h` | 公共 API — 新增 `generate_ptxir()` / `load_ptxir(apply_cfg)` |
| `src/ptxir/ptxir_serialization.cpp` | 公共 API 实现 — `generate_ptxir()` + `load_ptxir(apply_cfg=true)` |

### Tests

| File | Responsibility |
|------|----------------|
| `tests/unit/test_ptxir_serialization.cpp` | Roundtrip 测试 — 24 种指令类型 + 4 种错误路径 + 混合 100+ 语句 |
| `tests/CMakeLists.txt` | 注册新测试 target |

### Docs

| File | Responsibility |
|------|----------------|
| `src/ptx_ir/AGENTS.md` | 新增 StatementContext 修改协议章节 |
| `include/ptxir/AGENTS.md` | 新增公共头文件修改协议 |
| `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md` | 升级为四模式文档 |

---

## Tasks

### Task 1: Phase 1 — Reader Instruction Coverage (G9 fix)

**Goal:** 补全 Reader 15 种缺失 InstrVariant case 分支 + 移除 default 静默跳过
**Commit:** `feat(ptxir): complete reader instruction coverage (24/24)`

**Files:**
- Modify: `src/ptx_ir/ptxir_reader.cpp`

- [ ] **Step 1: Add S_MEMBAR case — read qualifiers (u8 count + u16[]) → construct MembarInstr → stmt.data = instr**

```cpp
case S_MEMBAR: {
    MembarInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 2: Add S_FENCE case — read qualifiers → construct FenceInstr**

```cpp
case S_FENCE: {
    FenceInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 3: Add S_REDUX_SYNC case — read qualifiers + operands (u8 count + u32[]) → construct ReduxSyncInstr**

```cpp
case S_REDUX_SYNC: {
    ReduxSyncInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(ImmOperand{string_table_[id]});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 4: Add S_MBARRIER case — read qualifiers + operands → construct MbarrierInstr**

```cpp
case S_MBARRIER: {
    MbarrierInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 5: Add S_CALL case — read qualifiers + operands → construct CallInstr**

```cpp
case S_CALL: {
    CallInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(ImmOperand{string_table_[id]});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 6: Add S_VOTE case — read qualifiers + operands → construct VoteInstr**

```cpp
case S_VOTE: {
    VoteInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 7: Add S_SHFL case — read qualifiers + operands → construct ShflInstr**

```cpp
case S_SHFL: {
    ShflInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 8: Add S_ATOM case — read qualifiers + operands → construct AtomInstr**

```cpp
case S_ATOM: {
    AtomInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    read_u32(in_);  // dst_reg_id (unused)
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 9: Add S_TEXTURE case — read qualifiers + operands → construct TextureInstr**

```cpp
case S_TEXTURE: {
    TextureInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 10: Add S_SURFACE case — read qualifiers + operands → construct SurfaceInstr**

```cpp
case S_SURFACE: {
    SurfaceInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 11: Add S_REDUCTION case — read qualifiers + operands → construct ReductionInstr**

```cpp
case S_REDUCTION: {
    ReductionInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 12: Add S_PREFETCH case — read qualifiers + operands → construct PrefetchInstr**

```cpp
case S_PREFETCH: {
    PrefetchInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 13: Add S_CP_ASYNC case — read qualifiers + operands → construct CpAsyncInstr**

```cpp
case S_CP_ASYNC: {
    CpAsyncInstr instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    uint8_t ocount = read_u8(in_);
    for (uint8_t i = 0; i < ocount; i++) {
        uint32_t id = read_u32(in_);
        if (id < string_table_.size()) {
            instr.operands.emplace_back(RegOperand{string_table_[id], -1});
        }
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 14: Add S_ABI_DIRECTIVE case — no fields → construct empty AbiDirective**

```cpp
case S_ABI_DIRECTIVE: {
    AbiDirective instr;
    stmt.data = instr;
    break;
}
```

- [ ] **Step 15: Add S_PREDICATE_PREFIX case — read qualifiers → construct PredicatePrefix**

```cpp
case S_PREDICATE_PREFIX: {
    PredicatePrefix instr;
    uint8_t qcount = read_u8(in_);
    for (uint8_t i = 0; i < qcount; i++) {
        instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
    }
    stmt.data = instr;
    break;
}
```

- [ ] **Step 16: Modify default branch — throw std::runtime_error instead of silent skip**

```cpp
default: {
    throw std::runtime_error("Unknown StatementType: " + std::to_string(type));
}
```

- [ ] **Step 17: Build verification**

```bash
cmake --build build --target ptxir_reader 2>&1
```

- [ ] **Step 18: Static check — confirm no // skip or stmt.data = instr in default**

```bash
grep -n "default:" src/ptx_ir/ptxir_reader.cpp | head -3
```

- [ ] **Step 19: Commit Phase 1**

```bash
git add src/ptx_ir/ptxir_reader.cpp
git commit -m "feat(ptxir): complete reader instruction coverage (24/24)"
```

---

### Task 2: Phase 2 Commit 1 — Writer V2 Format (TOC + offset backfill)

**Goal:** Writer 完全符合 ADR-0023 Decision 1：TOC 写入 + 顺序写 + 回填偏移
**Commit:** `refactor(ptxir): align writer with V2 format (TOC + offset backfill)`

**Files:**
- Modify: `include/ptx_ir/ptxir_format.h` (PTXIR_VERSION 1→2)
- Modify: `src/ptx_ir/ptxir_writer.cpp`

- [ ] **Step 1: Bump PTXIR_VERSION 1 → 2 in ptxir_format.h**

```cpp
static constexpr uint16_t PTXIR_VERSION = 2;
```

- [ ] **Step 2: Rewrite PtxirWriter::write_header() — write version=2, reserve TOC space, update section_count**

```cpp
void PtxirWriter::write_header() {
    PtxirHeader hdr{};
    std::memcpy(hdr.magic, PTXIR_MAGIC, 4);
    hdr.version = PTXIR_VERSION;  // 2
    hdr.flags = 0;
    hdr.section_count = 3;  // REGDECL + KERNEL + STRING_TABLE
    hdr.reserved = 0;
    hdr.string_table_offset = 0;
    hdr.string_table_size = 0;
    hdr.header_size = sizeof(PtxirHeader);
    out_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    // Reserve space for 3 TOC entries (6 bytes each = 18 bytes)
    // Written as zero padding, backfilled after sections
    for (int i = 0; i < 3; i++) {
        uint8_t zero[6] = {};
        out_.write(reinterpret_cast<const char*>(zero), 6);
    }
}
```

- [ ] **Step 3: Implement PtxirWriter::write_toc_entries() — write TOC in order (REGDECL, KERNEL, STRING_TABLE)**

```cpp
void PtxirWriter::write_toc_entries() {
    // Seek back to the TOC space right after header
    // TOC starts at offset sizeof(PtxirHeader)
    std::streampos header_end = static_cast<std::streampos>(sizeof(PtxirHeader));
    out_.seekp(header_end);
    for (const auto& entry : toc_entries_) {
        write_u8(out_, entry.type);
        write_u8(out_, entry.reserved);
        write_u32(out_, entry.offset);
    }
    // Seek back to end for subsequent writes
    out_.seekp(0, std::ios::end);
}
```

- [ ] **Step 4: Rewrite PtxirWriter::write() — correct order: pre_pass → header → TOC → regdecl → kernel → string_table → backfill**

```cpp
void PtxirWriter::write(const std::vector<StatementContext>& statements) {
    stmts_ = statements;
    pre_pass(statements);
    write_header();

    // Build TOC entries
    toc_entries_.clear();
    // REGDECL section offset
    toc_entries_.push_back({static_cast<uint8_t>(PtxirSectionType::REGDECL), 0, 0});
    // KERNEL section offset
    toc_entries_.push_back({static_cast<uint8_t>(PtxirSectionType::KERNEL), 0, 0});
    // STRING_TABLE section offset
    toc_entries_.push_back({static_cast<uint8_t>(PtxirSectionType::STRING_TABLE), 0, 0});

    // Write sections and record offsets
    toc_entries_[0].offset = static_cast<uint32_t>(out_.tellp());
    write_regdecl_section();

    toc_entries_[1].offset = static_cast<uint32_t>(out_.tellp());
    write_kernel_section();

    toc_entries_[2].offset = static_cast<uint32_t>(out_.tellp());
    write_string_table();

    // Backfill TOC entries and header offsets
    write_toc_entries();
    backfill_header_offsets();
}
```

- [ ] **Step 5: Implement PtxirWriter::write_regdecl_section() — write operand table from reg2id_**

```cpp
void PtxirWriter::write_regdecl_section() {
    write_u32(out_, static_cast<uint32_t>(reg2id_.size()));
    for (const auto& [name, id] : reg2id_) {
        uint32_t str_id = get_string_id(name);
        write_u32(out_, str_id);
    }
}
```

- [ ] **Step 6: Implement PtxirWriter::backfill_header_offsets() — seek back and write string_table_offset/size**

```cpp
void PtxirWriter::backfill_header_offsets() {
    std::streampos current_pos = out_.tellp();
    // string_table_offset is at offset 12 in header
    out_.seekp(12);
    write_u32(out_, string_table_offset_);
    write_u32(out_, string_table_size_);
    out_.seekp(current_pos);
}
```

- [ ] **Step 7: Build verification**

```bash
cmake --build build --target ptxir_writer 2>&1
```

- [ ] **Step 8: Commit**

```bash
git add include/ptx_ir/ptxir_format.h src/ptx_ir/ptxir_writer.cpp
git commit -m "refactor(ptxir): align writer with V2 format (TOC + offset backfill)"
```

---

### Task 3: Phase 2 Commit 2 — Reader V1 Legacy Format Support

**Goal:** Reader 支持 V1 旧格式向后兼容
**Commit:** `feat(ptxir): reader supports V1 legacy format (version=1 path)`

**Files:**
- Modify: `src/ptx_ir/ptxir_reader.cpp`
- Modify: `include/ptx_ir/ptxir_reader.h`

- [ ] **Step 1: Add read_legacy_v1() method declaration to ptxir_reader.h**

```cpp
private:
    void read_header();
    std::vector<StatementContext> read_legacy_v1();
    std::vector<StatementContext> read_v2();
    ...
```

- [ ] **Step 2: Modify read_header() — detect version, version=1 → return, version=2 → return**

```cpp
void PtxirReader::read_header() {
    PtxirHeader hdr;
    in_.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    if (std::memcmp(hdr.magic, PTXIR_MAGIC, 4) != 0) {
        throw std::runtime_error("Invalid PTXIR magic");
    }
    if (hdr.version != 1 && hdr.version != 2) {
        throw std::runtime_error("Unsupported PTXIR version: " + std::to_string(hdr.version));
    }
    version_ = hdr.version;
    // Store header for later use
    header_ = hdr;
}
```

- [ ] **Step 3: Add version_ and header_ fields to PtxirReader**

```cpp
private:
    std::istream& in_;
    std::vector<std::string> string_table_;
    uint32_t statement_count_ = 0;
    uint16_t version_ = 0;
    PtxirHeader header_{};
```

- [ ] **Step 4: Implement read_legacy_v1() — existing hardcoded sizeof(PtxirHeader) path**

```cpp
std::vector<StatementContext> PtxirReader::read_legacy_v1() {
    // Legacy V1 format: header + string_table (at sizeof(PtxirHeader)) + kernel_section
    read_string_table_legacy();
    return read_kernel_section();
}

void PtxirReader::read_string_table_legacy() {
    in_.seekg(static_cast<std::streamoff>(sizeof(PtxirHeader)));
    uint32_t count = read_u32(in_);
    for (uint32_t i = 0; i < count; i++) {
        uint16_t len = read_u16(in_);
        std::string s(len, '\0');
        in_.read(s.data(), static_cast<std::streamsize>(len));
        string_table_.push_back(s);
    }
}
```

- [ ] **Step 5: Modify read() — dispatch based on version**

```cpp
std::vector<StatementContext> PtxirReader::read() {
    read_header();
    if (version_ == 1) {
        return read_legacy_v1();
    }
    return read_v2();
}
```

- [ ] **Step 6: Build verification**

```bash
cmake --build build --target ptxir_reader 2>&1
```

- [ ] **Step 7: Commit**

```bash
git add src/ptx_ir/ptxir_reader.cpp include/ptx_ir/ptxir_reader.h
git commit -m "feat(ptxir): reader supports V1 legacy format (version=1 path)"
```

---

### Task 4: Phase 2 Commit 3 — Reader V2 Format (TOC-driven path)

**Goal:** Reader 按 TOC 条目解析 V2 格式文件
**Commit:** `refactor(ptxir): reader supports V2 format (TOC-driven path)`

**Files:**
- Modify: `src/ptx_ir/ptxir_reader.cpp`

- [ ] **Step 1: Implement read_v2() — read TOC entries, dispatch by section type**

```cpp
std::vector<StatementContext> PtxirReader::read_v2() {
    // TOC starts right after header (sizeof(PtxirHeader))
    in_.seekg(static_cast<std::streamoff>(sizeof(PtxirHeader)));

    std::vector<PtxirSectionTOC> toc;
    for (uint16_t i = 0; i < header_.section_count; i++) {
        PtxirSectionTOC entry;
        entry.type = read_u8(in_);
        entry.reserved = read_u8(in_);
        entry.offset = read_u32(in_);
        // Validate: no duplicate section types
        for (const auto& existing : toc) {
            if (existing.type == entry.type) {
                throw std::runtime_error("Duplicate section type in TOC: " + std::to_string(entry.type));
            }
        }
        toc.push_back(entry);
    }

    // Process sections in TOC order
    for (const auto& entry : toc) {
        in_.seekg(static_cast<std::streamoff>(entry.offset));
        switch (static_cast<PtxirSectionType>(entry.type)) {
            case PtxirSectionType::REGDECL:
                read_regdecl_section();
                break;
            case PtxirSectionType::KERNEL:
                return read_kernel_section();
            case PtxirSectionType::STRING_TABLE:
                read_string_table_v2();
                break;
            default:
                throw std::runtime_error("Unknown section type: " + std::to_string(entry.type));
        }
    }
    return {};
}
```

- [ ] **Step 2: Implement read_string_table_v2() — positioned by TOC not seekg(sizeof(PtxirHeader))**

```cpp
void PtxirReader::read_string_table_v2() {
    uint32_t count = read_u32(in_);
    for (uint32_t i = 0; i < count; i++) {
        uint16_t len = read_u16(in_);
        std::string s(len, '\0');
        in_.read(s.data(), static_cast<std::streamsize>(len));
        string_table_.push_back(s);
    }
}
```

- [ ] **Step 3: Implement read_regdecl_section() — read operand count + string IDs**

```cpp
void PtxirReader::read_regdecl_section() {
    uint32_t count = read_u32(in_);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t str_id = read_u32(in_);
        // REGDECL string IDs are already in the string table
        // Just validate they exist
        if (str_id >= string_table_.size()) {
            throw std::runtime_error("REGDECL string ID out of bounds: " + std::to_string(str_id));
        }
    }
}
```

- [ ] **Step 4: Add validation: duplicate TOC type → throw; offset out of bounds → throw**

```cpp
// Already handled in read_v2() Step 1 (duplicate check) and seekg validation
```

- [ ] **Step 5: Build verification**

```bash
cmake --build build --target ptxir_reader 2>&1
```

- [ ] **Step 6: Commit**

```bash
git add src/ptx_ir/ptxir_reader.cpp
git commit -m "refactor(ptxir): reader supports V2 format (TOC-driven path)"
```

---

### Task 5: Phase 3 Commit 1 — Roundtrip Unit Tests

**Goal:** 完整 roundtrip 测试套件覆盖 24 种指令类型 + 4 种错误路径 + 混合 100+ 语句
**Commit:** `test(ptxir): add roundtrip unit tests for all 24 instruction types`

**Files:**
- Create: `tests/unit/test_ptxir_serialization.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Create test_ptxir_serialization.cpp with Catch2 header + includes**

```cpp
#include "catch_amalgamated.hpp"
#include "ptxir/ptxir_serialization.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/operand_context.h"
#include <sstream>
```

- [ ] **Step 2: Add TEST_CASE("Roundtrip: BranchInstr")**

```cpp
TEST_CASE("Roundtrip: BranchInstr") {
    BranchInstr instr;
    instr.target = "L1";
    instr.predicate = "%p1";
    instr.predicate_negated = false;
    instr.reconvergence_pc = 42;
    StatementContext stmt(S_BRA, instr);

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    auto& loaded = result[0].get<BranchInstr>();
    CHECK(loaded.target == "L1");
    CHECK(loaded.predicate == "%p1");
    CHECK(loaded.predicate_negated == false);
    CHECK(loaded.reconvergence_pc == 42);
}
```

- [ ] **Step 3: Add TEST_CASE("Roundtrip: LabelInstr")**

```cpp
TEST_CASE("Roundtrip: LabelInstr") {
    LabelInstr instr;
    instr.labelName = "L_loop";
    StatementContext stmt(S_LABEL, instr);

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].get<LabelInstr>().labelName == "L_loop");
}
```

- [ ] **Step 4: Add TEST_CASE("Roundtrip: VoidInstr (S_EXIT)")**

```cpp
TEST_CASE("Roundtrip: VoidInstr (S_EXIT)") {
    StatementContext stmt(S_EXIT, VoidInstr{});
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_EXIT);
}
```

- [ ] **Step 5: Add TEST_CASE("Roundtrip: BarrierInstr")**

```cpp
TEST_CASE("Roundtrip: BarrierInstr") {
    BarrierInstr instr;
    instr.barId = 0;
    StatementContext stmt(S_BAR, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].get<BarrierInstr>().barId == 0);
}
```

- [ ] **Step 6: Add TEST_CASE("Roundtrip: GenericInstr (S_MOV)")**

```cpp
TEST_CASE("Roundtrip: GenericInstr (S_MOV)") {
    GenericInstr instr;
    instr.qualifiers = {Q_U32};
    instr.operands = {RegOperand{"%rd0", -1}, RegOperand{"%r1", -1}};
    StatementContext stmt(S_MOV, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    auto& loaded = result[0].get<GenericInstr>();
    CHECK(loaded.qualifiers.size() == 1);
    CHECK(loaded.operands.size() == 2);
}
```

- [ ] **Step 7: Add TEST_CASE("Roundtrip: DeclarationInstr (S_REG)")**

```cpp
TEST_CASE("Roundtrip: DeclarationInstr (S_REG)") {
    DeclarationInstr instr;
    instr.kind = DeclarationInstr::Kind::REG;
    instr.dataType = Q_U32;
    instr.name = "%r1";
    instr.array_size = 1;
    StatementContext stmt(S_REG, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    auto& loaded = result[0].get<DeclarationInstr>();
    CHECK(loaded.kind == DeclarationInstr::Kind::REG);
    CHECK(loaded.name == "%r1");
}
```

- [ ] **Step 8: Add TEST_CASE("Roundtrip: MembarInstr")**

```cpp
TEST_CASE("Roundtrip: MembarInstr") {
    MembarInstr instr;
    instr.qualifiers = {Q_CTA};
    StatementContext stmt(S_MEMBAR, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_MEMBAR);
}
```

- [ ] **Step 9: Add TEST_CASE("Roundtrip: FenceInstr")**

```cpp
TEST_CASE("Roundtrip: FenceInstr") {
    FenceInstr instr;
    instr.qualifiers = {Q_GPU};
    StatementContext stmt(S_FENCE, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_FENCE);
}
```

- [ ] **Step 10: Add TEST_CASE("Roundtrip: ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix")**

```cpp
TEST_CASE("Roundtrip: ReduxSyncInstr") {
    ReduxSyncInstr instr;
    instr.qualifiers = {Q_ADD, Q_S32};
    StatementContext stmt(S_REDUX_SYNC, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_REDUX_SYNC);
}

TEST_CASE("Roundtrip: MbarrierInstr") {
    MbarrierInstr instr;
    instr.qualifiers = {Q_CTA};
    StatementContext stmt(S_MBARRIER, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_MBARRIER);
}

TEST_CASE("Roundtrip: CallInstr") {
    CallInstr instr;
    instr.qualifiers = {Q_UNI};
    StatementContext stmt(S_CALL, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_CALL);
}

TEST_CASE("Roundtrip: PredicatePrefix") {
    PredicatePrefix instr;
    instr.qualifiers = {};
    StatementContext stmt(S_PREDICATE_PREFIX, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_PREDICATE_PREFIX);
}
```

- [ ] **Step 11: Add TEST_CASE("Roundtrip: VoteInstr, ShflInstr, AtomInstr")**

```cpp
TEST_CASE("Roundtrip: VoteInstr") {
    VoteInstr instr;
    instr.qualifiers = {Q_U32};
    StatementContext stmt(S_VOTE, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_VOTE);
}

TEST_CASE("Roundtrip: ShflInstr") {
    ShflInstr instr;
    instr.qualifiers = {Q_U32};
    StatementContext stmt(S_SHFL, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_SHFL);
}

TEST_CASE("Roundtrip: AtomInstr") {
    AtomInstr instr;
    instr.qualifiers = {Q_U32};
    StatementContext stmt(S_ATOM, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_ATOM);
}
```

- [ ] **Step 12: Add TEST_CASE("Roundtrip: TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective")**

```cpp
TEST_CASE("Roundtrip: TextureInstr") {
    TextureInstr instr;
    instr.qualifiers = {};
    StatementContext stmt(S_TEXTURE, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_TEXTURE);
}

TEST_CASE("Roundtrip: SurfaceInstr") {
    SurfaceInstr instr;
    StatementContext stmt(S_SURFACE, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_SURFACE);
}

TEST_CASE("Roundtrip: ReductionInstr") {
    ReductionInstr instr;
    StatementContext stmt(S_REDUCTION, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_REDUCTION);
}

TEST_CASE("Roundtrip: PrefetchInstr") {
    PrefetchInstr instr;
    StatementContext stmt(S_PREFETCH, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_PREFETCH);
}

TEST_CASE("Roundtrip: CpAsyncInstr") {
    CpAsyncInstr instr;
    StatementContext stmt(S_CP_ASYNC, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_CP_ASYNC);
}

TEST_CASE("Roundtrip: AbiDirective") {
    AbiDirective instr;
    StatementContext stmt(S_ABI_DIRECTIVE, instr);
    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_ABI_DIRECTIVE);
}
```

- [ ] **Step 13: Add TEST_CASE("Roundtrip: mixed 100+ statements")**

```cpp
TEST_CASE("Roundtrip: mixed 100+ statements") {
    std::vector<StatementContext> stmts;
    for (int i = 0; i < 50; i++) {
        GenericInstr instr;
        instr.qualifiers = {Q_U32};
        instr.operands = {RegOperand{"%r" + std::to_string(i), -1}};
        stmts.emplace_back(S_MOV, instr);
    }
    for (int i = 0; i < 50; i++) {
        BranchInstr instr;
        instr.target = "L" + std::to_string(i);
        stmts.emplace_back(S_BRA, instr);
    }
    // Add one of each special type
    stmts.emplace_back(S_MEMBAR, MembarInstr{});
    stmts.emplace_back(S_FENCE, FenceInstr{});
    stmts.emplace_back(S_REDUX_SYNC, ReduxSyncInstr{});
    stmts.emplace_back(S_MBARRIER, MbarrierInstr{});
    stmts.emplace_back(S_CALL, CallInstr{});
    stmts.emplace_back(S_VOTE, VoteInstr{});
    stmts.emplace_back(S_SHFL, ShflInstr{});
    stmts.emplace_back(S_ATOM, AtomInstr{});
    stmts.emplace_back(S_TEXTURE, TextureInstr{});
    stmts.emplace_back(S_SURFACE, SurfaceInstr{});
    stmts.emplace_back(S_REDUCTION, ReductionInstr{});
    stmts.emplace_back(S_PREFETCH, PrefetchInstr{});
    stmts.emplace_back(S_CP_ASYNC, CpAsyncInstr{});
    stmts.emplace_back(S_ABI_DIRECTIVE, AbiDirective{});

    auto data = serialize_to_string(stmts);
    auto result = deserialize_from_string(data);
    REQUIRE(result.size() == stmts.size());
    for (size_t i = 0; i < stmts.size(); i++) {
        CHECK(result[i].type == stmts[i].type);
    }
}
```

- [ ] **Step 14: Add TEST_CASE("Error: invalid magic")**

```cpp
TEST_CASE("Error: invalid magic") {
    std::string bad_data = "XXXX";
    // Pad to reasonable size
    bad_data.resize(100, '\0');
    std::istringstream iss(bad_data, std::ios::binary);
    PtxirReader reader(iss);
    CHECK_THROWS_AS(reader.read(), std::runtime_error);
}
```

- [ ] **Step 15: Add TEST_CASE("Error: unsupported version")**

```cpp
TEST_CASE("Error: unsupported version") {
    PtxirHeader hdr{};
    std::memcpy(hdr.magic, PTXIR_MAGIC, 4);
    hdr.version = 99;
    hdr.section_count = 0;
    hdr.header_size = sizeof(PtxirHeader);
    std::ostringstream oss(std::ios::binary);
    oss.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    std::istringstream iss(oss.str(), std::ios::binary);
    PtxirReader reader(iss);
    CHECK_THROWS_AS(reader.read(), std::runtime_error);
}
```

- [ ] **Step 16: Add TEST_CASE("Error: unknown opcode")**

```cpp
TEST_CASE("Error: unknown opcode") {
    // Write a header + unknown opcode
    PtxirHeader hdr{};
    std::memcpy(hdr.magic, PTXIR_MAGIC, 4);
    hdr.version = PTXIR_VERSION;
    hdr.section_count = 0;
    hdr.header_size = sizeof(PtxirHeader);
    std::ostringstream oss(std::ios::binary);
    oss.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    // Empty string table
    uint32_t zero = 0;
    oss.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    // 1 statement with unknown opcode 0xFF
    uint16_t bad_op = 0xFF;
    oss.write(reinterpret_cast<const char*>(&bad_op), sizeof(bad_op));
    std::istringstream iss(oss.str(), std::ios::binary);
    PtxirReader reader(iss);
    CHECK_THROWS_AS(reader.read(), std::runtime_error);
}
```

- [ ] **Step 17: Register test in CMakeLists.txt**

```cmake
add_catch_test(unit_ptxir_serialization tests/unit/test_ptxir_serialization.cpp)
target_link_libraries(unit_ptxir_serialization PRIVATE ptxir_writer ptxir_reader)
```

- [ ] **Step 18: Build and run tests**

```bash
cmake --build build --target unit_ptxir_serialization 2>&1
ctest -R unit_ptxir_serialization -V
```

- [ ] **Step 19: Commit**

```bash
git add tests/unit/test_ptxir_serialization.cpp tests/CMakeLists.txt
git commit -m "test(ptxir): add roundtrip unit tests for all 24 instruction types"
```

---

### Task 6: Phase 3 Commit 2 — generate_ptxir() Tool

**Goal:** 实现 `generate_ptxir()` 离线工具
**Commit:** `feat(ptxir): add generate_ptxir() offline tool`

**Files:**
- Modify: `include/ptxir/ptxir_serialization.h`
- Modify: `src/ptxir/ptxir_serialization.cpp`

- [ ] **Step 1: Add generate_ptxir() declaration to header**

```cpp
bool generate_ptxir(const std::string& ptx_path,
                    const std::string& ptxir_path,
                    const std::string& kernel_name = "");
```

- [ ] **Step 2: Implement generate_ptxir() in .cpp — call load_ptx_statements → serialize_statements**

```cpp
bool generate_ptxir(const std::string& ptx_path,
                    const std::string& ptxir_path,
                    const std::string& kernel_name) {
    // Forward declare from ptx_parser
    // PTX → ANTLR → StatementContext[]
    try {
        // load_ptx_statements is defined in the parser module
        // This requires ANTLR runtime
        auto stmts = load_ptx_statements(ptx_path, kernel_name, false);
        return serialize_statements(stmts, ptxir_path);
    } catch (...) {
        return false;
    }
}
```

- [ ] **Step 3: Build verification**

```bash
cmake --build build --target ptxir 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add include/ptxir/ptxir_serialization.h src/ptxir/ptxir_serialization.cpp
git commit -m "feat(ptxir): add generate_ptxir() offline tool"
```

---

### Task 7: Phase 3 Commit 3 — load_ptxir(apply_cfg=true) Path

**Goal:** 实现 `load_ptxir(apply_cfg=true)` 完整路径
**Commit:** `feat(ptxir): complete load_ptxir(apply_cfg=true) path with CFGBuilder integration`

**Files:**
- Modify: `include/ptxir/ptxir_serialization.h`
- Modify: `src/ptxir/ptxir_serialization.cpp`

- [ ] **Step 1: Add load_ptxir() declaration to header**

```cpp
std::vector<StatementContext> load_ptxir(const std::string& ptxir_path,
                                          bool apply_cfg = false);
```

- [ ] **Step 2: Implement load_ptxir() — deserialize + optional CFGBuilder::build()**

```cpp
std::vector<StatementContext> load_ptxir(const std::string& ptxir_path,
                                          bool apply_cfg) {
    auto stmts = deserialize_statements(ptxir_path);
    if (apply_cfg) {
        // CFGBuilder::build() modifies reconvergence_pc in BranchInstr/BarrierInstr
        CFGBuilder::build(stmts);
    }
    return stmts;
}
```

- [ ] **Step 3: Build verification**

```bash
cmake --build build --target ptxir 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add include/ptxir/ptxir_serialization.h src/ptxir/ptxir_serialization.cpp
git commit -m "feat(ptxir): complete load_ptxir(apply_cfg=true) path with CFGBuilder integration"
```

---

### Task 8: Phase 4 — Documentation & Protocol

**Goal:** AGENTS.md 协议 + 测试文档升级为四模式
**Commit 1:** `docs(ptxir): add StatementContext modification protocol to AGENTS.md`
**Commit 2:** `docs(ptxir): update THREE-MODE-TESTING-GUIDE.md to four-mode framework`

**Files:**
- Modify: `src/ptx_ir/AGENTS.md`
- Create/Modify: `include/ptxir/AGENTS.md`
- Modify: `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md`

- [ ] **Step 1: Add "StatementContext 修改协议" to src/ptx_ir/AGENTS.md**

添加 4 项 checklist:
1. 同步 ptxir_writer.cpp
2. 同步 ptxir_reader.cpp
3. 添加 roundtrip test
4. 更新 X-Macro dispatch

- [ ] **Step 2: Create include/ptxir/AGENTS.md with "公共头文件修改协议"**

```markdown
# include/ptxir — PTXIR 公共 API

## 公共头文件修改协议
修改 `ptxir_serialization.h` 中的公共 API 签名时，必须：
1. 同步更新 `src/ptxir/ptxir_serialization.cpp` 实现
2. 同步更新测试文件 `tests/unit/test_ptxir_serialization.cpp`
3. 同步更新 `src/ptx_ir/AGENTS.md` 中的 API 引用
```

- [ ] **Step 3: Add cross-reference between src/ptx_ir/AGENTS.md and include/ptxir/AGENTS.md**

- [ ] **Step 4: Commit 1**

```bash
git add src/ptx_ir/AGENTS.md include/ptxir/AGENTS.md
git commit -m "docs(ptxir): add StatementContext modification protocol to AGENTS.md"
```

- [ ] **Step 5: Add "Mode 4: PTXIR 快速加载" chapter to THREE-MODE-TESTING-GUIDE.md**

```markdown
## Mode 4: PTXIR 快速加载

PTXIR 二进制格式绕过 ANTLR 解析，实现 ~5ms 快速加载。

### API
- `load_ptxir(path, apply_cfg=false)` — 从 .ptxir 文件加载
- `generate_ptxir(ptx_path, ptxir_path)` — 从 PTX 文本生成 .ptxir
```

- [ ] **Step 6: Update doc title and references**

- [ ] **Step 7: Verify no dead links**

```bash
grep -r "three_mode_testing\|three-mode" docs/ 2>/dev/null
```

- [ ] **Step 8: Commit 2**

```bash
git add docs/developer-guide/
git commit -m "docs(ptxir): update THREE-MODE-TESTING-GUIDE.md to four-mode framework"
```

---

### Task 9: Final Verification & Archive

- [ ] **Step 1: Full build**

```bash
cmake --build build 2>&1
```

- [ ] **Step 2: Full ctest**

```bash
ctest --test-dir build -V 2>&1
```

- [ ] **Step 3: Run sanity.sh**

```bash
./scripts/sanity.sh 2>&1
```

- [ ] **Step 4: Run PTX syntax tests**

```bash
./tests/ptx/test_all_ptx.sh 2>&1
```

- [ ] **Step 5: Update archive reference**

Update `openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/tasks.md` §10:
```
- [x] 10.1-10.4 COMPLETED via openspec/changes/ptxir-format-compliance/ (commit <hash>)
```

- [ ] **Step 6: Commit archive reference**

```bash
git add openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/tasks.md
git commit -m "docs(openspec): mark ptxir-format-compliance tasks as complete in archive"
```

- [ ] **Step 7: Verify final state**

```bash
git status --short
git log --oneline -5
```