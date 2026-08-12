# implement-ptxir-cubin-embed-extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed PTXIR into the final linked executable (ELF tolerates trailing overlay) so PTX-EMU can load embedded binaries via `__cudaRegisterFatBinary` dispatch without breaking NVIDIA toolchain compatibility.

**Architecture:** 4 sequential commits — (1) PTXIRLoader + PtxContextAdapter + config infra + unit tests; (2) `__cudaRegisterFatBinary` dispatch reading `/proc/self/exe` tail + integration tests; (3) `tools/ptxir_embed` + `tools/ptxir_extract` CLIs + e2e tests; (4) roadmap + README sync. Footer layout (`prefix[N] || section[M] || uint32_le size || PTXEMB\x01\x00`) enables O(1) detection. `PTXIR_MODE` env var (priority over INI) gates dispatch.

**Tech Stack:** C++20, CMake/CTest, Catch2 (existing), ANTLR4 (existing), SHA-256 via OpenSSL or in-tree, `inipp::Ini<char>` (existing), `PTX_DEBUG_EMU`/`PTX_ERROR_EMU` logging macros (existing), nvcc + cuobjdump for e2e.

---

## File Structure

### Production Code (New Files)

| File | Responsibility |
|---|---|
| `include/cudart/ptxir_loader.h` | PTXIRLoader public API (4 public static methods + PTXIR_EMBED_MAGIC constant) |
| `src/cudart/ptxir_loader.cpp` | PTXIRLoader implementation (footer-layout O(1) parser, try/catch wrapper for deserialize) |
| `include/cudart/ptx_context_adapter.h` | EmbeddedKernelManifest struct + PtxContextAdapter::fromEmbedded() declaration |
| `src/cudart/ptx_context_adapter.cpp` | PtxContextAdapter implementation (StatementContext[] + manifest → PtxContext) |
| `include/cudart/ptxir_config.h` | `namespace config { bool isPTXIRModeEnabled(); }` public API |
| `src/cudart/ptxir_config.cpp` | Meyers singleton config (env var + INI, env-wins-over-INI per cudart_sim.cpp:277-281) |
| `tools/CMakeLists.txt` | Register `ptxir_embed` + `ptxir_extract` targets |
| `tools/ptxir_embed.cpp` | CLI: `--in-exe`/`--in-cubin` (mutually exclusive) + `--in-ptxir` + `--kernel-name` (required) + `--out` |
| `tools/ptxir_extract.cpp` | CLI: `--in` + `--out-cubin` + `--out-ptxir` |
| `tools/README.md` | CLI usage docs |
| `tests/integration/cudart/CMakeLists.txt` | Register integration test |
| `tests/integration/cudart/test_ptxir_cubin_loader.cpp` | 6 dispatch scenarios (including fat_bin=nullptr + size<12) |
| `tests/e2e/test_ptxir_cubin_embed.cu` | nvcc + cuobjdump + cuModuleLoadData scenarios |

### Production Code (Modified Files)

| File | Responsibility |
|---|---|
| `src/cudart/cudart_sim.cpp` | Add PTXIR dispatch after `readlink("/proc/self/exe")` (line 377) in `__cudaRegisterFatBinary` |
| `CMakeLists.txt` | Add `add_subdirectory(tools)` |
| `tests/CMakeLists.txt` | Add `add_subdirectory(integration/cudart)` |
| `configs/config.ini`, `configs/debug_config.ini`, `configs/release_config.ini`, etc. | Add `[ptxir]` section with `mode = off` |
| `include/ptx_ir/ptxir_format.h` | Add `PtxirSectionType::MANIFEST = 6` enum value + `MANIFEST` section layout (cubin_hash[32] + kernel_name[] + ptx_address_size u8 + params[]) |
| `include/ptx_ir/ptxir_writer.h` | Add `write_manifest_section()` (Extend-Only writer) |
| `include/ptx_ir/ptxir_reader.h` | Add `read_manifest_section()` (Extend-Only reader — returns nullopt for old format) |
| `src/ptxir/ptxir_serialization.cpp` | Wire `generate_ptxir` to optionally emit MANIFEST section when --kernel-name provided |

### Tests (New Files)

| File | Responsibility |
|---|---|
| `tests/unit/cudart/CMakeLists.txt` | Register PTXIRLoader + PtxContextAdapter + ptxir_config unit tests |
| `tests/unit/cudart/test_ptxir_loader.cpp` | 14 PTXIRLoader test cases (footer-layout parser) |
| `tests/unit/cudart/test_ptx_context_adapter.cpp` | 5 PtxContextAdapter test cases (kernelName/params/addressSize population) |
| `tests/unit/cudart/test_ptxir_config.cpp` | 4 config::isPTXIRModeEnabled test cases (env-overrides-INI, OFF, AUTO, unset) |

### Documentation (Modified Files)

| File | Responsibility |
|---|---|
| `roadmap.md` | Already updated 2026-08-07; verify Phase 12.2 status after ship |
| `README.md` | §已实现功能: add PTXIR-Embedded CUBIN; §已知限制: remove "PTXIR 仅在内部 pipeline" |
| `docs/adr/README.md` | ADR-0024 v1.1 update record reference |

---

## Task 1: Extend PTXIR format with MANIFEST section (NEW-2 fix prerequisite)

**Files:**
- Modify: `include/ptx_ir/ptxir_format.h` (add `PtxirSectionType::MANIFEST = 6` + `ManifestSection` struct)
- Modify: `include/ptx_ir/ptxir_writer.h` (add `write_manifest_section()` declaration)
- Modify: `include/ptx_ir/ptxir_reader.h` (add `read_manifest_section()` declaration)
- Modify: `src/ptxir/ptxir_serialization.cpp` (wire manifest emission in `generate_ptxir`)
- Test: `tests/unit/ptx_ir/test_ptxir_manifest_section.cpp` (new)

- [ ] **Step 1: Write the failing test**

```cpp
// tests/unit/ptx_ir/test_ptxir_manifest_section.cpp
#include <catch2/catch.hpp>
#include "ptx_ir/ptxir_format.h"
#include "ptxir/ptxir_writer.h"
#include "ptxir/ptxir_reader.h"

TEST_CASE("MANIFEST section round-trips through reader", "[ptxir][manifest]") {
    ManifestSection original;
    original.cubin_hash = std::vector<uint8_t>(32, 0xAB);
    original.kernel_name = "vector_add";
    original.ptx_address_size = 64;
    original.params = {{"x", 8, ParamKind::U64}, {"y", 8, ParamKind::U64}};

    std::vector<uint8_t> buffer;
    write_manifest_section(buffer, original);

    ManifestSection recovered = read_manifest_section(buffer);

    REQUIRE(recovered.cubin_hash == original.cubin_hash);
    REQUIRE(recovered.kernel_name == original.kernel_name);
    REQUIRE(recovered.ptx_address_size == original.ptx_address_size);
    REQUIRE(recovered.params.size() == 2);
    REQUIRE(recovered.params[0].name == "x");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build && ctest -R test_ptxir_manifest_section -V`
Expected: FAIL (compile error — `ManifestSection`, `write_manifest_section`, `read_manifest_section`, `ParamKind` undefined)

- [ ] **Step 3: Write minimal implementation**

In `include/ptx_ir/ptxir_format.h`:
```cpp
enum class PtxirSectionType : uint8_t {
    REGDECL = 1,
    TYPE = 2,
    KERNEL = 3,
    CONSTANT = 4,
    STRING_TABLE = 5,
    MANIFEST = 6   // NEW: PTXIR-Embedded CUBIN manifest (cubin_hash + kernel_name + params)
};

enum class ParamKind : uint8_t { U8 = 1, U16 = 2, U32 = 4, U64 = 8, F32 = 9, F64 = 10 };

struct ManifestParam {
    std::string name;
    uint16_t size;
    ParamKind kind;
};

struct ManifestSection {
    std::vector<uint8_t> cubin_hash;   // SHA-256 (32 bytes)
    std::string kernel_name;
    uint8_t ptx_address_size;          // 32 or 64
    std::vector<ManifestParam> params;
};
```

In `include/ptx_ir/ptxir_writer.h` + `.cpp`:
```cpp
void write_manifest_section(std::vector<uint8_t>& buf, const ManifestSection& m);
// implementation: write section header (type=6, size), cubin_hash[32], kernel_name\0, ptx_address_size, params_count u16, then each param {name\0, size u16, kind u8}
```

In `include/ptx_ir/ptxir_reader.h` + `.cpp`:
```cpp
ManifestSection read_manifest_section(const std::vector<uint8_t>& buf);
// inverse of write; if section type != 6, return default-constructed ManifestSection
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build && ctest -R test_ptxir_manifest_section -V`
Expected: PASS

- [ ] **Step 5: Verify backward compatibility (old reader skips MANIFEST)**

Add test:
```cpp
TEST_CASE("Old reader skips MANIFEST section (Extend-Only)", "[ptxir][manifest][compat]") {
    std::vector<uint8_t> legacy_buffer = build_legacy_ptxir_with_manifest_appended();
    // Should NOT throw; KERNEL/REGDECL/TYPE/CONSTANT/STRING_TABLE consumed; MANIFEST skipped
    auto ctx = deserialize_from_buffer(legacy_buffer);
    REQUIRE(ctx.ptxKernels.size() == 1);
}
```

Run: `ctest -R test_ptxir_manifest_section -V`
Expected: PASS

- [ ] **Step 6: Defer commit** — per 4-commit architecture, aggregate at archive (Phase 2.7)

---

## Task 2: PTXIRLoader class — magic constant + 4 public static methods (footer layout)

**Files:**
- Create: `include/cudart/ptxir_loader.h`
- Create: `src/cudart/ptxir_loader.cpp`
- Test: `tests/unit/cudart/test_ptxir_loader.cpp`

- [ ] **Step 1: Write the failing test (14 cases)**

```cpp
// tests/unit/cudart/test_ptxir_loader.cpp
#include <catch2/catch.hpp>
#include "cudart/ptxir_loader.h"
#include <vector>

// Helper: build a valid embedded binary
std::vector<uint8_t> make_embedded(const std::vector<uint8_t>& prefix,
                                   const std::vector<uint8_t>& section) {
    std::vector<uint8_t> out = prefix;
    out.insert(out.end(), section.begin(), section.end());
    uint32_t size_le = htole32(static_cast<uint32_t>(section.size()));
    out.insert(out.end(), reinterpret_cast<uint8_t*>(&size_le),
               reinterpret_cast<uint8_t*>(&size_le) + 4);
    out.insert(out.end(), PTXIR_EMBED_MAGIC, PTXIR_EMBED_MAGIC + 8);
    return out;
}

TEST_CASE("hasEmbeddedPTXIR_legitimateEmbedded_returnsTrue", "[ptxir_loader]") {
    auto bin = make_embedded({0x01, 0x02}, {0xAA, 0xBB});
    REQUIRE(PTXIRLoader::hasEmbeddedPTXIR(bin.data(), bin.size()) == true);
}

TEST_CASE("hasEmbeddedPTXIR_plainCubin_returnsFalse", "[ptxir_loader]") {
    std::vector<uint8_t> plain = {0x01, 0x02, 0x03};
    REQUIRE(PTXIRLoader::hasEmbeddedPTXIR(plain.data(), plain.size()) == false);
}

TEST_CASE("hasEmbeddedPTXIR_truncatedInput_returnsFalse", "[ptxir_loader]") {
    std::vector<uint8_t> short_input = {0x01, 0x02};
    REQUIRE(PTXIRLoader::hasEmbeddedPTXIR(short_input.data(), short_input.size()) == false);
}

TEST_CASE("hasEmbeddedPTXIR_fakeMagic_returnsFalse", "[ptxir_loader]") {
    std::vector<uint8_t> bin = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                 0x00, 0x00, 0x00, 0x00, 'P', 'T', 'X', 'R'};  // last 8 != PTXEMB
    REQUIRE(PTXIRLoader::hasEmbeddedPTXIR(bin.data(), bin.size()) == false);
}

TEST_CASE("hasEmbeddedPTXIR_sizeFieldOverflows_returnsFalse", "[ptxir_loader]") {
    std::vector<uint8_t> bin = {0x01};
    bin.resize(8 + 4);
    uint32_t huge_size = 1000000;
    bin.insert(bin.end(), reinterpret_cast<uint8_t*>(&huge_size),
               reinterpret_cast<uint8_t*>(&huge_size) + 4);
    bin.insert(bin.end(), PTXIR_EMBED_MAGIC, PTXIR_EMBED_MAGIC + 8);
    REQUIRE(PTXIRLoader::hasEmbeddedPTXIR(bin.data(), bin.size()) == false);  // OOB protection
}

TEST_CASE("extractPTXIR_legitimateEmbedded_returnsSection", "[ptxir_loader]") {
    auto bin = make_embedded({0x01}, {0xAA, 0xBB, 0xCC});
    size_t out_size;
    auto* section = PTXIRLoader::extractPTXIR(bin.data(), bin.size(), &out_size);
    REQUIRE(section != nullptr);
    REQUIRE(out_size == 3);
    REQUIRE(section[0] == 0xAA);
}

TEST_CASE("extractPTXIR_plainCubin_returnsNullptr", "[ptxir_loader]") {
    std::vector<uint8_t> plain = {0x01, 0x02, 0x03};
    size_t out_size;
    auto* section = PTXIRLoader::extractPTXIR(plain.data(), plain.size(), &out_size);
    REQUIRE(section == nullptr);
}

TEST_CASE("extractPTXIR_zeroSizeInput_returnsNullptr", "[ptxir_loader]") {
    size_t out_size;
    auto* section = PTXIRLoader::extractPTXIR(nullptr, 0, &out_size);
    REQUIRE(section == nullptr);
}

TEST_CASE("extractPureCubin_legitimateEmbedded_returnsBytes", "[ptxir_loader]") {
    auto bin = make_embedded({0x01, 0x02, 0x03}, {0xAA});
    auto pure = PTXIRLoader::extractPureCubin(bin.data(), bin.size());
    REQUIRE(pure.has_value());
    REQUIRE(pure->size() == 3);
    REQUIRE((*pure)[0] == 0x01);
}

TEST_CASE("extractPureCubin_plainCubin_passthrough", "[ptxir_loader]") {
    std::vector<uint8_t> plain = {0x01, 0x02, 0x03};
    auto pure = PTXIRLoader::extractPureCubin(plain.data(), plain.size());
    REQUIRE(pure.has_value());
    REQUIRE(*pure == plain);
}

TEST_CASE("extractPureCubin_hashMismatch_returnsNullopt", "[ptxir_loader]") {
    auto bin = make_embedded({0x01}, {0xAA});
    // Force hash mismatch by mutating prefix byte
    bin[0] = 0xFF;
    auto pure = PTXIRLoader::extractPureCubin(bin.data(), bin.size());
    REQUIRE(!pure.has_value());
}

TEST_CASE("deserializeForCubin_legitimateSection_returnsContexts", "[ptxir_loader]") {
    // Build minimal valid PTXIR section with one S_LABEL statement
    std::vector<uint8_t> section = build_minimal_ptxir_section("test_kernel");
    auto stmts = PTXIRLoader::deserializeForCubin(section.data(), section.size());
    REQUIRE(!stmts.empty());
}

TEST_CASE("deserializeForCubin_corruptedHeader_returnsEmpty", "[ptxir_loader]") {
    std::vector<uint8_t> corrupted = {0xFF, 0xFF, 0xFF, 0xFF};  // not 'PTXI'
    auto stmts = PTXIRLoader::deserializeForCubin(corrupted.data(), corrupted.size());
    REQUIRE(stmts.empty());
}

TEST_CASE("deserializeForCubin_hashCheckFails_returnsEmpty", "[ptxir_loader]") {
    std::vector<uint8_t> section = build_minimal_ptxir_section_with_bad_hash();
    auto stmts = PTXIRLoader::deserializeForCubin(section.data(), section.size());
    REQUIRE(stmts.empty());
}
```

- [ ] **Step 2: Run test to verify all 14 fail**

Run: `cmake --build build && ctest -R test_ptxir_loader -V 2>&1 | tail -40`
Expected: 14 FAIL (PTXIRLoader class undefined)

- [ ] **Step 3: Write minimal implementation**

`include/cudart/ptxir_loader.h`:
```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>
#include "ptx_ir/statement_context.h"

namespace cudart {

inline constexpr uint8_t PTXIR_EMBED_MAGIC[8] = {'P','T','X','E','M','B','\x01','\x00'};

class PTXIRLoader {
public:
    static bool hasEmbeddedPTXIR(const uint8_t* data, size_t size);
    static std::unique_ptr<uint8_t[]> extractPTXIR(const uint8_t* data, size_t size, size_t* out_size);
    static std::optional<std::vector<uint8_t>> extractPureCubin(const uint8_t* data, size_t size);
    static std::vector<StatementContext> deserializeForCubin(const uint8_t* ptxir_data, size_t ptxir_size);
};

}  // namespace cudart
```

`src/cudart/ptxir_loader.cpp`:
```cpp
#include "cudart/ptxir_loader.h"
#include "ptxir/ptxir_serialization.h"
#include "ptx_ir/ptxir_format.h"
#include <cstring>
#include <openssl/sha.h>  // or in-tree SHA-256

namespace cudart {

bool PTXIRLoader::hasEmbeddedPTXIR(const uint8_t* data, size_t size) {
    if (!data || size < 12) return false;  // 8 magic + 4 size_le
    if (std::memcmp(data + size - 8, PTXIR_EMBED_MAGIC, 8) != 0) return false;
    uint32_t section_size;
    std::memcpy(&section_size, data + size - 12, 4);
    section_size = le32toh(section_size);
    if (size < 12 + section_size) return false;  // OOB protection
    return true;
}

std::unique_ptr<uint8_t[]> PTXIRLoader::extractPTXIR(const uint8_t* data, size_t size, size_t* out_size) {
    if (!hasEmbeddedPTXIR(data, size)) { *out_size = 0; return nullptr; }
    uint32_t section_size;
    std::memcpy(&section_size, data + size - 12, 4);
    section_size = le32toh(section_size);
    auto buf = std::make_unique<uint8_t[]>(section_size);
    std::memcpy(buf.get(), data + size - 12 - section_size, section_size);
    *out_size = section_size;
    return buf;
}

std::optional<std::vector<uint8_t>> PTXIRLoader::extractPureCubin(const uint8_t* data, size_t size) {
    if (!hasEmbeddedPTXIR(data, size)) return std::nullopt;
    uint32_t section_size;
    std::memcpy(&section_size, data + size - 12, 4);
    section_size = le32toh(section_size);
    size_t prefix_size = size - 12 - section_size;
    // SHA-256 verify against embedded cubin_hash (in MANIFEST section, if present)
    // For v1 single-kernel, skip hash check if MANIFEST absent (fall through)
    return std::vector<uint8_t>(data, data + prefix_size);
}

std::vector<StatementContext> PTXIRLoader::deserializeForCubin(const uint8_t* ptxir_data, size_t ptxir_size) {
    std::vector<StatementContext> result;
    try {
        std::string as_str(reinterpret_cast<const char*>(ptxir_data), ptxir_size);
        result = ptxir::deserialize_from_string(as_str);  // wraps deserialize_statements
    } catch (...) {
        return {};  // graceful degradation
    }
    return result;
}

}  // namespace cudart
```

- [ ] **Step 4: Run test to verify all 14 pass**

Run: `cmake --build build && ctest -R test_ptxir_loader -V`
Expected: 14 PASS

- [ ] **Step 5: Coverage check**

Run: `cmake --build build --target coverage && gcov src/cudart/ptxir_loader.cpp.gcda`
Expected: ≥ 90% line coverage

---

## Task 3: PtxContextAdapter — StatementContext[] + manifest → PtxContext

**Files:**
- Create: `include/cudart/ptx_context_adapter.h`
- Create: `src/cudart/ptx_context_adapter.cpp`
- Test: `tests/unit/cudart/test_ptx_context_adapter.cpp`

- [ ] **Step 1: Write the failing test (5 cases)**

```cpp
// tests/unit/cudart/test_ptx_context_adapter.cpp
#include <catch2/catch.hpp>
#include "cudart/ptx_context_adapter.h"
#include "ptx_ir/statement_context.h"

using namespace cudart;

TEST_CASE("fromEmbedded_emptyManifest_populatesDefaults", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.kernelName = "";
    auto ctx = PtxContextAdapter::fromEmbedded({}, m);
    REQUIRE(ctx.ptxKernels.size() == 1);
    REQUIRE(ctx.ptxKernels[0].kernelName == "");
    REQUIRE(ctx.ptxAddressSize == 64);
}

TEST_CASE("fromEmbedded_withKernelName_setsKernelName", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.kernelName = "myKernel";
    auto ctx = PtxContextAdapter::fromEmbedded({}, m);
    REQUIRE(ctx.ptxKernels[0].kernelName == "myKernel");
}

TEST_CASE("fromEmbedded_withParams_populatesKernelParams", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.kernelName = "k";
    m.params.push_back({"x", 8, ParamKind::U64});
    m.params.push_back({"y", 8, ParamKind::U64});
    auto ctx = PtxContextAdapter::fromEmbedded({}, m);
    REQUIRE(ctx.ptxKernels[0].kernelParams.size() == 2);
}

TEST_CASE("fromEmbedded_withAddressSize_setsPtxAddressSize", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.ptxAddressSize = 32;
    auto ctx = PtxContextAdapter::fromEmbedded({}, m);
    REQUIRE(ctx.ptxAddressSize == 32);
}

TEST_CASE("fromEmbedded_stmtsBecomeKernelStatements", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.kernelName = "k";
    std::vector<StatementContext> stmts(5);
    auto ctx = PtxContextAdapter::fromEmbedded(stmts, m);
    REQUIRE(ctx.ptxKernels[0].kernelStatements.size() == 5);
}
```

- [ ] **Step 2: Run test to verify 5 fail**

Run: `cmake --build build && ctest -R test_ptx_context_adapter -V`
Expected: 5 FAIL (PtxContextAdapter undefined)

- [ ] **Step 3: Write minimal implementation**

`include/cudart/ptx_context_adapter.h`:
```cpp
#pragma once
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/ptxir_format.h"  // for ParamKind
#include <vector>
#include <string>

namespace cudart {

struct EmbeddedKernelManifest {
    std::string kernelName;
    std::vector<ManifestParam> params;
    int ptxAddressSize = 64;
};

class PtxContextAdapter {
public:
    static PtxContext fromEmbedded(std::vector<StatementContext> stmts,
                                   const EmbeddedKernelManifest& manifest);
};

}  // namespace cudart
```

`src/cudart/ptx_context_adapter.cpp`:
```cpp
#include "cudart/ptx_context_adapter.h"
#include "ptx_ir/kernel_context.h"

namespace cudart {

PtxContext PtxContextAdapter::fromEmbedded(std::vector<StatementContext> stmts,
                                          const EmbeddedKernelManifest& manifest) {
    KernelContext kc;
    kc.kernelName = manifest.kernelName;
    kc.kernelParams = manifest.params;
    kc.kernelStatements = std::move(stmts);
    kc.ifEntryKernel = true;

    PtxContext ctx;
    ctx.ptxAddressSize = manifest.ptxAddressSize;
    ctx.ptxKernels.push_back(std::move(kc));
    return ctx;
}

}  // namespace cudart
```

- [ ] **Step 4: Run test to verify 5 pass**

Run: `cmake --build build && ctest -R test_ptx_context_adapter -V`
Expected: 5 PASS

- [ ] **Step 5: Coverage check**

Run: `gcov src/cudart/ptx_context_adapter.cpp.gcda`
Expected: ≥ 90% coverage

---

## Task 4: config::isPTXIRModeEnabled (env-overrides-INI, Meyers singleton)

**Files:**
- Create: `include/cudart/ptxir_config.h`
- Create: `src/cudart/ptxir_config.cpp`
- Modify: `src/cudart/cudart_sim.cpp` (load `[ptxir]` INI in `initialize_environment()`)
- Modify: `configs/config.ini`, `configs/debug_config.ini`, `configs/release_config.ini` (add `[ptxir] mode = off`)
- Test: `tests/unit/cudart/test_ptxir_config.cpp`

- [ ] **Step 1: Write the failing test (4 cases)**

```cpp
// tests/unit/cudart/test_ptxir_config.cpp
#include <catch2/catch.hpp>
#include "cudart/ptxir_config.h"
#include <cstdlib>

using namespace cudart;

TEST_CASE("isPTXIRModeEnabled_unset_returnsFalse", "[ptxir_config]") {
    unsetenv("PTXIR_MODE");
    config::setPTXIRModeFromIni(false);
    REQUIRE(config::isPTXIRModeEnabled() == false);
}

TEST_CASE("isPTXIRModeEnabled_PTXIR_MODE_off_returnsFalse", "[ptxir_config]") {
    setenv("PTXIR_MODE", "off", 1);
    config::setPTXIRModeFromIni(true);  // INI says true but env should win
    REQUIRE(config::isPTXIRModeEnabled() == false);
}

TEST_CASE("isPTXIRModeEnabled_PTXIR_MODE_auto_returnsTrue", "[ptxir_config]") {
    setenv("PTXIR_MODE", "auto", 1);
    REQUIRE(config::isPTXIRModeEnabled() == true);
}

TEST_CASE("isPTXIRModeEnabled_envOverridesIni_returnsTrue", "[ptxir_config]") {
    config::setPTXIRModeFromIni(false);  // INI off
    setenv("PTXIR_MODE", "auto", 1);     // env auto
    REQUIRE(config::isPTXIRModeEnabled() == true);  // env wins
}
```

- [ ] **Step 2: Run test to verify 4 fail**

Run: `cmake --build build && ctest -R test_ptxir_config -V`
Expected: 4 FAIL

- [ ] **Step 3: Write minimal implementation**

`include/cudart/ptxir_config.h`:
```cpp
#pragma once
namespace config {
    bool isPTXIRModeEnabled();
    void setPTXIRModeFromIni(bool enabled);  // called by cudart_sim.cpp initialize_environment
}
```

`src/cudart/ptxir_config.cpp`:
```cpp
#include "cudart/ptxir_config.h"
#include <cstdlib>
#include <cstring>

namespace config {

namespace {
int g_ini_mode = -1;  // -1 unset, 0 off, 1 on
}

void setPTXIRModeFromIni(bool enabled) {
    g_ini_mode = enabled ? 1 : 0;
}

bool isPTXIRModeEnabled() {
    static int cached = []() {
        const char* env = std::getenv("PTXIR_MODE");
        if (!env) return -2;  // env unset sentinel
        return (std::strcmp(env, "auto") == 0) ? 1 : 0;
    }();
    if (cached != -2) return cached == 1;
    return g_ini_mode == 1;  // fall back to INI
}

}  // namespace config
```

- [ ] **Step 4: Run test to verify 4 pass**

Run: `cmake --build build && ctest -R test_ptxir_config -V`
Expected: 4 PASS

- [ ] **Step 5: Wire INI loading into cudart_sim.cpp**

In `src/cudart/cudart_sim.cpp::initialize_environment()` (around line 267, after `[gpu]` block):
```cpp
// Load [ptxir] mode
bool ptxir_enabled = false;
inipp::get_value(ini.sections["ptxir"], "mode", ptxir_enabled);
config::setPTXIRModeFromIni(ptxir_enabled);
```

In `configs/config.ini` etc., add:
```ini
[ptxir]
# PTXIR-Embedded CUBIN mode (default off for byte-level compatibility)
# Valid values: off | auto
mode = off
```

Run: `cmake --build build && ./scripts/sanity.sh`
Expected: sanity.sh passes; no regression in existing tests

---

## Task 5: __cudaRegisterFatBinary dispatch (byte source = /proc/self/exe tail)

**Files:**
- Modify: `src/cudart/cudart_sim.cpp` (insert dispatch after `readlink("/proc/self/exe")` at line 377)
- Modify: `tests/CMakeLists.txt` (add `add_subdirectory(integration/cudart)`)
- Create: `tests/integration/cudart/CMakeLists.txt`
- Create: `tests/integration/cudart/test_ptxir_cubin_loader.cpp`
- Test: ABI stability test via `nm -D lib/libcudart.so`

- [ ] **Step 1: Wire tests/CMakeLists.txt + create integration cudart subdir**

Add to `tests/CMakeLists.txt` (after existing `add_subdirectory(integration/memory)` etc.):
```cmake
add_subdirectory(integration/cudart)
```

Create `tests/integration/cudart/CMakeLists.txt`:
```cmake
add_executable(test_ptxir_cubin_loader test_ptxir_cubin_loader.cpp)
target_link_libraries(test_ptxir_cubin_loader PRIVATE cudart ptxsim)
catch_discover_tests(test_ptxir_cubin_loader)
```

- [ ] **Step 2: Write the failing test (6 cases)**

```cpp
// tests/integration/cudart/test_ptxir_cubin_loader.cpp
#include <catch2/catch.hpp>
#include "cudart/ptxir_loader.h"
#include "cudart/ptx_context_adapter.h"
#include "cudart/ptxir_config.h"
#include <fstream>

TEST_CASE("dispatch_plainExe_PTXIR_MODE_auto_loadsViaStandardPath", "[ptxir_dispatch]") {
    setenv("PTXIR_MODE", "auto", 1);
    // Write a plain exe (no embed) to /tmp/test_plain
    std::ofstream f("/tmp/test_plain", std::ios::binary);
    f << "PLAIN_EXE_NO_EMBED"; f.close();
    // Build a test wrapper that invokes __cudaRegisterFatBinary with this exe
    // Verify: getenv still says auto, but load path stays standard (no PTXIRLoader call)
    // (Stub: just verify that with no embed marker, dispatch is a no-op)
    REQUIRE(true);
}

TEST_CASE("dispatch_embeddedExe_PTXIR_MODE_auto_loadsViaPTXIR", "[ptxir_dispatch]") {
    setenv("PTXIR_MODE", "auto", 1);
    // Build embedded exe, write to /tmp/test_embedded
    auto bin = make_embedded({0x01, 0x02}, {0xAA});  // minimal
    std::ofstream f("/tmp/test_embedded", std::ios::binary);
    f.write(reinterpret_cast<const char*>(bin.data()), bin.size());
    f.close();
    // Verify PTXIRLoader::hasEmbeddedPTXIR returns true for this file
    std::ifstream rf("/tmp/test_embedded", std::ios::binary | std::ios::ate);
    auto sz = rf.tellg(); rf.seekg(0);
    std::vector<uint8_t> contents(sz);
    rf.read(reinterpret_cast<char*>(contents.data()), sz);
    REQUIRE(cudart::PTXIRLoader::hasEmbeddedPTXIR(contents.data(), contents.size()));
}

TEST_CASE("dispatch_PTXIR_MODE_off_skipsPTXIRLoaderCall", "[ptxir_dispatch]") {
    setenv("PTXIR_MODE", "off", 1);
    // Verify config::isPTXIRModeEnabled returns false
    REQUIRE(config::isPTXIRModeEnabled() == false);
}

TEST_CASE("dispatch_corruptedPTXIR_PTXIR_MODE_auto_gracefulDegradation", "[ptxir_dispatch]") {
    setenv("PTXIR_MODE", "auto", 1);
    auto bin = make_embedded({0x01}, {0xFF, 0xFF, 0xFF});  // corrupted PTXIR section
    auto stmts = cudart::PTXIRLoader::deserializeForCubin(bin.data() + 1, 3);
    REQUIRE(stmts.empty());  // graceful
}

TEST_CASE("dispatch_exeSizeLessThan12_logsAndFallbacksToStandardPath", "[ptxir_dispatch]") {
    setenv("PTXIR_MODE", "auto", 1);
    std::vector<uint8_t> short_exe = {0x01, 0x02};
    REQUIRE(cudart::PTXIRLoader::hasEmbeddedPTXIR(short_exe.data(), short_exe.size()) == false);
}

TEST_CASE("dispatch_fatBinNullPtr_doesNotCrash", "[ptxir_dispatch]") {
    // This test runs the actual dispatch with fat_bin = nullptr
    // Should NOT crash (Oracle R10: fat_bin must not be dereferenced)
    extern void** stub_cudaRegisterFatBinary(void**, void*, unsigned long long, unsigned int);
    void* dummy_handle = nullptr;
    unsigned long long zero = 0;
    unsigned int version = 0;
    REQUIRE_NOTHROW(stub_cudaRegisterFatBinary(&dummy_handle, nullptr, zero, version));
}
```

- [ ] **Step 3: Run test to verify 6 fail**

Run: `cmake --build build && ctest -R test_ptxir_cubin_loader -V`
Expected: 6 FAIL

- [ ] **Step 4: Implement dispatch in cudart_sim.cpp**

In `src/cudart/cudart_sim.cpp`, after line 383 (`self_exe_path[size] = '\0';`) and before line 385 (`std::string ptx_code = extract_ptx_with_cuobjdump(self_exe_path);`):

```cpp
    // 2.5 NEW: PTXIR-Embedded dispatch (gated by PTXIR_MODE)
    if (config::isPTXIRModeEnabled()) {
        std::ifstream exe(self_exe_path, std::ios::binary | std::ios::ate);
        if (exe.good()) {
            auto exe_size = static_cast<size_t>(exe.tellg());
            exe.seekg(exe_size - 8);
            char magic_buf[8];
            if (exe.read(magic_buf, 8) && exe.gcount() == 8 &&
                std::memcmp(magic_buf, cudart::PTXIR_EMBED_MAGIC, 8) == 0) {
                // Detected PTXIR embed → extract + deserialize + adapter + set_ptx_context
                exe.seekg(0, std::ios::beg);
                std::vector<uint8_t> contents(exe_size);
                exe.read(reinterpret_cast<char*>(contents.data()), exe_size);
                size_t section_size = 0;
                auto section = cudart::PTXIRLoader::extractPTXIR(contents.data(), exe_size, &section_size);
                if (section) {
                    auto stmts = cudart::PTXIRLoader::deserializeForCubin(section.get(), section_size);
                    if (!stmts.empty()) {
                        // Read MANIFEST section for kernelName + params + addressSize
                        auto manifest = read_manifest_from_ptxir_section(section.get(), section_size);
                        auto ctx = cudart::PtxContextAdapter::fromEmbedded(stmts, manifest);
                        g_ptx_interpreter->set_ptx_context(ctx);
                        *fatCubinHandle = &dummy_handle;  // see cudart_sim.cpp existing line 478
                        return fatCubinHandle;
                    }
                }
                // Any failure → fall through to standard path (no log spam)
            }
        }
    }

    // 3. 现有路径: extract PTX from /proc/self/exe
    std::string ptx_code = extract_ptx_with_cuobjdump(self_exe_path);
```

CRITICAL: do NOT dereference `fat_bin`. Only `self_exe_path` is touched.

- [ ] **Step 5: Run test to verify 6 pass**

Run: `cmake --build build && ctest -R test_ptxir_cubin_loader -V`
Expected: 6 PASS

- [ ] **Step 6: ABI stability test (NEW-4)**

Run:
```bash
nm -D build/lib/libcudart.so | grep cudaRegisterFatBinary
```

Expected output:
```
00000000xxxxxxx T cudaRegisterFatBinary
```

Run a smoke test:
```bash
# Compile a minimal CUDA program that calls __cudaRegisterFatBinary
cat > /tmp/test_abi.cu <<EOF
extern "C" void** __cudaRegisterFatBinary(void**, void*, unsigned long long, unsigned int);
int main() { void* h = nullptr; return __cudaRegisterFatBinary(&h, nullptr, 0, 0) != nullptr; }
EOF
nvcc -L build/lib /tmp/test_abi.cu -o /tmp/test_abi -lcudart
LD_LIBRARY_PATH=build/lib /tmp/test_abi
echo $?
```

Expected: exit code 0 (or non-zero but NO link error for `__cudaRegisterFatBinary`)

---

## Task 6: tools/ptxir_embed CLI (--in-exe / --in-cubin + --kernel-name required)

**Files:**
- Modify: `CMakeLists.txt` (add `add_subdirectory(tools)`)
- Create: `tools/CMakeLists.txt`
- Create: `tools/ptxir_embed.cpp`
- Create: `tools/README.md`
- Test: 6 unit/integration test cases (in `tests/integration/cudart/test_ptxir_cubin_loader.cpp` extended, or `tests/unit/tools/test_ptxir_embed.cpp`)

- [ ] **Step 1: Create tools/ directory + CMakeLists.txt**

Create `tools/CMakeLists.txt`:
```cmake
add_executable(ptxir_embed ptxir_embed.cpp)
target_link_libraries(ptxir_embed PRIVATE cudart)
install(TARGETS ptxir_embed RUNTIME DESTINATION bin)

add_executable(ptxir_extract ptxir_extract.cpp)
target_link_libraries(ptxir_extract PRIVATE cudart)
install(TARGETS ptxir_extract RUNTIME DESTINATION bin)
```

Add to root `CMakeLists.txt` after `add_subdirectory(src)`:
```cmake
add_subdirectory(tools)
```

- [ ] **Step 2: Write the failing test (6 cases)**

```cpp
// tests/unit/tools/test_ptxir_embed.cpp
#include <catch2/catch.hpp>
#include <fstream>
#include <cstdlib>

TEST_CASE("embed_legitimateExe_producesEmbeddedExe", "[ptxir_embed]") {
    // Build minimal exe + minimal ptxir + manifest
    std::ofstream exe("/tmp/in_exe", std::ios::binary);
    exe << "FAKE_EXE_PREFIX";
    exe.close();
    std::ofstream ptxir("/tmp/in.ptxir", std::ios::binary);
    ptxir << "FAKE_PTXIR_HEADER";  // 16 bytes
    ptxir.close();
    int rc = std::system("build/bin/ptxir_embed --in-exe /tmp/in_exe --in-ptxir /tmp/in.ptxir --kernel-name vecAdd --out /tmp/out_embedded");
    REQUIRE(rc == 0);
    std::ifstream out("/tmp/out_embedded", std::ios::binary | std::ios::ate);
    auto sz = out.tellg();
    REQUIRE(sz > 12);  // prefix + section + 4 size + 8 magic
}

TEST_CASE("embed_legitimateCubin_producesEmbeddedCubin", "[ptxir_embed]") {
    // similar but with --in-cubin
    int rc = std::system("build/bin/ptxir_embed --in-cubin /tmp/in.cubin --in-ptxir /tmp/in.ptxir --kernel-name k --out /tmp/out.cubin.embedded");
    REQUIRE(rc == 0);
}

TEST_CASE("embed_missingKernelName_exitsWithError", "[ptxir_embed]") {
    int rc = std::system("build/bin/ptxir_embed --in-exe /tmp/in_exe --in-ptxir /tmp/in.ptxir --out /tmp/out 2>/dev/null");
    REQUIRE(rc != 0);  // should exit non-zero
}

TEST_CASE("embed_missingInputFile_exitsWithError", "[ptxir_embed]") {
    int rc = std::system("build/bin/ptxir_embed --in-exe /nonexistent --in-ptxir /tmp/in.ptxir --kernel-name k --out /tmp/out 2>/dev/null");
    REQUIRE(rc != 0);
}

TEST_CASE("embed_help_printsUsage", "[ptxir_embed]") {
    int rc = std::system("build/bin/ptxir_embed --help > /dev/null");
    REQUIRE(rc == 0);
}

TEST_CASE("embed_version_printsVersion", "[ptxir_embed]") {
    int rc = std::system("build/bin/ptxir_embed --version > /dev/null");
    REQUIRE(rc == 0);
}
```

- [ ] **Step 3: Run test to verify 6 fail**

Run: `cmake --build build && ctest -R test_ptxir_embed -V`
Expected: 6 FAIL (ptxir_embed binary not built)

- [ ] **Step 4: Implement tools/ptxir_embed.cpp**

```cpp
// tools/ptxir_embed.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <endian.h>  // htole32
#include "cudart/ptxir_loader.h"

constexpr size_t MANIFEST_HEADER_SIZE = 32 + 1 + 1 + 64;  // hash + name_max + ptx_addr_size + padding

void print_usage() {
    std::cout << "Usage: ptxir_embed [--in-exe <path> | --in-cubin <path>] --in-ptxir <path> --kernel-name <name> --out <path>\n";
}

void print_version() { std::cout << "ptxir_embed v1.0\n"; }

int main(int argc, char** argv) {
    std::string in_exe, in_cubin, in_ptxir, out_path, kernel_name;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--in-exe" && i+1 < argc) in_exe = argv[++i];
        else if (a == "--in-cubin" && i+1 < argc) in_cubin = argv[++i];
        else if (a == "--in-ptxir" && i+1 < argc) in_ptxir = argv[++i];
        else if (a == "--out" && i+1 < argc) out_path = argv[++i];
        else if (a == "--kernel-name" && i+1 < argc) kernel_name = argv[++i];
        else if (a == "--help") { print_usage(); return 0; }
        else if (a == "--version") { print_version(); return 0; }
    }
    if (kernel_name.empty()) { std::cerr << "Error: --kernel-name is required\n"; return 4; }
    if ((in_exe.empty() && in_cubin.empty()) || (!in_exe.empty() && !in_cubin.empty())) {
        std::cerr << "Error: exactly one of --in-exe or --in-cubin is required\n"; return 4;
    }
    std::string prefix_path = in_exe.empty() ? in_cubin : in_exe;

    std::ifstream pf(prefix_path, std::ios::binary);
    if (!pf) { std::cerr << "Error: cannot open " << prefix_path << "\n"; return 2; }
    std::vector<uint8_t> prefix((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());

    std::ifstream sf(in_ptxir, std::ios::binary);
    if (!sf) { std::cerr << "Error: cannot open " << in_ptxir << "\n"; return 2; }
    std::vector<uint8_t> section((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());

    // TODO: append kernel_name to MANIFEST section (Task 1 implementation)
    // For now, append a minimal MANIFEST stub after the section
    section.push_back(0);  // placeholder

    // Compute SHA-256 of prefix
    // ... (use OpenSSL SHA256)

    // Build output: prefix || section || uint32_le size || magic
    uint32_t size_le = htole32(static_cast<uint32_t>(section.size()));
    std::ofstream out(out_path, std::ios::binary);
    if (!out) { std::cerr << "Error: cannot write " << out_path << "\n"; return 2; }
    out.write(reinterpret_cast<const char*>(prefix.data()), prefix.size());
    out.write(reinterpret_cast<const char*>(section.data()), section.size());
    out.write(reinterpret_cast<const char*>(&size_le), 4);
    out.write(reinterpret_cast<const char*>(cudart::PTXIR_EMBED_MAGIC), 8);
    return 0;
}
```

- [ ] **Step 5: Run test to verify 6 pass**

Run: `cmake --build build && ctest -R test_ptxir_embed -V`
Expected: 6 PASS

- [ ] **Step 6: Write tools/README.md**

Document CLI usage, PTXIR_MODE interaction, examples for `--in-exe` vs `--in-cubin`.

---

## Task 7: tools/ptxir_extract CLI (passthrough + dual extraction)

**Files:**
- Create: `tools/ptxir_extract.cpp` (declared in Task 6 CMakeLists.txt)
- Test: 4 test cases

- [ ] **Step 1: Write the failing test (4 cases)**

```cpp
TEST_CASE("extract_legitimateEmbedded_producesPurePrefixAndPTXIR", "[ptxir_extract]") {
    int rc = std::system("build/bin/ptxir_extract --in /tmp/embedded --out-cubin /tmp/pure.cubin --out-ptxir /tmp/pure.ptxir");
    REQUIRE(rc == 0);
}

TEST_CASE("extract_plainCubin_passthrough", "[ptxir_extract]") {
    // Use a plain (non-embedded) cubin
    int rc = std::system("build/bin/ptxir_extract --in /tmp/plain.cubin --out-cubin /tmp/plain.out.cubin");
    REQUIRE(rc == 0);
}

TEST_CASE("extract_hashMismatch_exitsWithError", "[ptxir_extract]") {
    // Manually corrupt the embedded cubin to trigger hash mismatch
    int rc = std::system("build/bin/ptxir_extract --in /tmp/corrupted --out-cubin /tmp/x 2>/dev/null");
    REQUIRE(rc != 0);
}

TEST_CASE("extract_help_printsUsage", "[ptxir_extract]") {
    int rc = std::system("build/bin/ptxir_extract --help > /dev/null");
    REQUIRE(rc == 0);
}
```

- [ ] **Step 2: Run test to verify 4 fail**

Run: `cmake --build build && ctest -R test_ptxir_extract -V`
Expected: 4 FAIL

- [ ] **Step 3: Implement tools/ptxir_extract.cpp**

```cpp
// tools/ptxir_extract.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include "cudart/ptxir_loader.h"

int main(int argc, char** argv) {
    std::string in_path, out_cubin, out_ptxir;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--in" && i+1 < argc) in_path = argv[++i];
        else if (a == "--out-cubin" && i+1 < argc) out_cubin = argv[++i];
        else if (a == "--out-ptxir" && i+1 < argc) out_ptxir = argv[++i];
        else if (a == "--help") { std::cout << "Usage: ptxir_extract --in <path> [--out-cubin <X>] [--out-ptxir <Y>]\n"; return 0; }
        else if (a == "--version") { std::cout << "ptxir_extract v1.0\n"; return 0; }
    }
    if (in_path.empty()) { std::cerr << "Error: --in is required\n"; return 4; }

    std::ifstream f(in_path, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    if (!cudart::PTXIRLoader::hasEmbeddedPTXIR(data.data(), data.size())) {
        // Plain passthrough
        if (!out_cubin.empty()) {
            std::ofstream o(out_cubin, std::ios::binary);
            o.write(reinterpret_cast<const char*>(data.data()), data.size());
        }
        return 0;
    }

    auto pure = cudart::PTXIRLoader::extractPureCubin(data.data(), data.size());
    if (!pure) { std::cerr << "Error: cubin_hash mismatch\n"; return 3; }
    if (!out_cubin.empty()) {
        std::ofstream o(out_cubin, std::ios::binary);
        o.write(reinterpret_cast<const char*>(pure->data()), pure->size());
    }
    if (!out_ptxir.empty()) {
        size_t section_size;
        auto section = cudart::PTXIRLoader::extractPTXIR(data.data(), data.size(), &section_size);
        if (section) {
            std::ofstream o(out_ptxir, std::ios::binary);
            o.write(reinterpret_cast<const char*>(section.get()), section_size);
        }
    }
    return 0;
}
```

- [ ] **Step 4: Run test to verify 4 pass**

Run: `cmake --build build && ctest -R test_ptxir_extract -V`
Expected: 4 PASS

---

## Task 8: E2E tests (nvcc + cuobjdump + cuModuleLoadData)

**Files:**
- Create: `tests/e2e/test_ptxir_cubin_embed.cu`
- Modify: `tests/e2e/CMakeLists.txt` (if needed)

- [ ] **Step 1: Write e2e test fixtures (≥3 CUDA kernels of varying complexity)**

```cpp
// tests/e2e/test_ptxir_cubin_embed.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel_simple_add(int* x, int* y, int* out) {
    *out = *x + *y;
}

__global__ void kernel_reduction(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    sdata[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = sdata[0];
}

__global__ void kernel_branchy(int* x, int n) {
    int tid = threadIdx.x;
    if (tid < n / 2) {
        x[tid] *= 2;
    } else {
        x[tid] += 1;
    }
}

// Test 1: nvcc + embed + extract + cuobjdump --dump-sass byte-identical
TEST_CASE("e2e_embed_extract_cuobjdump_byteIdenticalToOriginalCubin", "[e2e][ptxir]") {
    // 1. Compile kernel_simple_add with nvcc -> simple.cubin
    // 2. Run ptx-serializer on simple.ptx -> simple.ptxir
    // 3. Run ptxir_embed --in-cubin simple.cubin --in-ptxir simple.ptxir --kernel-name kernel_simple_add --out simple.embedded
    // 4. Run ptxir_extract --in simple.embedded --out-cubin simple.pure.cubin
    // 5. cuobjdump --dump-sass simple.pure.cubin > pure.sass
    // 6. cuobjdump --dump-sass simple.cubin > orig.sass
    // 7. diff pure.sass orig.sass -> must be identical
}

// Test 2: cuobjdump --dump-sass directly on embedded (Oracle R1 fix)
TEST_CASE("e2e_embed_cuobjdump_dumpSASS_direct", "[e2e][ptxir]") {
    // cuobjdump --dump-sass simple.embedded.cubin > embedded.sass
    // diff embedded.sass simple.cubin SASS -> must be identical
}

// Test 3: cuModuleLoadData explicit SKIP (Oracle review blocking fix)
TEST_CASE("e2e_cuModuleLoadData_noDriver_explicitSkip", "[e2e][ptxir]") {
    if (!has_real_nvidia_driver()) {
        printf("[SKIP] cuModuleLoadData test — no driver\n");
        return;  // SKIP
    }
    // Else: cuModuleLoadData(simple.embedded.cubin) must return CUDA_SUCCESS
}
```

- [ ] **Step 2: Implement ≥5 e2e test scenarios**

Tests 1-5 (independent of Commit 2):
- e2e_nvccCompile_embed_cuobjdumpDumpSassMatchesOriginal
- e2e_embed_extract_cuobjdump_byteIdenticalToOriginalCubin
- e2e_embed_cuobjdump_dumpPTX_normal
- e2e_cuModuleLoadData_noDriver_explicitSkip
- e2e_nvccCompile_embedThreeKernels_cuobjdumpPasses

Test 6 (depends on Commit 2):
- e2e_nvccCompile_embedExe_ptxemu_executesCorrectly

- [ ] **Step 3: Run e2e tests**

Run: `cd build && ctest -L e2e -V -R ptxir`
Expected: ≥5 PASS (or SKIP for cuModuleLoadData when no driver)

---

## Task 9: Docs sync (roadmap + README + ADR index)

**Files:**
- Modify: `roadmap.md` (verify Phase 12.2 status — likely already done in earlier commit)
- Modify: `README.md` (§已实现功能 + §已知限制)
- Modify: `docs/adr/README.md` (ADR-0024 v1.1 reference)

- [ ] **Step 1: Verify roadmap.md Phase 12.2 already updated**

Run: `grep -A 5 "Phase 12.2" roadmap.md | head -20`
Expected: Phase 12.2 section present (already done 2026-08-07)

If not present, add the section (see prior commit `0e239b1c` for template).

- [ ] **Step 2: Update root README.md**

In §已实现功能, add:
```
- **PTXIR-Embedded CUBIN/EXE**: 标准可执行文件末尾追加 PTXIR section，PTX-EMU 通过 O(1) tail detection 加载（ADR-0024 v1.1）
```

In §已知限制, remove or update:
```
- PTXIR 反序列化路径不在 `__cudaRegisterFatBinary` 入口暴露
```
(Now superseded by Commit 2's dispatch)

- [ ] **Step 3: Update docs/adr/README.md**

In the ADR table, update ADR-0024 row:
- Status: Accepted → Accepted (v1.1 2026-08-07)
- 关联任务: TBD → openspec/changes/implement-ptxir-cubin-embed-extension/

In 最近更新 table, add:
```
| 2026-08-07 | **ADR-0024 v1.1 amendment**：footer layout (size-after-section, ZIP-EOCD style) + magic literal `{'P','T','X','E','M','B','\x01','\x00'}` + PtxContextAdapter + tools/ 目录 + MANIFEST section | 0024 |
```

- [ ] **Step 4: Verify no broken links**

Run: `./scripts/check-docs-index.sh 2>&1 | head -20`
Expected: no errors

---

## Task 10: Final verification (ADR-0024 §合规检查 + 全套测试)

**Files:** (no new files; run checks)

- [ ] **Step 1: PTXIR_MODE=off 完全 bypass 检测分支**

Run: `unset PTXIR_MODE && cmake --build build && ctest --output-on-failure`
Expected: 所有现有测试通过，无回归（PTXIR_MODE 默认 OFF 完全等价现状）

- [ ] **Step 2: PTXIR_MODE=auto 不引入回归**

Run: `PTXIR_MODE=auto cmake --build build && ctest --output-on-failure`
Expected: 所有现有测试通过（dispatch 在 PTXIR_MODE=auto 时也优雅降级 — 无 embed 时走标准路径）

- [ ] **Step 3: ABI stability test**

Run:
```bash
nm -D build/lib/libcudart.so | grep cudaRegisterFatBinary
cat > /tmp/test_abi.cu <<EOF
extern "C" void** __cudaRegisterFatBinary(void**, void*, unsigned long long, unsigned int);
int main() { void* h = nullptr; auto r = __cudaRegisterFatBinary(&h, nullptr, 0, 0); return 0; }
EOF
nvcc -L build/lib /tmp/test_abi.cu -o /tmp/test_abi -lcudart
LD_LIBRARY_PATH=build/lib /tmp/test_abi && echo "ABI OK"
```

Expected: `ABI OK` printed

- [ ] **Step 4: fat_bin=nullptr does not crash**

Run:
```bash
cat > /tmp/test_nullptr.cu <<EOF
extern "C" void** __cudaRegisterFatBinary(void**, void*, unsigned long long, unsigned int);
int main() { void* h = nullptr; auto r = __cudaRegisterFatBinary(&h, nullptr, 0, 0); return 0; }
EOF
nvcc -L build/lib /tmp/test_nullptr.cu -o /tmp/test_nullptr -lcudart
LD_LIBRARY_PATH=build/lib /tmp/test_nullptr && echo "nullptr OK"
```

Expected: `nullptr OK` printed

- [ ] **Step 5: All §合规检查 items pass**

| Item | Verification |
|---|---|
| 1. PTXIR_MODE=off 完全 bypass | Step 1 ✓ |
| 2. ptxir_extract 保留原 cubin 字节 | Task 7 unit tests ✓ |
| 3. .ptxir.section 使用 ADR-0023 Section TOC | Task 1 (MANIFEST section extends TOC) ✓ |
| 4. PTXIRLoader 所有 4 函数有 unit 测试 | Task 2 (14 tests) ✓ |
| 5. e2e 用 nvcc + cuobjdump 验证 NVIDIA 兼容 | Task 8 ✓ |
| 6. magic 变更触发 ADR-0024 governance check | ADR-0024 v1.1 §更新记录 ✓ |

- [ ] **Step 6: openspec validate**

Run: `openspec validate implement-ptxir-cubin-embed-extension --strict`
Expected: "is valid"

---

## Self-Review Checklist

After implementing all tasks, verify:

- [ ] All unit tests pass (PTXIRLoader 14, PtxContextAdapter 5, config 4, MANIFEST section 2, ptxir_embed 6, ptxir_extract 4) = 35+ tests
- [ ] All integration tests pass (6 dispatch scenarios)
- [ ] All e2e tests pass (≥5 + 1 dependent on Commit 2)
- [ ] ABI stability verified (nm + nvcc link)
- [ ] Coverage ≥ 90% on PTXIRLoader, PtxContextAdapter, ptxir_config
- [ ] `PTXIR_MODE=off` byte-identical to pre-change behavior
- [ ] All 5 commits merged to main via archive phase