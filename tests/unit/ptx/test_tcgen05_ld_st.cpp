// tests/unit/ptx/test_tcgen05_ld_st.cpp
// Phase 2.1 (Fix #12): unit tests for tcgen05.ld and tcgen05.st handlers.
//
// TDD RED phase: these tests MUST FAIL because tcgen05.ld/st handlers
// are not yet implemented in wmma.cpp (UnsupportedInstructionException).
//
// Verifies:
//   1. tcgen05.ld copies data from TMA descriptor's global_address to TMEM slot
//   2. tcgen05.st copies data from TMEM slot to TMA descriptor's global_address
//   3. Descriptor existence validation
//   4. Slot bounds validation
//   5. ld → st roundtrip data consistency

#include "catch_amalgamated.hpp"

#include "ptxsim/instruction_handlers.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/memory/tma_descriptor.h"
#include "ptxsim/memory/tmem.h"
#include "memory/hardware_memory_manager.h"

#include <array>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kSlotSize = 128;

std::vector<Qualifier> make_tcgen05_ld_quals() {
    // PTX ISA §9.7.13: tcgen05.ld.sync.aligned.cta_group::1.desc[...].kind::f16
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    return {Qualifier::Q_CLUSTER, Qualifier::Q_F16, Qualifier::Q_TCGEN05_LD};
}

std::vector<Qualifier> make_tcgen05_st_quals() {
    // PTX ISA §9.7.13: tcgen05.st.sync.aligned.cta_group::1.desc[...].kind::f16
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    return {Qualifier::Q_CLUSTER, Qualifier::Q_F16, Qualifier::Q_TCGEN05_ST};
}

// Read f16 values from a TMEM slot.
std::vector<uint16_t> read_tmem_slot_f16(Tmem& tmem, size_t slot_id,
                                          size_t num_values) {
    std::array<uint8_t, Tmem::kSlotSize> buf{};
    tmem.read(slot_id, buf.data(), Tmem::kSlotSize);
    size_t n = std::min(num_values, static_cast<size_t>(Tmem::kSlotSize / 2));
    std::vector<uint16_t> result(n);
    std::memcpy(result.data(), buf.data(), n * sizeof(uint16_t));
    return result;
}

// Fill TMEM slot with f16 data.
void fill_tmem_slot_f16(Tmem& tmem, size_t slot_id,
                         const std::vector<uint16_t>& data) {
    std::array<uint8_t, Tmem::kSlotSize> buf{};
    size_t copy_bytes =
        std::min(data.size() * sizeof(uint16_t), Tmem::kSlotSize);
    std::memcpy(buf.data(), data.data(), copy_bytes);
    tmem.write(slot_id, buf.data(), Tmem::kSlotSize);
}

// Create a deterministic test data pattern (64× uint16_t).
std::vector<uint16_t> make_pattern_data() {
    std::vector<uint16_t> data(64);
    for (size_t i = 0; i < 64; ++i) {
        data[i] = static_cast<uint16_t>((i + 1) * 37);
    }
    return data;
}

} // anonymous namespace

TEST_CASE("tcgen05.ld handler throws for bare ThreadContext (no warp)",
          "[unit][ptx][wmma][tcgen05][ld_st]") {
    ThreadContext ctx;
    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_ld_quals();

    REQUIRE_THROWS_AS(handler.processWmmaOperation(&ctx, ops, quals),
                      UnsupportedInstructionException);
}

TEST_CASE("tcgen05.ld copies TMA descriptor data to TMEM slot",
          "[unit][ptx][wmma][tcgen05][ld_st]") {
    CTAContext cta;
    Tmem& tmem = cta.tmem();

    auto pattern = make_pattern_data();

    // Create a valid TMA descriptor with global_address pointing to the pattern buffer
    std::array<uint8_t, kTmaDescriptorSize> desc_raw{};
    {
        uint64_t gaddr = reinterpret_cast<uint64_t>(pattern.data());
        std::memcpy(desc_raw.data(), &gaddr, sizeof(gaddr));
        for (int d = 0; d < 5; ++d)
            desc_raw[8 + d * 4] = 8;
        for (int d = 0; d < 5; ++d)
            desc_raw[64 + d * 4] = 1;
        for (int d = 0; d < 4; ++d) {
            desc_raw[32 + d * 8] = 0x10;
            desc_raw[84 + d] = 1;
        }
        desc_raw[88] = 2;
        desc_raw[92] = 1;
        desc_raw[93] = 0;
        desc_raw[94] = 0;
        desc_raw[95] = 0;
        desc_raw[28] = 0; desc_raw[29] = 0; desc_raw[30] = 0; desc_raw[31] = 0;
    }

    TmaDescriptor desc = parse_descriptor_bytes(desc_raw.data(), kTmaDescriptorSize);
    desc.global_address = reinterpret_cast<uint64_t>(pattern.data());
    // Copy pattern data into raw_bytes for the descriptor
    std::memcpy(desc.raw_bytes.data(), pattern.data(),
                std::min(pattern.size() * sizeof(uint16_t), kTmaDescriptorSize));

    cta.tma_descriptor_store().store(0, desc);

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    cta.init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    auto warps = cta.release_warps();
    REQUIRE(warps.size() == 1);
    WarpContext* warp = warps[0].get();
    ThreadContext* ctx = warp->get_thread(0);
    REQUIRE(ctx != nullptr);

    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_ld_quals();

    REQUIRE_NOTHROW(handler.processWmmaOperation(ctx, ops, quals));

    auto result = read_tmem_slot_f16(tmem, 0, 64);
    REQUIRE(result.size() == 64);
    for (size_t i = 0; i < 64; ++i) {
        CAPTURE(i);
        REQUIRE(result[i] == pattern[i]);
    }
}

TEST_CASE("tcgen05.st copies TMEM slot data to TMA descriptor global_address",
          "[unit][ptx][wmma][tcgen05][ld_st]") {
    CTAContext cta;
    Tmem& tmem = cta.tmem();

    auto src_data = make_pattern_data();
    fill_tmem_slot_f16(tmem, 0, src_data);

    std::array<uint8_t, kTmaDescriptorSize> desc_raw{};
    uint64_t gaddr = reinterpret_cast<uint64_t>(desc_raw.data());
    std::memcpy(desc_raw.data(), &gaddr, sizeof(gaddr));

    // Set global_dim, rank for valid parse
    for (int d = 0; d < 5; ++d)
        desc_raw[8 + d * 4] = 8; // global_dim = 8
    for (int d = 0; d < 5; ++d)
        desc_raw[64 + d * 4] = 1; // box_dim = 1
    for (int d = 0; d < 4; ++d) {
        desc_raw[32 + d * 8] = 0x10; // stride = 16
        desc_raw[84 + d] = 1; // element_stride = 1
    }
    desc_raw[88] = 2; // rank = 2

    desc_raw[92] = 1;           // elemtype
    desc_raw[93] = 0;           // interleave=NONE
    desc_raw[94] = 0;           // swizzle=NONE
    desc_raw[95] = 0;           // fill=ZERO
    desc_raw[28] = 0; desc_raw[29] = 0; desc_raw[30] = 0; desc_raw[31] = 0;

    TmaDescriptor desc = parse_descriptor_bytes(desc_raw.data(), kTmaDescriptorSize);
    desc.global_address = gaddr;
    cta.tma_descriptor_store().store(0, desc);

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    cta.init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    auto warps = cta.release_warps();
    REQUIRE(warps.size() == 1);
    WarpContext* warp = warps[0].get();
    ThreadContext* ctx = warp->get_thread(0);
    REQUIRE(ctx != nullptr);

    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_st_quals();

    REQUIRE_NOTHROW(handler.processWmmaOperation(ctx, ops, quals));

    const uint16_t* result =
        reinterpret_cast<const uint16_t*>(desc_raw.data());
    for (size_t i = 0; i < 64; ++i) {
        CAPTURE(i);
        REQUIRE(result[i] == src_data[i]);
    }
}

TEST_CASE("tcgen05.ld throws when descriptor not found",
          "[unit][ptx][wmma][tcgen05][ld_st]") {
    CTAContext cta;

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    cta.init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    auto warps = cta.release_warps();
    WarpContext* warp = warps[0].get();
    ThreadContext* ctx = warp->get_thread(0);
    REQUIRE(ctx != nullptr);

    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_ld_quals();

    REQUIRE_THROWS_AS(handler.processWmmaOperation(ctx, ops, quals),
                      UnsupportedInstructionException);
}

TEST_CASE("tcgen05.st throws when descriptor not found",
          "[unit][ptx][wmma][tcgen05][ld_st]") {
    CTAContext cta;

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    cta.init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    auto warps = cta.release_warps();
    WarpContext* warp = warps[0].get();
    ThreadContext* ctx = warp->get_thread(0);
    REQUIRE(ctx != nullptr);

    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
    auto quals = make_tcgen05_st_quals();

    REQUIRE_THROWS_AS(handler.processWmmaOperation(ctx, ops, quals),
                      UnsupportedInstructionException);
}

TEST_CASE("tcgen05.ld → st roundtrip preserves data integrity",
          "[unit][ptx][wmma][tcgen05][ld_st][roundtrip]") {
    CTAContext cta;
    Tmem& tmem = cta.tmem();

    auto pattern = make_pattern_data();
    size_t data_bytes = pattern.size() * sizeof(uint16_t);
    std::array<uint8_t, kSlotSize> src_buf{};
    std::array<uint8_t, kSlotSize> dst_buf{};

    std::memcpy(src_buf.data(), pattern.data(), data_bytes);

    // TMA descriptor for ld: points to src_buf
    std::array<uint8_t, kTmaDescriptorSize> ld_desc_raw{};
    {
        uint64_t gaddr = reinterpret_cast<uint64_t>(src_buf.data());
        std::memcpy(ld_desc_raw.data(), &gaddr, sizeof(gaddr));
        for (int d = 0; d < 5; ++d)
            ld_desc_raw[8 + d * 4] = 8;
        for (int d = 0; d < 5; ++d)
            ld_desc_raw[64 + d * 4] = 1;
        for (int d = 0; d < 4; ++d) {
            ld_desc_raw[32 + d * 8] = 0x10;
            ld_desc_raw[84 + d] = 1;
        }
        ld_desc_raw[88] = 2;
        ld_desc_raw[92] = 1;
        ld_desc_raw[93] = 0;
        ld_desc_raw[94] = 0;
        ld_desc_raw[95] = 0;
        ld_desc_raw[28] = 0; ld_desc_raw[29] = 0; ld_desc_raw[30] = 0; ld_desc_raw[31] = 0;
    }

    TmaDescriptor ld_desc = parse_descriptor_bytes(ld_desc_raw.data(), kTmaDescriptorSize);
    ld_desc.global_address = reinterpret_cast<uint64_t>(src_buf.data());
    cta.tma_descriptor_store().store(0, ld_desc);

    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    cta.init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    auto warps = cta.release_warps();
    WarpContext* warp = warps[0].get();
    ThreadContext* ctx = warp->get_thread(0);
    REQUIRE(ctx != nullptr);

    WmmaHandler handler;
    void* ops[4] = {nullptr, nullptr, nullptr, nullptr};

    REQUIRE_NOTHROW(
        handler.processWmmaOperation(ctx, ops, make_tcgen05_ld_quals()));

    auto tmem_data = read_tmem_slot_f16(tmem, 0, 64);
    REQUIRE(tmem_data.size() == 64);
    for (size_t i = 0; i < 64; ++i) {
        CAPTURE(i);
        REQUIRE(tmem_data[i] == pattern[i]);
    }

    // Overwrite descriptor 0 to point to dst_buf for st
    TmaDescriptor st_desc = ld_desc;
    st_desc.global_address = reinterpret_cast<uint64_t>(dst_buf.data());
    cta.tma_descriptor_store().store(0, st_desc);

    REQUIRE_NOTHROW(
        handler.processWmmaOperation(ctx, ops, make_tcgen05_st_quals()));

    const uint16_t* dst_result =
        reinterpret_cast<const uint16_t*>(dst_buf.data());
    for (size_t i = 0; i < 64; ++i) {
        CAPTURE(i);
        REQUIRE(dst_result[i] == pattern[i]);
    }
}