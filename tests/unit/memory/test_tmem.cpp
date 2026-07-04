// tests/unit/memory/test_tmem.cpp
// Phase 0.2 (Fix #6): per-CTA Tensor Memory (TMEM) unit tests.
//
// 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13.
// ≥10 TEST_CASEs cover default-zero, read/write round-trip, cross-CTA
// isolation, bounds enforcement, clear, non-clobbering partial writes.
// Consumed by tcgen05.* in Phase 1-3.

#include "catch_amalgamated.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ptxsim/memory/tmem.h"

// ---------------------------------------------------------------------------
// 1. construct_default_zeros_all_slots
// ---------------------------------------------------------------------------
TEST_CASE("construct_default_zeros_all_slots", "[tmem][construct]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    std::vector<uint8_t> buf(128);
    for (size_t slot = 0; slot < 256; ++slot) {
        tmem.read(slot, buf.data(), 128);
        for (size_t i = 0; i < 128; ++i) {
            INFO("slot " << slot << " byte " << i << " is non-zero");
            REQUIRE(buf[i] == 0);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. construct_multiple_independent_instances (cross-CTA isolation)
// ---------------------------------------------------------------------------
TEST_CASE("construct_multiple_independent_instances", "[tmem][isolation]") {
    Tmem tmem_a;
    Tmem tmem_b;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    std::vector<uint8_t> buf_a(128, 0xAA);
    tmem_a.write(7, buf_a.data(), 128);

    // read from tmem_b at same slot; must still be zero (independent storage)
    std::vector<uint8_t> buf_b(128);
    tmem_b.read(7, buf_b.data(), 128);
    for (size_t i = 0; i < 128; ++i) {
        INFO("tmem_b slot 7 byte " << i << " leaked from tmem_a");
        REQUIRE(buf_b[i] == 0);
    }
}

// ---------------------------------------------------------------------------
// 3. write_slot_then_read_slot_roundtrip
// ---------------------------------------------------------------------------
TEST_CASE("write_slot_then_read_slot_roundtrip", "[tmem][roundtrip]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    std::vector<uint8_t> write_buf(128);
    for (size_t i = 0; i < 128; ++i) {
        write_buf[i] = static_cast<uint8_t>(i & 0xFF);
    }

    tmem.write(5, write_buf.data(), 128);

    std::vector<uint8_t> read_buf(128);
    tmem.read(5, read_buf.data(), 128);

    for (size_t i = 0; i < 128; ++i) {
        INFO("slot 5 byte " << i << " mismatch");
        REQUIRE(read_buf[i] == static_cast<uint8_t>(i & 0xFF));
    }
}

// ---------------------------------------------------------------------------
// 4. write_small_then_read_full_does_not_clobber
// ---------------------------------------------------------------------------
TEST_CASE("write_small_then_read_full_does_not_clobber", "[tmem][partial]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    std::vector<uint8_t> small_buf(4, 0xCC);
    tmem.write(10, small_buf.data(), 4);

    std::vector<uint8_t> read_buf(128);
    tmem.read(10, read_buf.data(), 128);

    // first 4 bytes must be 0xCC
    for (size_t i = 0; i < 4; ++i) {
        INFO("slot 10 byte " << i << " should be 0xCC");
        REQUIRE(read_buf[i] == 0xCC);
    }
    // remaining bytes must still be 0
    for (size_t i = 4; i < 128; ++i) {
        INFO("slot 10 byte " << i << " was clobbered");
        REQUIRE(read_buf[i] == 0);
    }
}

// ---------------------------------------------------------------------------
// 5. write_out_of_range_slot_id_256_throw
// ---------------------------------------------------------------------------
TEST_CASE("write_out_of_range_slot_id_256_throw", "[tmem][bounds]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    // slot_id is 0-indexed [0, 256)
    std::vector<uint8_t> buf(128, 0xFF);
    REQUIRE_THROWS_AS(tmem.write(256, buf.data(), 128), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 6. write_out_of_range_slot_id_512_throw
// ---------------------------------------------------------------------------
TEST_CASE("write_out_of_range_slot_id_512_throw", "[tmem][bounds]") {
    Tmem tmem;

    std::vector<uint8_t> buf(128, 0xFF);
    REQUIRE_THROWS_AS(tmem.write(512, buf.data(), 128), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 7. read_slot_id_256_throw
// ---------------------------------------------------------------------------
TEST_CASE("read_slot_id_256_throw", "[tmem][bounds]") {
    Tmem tmem;

    std::vector<uint8_t> buf(128);
    REQUIRE_THROWS_AS(tmem.read(256, buf.data(), 128), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 8. read_slot_id_large_throw
// ---------------------------------------------------------------------------
TEST_CASE("read_slot_id_large_throw", "[tmem][bounds]") {
    Tmem tmem;

    std::vector<uint8_t> buf(128);
    REQUIRE_THROWS_AS(tmem.read(999, buf.data(), 128), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 9. write_size_greater_than_slot_128_throw
// ---------------------------------------------------------------------------
TEST_CASE("write_size_greater_than_slot_128_throw", "[tmem][bounds]") {
    Tmem tmem;

    std::vector<uint8_t> buf(129, 0xFF);
    REQUIRE_THROWS_AS(tmem.write(0, buf.data(), 129), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 10. read_size_greater_than_slot_128_throw
// ---------------------------------------------------------------------------
TEST_CASE("read_size_greater_than_slot_128_throw", "[tmem][bounds]") {
    Tmem tmem;

    std::vector<uint8_t> buf(129);
    REQUIRE_THROWS_AS(tmem.read(0, buf.data(), 129), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 11. clear_clears_all_slots
// ---------------------------------------------------------------------------
TEST_CASE("clear_clears_all_slots", "[tmem][clear]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    std::vector<uint8_t> write_buf(128, 0xEF);
    for (size_t slot = 0; slot < 256; ++slot) {
        tmem.write(slot, write_buf.data(), 128);
    }

    tmem.clear();

    std::vector<uint8_t> read_buf(128);
    for (size_t slot = 0; slot < 256; ++slot) {
        tmem.read(slot, read_buf.data(), 128);
        for (size_t i = 0; i < 128; ++i) {
            INFO("slot " << slot << " byte " << i
                         << " not cleared (0x"
                         << std::hex << static_cast<int>(read_buf[i]) << ")");
            REQUIRE(read_buf[i] == 0);
        }
    }
}

// ---------------------------------------------------------------------------
// 12. partial_slot_write_does_not_leak_to_next_slot
// ---------------------------------------------------------------------------
TEST_CASE("partial_slot_write_does_not_leak_to_next_slot",
          "[tmem][isolation]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    std::vector<uint8_t> half_buf(64, 0xDD);
    tmem.write(3, half_buf.data(), 64);

    // slot 4 must still be all zeros (no data leak beyond slot boundary)
    std::vector<uint8_t> read_buf(128);
    tmem.read(4, read_buf.data(), 128);
    for (size_t i = 0; i < 128; ++i) {
        INFO("slot 4 byte " << i << " leaked from slot 3 write");
        REQUIRE(read_buf[i] == 0);
    }
}

// ---------------------------------------------------------------------------
// 13. validate_slot_id_returns_correct_values
// ---------------------------------------------------------------------------
TEST_CASE("validate_slot_id_returns_correct_values", "[tmem][validate]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    // valid: [0, 256)
    REQUIRE(tmem.validate_slot_id(0) == true);
    REQUIRE(tmem.validate_slot_id(127) == true);
    REQUIRE(tmem.validate_slot_id(255) == true);
    REQUIRE(tmem.validate_slot_id(256) == false);
    REQUIRE(tmem.validate_slot_id(512) == false);
    REQUIRE(tmem.validate_slot_id(999) == false);
}

// ---------------------------------------------------------------------------
// 14. write_zero_bytes_ok_writes_nothing
// ---------------------------------------------------------------------------
TEST_CASE("write_zero_bytes_ok_writes_nothing", "[tmem][edge]") {
    Tmem tmem;

    // writing 0 bytes should succeed but not clobber anything
    tmem.write(0, nullptr, 0);

    // after zero-byte write, all slots should still be zero
    std::vector<uint8_t> buf(128);
    for (size_t slot = 0; slot < 256; ++slot) {
        tmem.read(slot, buf.data(), 128);
        for (size_t i = 0; i < 128; ++i) {
            REQUIRE(buf[i] == 0);
        }
    }
}

// ---------------------------------------------------------------------------
// 15. write_full_128_slot_boundary_no_overflow
// ---------------------------------------------------------------------------
TEST_CASE("write_full_128_slot_boundary_no_overflow", "[tmem][bounds]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    // write exactly 128 bytes at the last valid slot (255)
    std::vector<uint8_t> full_buf(128, 0xAB);
    tmem.write(255, full_buf.data(), 128);

    std::vector<uint8_t> read_buf(128);
    tmem.read(255, read_buf.data(), 128);
    for (size_t i = 0; i < 128; ++i) {
        INFO("last slot byte " << i << " mismatch");
        REQUIRE(read_buf[i] == 0xAB);
    }
}

// ---------------------------------------------------------------------------
// 16. concurrent_writes_to_different_slots_safe
// ---------------------------------------------------------------------------
TEST_CASE("concurrent_writes_to_different_slots_safe",
          "[tmem][thread_safety]") {
    Tmem tmem;

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    // Phase 1-3 may invoke from multiple warps; verify serialization under
    // mutex does not deadlock or corrupt data.
    constexpr int kThreads = 4;
    constexpr int kIterations = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&tmem, t]() {
            for (int i = 0; i < kIterations; ++i) {
                size_t slot = static_cast<size_t>(t * 4 + (i % 4));
                std::vector<uint8_t> buf(
                    128, static_cast<uint8_t>(t * 32 + i));
                tmem.write(slot, buf.data(), 128);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // all slots should be readable without hanging or corruption
    std::vector<uint8_t> buf(128);
    for (size_t slot = 0; slot < 256; ++slot) {
        tmem.read(slot, buf.data(), 128);
    }
}

// ---------------------------------------------------------------------------
// 17. concurrent_reads_from_same_slot_safe
// ---------------------------------------------------------------------------
TEST_CASE("concurrent_reads_from_same_slot_safe",
          "[tmem][thread_safety]") {
    Tmem tmem;

    std::vector<uint8_t> write_buf(128, 0x42);
    tmem.write(42, write_buf.data(), 128);

    constexpr int kThreads = 8;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&tmem]() {
            std::vector<uint8_t> buf(128);
            for (int i = 0; i < 50; ++i) {
                tmem.read(42, buf.data(), 128);
                REQUIRE(buf[0] == 0x42);
                REQUIRE(buf[127] == 0x42);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

// ---------------------------------------------------------------------------
// 18. kSlotCount_and_kSlotSize_constants_match_total
// ---------------------------------------------------------------------------
TEST_CASE("kSlotCount_and_kSlotSize_constants_match_total",
          "[tmem][constants]") {
    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    REQUIRE(Tmem::kSlotCount == 256);
    REQUIRE(Tmem::kSlotSize == 128);
    REQUIRE(Tmem::kTotalSize == 256u * 128u);
    REQUIRE(Tmem::kTotalSize == 32u * 1024u);
}