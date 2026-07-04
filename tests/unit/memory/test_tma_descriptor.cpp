// tests/unit/memory/test_tma_descriptor.cpp
// Phase 0.1 (Fix #5): TMA TensorMap descriptor parser unit tests.
//
// Validates that parse_descriptor_bytes() correctly extracts every field of
// the 128-byte opaque CUtensorMap layout used by Blackwell tcgen05.
//
// All magic numbers / byte offsets below mirror the layout documented in
// src/ptxsim/memory/tma_descriptor.h, which is itself derived from:
//   * NVIDIA PTX ISA §9.7.13 (tensormap.replace field ordinals)
//   * CUDA Driver API CUtensorMap (cuTensorMapEncodeTiled)
//   * LLVM NVPTX target LowerCUtarMap pass
//
// Per ptx-lessons-learned §5 (qualifier-type judgment) and the Phase 0.1
// design notes, NO byte offset here has been verified against real
// hardware — every assertion carries UNVERIFIED-AGAINST-HARDWARE in the
// header. Tests document the assumed layout; future hardware validation
// may require shifting offsets.

#include "catch_amalgamated.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ptxsim/memory/tma_descriptor.h"

namespace {

// Build a zero-initialized 128-byte descriptor buffer that callers can
// patch field-by-field before handing to parse_descriptor_bytes().
std::vector<uint8_t> make_zero_descriptor() {
    return std::vector<uint8_t>(kTmaDescriptorSize, 0);
}

// Field encoders. Each writes a little-endian value at the documented byte
// offset. Centralizing here keeps test fixtures readable and ensures every
// test uses the SAME offset convention as the production parser.
//
// Offsets per src/ptxsim/memory/tma_descriptor.h:
//   0..7   global_address     (uint64)
//   8..27  global_dim[0..4]   (5 × uint32)
//   28..31 RESERVED
//   32..63 global_stride[0..3] (4 × uint64, bytes)
//   64..83 box_dim[0..4]      (5 × uint32)
//   84..87 element_stride     (4 × uint8 packed, or 1 elem-stride register)
//   88..91 rank + control flags
//   92     elemtype           (CU_TENSOR_MAP_DATA_TYPE_*)
//   93     interleave_layout  (0=none,1=16B,2=32B)
//   94     swizzle_mode       (0=none,1=32B,2=64B,3=128B)
//   95     fill_mode          (0=zero,1=NaN)
//   96..127 RESERVED / im2col
void put_u64(std::vector<uint8_t>& buf, size_t off, uint64_t v) {
    for (size_t i = 0; i < 8; ++i) buf[off + i] = static_cast<uint8_t>(v >> (8 * i));
}
void put_u32(std::vector<uint8_t>& buf, size_t off, uint32_t v) {
    for (size_t i = 0; i < 4; ++i) buf[off + i] = static_cast<uint8_t>(v >> (8 * i));
}
void put_u8(std::vector<uint8_t>& buf, size_t off, uint8_t v) {
    buf[off] = v;
}

// Build a valid f16 descriptor used as the canonical fixture for many tests.
// Layout choices (all UNVERIFIED-AGAINST-HARDWARE, see header):
//   global_address = 0x1000 (16-byte aligned)
//   global_dim     = {256, 128, 0, 0, 0} (rank 2)
//   global_stride  = {512, 65536, 0, 0} (bytes, multiples of 16)
//   box_dim        = {16, 16, 1, 1, 1}
//   element_stride = {1, 1, 1, 1} packed
//   rank           = 2 (low byte of offset 88)
//   elemtype       = 6 (CU_TENSOR_MAP_DATA_TYPE_FLOAT16)
//   interleave     = 0 (NONE)
//   swizzle        = 0 (NONE)
//   fill_mode      = 0 (ZERO)
std::vector<uint8_t> make_valid_f16_descriptor() {
    auto buf = make_zero_descriptor();
    put_u64(buf, 0, 0x1000);                          // global_address
    put_u32(buf, 8, 256);                             // global_dim[0]
    put_u32(buf, 12, 128);                            // global_dim[1]
    put_u32(buf, 16, 0);                              // global_dim[2]
    put_u32(buf, 20, 0);                              // global_dim[3]
    put_u32(buf, 24, 0);                              // global_dim[4]
    put_u64(buf, 32, 512);                            // global_stride[0]
    put_u64(buf, 40, 65536);                          // global_stride[1]
    put_u64(buf, 48, 0);                              // global_stride[2]
    put_u64(buf, 56, 0);                              // global_stride[3]
    put_u32(buf, 64, 16);                             // box_dim[0]
    put_u32(buf, 68, 16);                             // box_dim[1]
    put_u32(buf, 72, 1);                              // box_dim[2]
    put_u32(buf, 76, 1);                              // box_dim[3]
    put_u32(buf, 80, 1);                              // box_dim[4]
    put_u8(buf, 84, 1);                               // element_stride[0]
    put_u8(buf, 85, 1);                               // element_stride[1]
    put_u8(buf, 86, 1);                               // element_stride[2]
    put_u8(buf, 87, 1);                               // element_stride[3]
    put_u32(buf, 88, 2);                              // rank=2 (low byte)
    put_u8(buf, 92, 6);                               // elemtype FLOAT16
    put_u8(buf, 93, 0);                               // interleave NONE
    put_u8(buf, 94, 0);                               // swizzle NONE
    put_u8(buf, 95, 0);                               // fill_mode ZERO
    return buf;
}

}  // namespace

// ============================================================================
// TEST_CASE 1: parse_known_128b_descriptor_f16
// Verifies the canonical f16 fixture round-trips through the parser with
// every field correctly extracted. This is the "happy path" anchor — if it
// fails, every other test is suspect.
// ============================================================================
TEST_CASE("parse_known_128b_descriptor_f16", "[tma][memory]") {
    auto bytes = make_valid_f16_descriptor();
    TmaDescriptor d = parse_descriptor_bytes(bytes.data(), bytes.size());

    REQUIRE(d.global_address == 0x1000);
    REQUIRE(d.global_dim[0] == 256);
    REQUIRE(d.global_dim[1] == 128);
    REQUIRE(d.global_dim[2] == 0);
    REQUIRE(d.global_dim[3] == 0);
    REQUIRE(d.global_dim[4] == 0);
    REQUIRE(d.global_stride[0] == 512);
    REQUIRE(d.global_stride[1] == 65536);
    REQUIRE(d.global_stride[2] == 0);
    REQUIRE(d.global_stride[3] == 0);
    REQUIRE(d.box_dim[0] == 16);
    REQUIRE(d.box_dim[1] == 16);
    REQUIRE(d.box_dim[2] == 1);
    REQUIRE(d.box_dim[3] == 1);
    REQUIRE(d.box_dim[4] == 1);
    REQUIRE(d.element_stride[0] == 1);
    REQUIRE(d.element_stride[1] == 1);
    REQUIRE(d.element_stride[2] == 1);
    REQUIRE(d.element_stride[3] == 1);
    REQUIRE(d.rank == 2);
    REQUIRE(d.elemtype == 6);   // FLOAT16
    REQUIRE(d.interleave_layout == 0);
    REQUIRE(d.swizzle_mode == 0);
    REQUIRE(d.fill_mode == 0);
}

// ============================================================================
// TEST_CASE 2: parse_dtype_variants
// Walks every dtype enum value the parser must recognize. Phase 0.1 does
// not interpret dtype semantics; it only round-trips the byte. Future
// tcgen05.mma handlers will consume d.elemtype.
// ============================================================================
TEST_CASE("parse_dtype_variants", "[tma][memory]") {
    const std::array<uint8_t, 12> dtypes = {
        0,   // UINT8
        1,   // UINT16
        2,   // UINT32
        3,   // INT32
        4,   // UINT64
        5,   // INT64
        6,   // FLOAT16
        7,   // FLOAT32
        9,   // FLOAT64
        10,  // BFLOAT16
        11,  // TFLOAT32
        12,  // TFLOAT32_FTZ
    };
    for (uint8_t dt : dtypes) {
        auto bytes = make_valid_f16_descriptor();
        put_u8(bytes, 92, dt);
        TmaDescriptor d = parse_descriptor_bytes(bytes.data(), bytes.size());
        REQUIRE(d.elemtype == dt);
    }
}

// ============================================================================
// TEST_CASE 3: parse_swizzle_variants
// Swizzle mode affects how tcgen05.ld decomposes addresses. Phase 0.1 must
// preserve the raw encoding; downstream consumers in Phase 1-3 will map
// 0/1/2/3 to NONE/32B/64B/128B.
// ============================================================================
TEST_CASE("parse_swizzle_variants", "[tma][memory]") {
    const std::array<uint8_t, 4> swizzles = {0, 1, 2, 3};
    for (uint8_t sw : swizzles) {
        auto bytes = make_valid_f16_descriptor();
        put_u8(bytes, 94, sw);
        TmaDescriptor d = parse_descriptor_bytes(bytes.data(), bytes.size());
        REQUIRE(d.swizzle_mode == sw);
    }
}

// ============================================================================
// TEST_CASE 4: parse_interleave_variants
// Interleave layout changes the granularity of element packing. 0/1/2 map
// to NONE/16B/32B per CUDA Driver API.
// ============================================================================
TEST_CASE("parse_interleave_variants", "[tma][memory]") {
    const std::array<uint8_t, 3> interleave = {0, 1, 2};
    for (uint8_t il : interleave) {
        auto bytes = make_valid_f16_descriptor();
        put_u8(bytes, 93, il);
        TmaDescriptor d = parse_descriptor_bytes(bytes.data(), bytes.size());
        REQUIRE(d.interleave_layout == il);
    }
}

// ============================================================================
// TEST_CASE 5: parse_rank_variants
// TMA descriptors may carry rank 1..5. The parser must extract the low
// byte of offset 88 without corrupting adjacent control bits.
// ============================================================================
TEST_CASE("parse_rank_variants", "[tma][memory]") {
    const std::array<uint32_t, 5> ranks = {1, 2, 3, 4, 5};
    for (uint32_t r : ranks) {
        auto bytes = make_valid_f16_descriptor();
        put_u32(bytes, 88, r);
        TmaDescriptor d = parse_descriptor_bytes(bytes.data(), bytes.size());
        REQUIRE(d.rank == r);
    }
}

// ============================================================================
// TEST_CASE 6: parse_invalid_size_too_small
// Buffers smaller than 128 bytes must throw std::runtime_error. Silent
// acceptance would let tcgen05 handlers read uninitialized memory.
// ============================================================================
TEST_CASE("parse_invalid_size_too_small", "[tma][memory]") {
    auto bytes = make_valid_f16_descriptor();
    bytes.resize(64);  // too small
    REQUIRE_THROWS_AS(
        parse_descriptor_bytes(bytes.data(), bytes.size()),
        std::runtime_error);
}

// ============================================================================
// TEST_CASE 7: parse_invalid_size_too_large
// Buffers larger than 128 bytes also throw — the opaque descriptor has a
// fixed size per PTX ISA §9.7.13. Allowing slack would mask upstream
// allocation bugs.
// ============================================================================
TEST_CASE("parse_invalid_size_too_large", "[tma][memory]") {
    auto bytes = make_valid_f16_descriptor();
    bytes.resize(256);  // too large
    REQUIRE_THROWS_AS(
        parse_descriptor_bytes(bytes.data(), bytes.size()),
        std::runtime_error);
}

// ============================================================================
// TEST_CASE 8: parse_null_bytes_throws
// Defensive: a null pointer must not crash — must throw runtime_error.
// ============================================================================
TEST_CASE("parse_null_bytes_throws", "[tma][memory]") {
    REQUIRE_THROWS_AS(
        parse_descriptor_bytes(nullptr, kTmaDescriptorSize),
        std::runtime_error);
}

// ============================================================================
// TEST_CASE 9: parse_reserved_bytes_nonzero_kept
// Reserved bytes at offsets 28-31 and 96-127 are not interpreted today.
// Phase 0.1 preserves them (zeroed by fixtures, but non-zero input must
// not throw — it may carry future im2col fields). We assert the parser
// does not inspect reserved bytes and they round-trip via raw_bytes().
// ============================================================================
TEST_CASE("parse_reserved_bytes_nonzero_kept", "[tma][memory]") {
    auto bytes = make_valid_f16_descriptor();
    // Stamp reserved regions with non-zero markers.
    put_u32(bytes, 28, 0xDEADBEEF);
    for (size_t i = 96; i < kTmaDescriptorSize; ++i) bytes[i] = 0xAB;
    TmaDescriptor d = parse_descriptor_bytes(bytes.data(), bytes.size());
    // Parsing must succeed despite non-zero reserved bytes.
    REQUIRE(d.global_address == 0x1000);
    REQUIRE(d.rank == 2);
    // raw_bytes() must preserve the original 128 bytes verbatim.
    REQUIRE(d.raw_bytes.size() == kTmaDescriptorSize);
    REQUIRE(d.raw_bytes[28] == 0xEF);
    REQUIRE(d.raw_bytes[29] == 0xBE);
    REQUIRE(d.raw_bytes[30] == 0xAD);
    REQUIRE(d.raw_bytes[31] == 0xDE);
    REQUIRE(d.raw_bytes[96] == 0xAB);
    REQUIRE(d.raw_bytes[127] == 0xAB);
}

// ============================================================================
// TEST_CASE 10: TmaDescriptorStore_basic
// Store / load round-trip by CTA id. Phase 0.5 will wire this into
// CTAContext; here we exercise the standalone container.
// ============================================================================
TEST_CASE("TmaDescriptorStore_basic", "[tma][memory]") {
    TmaDescriptorStore store;
    auto bytes = make_valid_f16_descriptor();
    TmaDescriptor d = parse_descriptor_bytes(bytes.data(), bytes.size());

    store.store(0, d);
    REQUIRE(store.has(0));
    const auto* loaded = store.load(0);
    REQUIRE(loaded != nullptr);
    REQUIRE(loaded->global_address == d.global_address);
    REQUIRE(loaded->rank == d.rank);
    REQUIRE(loaded->elemtype == d.elemtype);
}

// ============================================================================
// TEST_CASE 11: TmaDescriptorStore_multi_cta_isolation
// Two CTAs storing descriptors with different values must not collide.
// This locks in per-CTA isolation before Phase 0.5 wires the store into
// CTAContext.
// ============================================================================
TEST_CASE("TmaDescriptorStore_multi_cta_isolation", "[tma][memory]") {
    TmaDescriptorStore store;
    auto b1 = make_valid_f16_descriptor();
    auto b2 = make_valid_f16_descriptor();
    put_u64(b2, 0, 0x2000);  // different address

    TmaDescriptor d1 = parse_descriptor_bytes(b1.data(), b1.size());
    TmaDescriptor d2 = parse_descriptor_bytes(b2.data(), b2.size());

    store.store(7, d1);
    store.store(11, d2);

    REQUIRE(store.has(7));
    REQUIRE(store.has(11));
    REQUIRE_FALSE(store.has(99));
    REQUIRE(store.load(7)->global_address == 0x1000);
    REQUIRE(store.load(11)->global_address == 0x2000);
}

// ============================================================================
// TEST_CASE 12: TmaDescriptorStore_overwrite
// Storing twice to the same CTA id overwrites the previous descriptor.
// tcgen05 host API may re-issue descriptors for the same slot.
// ============================================================================
TEST_CASE("TmaDescriptorStore_overwrite", "[tma][memory]") {
    TmaDescriptorStore store;
    auto b1 = make_valid_f16_descriptor();
    auto b2 = make_valid_f16_descriptor();
    put_u64(b2, 0, 0x3000);

    TmaDescriptor d1 = parse_descriptor_bytes(b1.data(), b1.size());
    TmaDescriptor d2 = parse_descriptor_bytes(b2.data(), b2.size());

    store.store(3, d1);
    REQUIRE(store.load(3)->global_address == 0x1000);
    store.store(3, d2);
    REQUIRE(store.load(3)->global_address == 0x3000);
}

// ============================================================================
// TEST_CASE 13: TmaDescriptorStore_clear
// Clearing must drop all entries. Used at CTA teardown (Phase 0.5).
// ============================================================================
TEST_CASE("TmaDescriptorStore_clear", "[tma][memory]") {
    TmaDescriptorStore store;
    auto bytes = make_valid_f16_descriptor();
    TmaDescriptor d = parse_descriptor_bytes(bytes.data(), bytes.size());
    store.store(0, d);
    store.store(1, d);
    store.store(2, d);
    REQUIRE(store.has(0));
    store.clear();
    REQUIRE_FALSE(store.has(0));
    REQUIRE_FALSE(store.has(1));
    REQUIRE_FALSE(store.has(2));
    REQUIRE(store.load(0) == nullptr);
}

// ============================================================================
// TEST_CASE 14: parse_global_stride_alignment_constraints
// Validates the constraint: global_stride[i] must be multiple of 16 when
// interleave=NONE (32 when interleave=32B). Misaligned stride must throw.
//
// PTX ISA §9.7.13 / CUDA Driver cuTensorMapEncodeTiled: stride is in bytes
// and must respect the interleave granularity.
// ============================================================================
TEST_CASE("parse_global_stride_alignment_constraints", "[tma][memory]") {
    SECTION("aligned strides accepted") {
        auto bytes = make_valid_f16_descriptor();
        // 512 and 65536 are multiples of 16 — already set by fixture.
        REQUIRE_NOTHROW(parse_descriptor_bytes(bytes.data(), bytes.size()));
    }
    SECTION("misaligned stride (no interleave) throws") {
        auto bytes = make_valid_f16_descriptor();
        put_u64(bytes, 32, 17);  // not a multiple of 16
        REQUIRE_THROWS_AS(
            parse_descriptor_bytes(bytes.data(), bytes.size()),
            std::runtime_error);
    }
    SECTION("stride=0 accepted for unused dims") {
        auto bytes = make_valid_f16_descriptor();
        // Rank-2 means strides[2..3] are unused; 0 is the conventional fill.
        REQUIRE_NOTHROW(parse_descriptor_bytes(bytes.data(), bytes.size()));
    }
}

// ============================================================================
// TEST_CASE 15: parse_box_dim_bounds
// box_dim[i] must be in [1, 256]. Out-of-range values must throw.
// ============================================================================
TEST_CASE("parse_box_dim_bounds", "[tma][memory]") {
    SECTION("valid bounds accepted") {
        auto bytes = make_valid_f16_descriptor();
        put_u32(bytes, 64, 1);
        put_u32(bytes, 68, 256);
        REQUIRE_NOTHROW(parse_descriptor_bytes(bytes.data(), bytes.size()));
    }
    SECTION("box_dim=0 throws") {
        auto bytes = make_valid_f16_descriptor();
        put_u32(bytes, 64, 0);
        REQUIRE_THROWS_AS(
            parse_descriptor_bytes(bytes.data(), bytes.size()),
            std::runtime_error);
    }
    SECTION("box_dim>256 throws") {
        auto bytes = make_valid_f16_descriptor();
        put_u32(bytes, 64, 257);
        REQUIRE_THROWS_AS(
            parse_descriptor_bytes(bytes.data(), bytes.size()),
            std::runtime_error);
    }
}

// ============================================================================
// TEST_CASE 16: parse_rank_bounds
// rank must be in [1, 5]. Out-of-range values throw.
// ============================================================================
TEST_CASE("parse_rank_bounds", "[tma][memory]") {
    SECTION("rank=0 throws") {
        auto bytes = make_valid_f16_descriptor();
        put_u32(bytes, 88, 0);
        REQUIRE_THROWS_AS(
            parse_descriptor_bytes(bytes.data(), bytes.size()),
            std::runtime_error);
    }
    SECTION("rank=6 throws") {
        auto bytes = make_valid_f16_descriptor();
        put_u32(bytes, 88, 6);
        REQUIRE_THROWS_AS(
            parse_descriptor_bytes(bytes.data(), bytes.size()),
            std::runtime_error);
    }
}

// ============================================================================
// TEST_CASE 17: parse_global_address_alignment
// global_address must be 16-byte aligned (PTX ISA §9.7.13 / Driver API).
// Unaligned address throws.
// ============================================================================
TEST_CASE("parse_global_address_alignment", "[tma][memory]") {
    auto bytes = make_valid_f16_descriptor();
    put_u64(bytes, 0, 0x1001);  // not 16-byte aligned
    REQUIRE_THROWS_AS(
        parse_descriptor_bytes(bytes.data(), bytes.size()),
        std::runtime_error);
}

// ============================================================================
// TEST_CASE 18: parse_element_stride_bounds
// element_stride[i] must be in [1, 8]. Out-of-range throws.
// ============================================================================
TEST_CASE("parse_element_stride_bounds", "[tma][memory]") {
    SECTION("valid stride=8 accepted") {
        auto bytes = make_valid_f16_descriptor();
        put_u8(bytes, 84, 8);
        REQUIRE_NOTHROW(parse_descriptor_bytes(bytes.data(), bytes.size()));
    }
    SECTION("element_stride=0 throws") {
        auto bytes = make_valid_f16_descriptor();
        put_u8(bytes, 84, 0);
        REQUIRE_THROWS_AS(
            parse_descriptor_bytes(bytes.data(), bytes.size()),
            std::runtime_error);
    }
    SECTION("element_stride=9 throws") {
        auto bytes = make_valid_f16_descriptor();
        put_u8(bytes, 84, 9);
        REQUIRE_THROWS_AS(
            parse_descriptor_bytes(bytes.data(), bytes.size()),
            std::runtime_error);
    }
}
