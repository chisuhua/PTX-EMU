// src/ptxsim/memory/tma_descriptor.h
// Phase 0.1 (Fix #5): Blackwell TMA TensorMap descriptor parser.
//
// 128-byte opaque CUtensorMap layout (PTX ISA §9.7.13) parsed from raw
// bytes captured from device memory. Used by tcgen05.ld in Phase 1-3.
//
// ---------------------------------------------------------------------------
// LAYOUT NOTES — UNVERIFIED-AGAINST-HARDWARE
// ---------------------------------------------------------------------------
// The byte offsets below were INFERRED from three sources, none of which
// is a hardware dump:
//   1. NVIDIA PTX ISA §9.7.13 "tensormap.replace" field ordinals
//      (the .replace intrinsic lets PTX patch one field at a time, which
//       exposes the ordinal → field mapping, but NOT the byte offset).
//   2. CUDA Driver API CUtensorMap (cuTensorMapEncodeTiled) param order
//      and declared C type sizes — gives field *size* and *order* but
//      not packing/padding.
//   3. LLVM NVPTX target LowerCUtensorMap pass — gives a working layout
//      that the Linux driver accepts, but NVIDIA may pack differently.
//
// Gate G5 (per proposal.md): a manual cross-check of these offsets against
// a real cuTensorMapEncodeTiled output (or a future cuobjdump -xptx dump)
// is REQUIRED before any tcgen05.ld handler consumes this struct. Until
// then, every magic number in tma_descriptor.cpp is annotated
//   // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
// and any offset drift would surface as a unit-test failure (the tests
// in tests/unit/memory/test_tma_descriptor.cpp encode the same assumed
// offsets, so a coordinated shift is required to silently regress).
// ---------------------------------------------------------------------------

#ifndef PTXSIM_MEMORY_TMA_DESCRIPTOR_H
#define PTXSIM_MEMORY_TMA_DESCRIPTOR_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

// Total size of an opaque CUtensorMap descriptor.
// PTX ISA §9.7.13: "128 bytes (16 × uint64_t), alignas(64)".
// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
inline constexpr size_t kTmaDescriptorSize = 128;

// Maximum number of dimensions supported by a TMA tensor map.
// PTX ISA §9.7.13: "rank in [1, 5]".
// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
inline constexpr uint32_t kTmaMaxRank = 5;

// Parsed TMA TensorMap descriptor.
//
// Field order mirrors the documented CUDA Driver API
// cuTensorMapEncodeTiled parameter list. Every numeric field carries a
// comment citing the PTX ISA section that defines it; see also the
// per-offset annotations in tma_descriptor.cpp.
struct TmaDescriptor {
    // offset 0..7: global_address — base device pointer. Must be 16-byte
    // aligned (PTX ISA §9.7.13 / Driver API).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint64_t global_address = 0;

    // offset 8..27: global_dim[0..4] — per-dimension element count. Unused
    // dims (rank < 5) are conventionally 0.
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint32_t global_dim[kTmaMaxRank] = {0, 0, 0, 0, 0};

    // offset 28..31: RESERVED. PTX ISA does not document this slot; it is
    // skipped by the parser. raw_bytes preserves the original value.
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13

    // offset 32..63: global_stride[0..3] — byte stride per dim, must be a
    // multiple of 16 (or 32 when interleave=32B). Unused dims are 0.
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint64_t global_stride[4] = {0, 0, 0, 0};

    // offset 64..83: box_dim[0..4] — per-dim box size, in [1, 256].
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint32_t box_dim[kTmaMaxRank] = {0, 0, 0, 0, 0};

    // offset 84..87: element_stride[0..3] — packed 4× uint8, in [1, 8].
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint8_t element_stride[4] = {0, 0, 0, 0};

    // offset 88..91: rank + control flags. Only the low byte is rank;
    // higher bytes are reserved for future control bits.
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint32_t rank = 0;

    // offset 92: elemtype — CU_TENSOR_MAP_DATA_TYPE_* enum (see .cpp for
    // the full value table). Phase 0.1 round-trips the raw byte.
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint8_t elemtype = 0;

    // offset 93: interleave_layout — 0=NONE, 1=16B, 2=32B.
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint8_t interleave_layout = 0;

    // offset 94: swizzle_mode — 0=NONE, 1=32B, 2=64B, 3=128B, 4=96B
    // (Blackwell extension).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint8_t swizzle_mode = 0;

    // offset 95: fill_mode — 0=ZERO, 1=NaN (OOB fill behavior).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint8_t fill_mode = 0;

    // offset 96..127: RESERVED / im2col fields. PTX ISA §9.7.13 reserves
    // this for future use; Phase 0.1 preserves the raw bytes verbatim.
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13

    // Verbatim copy of the original 128-byte buffer. Retained so future
    // tcgen05 handlers can re-read reserved regions without re-parsing.
    std::vector<uint8_t> raw_bytes;
};

// Per-CTA store of parsed TMA descriptors.
//
// Phase 0.5 wires this into CTAContext. Phase 0.1 ships the standalone
// container with CTA-id-keyed insert / lookup / clear semantics, and
// asserts per-CTA isolation (two CTAs writing different descriptors must
// not collide).
//
// Thread-safety: NOT thread-safe. Phase 0 callers are single-threaded
// (test harness). Phase 0.5 will revisit when CTAContext lifetime is
// wired up.
class TmaDescriptorStore {
public:
    TmaDescriptorStore() = default;
    ~TmaDescriptorStore() = default;

    TmaDescriptorStore(const TmaDescriptorStore&) = delete;
    TmaDescriptorStore& operator=(const TmaDescriptorStore&) = delete;

    // Store (or overwrite) the descriptor for `cta_id`. Copies the struct
    // (including raw_bytes) into internal storage.
    void store(uint32_t cta_id, const TmaDescriptor& descriptor);

    // Returns nullptr if `cta_id` has no descriptor.
    const TmaDescriptor* load(uint32_t cta_id) const;

    bool has(uint32_t cta_id) const;

    // Drop every entry. Called at CTA teardown (Phase 0.5).
    void clear();

private:
    std::map<uint32_t, TmaDescriptor> store_;
};

// Parse a 128-byte opaque CUtensorMap descriptor from raw bytes.
//
// Throws std::runtime_error on any structural violation:
//   * `bytes == nullptr`
//   * `size != kTmaDescriptorSize`
//   * global_address not 16-byte aligned
//   * global_stride[i] not a multiple of 16 (when interleave=NONE) / 32
//   * box_dim[i] not in [1, 256]
//   * element_stride[i] not in [1, 8]
//   * rank not in [1, 5]
//
// Does NOT throw on: non-zero reserved bytes, unknown elemtype values
// (Phase 0.1 round-trips the raw byte — full dtype semantics live in
// Phase 1-3 tcgen05 handlers).
//
// All magic numbers / offsets in the implementation are annotated
// `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13`.
TmaDescriptor parse_descriptor_bytes(const void* bytes, size_t size);

#endif  // PTXSIM_MEMORY_TMA_DESCRIPTOR_H
