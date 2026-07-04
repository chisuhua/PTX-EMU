// src/ptxsim/memory/tma_descriptor.cpp
// Phase 0.1 (Fix #5): TMA descriptor parsing implementation.
//
// 128-byte opaque CUtensorMap layout parsing with strict validation.
// See header for layout notes (all offsets UNVERIFIED-AGAINST-HARDWARE).
//
// Per ptx-lessons-learned §1（跨模块间接状态翻译）and the Phase 0 设计
// 检查清单，每个 thrown 异常必须提供清晰的规则文档，避免下游
// 维护者（如 Phase 1-3 tcgen05 handler 开发者）误判"为什么这段代码
// 不能通过"。

#include "ptxsim/memory/tma_descriptor.h"

#include <array>
#include <stdexcept>

// Helper: throw runtime_error with structured message.
inline void throw_error(const char* msg) {
    throw std::runtime_error(msg);
}

// note_reserved_bytes: explicit no-op observation hook for bytes 28-31.
// PTX ISA §9.7.13 reserves offset 28 (bytes 28-31). Modern toolchains
// zero this slot, but Blackwell im2col variants may carry pixelBoxLower/
// pixelBoxUpper/channelsPerPixel/pixelsPerColumn there. Phase 0.1 does not
// parse im2col, so this function intentionally ignores the bytes. It exists
// as a single point of control: a future change to support im2col can lift
// this into a real validator without hunting through parse_descriptor_bytes().
//
// Returning the bool "observed_nonzero" is a deliberate hook for loggers
// (debug build only) and tests; production code MUST NOT throw or alter
// descriptor state based on this signal.
// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
inline bool note_reserved_bytes(const uint8_t* off, const char* /*name*/) {
    return (*off != 0 || *(off + 1) != 0 ||
            *(off + 2) != 0 || *(off + 3) != 0);
}

// validate_alignment: 16-byte alignment check.
// GRANT: global_address [PTX ISA §9.7.13 / Driver API]。
inline bool is_16byte_aligned(uint64_t addr) {
    return (addr & 0xF) == 0;
}

// validate_stride_multiple: checks stride[i] % granularity == 0.
// interleave=NONE → granularity of 16B (PTX ISA §9.7.13 / Driver API)。
inline bool is_stride_aligned_to_16(uint64_t stride) {
    return (stride % 16) == 0;
}

// validate_box_dim: checks each dim is in [1, 256].
inline bool is_box_dim_valid(uint32_t dim) {
    return dim >= 1 && dim <= 256;
}

// validate_element_stride: checks each stride in [1, 8].
inline bool is_element_stride_valid(uint8_t stride) {
    return stride >= 1 && stride <= 8;
}

// validate_rank: checks rank is in [1, 5].
inline bool is_rank_valid(uint32_t rank) {
    return rank >= 1 && rank <= 5;
}

// 位操作辅助：lev硬化 endian packing rules。
inline uint64_t read_u64_le(const uint8_t* off) {
    return static_cast<uint64_t>(off[0]) |
           (static_cast<uint64_t>(off[1]) << 8) |
           (static_cast<uint64_t>(off[2]) << 16) |
           (static_cast<uint64_t>(off[3]) << 24) |
           (static_cast<uint64_t>(off[4]) << 32) |
           (static_cast<uint64_t>(off[5]) << 40) |
           (static_cast<uint64_t>(off[6]) << 48) |
           (static_cast<uint64_t>(off[7]) << 56);
}
inline uint32_t read_u32_le(const uint8_t* off) {
    return static_cast<uint32_t>(off[0]) |
           (static_cast<uint32_t>(off[1]) << 8) |
           (static_cast<uint32_t>(off[2]) << 16) |
           (static_cast<uint32_t>(off[3]) << 24);
}

TmaDescriptor parse_descriptor_bytes(const void* bytes, size_t size) {
    // 阶段 0: 空指针与尺寸校验（防御性编程）。
    if (bytes == nullptr) {
        throw_error("parse_descriptor_bytes: null input pointer");
    }
    if (size != kTmaDescriptorSize) {
        throw_error("parse_descriptor_bytes: descriptor size must be 128 bytes");
    }

    const uint8_t* data = static_cast<const uint8_t*>(bytes);
    TmaDescriptor desc;
    desc.raw_bytes.assign(data, data + size);

    // offset 0..7: global_address (uint64).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    desc.global_address = read_u64_le(data + 0);
    if (!is_16byte_aligned(desc.global_address)) {
        throw_error("parse_descriptor_bytes: global_address not 16-byte aligned");
    }

    // offset 8..27: global_dim[0..4] (5 × uint32).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    for (size_t i = 0; i < 5; ++i) {
        desc.global_dim[i] = read_u32_le(data + 8 + i * 4);
    }

    // offset 28..31: RESERVED. Observe-only hook; see note_reserved_bytes().
    (void)note_reserved_bytes(data + 28, "global_dummy_for_reserved");

    // offset 32..63: global_stride[0..3] (4 × uint64).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    for (size_t i = 0; i < 4; ++i) {
        desc.global_stride[i] = read_u64_le(data + 32 + i * 8);
    }

    // 输入校验：global_stride 必须为 0 或 16 的倍数（interleave=NONE）。
    // PTX ISA §9.7.13 / CUDA Driver cuTensorMapEncodeTiled：stride 是字节
    // 数且必须对齐到 interleave 粒度。Phase 0.1 仅支持 interleave=NONE
    // → 粒度为 16B。非零且非 16 倍数的 stride 抛 runtime_error，避免
    // Phase 1-3 tcgen05.ld handler 拿到错误对齐的 stride 后产生非确定性
    // 地址翻译（参见 ptx-lessons-learned §5 — 类型/对齐误判是最隐蔽的
    // bug 来源）。
    for (size_t i = 0; i < 4; ++i) {
        if (desc.global_stride[i] != 0 &&
            !is_stride_aligned_to_16(desc.global_stride[i])) {
            throw_error("parse_descriptor_bytes: global_stride not 16-byte aligned");
        }
    }

    // offset 64..83: box_dim[0..4] (5 × uint32).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    for (size_t i = 0; i < 5; ++i) {
        desc.box_dim[i] = read_u32_le(data + 64 + i * 4);

        // 输入校验：box_dim 在 [1, 256] 范围内。
        if (!is_box_dim_valid(desc.box_dim[i])) {
            throw_error("parse_descriptor_bytes: box_dim out of valid range [1,256]");
        }
    }

    // offset 84..87: element_stride[0..3] (4 × uint8 packed)。
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    for (size_t i = 0; i < 4; ++i) {
        desc.element_stride[i] = data[84 + i];
    }

    // 输入校验：element_stride 在 [1, 8] 范围内。
    for (size_t i = 0; i < 4; ++i) {
        if (!is_element_stride_valid(desc.element_stride[i])) {
            throw_error("parse_descriptor_bytes: element_stride out of valid range [1,8]");
        }
    }

    // offset 88..91: rank + control flags. 仅解析低字节的 rank（PTX ISA §9.7.13）。
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    uint32_t rank_ctrl = read_u32_le(data + 88);
    desc.rank = rank_ctrl & 0xFF;

    // 输入校验：rank ∈ [1, 5]。
    if (!is_rank_valid(desc.rank)) {
        throw_error("parse_descriptor_bytes: rank out of valid range [1,5]");
    }

    // offset 92: elemtype (uint8).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    desc.elemtype = data[92];

    // offset 93: interleave_layout (uint8).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    desc.interleave_layout = data[93];

    // offset 94: swizzle_mode (uint8).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    desc.swizzle_mode = data[94];

    // offset 95: fill_mode (uint8).
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    desc.fill_mode = data[95];

    // offset 96..127: RESERVED / im2col fields, 逐字节复制到 raw_bytes。
    // 当前用于未来字段兼容性提升，不变动业务语义。
    return desc;
}

void TmaDescriptorStore::store(uint32_t cta_id, const TmaDescriptor& descriptor) {
    store_[cta_id] = descriptor;
}

const TmaDescriptor* TmaDescriptorStore::load(uint32_t cta_id) const {
    auto it = store_.find(cta_id);
    if (it == store_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool TmaDescriptorStore::has(uint32_t cta_id) const {
    return store_.find(cta_id) != store_.end();
}

void TmaDescriptorStore::clear() {
    store_.clear();
}
