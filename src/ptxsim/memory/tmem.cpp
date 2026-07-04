// src/ptxsim/memory/tmem.cpp
// Phase 0.2 (Fix #6): per-CTA Tensor Memory (TMEM) implementation.
//
// 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13.
// Per ptx-lessons-learned §2（递归锁死锁）: 每个 public 方法独立持锁，
// 且不调用任何其他 public 方法。read 使用 mutable mutex 保证 const 安全。

#include "ptxsim/memory/tmem.h"

#include <cstring>
#include <stdexcept>

// Helper: throw runtime_error with structured message (mirrors Phase 0.1 pattern).
inline void throw_error(const char* msg) { throw std::runtime_error(msg); }

Tmem::Tmem() { clear(); }

Tmem::~Tmem() = default;

bool Tmem::validate_slot_id(size_t slot_id) const {
    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    return slot_id < kSlotCount;
}

void Tmem::read(size_t slot_id, void* bytes, size_t size) const {
    std::lock_guard<std::mutex> lock(mu_);

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    if (!validate_slot_id(slot_id))
        throw_error("Tmem::read: slot_id out of range");

    if (size > kSlotSize)
        throw_error("Tmem::read: size exceeds 128-byte slot capacity");

    if (size == 0) return;

    size_t offset = slot_id * kSlotSize;
    std::memcpy(bytes, storage_.data() + offset, size);
}

void Tmem::write(size_t slot_id, const void* bytes, size_t size) {
    std::lock_guard<std::mutex> lock(mu_);

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    if (!validate_slot_id(slot_id))
        throw_error("Tmem::write: slot_id out of range");

    if (size > kSlotSize)
        throw_error("Tmem::write: size exceeds 128-byte slot capacity");

    if (size == 0) return;

    size_t offset = slot_id * kSlotSize;
    std::memcpy(storage_.data() + offset, bytes, size);
}

void Tmem::clear() {
    std::lock_guard<std::mutex> lock(mu_);

    // 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13
    storage_.fill(0);
}