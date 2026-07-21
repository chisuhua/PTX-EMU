#ifndef CUDART_STUB_BRIDGE_H
#define CUDART_STUB_BRIDGE_H

// StubBridge: 零延迟 CppTLMBridge 实现（auto-co-sim-standalone D1）
//
// 五点虚方法：
//   - submit_kernel: 记录 kernel_id，立即返回 0
//   - poll_kernel:   已知 id → 0（完成）；未知 id → UINT64_MAX（错误）
//   - synchronize_stream: 立即返回 0（总是同步）
//   - global_access: 返回 0（零延迟 stub，不建模 NoC）
//   - version:       返回 CPPTLMBRIDGE_VERSION
//
// submitted_ids_ 由 std::mutex 保护，与 PtxEmuDriverShim 保持一致的
// 线程安全模式（当前 host 单线程模型下无害，但预防未来 CppTLM 多线程调用）。

#include "cudart/cpptlm_bridge.h"

#include <cstdint>
#include <mutex>
#include <unordered_set>

class StubBridge : public CppTLMBridge {
public:
    int version() const override { return CPPTLMBRIDGE_VERSION; }

    int submit_kernel(uint64_t kernel_id,
                      const char* /*kernel_name*/,
                      uint32_t /*grid_x*/, uint32_t /*grid_y*/,
                      uint32_t /*grid_z*/,
                      uint32_t /*block_x*/, uint32_t /*block_y*/,
                      uint32_t /*block_z*/,
                      const void** /*kernel_args*/, size_t /*args_count*/,
                      size_t /*shared_mem*/,
                      uint64_t /*stream_id*/) override {
        std::lock_guard<std::mutex> lock(mu_);
        submitted_ids_.insert(kernel_id);
        return 0;
    }

    uint64_t poll_kernel(uint64_t kernel_id) override {
        std::lock_guard<std::mutex> lock(mu_);
        return submitted_ids_.count(kernel_id) ? 0 : UINT64_MAX;
    }

    int synchronize_stream(uint64_t /*stream_id*/) override {
        return 0;
    }

    uint64_t global_access(uint64_t /*device_addr*/, uint64_t /*val*/,
                           uint8_t /*type*/) override {
        return 0;  // zero-latency stub — no NoC model
    }

private:
    mutable std::mutex mu_;
    std::unordered_set<uint64_t> submitted_ids_;
};

#endif  // CUDART_STUB_BRIDGE_H
