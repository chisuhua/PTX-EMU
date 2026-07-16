// test_stream_sync_loop.cpp
// =============================================================================
// Unit test: cudaStreamSynchronize / cudaDeviceSynchronize polling loop (B2)
//
// Per Metis second-pass review (B2): sync functions were single-pass poll
// that returned even when kernels were still pending. Violated CUDA spec
// requirement that sync blocks until completion.
//
// This test verifies the contract:
//   1. cudaStreamSynchronize with a bridge attached does not crash or hang
//      when g_pending_kernels is empty (loop exits immediately).
//   2. cudaDeviceSynchronize with a bridge attached returns success.
//   3. nullptr bridge path is immediate (no polling).
//   4. A bridge whose poll_kernel returns >0 on first poll and 0 on second
//      does not cause sync to hang on empty pending set (loop exits via
//      "no pending" check, not via poll completion).
//
// NOTE: g_pending_kernels is static in cudart_sim.cpp, so we cannot directly
// populate it without a registered fat binary + cudaLaunchKernel. Full loop
// drain behavior is validated by the e2e suite (e2e_barrier_warp_sync etc.)
// which exercises cudaLaunchKernel -> cudaStreamSynchronize. This unit test
// guards the no-hang contract and the bridge-attached immediate-return path.
//
// Ref: docs/adr/0021-cpptlm-d1-full-integration.md
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cpptlm_bridge.h"
#include "cudart/cudart_intrinsics.h"  // cudaStream_t (void*), cudaSuccess, cudaError_t

#include <atomic>
#include <unordered_map>
#include <vector>

namespace {

// Mock bridge whose poll_kernel requires N polls before returning 0.
class PollCountingBridge : public CppTLMBridge {
public:
    int version() const override { return CPPTLMBRIDGE_VERSION; }

    int submit_kernel(uint64_t kernel_id, const char*, uint32_t, uint32_t,
                      uint32_t, uint32_t, uint32_t, uint32_t, const void**,
                      size_t, size_t, uint64_t) override {
        std::lock_guard<std::mutex> lk(mu_);
        poll_counts_[kernel_id] = 0;
        return 0;
    }

    uint64_t poll_kernel(uint64_t kernel_id) override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = poll_counts_.find(kernel_id);
        if (it == poll_counts_.end()) {
            return UINT64_MAX;  // unknown kernel - caller treats as completed
        }
        it->second += 1;
        if (it->second >= polls_required_) {
            poll_counts_.erase(it);
            return 0;  // completed
        }
        return static_cast<uint64_t>(polls_required_ - it->second);
    }

    int synchronize_stream(uint64_t) override { return 0; }
    uint64_t global_access(uint64_t, uint64_t, uint8_t) override { return 0; }

    size_t pending_count() {
        std::lock_guard<std::mutex> lk(mu_);
        return poll_counts_.size();
    }
    void set_polls_required(uint32_t n) { polls_required_ = n; }

private:
    std::mutex mu_;
    std::unordered_map<uint64_t, uint32_t> poll_counts_;
    uint32_t polls_required_ = 3;
};

}  // namespace

// Globals + ABI entry points defined in cudart_sim.cpp.
extern CppTLMBridge* g_cpptlm_bridge;
extern "C" {
void cpptlm_attach_bridge(CppTLMBridge* bridge);
void cpptlm_detach_bridge();
}

// CUDA runtime entry points (C linkage, defined in cudart_sim.cpp).
extern "C" {
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaDeviceSynchronize();
}

TEST_CASE("cudaStreamSynchronize with bridge, no pending kernels, returns immediately", "[cudart][stream][sync][b2]") {
    PollCountingBridge bridge;
    bridge.set_polls_required(3);
    cpptlm_attach_bridge(&bridge);
    REQUIRE(g_cpptlm_bridge == &bridge);

    // Empty g_pending_kernels -> the while-loop must exit via the
    // "no pending for this stream" check, NOT by polling to completion.
    // If the loop had no exit condition (pure spin), this would hang.
    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(1));
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);

    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);
}

TEST_CASE("cudaDeviceSynchronize with bridge returns success", "[cudart][stream][sync][b2]") {
    PollCountingBridge bridge;
    bridge.set_polls_required(1);
    cpptlm_attach_bridge(&bridge);
    REQUIRE(g_cpptlm_bridge == &bridge);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);
}

TEST_CASE("cudaStreamSynchronize nullptr bridge (default path) is immediate", "[cudart][stream][sync][b2]") {
    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);

    // nullptr bridge: sync is immediate (no polling). Must not hang.
    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(1));
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("cudaDeviceSynchronize with bridge, polls_required=2, empty pending, no hang", "[cudart][stream][sync][b2]") {
    // Regression guard: a bridge requiring multiple polls must not cause
    // sync to hang when there are no pending kernels. The loop's exit
    // condition (no pending kernels for stream/all) must fire on the
    // first iteration regardless of poll_kernel's configured behavior.
    PollCountingBridge bridge;
    bridge.set_polls_required(2);
    cpptlm_attach_bridge(&bridge);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);

    cpptlm_detach_bridge();
    REQUIRE(g_cpptlm_bridge == nullptr);
}
