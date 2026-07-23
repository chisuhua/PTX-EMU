#ifndef PTX_EMU_DRIVER_SHIM_H
#define PTX_EMU_DRIVER_SHIM_H

// CppTLM D1-Full P1: PTX-EMU side IPtxEmuDriver implementation
// Provides advance() / inject_*() / is_kernel_complete() / mark_complete()
// as the PTX execution backend for CppTLM.
//
// CppTLM retrieves this instance via cpptlm_set_driver() and
// reinterpret_casts back to IPtxEmuDriver* before calling virtual methods.
// This file is always compiled (CppTLM is always linked since commit 84212a9d).

#include "ptxsim/scoreboard_interface.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/tensor_core_interface.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

class GPUContext;

class PtxEmuDriverShim {
public:
    explicit PtxEmuDriverShim(GPUContext* ctx);
    ~PtxEmuDriverShim();

    PtxEmuDriverShim(const PtxEmuDriverShim&) = delete;
    PtxEmuDriverShim& operator=(const PtxEmuDriverShim&) = delete;

    // ---- Driver Interface ----
    // Return: 0=NoOp, 1=Executed, 2=KernelComplete, -1=Error
    int advance(uint32_t max_cycles, uint32_t& actual);

    void inject_scoreboard(uint32_t sm_id,
                           std::unique_ptr<IScoreboard> sb);
    void inject_pipeline(uint32_t sm_id,
                         std::unique_ptr<IPipelineLatencyProvider> pp);
    void inject_tensor_core(uint32_t sm_id,
                            std::unique_ptr<ITensorCoreTiming> tc);

    bool is_kernel_complete(uint64_t kernel_id) const;
    void mark_complete(uint64_t kernel_id);
    uint32_t num_sms() const;

    GPUContext* get_gpu_context() const { return ctx_; }

    // Cross-.so raw pointer transfer
    void* raw_ptr() { return static_cast<void*>(this); }

private:
    GPUContext* ctx_;

    mutable std::mutex mu_;
    std::unordered_map<uint64_t, bool> completion_;

    std::vector<std::unique_ptr<IScoreboard>>              scoreboards_;
    std::vector<std::unique_ptr<IPipelineLatencyProvider>> pipelines_;
    std::vector<std::unique_ptr<ITensorCoreTiming>>        tensor_cores_;
};

#endif // PTX_EMU_DRIVER_SHIM_H
