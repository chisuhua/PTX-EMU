#include "PtxEmuDriverShim.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/sm_context.h"
#include "utils/logger.h"

PtxEmuDriverShim::PtxEmuDriverShim(GPUContext* ctx) : ctx_(ctx) {
    PTX_DEBUG_EMU("PtxEmuDriverShim created (ctx=%p)", static_cast<void*>(ctx_));
}

PtxEmuDriverShim::~PtxEmuDriverShim() {
    PTX_DEBUG_EMU("PtxEmuDriverShim destroyed");
    // unique_ptr members auto-release injected objects
}

int PtxEmuDriverShim::advance(uint32_t max_cycles, uint32_t& actual) {
    if (!ctx_) return -1;  // Error: AdvanceResult::Error

    actual = 0;
    try {
        while (actual < max_cycles && ctx_->get_state() != EXIT) {
            ctx_->exe_once();
            ++actual;
        }

        if (ctx_->get_state() == EXIT) {
            // Mark all tracked kernels as complete
            std::lock_guard<std::mutex> lock(mu_);
            for (auto& [kid, done] : completion_) {
                done = true;
            }
            return 2;  // AdvanceResult::KernelComplete
        }
    } catch (const std::exception& e) {
        PTX_ERROR_EMU("PtxEmuDriverShim::advance exception: %s", e.what());
        return -1;  // AdvanceResult::Error
    } catch (...) {
        PTX_ERROR_EMU("PtxEmuDriverShim::advance unknown exception");
        return -1;
    }

    return actual > 0 ? 1 : 0;  // Executed / NoOp
}

void PtxEmuDriverShim::inject_scoreboard(
    uint32_t sm_id, std::unique_ptr<IScoreboard> sb) {

    if (sm_id >= num_sms() || !sb) return;

    auto* raw = sb.get();
    ctx_->get_sm(sm_id)->set_scoreboard(raw);
    scoreboards_.push_back(std::move(sb));

    PTX_DEBUG_EMU("PtxEmuDriverShim: injected scoreboard to SM %u (ptr=%p)",
                  sm_id, static_cast<void*>(raw));
}

void PtxEmuDriverShim::inject_pipeline(
    uint32_t sm_id, std::unique_ptr<IPipelineLatencyProvider> pp) {

    if (sm_id >= num_sms() || !pp) return;

    auto* raw = pp.get();
    ctx_->get_sm(sm_id)->set_pipeline_latency_provider(raw);
    pipelines_.push_back(std::move(pp));

    PTX_DEBUG_EMU("PtxEmuDriverShim: injected pipeline to SM %u (ptr=%p)",
                  sm_id, static_cast<void*>(raw));
}

void PtxEmuDriverShim::inject_tensor_core(
    uint32_t sm_id, std::unique_ptr<ITensorCoreTiming> tc) {

    if (sm_id >= num_sms() || !tc) return;

    auto* raw = tc.get();
    ctx_->get_sm(sm_id)->set_tensor_core_timing(raw);
    tensor_cores_.push_back(std::move(tc));

    PTX_DEBUG_EMU("PtxEmuDriverShim: injected tensor_core to SM %u (ptr=%p)",
                  sm_id, static_cast<void*>(raw));
}

bool PtxEmuDriverShim::is_kernel_complete(uint64_t kernel_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = completion_.find(kernel_id);
    return it != completion_.end() && it->second;
}

void PtxEmuDriverShim::mark_complete(uint64_t kernel_id) {
    std::lock_guard<std::mutex> lock(mu_);
    completion_[kernel_id] = true;
    PTX_DEBUG_EMU("PtxEmuDriverShim: kernel %lu marked complete", kernel_id);
}

uint32_t PtxEmuDriverShim::num_sms() const {
    return ctx_ ? static_cast<uint32_t>(ctx_->get_num_sms()) : 0;
}
