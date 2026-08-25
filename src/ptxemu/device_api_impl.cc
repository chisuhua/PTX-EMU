// device_api_impl.cc - PTX-EMU public device API implementation
//
// Phase 2 implementation (HSK-8 ack 738b412c, OpenSpec design.md Phase 2):
//   - IPtxEmuDevice thin adapter layer over PTX-EMU internal classes
//   - 1:1 mapping to S1 facade.cc 12 callsites (HSK-8 spec §CppTLM 端接受条件 #1)
//   - C++17 compatible (per spec/public-device-api §Requirement C++17)
//   - ThreadState enum maps to ptxsim::EXE_STATE (HSK-8 spec §Decision 6
//     static_assert 锁)
//
// Phase 2.1 minimal scope (本文件):
//   - PtxEmuDeviceImpl class
//   - exe_once / sm_exe_once / warp_exe_once 委托 GPUContext
//   - get_thread_state / get_warp_status 读取 GPUContext 状态 (stubs)
//   - 其他 set_* 方法返回 false (not implemented yet) — Phase 2.2/2.3
//   - create_device / destroy_device factory
//
// Phase 2.2 (device-api-delegation change): 3 set_* 方法委托实现:
//   - set_scoreboard: SMContext + IScoreboard 注册验证 (R7 mask 传播 deferred)
//   - set_active_mask: WarpContext overwrite (NOT OR-merge, per BUG-RETHANG)
//   - set_next_pc: ThreadContext::set_pc + commit_pc (NOT force_set_pc)
//
// Phase 2.3 follow-up: attach_timing via HSK-4 vendored interfaces
//   (per Decision 6 namespace bridge via static_cast<void*> round-trip)

#include <ptxemu/device_api.h>
#include <ptxsim/gpu_context.h>
#include <ptxsim/sm_context.h>
#include <ptxsim/execution_types.h>

#include <memory>

// Forward decl for global GPUContext singleton (declared in src/cudart/
// cudart_sim.cpp per ADR-0021 v1.1 amendment).
extern std::unique_ptr<GPUContext> g_gpu_context;

namespace ptxemu {

namespace {
// Map EXE_STATE (global namespace) to ptxemu::ThreadState
// (HSK-8 spec §Decision 6).
// Ordering MUST match ThreadState enum values in device_api.h.
ThreadState map_state(EXE_STATE s) {
    switch (s) {
        case EXE_STATE::IDLE:    return ThreadState::kIdle;
        case EXE_STATE::RUN:     return ThreadState::kRun;
        case EXE_STATE::EXIT:    return ThreadState::kExit;
        case EXE_STATE::BAR_SYNC: return ThreadState::kBarSync;
    }
    return ThreadState::kIdle;
}
}  // namespace

class PtxEmuDeviceImpl : public IPtxEmuDevice {
public:
    explicit PtxEmuDeviceImpl() = default;
    ~PtxEmuDeviceImpl() override = default;

    bool initialize(const DeviceConfig& config) override {
        config_ = config;
        initialized_ = true;
        return true;
    }

    void shutdown() override {
        initialized_ = false;
    }

    // Phase 2.1 - basic exe_once delegation (HSK-8 spec §CppTLM 端接受条件
    // #1: 1:1 mapping to S1 facade).
    int exe_once() override {
        // PTX-EMU uses global g_gpu_context singleton (per cudart/cudart_sim.cpp).
        // Phase 2.2 should migrate to instance-based but currently we keep
        // global access for backward compat with existing PTX-EMU infrastructure.
        if (!g_gpu_context) return -1;
        g_gpu_context->exe_once();
        return 0;
    }

    int sm_exe_once(uint32_t sm_id) override {
        if (!g_gpu_context) return -1;
        auto* sm = g_gpu_context->get_sm(sm_id);
        if (!sm) return -1;
        sm->exe_once();
        return 0;
    }

    int warp_exe_once(uint32_t /*sm_id*/, uint32_t /*warp_id*/) override {
        // Phase 2.2: SMContext::get_warp(warp_id) → WarpContext::execute
        // For now return -1 to indicate not implemented.
        return -1;
    }

    // Phase 2.2 — set_scoreboard delegation via SMContext.
    // Per design R7 (device-api-delegation/design.md): Phase 2.2 minimum
    // validates SMContext + IScoreboard registration; mask/warp_id
    // propagation deferred to Phase 2.2.1 follow-up change.
    // Per Decision 6 (namespace bridge): Phase 2.2 R7-constrained minimal
    // scope sidesteps the bridge entirely — no IScoreboard* parameter on
    // public surface; we only verify the existing wiring is in place.
    bool set_scoreboard(uint32_t sm_id, uint32_t warp_id,
                        uint64_t /*mask*/) override {
        if (!g_gpu_context) return false;
        auto* sm = g_gpu_context->get_sm(sm_id);
        if (!sm) return false;
        // R7: validate IScoreboard registration only.
        (void)warp_id;
        return sm->get_scoreboard() != nullptr;
    }

    // get_thread_state: read from GPUContext thread state.
    ThreadState get_thread_state(uint32_t /*sm_id*/, uint32_t /*warp_id*/,
                                 uint32_t /*lane_id*/) override {
        // Phase 2.2: SMContext → WarpContext → ThreadContext::state
        return ThreadState::kIdle;
    }

    // set_active_mask — overwrite semantics (per ptx-lessons-learned §1
    // BUG-RETHANG / BUG-POSTBARRIER-TWOHALVES: set_active_mask is
    // overwrite, NOT OR-merge; OR logic is encapsulated in
    // BarrierModule::release_warp_barrier).
    // ptx-barrier-mechanism skill: ret handler depends on overwrite
    // semantics to clear retired lanes correctly.
    bool set_active_mask(uint32_t sm_id, uint32_t warp_id,
                         uint64_t mask) override {
        if (!g_gpu_context) return false;
        auto* sm = g_gpu_context->get_sm(sm_id);
        if (!sm) return false;
        auto* warp = sm->get_warp(warp_id);
        if (!warp) return false;
        // OVERWRITE (NOT OR-merge). Direct delegation to WarpContext.
        warp->set_active_mask(static_cast<uint32_t>(mask));
        return true;
    }

    // set_next_pc — normal PC advancement (NOT force_set_pc per
    // ptx-lessons-learned ANTI-PATTERNS + AGENTS.md L85).
    bool set_next_pc(uint32_t sm_id, uint32_t warp_id,
                     uint32_t lane_id, uint32_t pc) override {
        if (!g_gpu_context) return false;
        auto* sm = g_gpu_context->get_sm(sm_id);
        if (!sm) return false;
        auto* warp = sm->get_warp(warp_id);
        if (!warp) return false;
        auto* thread = warp->get_thread(static_cast<int>(lane_id));
        if (!thread) return false;
        // AGENTS.md ANTI-PATTERNS L85: NEVER force_set_pc(). Use set_pc + commit_pc.
        thread->set_pc(static_cast<int>(pc));
        thread->commit_pc();
        return true;
    }

    // get_warp_status: snapshot of warp state.
    WarpStatus get_warp_status(uint32_t /*sm_id*/, uint32_t /*warp_id*/) override {
        // Phase 2.2: query WarpContext for lanes/active_mask/blocked_cycles
        WarpStatus s{};
        return s;
    }

    bool is_finished() override {
        // Phase 2.2: GPUContext::is_idle or similar
        if (!g_gpu_context) return true;
        return g_gpu_context->get_state() == EXE_STATE::IDLE;
    }

    // attach_timing — HSK-4 vendored interfaces injection (HSK-8 spec #6).
    void attach_timing(IScoreboard* /*sb*/, IPipelineLatencyProvider* /*pl*/,
                       ITensorCoreTiming* /*tc*/) override {
        // Phase 2.3: store + inject into SMContext
    }

private:
    DeviceConfig config_{};
    bool initialized_ = false;
};

// Factory (HSK-8 spec §CppTLM 端接受条件 #1 第 4 项).
std::unique_ptr<IPtxEmuDevice> create_device() {
    return std::make_unique<PtxEmuDeviceImpl>();
}

void destroy_device(IPtxEmuDevice* dev) {
    delete dev;
}

}  // namespace ptxemu