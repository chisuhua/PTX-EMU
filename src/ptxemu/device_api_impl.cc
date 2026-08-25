// device_api_impl.cc - PTX-EMU public device API implementation
//
// HSK-8 ack 738b412c, OpenSpec design.md Phase 2 + phase-2-2-1-3-1-followup:
//   - IPtxEmuDevice thin adapter layer over PTX-EMU internal classes
//   - 1:1 mapping to S1 facade.cc 12 callsites (HSK-8 spec §CppTLM 端接受条件 #1)
//   - C++17 compatible (per spec/public-device-api §Requirement C++17)
//   - ThreadState enum maps to ptxsim::EXE_STATE (HSK-8 spec §Decision 6)
//   - 12/12 IPtxEmuDevice methods wired; see include/ptxemu/AGENTS.md for
//     per-method delegation table

#include <ptxemu/device_api.h>
#include <ptxsim/gpu_context.h>
#include <ptxsim/sm_context.h>
#include <ptxsim/execution_types.h>
#include <ptxsim/scoreboard_interface.h>
#include <ptxsim/pipeline_interface.h>
#include <ptxsim/tensor_core_interface.h>
#include <ptxsim/warp_context.h>
#include <ptxsim/warp_state.h>
#include <ptxsim/thread_state.h>
#include <ptx_ir/statement_context.h>

#include <climits>
#include <cstdint>
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

// Map ptxsim::ThreadStatus (WarpState::threads[i].status) to
// ptxemu::ThreadState. Parallel to map_state(EXE_STATE) helper.
// ThreadStatus is from include/ptxsim/thread_state.h (Active/Blocked/
// Exited/Yielded), distinct from EXE_STATE — used by
// WarpContext::get_warp_state().threads[i].status for get_warp_status
// lane snapshot.
//
// Yielded maps to kIdle (conservative default): ThreadState enum is
// frozen at 4 values per HSK-8 spec §Decision 6 — adding a new value
// would break ABI (would require HSK-9 VERSION bump). Yielded
// semantically means "Active but yielded CPU" — mapping to kIdle is
// the conservative default.
ThreadState map_thread_status(ptxsim::ThreadStatus ts) {
    switch (ts) {
        case ptxsim::ThreadStatus::Active:  return ThreadState::kRun;
        case ptxsim::ThreadStatus::Blocked: return ThreadState::kBarSync;
        case ptxsim::ThreadStatus::Exited:  return ThreadState::kExit;
        case ptxsim::ThreadStatus::Yielded: return ThreadState::kIdle;
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

    // warp_exe_once: advance ONE warp by one statement via
    // WarpContext::execute_warp_instruction(StatementContext&, int).
    //
    // CRITICAL: This is a STATE-MUTATING hot path. Per ptx-instruction-pipeline
    // skill, must NOT bypass barrier/scoreboard invariants. Per
    // ptx-barrier-mechanism, must respect BarrierModule::release_warp_barrier
    // overwrite semantics (BUG-RETHANG / BUG-POSTBARRIER-TWOHALVES guard).
    //
    // Implementation mirrors the single-warp path in SMContext::exe_once
    // (sm_context.cpp L255-L321): get_lanes_by_pc → pick non-blocked PC →
    // extract StatementContext from sample lane → execute.
    //
    // Returns 0 on success, -1 if sm_id/warp_id invalid or no StatementContext
    // available. Returns 0 if no schedulable lanes (matches SMContext idle path).
    int warp_exe_once(uint32_t sm_id, uint32_t warp_id) override {
        if (!g_gpu_context) return -1;
        auto* sm = g_gpu_context->get_sm(sm_id);
        if (!sm) return -1;
        auto* warp = sm->get_warp(warp_id);
        if (!warp) return -1;

        auto lanes_by_pc = warp->get_lanes_by_pc();
        if (lanes_by_pc.empty()) {
            // No schedulable lanes (idle, all blocked, or all exited) — same
            // skip semantics as SMContext::exe_once.
            return 0;
        }

        // Pick first PC whose lanes are not all blocked on a barrier.
        int pick_pc = lanes_by_pc.begin()->first;
        const auto& ws = warp->get_warp_state();
        for (const auto& [pc, lanes] : lanes_by_pc) {
            bool all_non_blocked = true;
            for (int lane : lanes) {
                if (ws.threads[lane].is_blocked) {
                    all_non_blocked = false;
                    break;
                }
            }
            if (all_non_blocked) {
                pick_pc = pc;
                break;
            }
        }

        // Extract StatementContext from the sample lane (mirrors
        // SMContext::exe_once L262-L264).
        const auto& lanes = lanes_by_pc.begin()->second;
        int sample_lane = lanes[0];
        ThreadContext* thread = warp->get_thread(sample_lane);
        if (!thread) return -1;
        if (pick_pc < 0 || pick_pc >= thread->statements_size()) {
            // Out-of-bounds PC (e.g. barrier handler jumped to reconvergence_pc
            // beyond statement list). Skip this tick — matches scheduler_utils.h
            // out-of-bounds guard.
            return 0;
        }
        StatementContext* stmt = thread->get_statement_at(pick_pc);
        if (!stmt) return -1;

        warp->execute_warp_instruction(*stmt, pick_pc);
        return 0;
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

    // get_thread_state: read from ThreadContext::get_state() (which returns
    // EXE_STATE) and map to ptxemu::ThreadState via the existing map_state
    // helper (HSK-8 spec §Decision 6 static_assert lock).
    //
    // READ-ONLY (per state-modification-audit skill): no state mutation.
    //
    // CRITICAL: Per include/ptxsim/thread_context.h:205, ThreadContext exposes
    // get_state() (not direct field access). The pre-fix stub returned
    // hardcoded ThreadState::kIdle which violated HSK-8 spec §Decision 6
    // (returning constant regardless of underlying EXE_STATE).
    ThreadState get_thread_state(uint32_t sm_id, uint32_t warp_id,
                                 uint32_t lane_id) override {
        if (!g_gpu_context) return ThreadState::kIdle;
        auto* sm = g_gpu_context->get_sm(sm_id);
        if (!sm) return ThreadState::kIdle;
        auto* warp = sm->get_warp(warp_id);
        if (!warp) return ThreadState::kIdle;
        auto* thread = warp->get_thread(static_cast<int>(lane_id));
        if (!thread) return ThreadState::kIdle;
        return map_state(thread->get_state());
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

    // get_warp_status: snapshot of warp state via warp->get_warp_state().
    // Populates the EXISTING 5-field WarpStatus struct at
    // include/ptxemu/device_api.h:69-75 — no new fields, no sizeof change
    // (HSK-8 spec §Decision 5 sizeof visibility, PTXEMU_API_VERSION=1 frozen).
    //
    // READ-ONLY (per state-modification-audit skill): no state mutation.
    // Returns default-constructed WarpStatus on invalid sm_id/warp_id.
    WarpStatus get_warp_status(uint32_t sm_id, uint32_t warp_id) override {
        if (!g_gpu_context) return WarpStatus{};
        auto* sm = g_gpu_context->get_sm(sm_id);
        if (!sm) return WarpStatus{};
        auto* warp = sm->get_warp(warp_id);
        if (!warp) return WarpStatus{};

        WarpStatus s;
        s.warp_id = warp_id;
        s.sm_id = sm_id;

        const auto& ws = warp->get_warp_state();
        s.lanes.reserve(32);
        for (int i = 0; i < 32; ++i) {
            LaneStatus ls;
            ls.lane_id = static_cast<uint32_t>(i);
            ls.state = map_thread_status(ws.threads[i].status);
            ls.pc = ws.threads[i].pc;
            s.lanes.push_back(ls);
        }

        s.active_count = static_cast<uint32_t>(ws.count_active_lanes());

        // Sum blocked_cycles_remaining across threads, clamp to int32_t
        // range. 32 threads × uint32_t max = ~128 billion — extreme case
        // but preserved as defense (per design Decision Open Questions Q3).
        uint64_t total_blocked = 0;
        for (const auto& thread : ws.threads) {
            total_blocked += thread.blocked_cycles_remaining;
        }
        s.blocked_cycles = (total_blocked > static_cast<uint64_t>(INT32_MAX))
                               ? INT32_MAX
                               : static_cast<int32_t>(total_blocked);

        return s;
    }

    bool is_finished() override {
        // Phase 2.2: GPUContext::is_idle or similar
        if (!g_gpu_context) return true;
        return g_gpu_context->get_state() == EXE_STATE::IDLE;
    }

    // attach_timing — HSK-4 vendored interfaces injection (HSK-8 spec §6).
    // Per design Decision 6: namespace bridge via static_cast<void*>
    // round-trip. ptxemu::IScoreboard* (device_api.h forward decl) is
    // bridged to ::IScoreboard* (ptxsim/scoreboard_interface.h full def)
    // via void* intermediate. Same pattern for the other 2 interfaces.
    // Phase 2.3 prototype hardcodes sm_id=0 (attach_timing is a global
    // setup method, not per-SM).
    void attach_timing(IScoreboard* sb, IPipelineLatencyProvider* pl,
                       ITensorCoreTiming* tc) override {
        if (!g_gpu_context) return;
        auto* sm = g_gpu_context->get_sm(0);
        if (!sm) return;
        sm->set_scoreboard(
            static_cast<::IScoreboard*>(static_cast<void*>(sb)));
        sm->set_pipeline_latency_provider(
            static_cast<::IPipelineLatencyProvider*>(static_cast<void*>(pl)));
        sm->set_tensor_core_timing(
            static_cast<::ITensorCoreTiming*>(static_cast<void*>(tc)));
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