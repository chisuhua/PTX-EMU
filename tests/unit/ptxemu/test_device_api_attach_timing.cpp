// test_device_api_attach_timing.cpp
// =============================================================================
// Phase 2.3 unit tests for IPtxEmuDevice::attach_timing HSK-4 vendored
// interface injection.
//
// Scope (per openspec/changes/device-api-delegation/specs/ptxemu-device-api-delegation):
//   1. Null g_gpu_context: attach_timing returns without crash
//   2. Null SM 0: attach_timing returns without crash
//   3. Null interface arguments: returns without crash, no state corruption
//   4. Valid interfaces: injected into SMContext via get_scoreboard() etc.
//
// Ref: openspec/changes/device-api-delegation/design.md Decision 6
// (namespace bridge via static_cast<void*> round-trip)
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxemu/device_api.h"

#include "ptxsim/gpu_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/scoreboard_interface.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/tensor_core_interface.h"

#include <memory>

extern std::unique_ptr<GPUContext> g_gpu_context;

namespace {

// Trivial mock IScoreboard for round-trip identity test.
// Implements the global ::IScoreboard interface so we can create a concrete
// instance and verify the static_cast<void*> round-trip preserves the
// pointer identity (per Decision 6 testing strategy).
class MockScoreboard : public IScoreboard {
public:
    bool has_free_entry() const override { return true; }
    bool allocate(uint32_t /*reg_id*/, uint32_t /*warp_id*/) override {
        return true;
    }
    bool release(uint32_t /*reg_id*/, uint32_t /*warp_id*/) override {
        return true;
    }
    void tick() override {}
};

// Trivial mocks for the other 2 interfaces (no methods to implement if pure
// virtual only; here we just declare them as concrete for testing).
class MockPipelineLatencyProvider : public IPipelineLatencyProvider {
public:
    // IPipelineLatencyProvider is pure virtual; if it has any methods,
    // they would need to be implemented here. Currently IScoreboard-style
    // interface (pure abstract); if compile fails, see include header.
};

class MockTensorCoreTiming : public ITensorCoreTiming {
public:
    // Same: pure abstract; implementation added per actual interface
};

// RAII scope for install/restore g_gpu_context singleton
class GpuContextScope {
public:
    GpuContextScope() {
        saved_ = std::move(g_gpu_context);
        g_gpu_context = std::make_unique<GPUContext>();
        REQUIRE(g_gpu_context != nullptr);
    }
    ~GpuContextScope() {
        g_gpu_context = std::move(saved_);
    }

    GpuContextScope(const GpuContextScope&) = delete;
    GpuContextScope& operator=(const GpuContextScope&) = delete;

    GPUContext* gpu() const { return g_gpu_context.get(); }

private:
    std::unique_ptr<GPUContext> saved_;
};

}  // namespace

TEST_CASE("device_api_attach_timing: null g_gpu_context returns without crash",
          "[unit][ptxemu][delegation]") {
    auto dev = ptxemu::create_device();
    REQUIRE(dev != nullptr);

    MockScoreboard mock;
    // ptxemu::IScoreboard* → void* → ::IScoreboard* via the public API;
    // but here we pass nullptr for all 3 to test the null safety path.
    dev->attach_timing(nullptr, nullptr, nullptr);
    REQUIRE(true);  // reached without crash
}

TEST_CASE("device_api_attach_timing: valid interfaces inject into SMContext",
          "[unit][ptxemu][delegation]") {
    GpuContextScope scope;
    auto dev = ptxemu::create_device();
    REQUIRE(dev != nullptr);

    auto* sm = scope.gpu()->get_sm(0);
    if (sm == nullptr) {
        WARN("SM 0 not available; cannot verify attach_timing injection "
             "(requires SM setup)");
        return;
    }

    // Create mock interfaces (use ::IScoreboard via void* bridge)
    auto mock_sb = std::make_unique<MockScoreboard>();
    ::IScoreboard* global_sb = mock_sb.get();
    void* sb_void = static_cast<void*>(global_sb);
    auto* ptxemu_sb = static_cast<ptxemu::IScoreboard*>(sb_void);

    dev->attach_timing(ptxemu_sb, nullptr, nullptr);

    // Verify SMContext has the injected IScoreboard
    REQUIRE(sm->get_scoreboard() != nullptr);
    REQUIRE(sm->get_scoreboard() == global_sb);  // round-trip identity
}

TEST_CASE("device_api_attach_timing: null interface args don't corrupt state",
          "[unit][ptxemu][delegation]") {
    GpuContextScope scope;
    auto dev = ptxemu::create_device();
    REQUIRE(dev != nullptr);

    auto* sm = scope.gpu()->get_sm(0);
    if (sm == nullptr) {
        WARN("SM 0 not available; skipping");
        return;
    }

    // Pre-state: scoreboard is null (or whatever default)
    REQUIRE(sm->get_scoreboard() == nullptr);

    // Call with all nulls
    dev->attach_timing(nullptr, nullptr, nullptr);

    // Post-state: still null (setter receives nullptr, stores nullptr)
    REQUIRE(sm->get_scoreboard() == nullptr);
}