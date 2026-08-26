// device_api.h - PTX-EMU public device API (HSK-8 spec)
//
// HSK-8 acceptance criteria #1: PTX-EMU 仓 include/ptxemu/device_api.h
// 必须存在 (含 IPtxEmuDevice + 工厂 + PTXEMU_API_VERSION=1)。
// HSK-8 spec §"CppTLM 端接受条件" 锁定 5 条验收条件,本文件覆盖 #1 (公共头),
// Phase 2 PR 同时覆盖 #2 (ptxemu_core 库目标) + #4 (drift_check workflow)。
//
// C++17 子集: 禁止 std::format/requires/concept/<=>/consteval/constinit/
// likely attribute (per open spec/public-device-api §Requirement C++17)。
//
// 5 条契约 (HSK-8 spec §"HSK-8 核心契约"):
//   #1 公共头路径: include/ptxemu/device_api.h ✓
//   #2 命名空间: ptxemu ✓
//   #3 版本守卫宏: PTXEMU_API_VERSION = 1 ✓
//   #4 C++17 兼容: 仅使用 C++17 子集特性 ✓
//   #5 CMake 库目标: add_library(ptxemu_core STATIC ...) — Phase 2 PR

#ifndef PTXEMU_DEVICE_API_H
#define PTXEMU_DEVICE_API_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#define PTXEMU_API_VERSION 1

// Phase 0.3d: 公共面 pure data (无 mutable void* 等实现状态)。
// 5 条验收 #1: 包含 IPtxEmuDevice + 工厂 + VERSION (本文件)。

namespace ptxemu {

// Forward declarations for HSK-4 vendored interfaces (HSK-8 spec #6:
// HSK-4 复用 — IScoreboard/IPipelineLatencyProvider/ITensorCoreTiming
// 在 Phase 2 由 ptxemu_core 库实现引入, 本头仅前向声明)。
struct IScoreboard;
struct IPipelineLatencyProvider;
struct ITensorCoreTiming;

// Device configuration DTO (per HSK-8 spec §CppTLM 端接受条件 #1 第 3 项)。
struct DeviceConfig {
    uint32_t num_sms = 1;
    uint32_t max_warps_per_sm = 64;
    uint32_t max_threads_per_sm = 2048;
    std::size_t shared_mem_size_per_sm = 48 * 1024;
    uint32_t registers_per_sm = 65536;
    uint32_t max_blocks_per_sm = 32;
    uint32_t warp_size = 32;
};

// ThreadState enum (per HSK-8 spec §Decision 6 static_assert 锁:
// impl 层 static_assert(static_cast<uint32_t>(ptxemu::ThreadState::kIdle) ==
// static_cast<uint32_t>(::EXE_STATE::IDLE))。注: EXE_STATE 是全局命名空间
// 无作用域枚举 (include/ptxsim/execution_types.h:8), 不在 ptxsim 命名空间。
enum class ThreadState : uint32_t {
    kIdle = 0,    // ::EXE_STATE::IDLE
    kRun = 1,     // ::EXE_STATE::RUN
    kExit = 2,    // ::EXE_STATE::EXIT
    kBarSync = 3, // ::EXE_STATE::BAR_SYNC
};

// Per-lane status snapshot.
struct LaneStatus {
    uint32_t lane_id = 0;
    ThreadState state = ThreadState::kIdle;
    uint32_t pc = 0;
};

// Per-warp status snapshot.
struct WarpStatus {
    uint32_t warp_id = 0;
    uint32_t sm_id = 0;
    std::vector<LaneStatus> lanes;
    uint32_t active_count = 0;
    int32_t blocked_cycles = 0;
};

// IPtxEmuDevice — PTX-EMU 公共设备 API 抽象接口。
//
// 5+ abstract methods 覆盖 S1 facade.cc 12 callsites (HSK-8 spec §CppTLM
// 端接受条件 #1: 含 工厂 + PTXEMU_API_VERSION=1 + 抽象接口)。
//
// Method 命名规则: 1:1 映射 S1 facade 调用点, 避免 facade.cc 重写
// 时二次理解成本。
class IPtxEmuDevice {
public:
    virtual ~IPtxEmuDevice() = default;

    // Lifecycle
    virtual bool initialize(const DeviceConfig& config) = 0;
    virtual void shutdown() = 0;

    // Execution (HSK-8 spec §CppTLM 端接受条件 隐含要求 S1 facade
    // 调用点 1:1 映射)
    virtual int exe_once() = 0;                      // GPUContext::exe_once()
    virtual int sm_exe_once(uint32_t sm_id) = 0;    // SMContext::exe_once()
    virtual int warp_exe_once(uint32_t sm_id, uint32_t warp_id) = 0;
                                                     // WarpContext::exe_once()

    // Memory (S1 facade set_scoreboard 1:1 映射)
    virtual bool set_scoreboard(uint32_t sm_id, uint32_t warp_id, uint64_t mask) = 0;

    // Thread control (S1 facade thread 状态/控制 1:1 映射)
    virtual ThreadState get_thread_state(uint32_t sm_id, uint32_t warp_id, uint32_t lane_id) = 0;
    virtual bool set_active_mask(uint32_t sm_id, uint32_t warp_id, uint64_t mask) = 0;
    virtual bool set_next_pc(uint32_t sm_id, uint32_t warp_id, uint32_t lane_id, uint32_t pc) = 0;

    // Status query
    virtual WarpStatus get_warp_status(uint32_t sm_id, uint32_t warp_id) = 0;
    virtual bool is_finished() = 0;

    // HSK-4 vendored interfaces injection (HSK-8 spec §6 HSK-4 复用)
    // — attach_timing() 接收 HSK-4 已 vendored 3 接口, 不重复定义。
    virtual void attach_timing(IScoreboard* sb, IPipelineLatencyProvider* pl, ITensorCoreTiming* tc) = 0;

    // Static assert: PTXEMU_API_VERSION frozen at 1.
    // HSK-8 spec §Decision 3: 公共签名变更须签发 HSK-9 bump VERSION。
    static_assert(PTXEMU_API_VERSION == 1,
                  "PTXEMU_API_VERSION frozen at 1; 公共签名变更必须签发 HSK-9 bump VERSION");
};

// Factory (HSK-8 spec §CppTLM 端接受条件 #1 第 4 项)
// — std::unique_ptr<IPtxEmuDevice> create_device(const DeviceConfig&)
// — void destroy_device(IPtxEmuDevice*)
std::unique_ptr<IPtxEmuDevice> create_device();
void destroy_device(IPtxEmuDevice* dev);

}  // namespace ptxemu

#endif  // PTXEMU_DEVICE_API_H