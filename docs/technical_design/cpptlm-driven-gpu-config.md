# CppTLM-Driven GPU Architecture Configuration

> **状态**: Draft v2 (refined after explore) | **日期**: 2026-07-22 | **作者**: Sisyphus

## 1. 问题

EMU_COSIM=1 协同仿真模式下，GPU 架构参数（SM 数、warp/线程数、共享内存、
寄存器数、全局内存、时钟频率）有两个独立配置源：

| 配置源 | 位置 | 内容 |
|--------|------|------|
| PTX-EMU | `configs/ampere_a100.json` (默认) | SM 数=108, warp=64, 共享=160KB, 全局=40GB, 指令延迟 |
| CppTLM | 硬编码在 `pipeline_tlm.cc` / `tensor_core_tlm.cc` | A100 指令延迟表（固定） |

两个配置源**互不知晓**，且 CppTLM 无法配置不同架构（始终硬编码 A100 延迟）。
这违背协同仿真的原则：CppTLM 作为时序模型权威，应当统一驱动 PTX-EMU 的
执行环境参数。

## 2. 目标

1. **CppTLM JSON 成为唯一配置权威**：SM 数、寄存器、共享内存、全局内存、时钟
   频率、指令延迟均从 CppTLM 端 JSON 驱动
2. **运行时可选架构**：通过选择 JSON 文件切换 A100 / H100 / B200 配置
3. **向后兼容**：无 CppTLM 时（`g_cpptlm_bridge == nullptr`），PTX-EMU 回退到
   其自身 `config.ini` → JSON 路径
4. **最小 ABI 变更**：复用现有 `cpptlm_bridge.h` ABI 模式（弱符号）

## 3. 架构设计

```
CppTLM configs/templates/gpu_soc/gpu_soc_a100.json
  ┌──────────────────────────────────────────────┐
  │ {                                             │
  │   "name": "gpu_soc_a100",                     │
  │   "gpu_arch": {                               │  ← 新增
  │     "architecture": "Ampere",                 │
  │     "num_sms": 108,                           │
  │     "max_warps_per_sm": 64,                   │
  │     "max_threads_per_sm": 2048,               │
  │     "shared_mem_size_per_sm_kb": 160,         │
  │     "registers_per_sm": 65536,                │
  │     "warp_size": 32,                          │
  │     "global_mem_size_gb": 40,                 │
  │     "clock_rate_mhz": 1410                    │
  │   },                                          │
  │   "modules": [...],                           │  ← 现有
  │   "connections": [...]                        │  ← 现有
  │ }                                             │
  └───────────┬──────────────────────────────────┘
              │ ApuSoC::set_config() → 提取 gpu_arch
              ▼
  ArchitectureParams struct (CppTLM 侧)
              │
              │ cpptlm_set_gpu_config(shim, &params)  ← 新增 ABI
              ▼
  PtxEmuDriverShim::set_gpu_config()  (PTX-EMU 侧)
              │
              │ ctx_->config.num_sms = params.num_sms
              │ ctx_->config.max_warps_per_sm = ...
              ▼
  GPUContext::init()  使用覆盖后的 config 创建 SMs
```

### 3.1 调用时序（精确定位）

基于 `initialize_environment()` 的实际实现（`cudart_sim.cpp:245-341`）：

```
Step 1 (cudart_sim.cpp:259-281): INI 解析 → 获取 gpu_config_filename
                                    (可被 PTX_EMU_GPU_CONFIG 环境变量覆盖)
Step 2 (cudart_sim.cpp:284):       g_gpu_context = make_unique<GPUContext>(json_path)
                                    ├─ 构造函数 (gpu_context.cpp:15-32):
                                    │  load_json_config() → 解析 JSON → 填充 config
Step 3 (cudart_sim.cpp:290):       g_gpu_context->init()
                                    ├─ 创建 SimpleMemory
                                    ├─ 初始化 ResourceManager
                                    └─ 创建 config.num_sms 个 SMContext
Step 4 (cudart_sim.cpp:314):       new PtxEmuDriverShim(g_gpu_context.get())
Step 5 (cudart_sim.cpp:327):       cpptlm_set_driver(shim, api)  ← 跨 .so 注册
Step 6 (cudart_sim.cpp:333-339):   if (EMU_COSIM) g_cpptlm_bridge = &stub_bridge;
```

**关键发现**: `GPUConfig` 是私有成员 (`gpu_context.h:148`)，`init()` 前无程序化修改途径。
方案 A（推荐）：在 Step 2 和 Step 3 之间插入 config override：
```cpp
g_gpu_context->set_gpu_config(cpptlm_config);  // ← 新增，init() 前覆盖
g_gpu_context->init();                          // 使用覆盖后的 config 创建 SM
```

### 3.2 CppTLM 侧参数传递策略（修订）

**发现：GPU 叶子模块（KernelLaunchTLM 等）没有 `set_config()` 重写**。因此有两条可行路径：

#### 路径 A（推荐）：KernelLaunchTLM 添加 `on_config_loaded()` + setter

利用 SimObject 基类已有的 `config_` 字段（存有 raw JSON params），在 `on_config_loaded()` 中读取：

```cpp
// kernel_launch_tlm.hh
void on_config_loaded() override {
    ChStreamModuleBase::on_config_loaded();
    const auto& cfg = get_config();
    if (cfg.contains("gpu_arch"))
        gpu_arch_params_ = ArchitectureParams::from_json(cfg["gpu_arch"]);
}
```

```json
// gpu_soc_gb203_v1.json
{
  "modules": [{
    "name": "kernel_launch",
    "type": "KernelLaunchTLM",
    "params": {
      "kernel_launch_interval": 50,
      "gpu_arch": {
        "num_sms": 2,
        "max_warps_per_sm": 64,
        ...
      }
    }
  }]
}
```

工厂 `module_factory.cc:293-304` 自动将 `params` 传入 `set_config()`，KernelLaunchTLM 自动提取。

#### 路径 B（备选）：GpuCluster 层级携带

```cpp
// GpuCluster::set_config() 已经读取 params:
gpu_topology_.num_sm_per_tpc = ...;
// 新增:
gpu_arch_params_ = ArchitectureParams::from_json(params["gpu_arch"]);
```
然后传播到 KernelLaunchTLM 通过 programmatic setter。

---

## 4. 具体文件变更

### 4.1 CppTLM 侧

#### `configs/templates/gpu_soc/gpu_soc_a100.json` (新建)

```json
{
  "name": "gpu_soc_a100",
  "description": "NVIDIA A100 (Ampere) — 108 SMs, HBM2e 40GB",
  "gpu_arch": {
    "architecture": "Ampere",
    "device_name": "NVIDIA A100",
    "num_sms": 2,
    "max_warps_per_sm": 64,
    "max_threads_per_sm": 2048,
    "shared_mem_size_per_sm_kb": 160,
    "registers_per_sm": 65536,
    "warp_size": 32,
    "global_mem_size_gb": 40,
    "clock_rate_mhz": 1410
  },
  "modules": [
    {
      "name": "kernel_launch",
      "type": "KernelLaunchTLM",
      "params": { "kernel_launch_interval": 50 }
    },
    {
      "name": "compute_unit_0",
      "type": "GpuComputeUnitTLM",
      "params": { "execution_latency": 2 }
    },
    {
      "name": "shared_memory_0",
      "type": "SharedMemoryTLM",
      "params": { "size_kb": 64, "banks": 32 }
    },
    {
      "name": "noc",
      "type": "GpuMeshNoC",
      "params": { "dim": 2, "hops_latency": 2 }
    },
    {
      "name": "memory_cluster",
      "type": "MemoryClusterTLM",
      "params": { "channels": 4, "capacity_gb": 40 }
    }
  ]
}
```

> **注**: `num_sms: 2` 用于测试环境。生产环境 A100 应为 108。可通过环境变量
> `CPPTLM_GPU_SCALE_SMS` 缩放或直接使用 `gpu_soc_a100_prod.json`（108 SMs）。

#### `configs/templates/gpu_soc/gpu_soc_b200.json` (新建)

```json
{
  "name": "gpu_soc_b200",
  "description": "NVIDIA B200 (Blackwell) — 144 SMs, HBM3e 192GB",
  "gpu_arch": {
    "architecture": "Blackwell",
    "device_name": "NVIDIA B200",
    "num_sms": 2,
    "max_warps_per_sm": 80,
    "max_threads_per_sm": 2560,
    "shared_mem_size_per_sm_kb": 240,
    "registers_per_sm": 65536,
    "warp_size": 32,
    "global_mem_size_gb": 192,
    "clock_rate_mhz": 2000
  },
  "modules": [
    {
      "name": "kernel_launch",
      "type": "KernelLaunchTLM",
      "params": { "kernel_launch_interval": 50 }
    },
    {
      "name": "compute_unit_0",
      "type": "GpuComputeUnitTLM",
      "params": { "execution_latency": 2 }
    },
    {
      "name": "shared_memory_0",
      "type": "SharedMemoryTLM",
      "params": { "size_kb": 64, "banks": 32 }
    },
    {
      "name": "noc",
      "type": "GpuMeshNoC",
      "params": { "dim": 2, "hops_latency": 2 }
    },
    {
      "name": "memory_cluster",
      "type": "MemoryClusterTLM",
      "params": { "channels": 8, "capacity_gb": 192 }
    }
  ]
}
```

#### `include/tlm/gpu/gpu_arch_params.hh` (新建)

```cpp
#ifndef TLM_GPU_GPU_ARCH_PARAMS_HH
#define TLM_GPU_GPU_ARCH_PARAMS_HH

#include <cstdint>
#include <string>

namespace tlm {

struct ArchitectureParams {
    std::string architecture;     // "Ampere" | "Hopper" | "Blackwell"
    std::string device_name;      // "NVIDIA A100" | etc.

    uint32_t num_sms              = 2;
    uint32_t max_warps_per_sm     = 64;
    uint32_t max_threads_per_sm   = 2048;
    uint32_t shared_mem_kb_per_sm = 160;
    uint32_t registers_per_sm     = 65536;
    uint32_t warp_size            = 32;
    uint32_t global_mem_gb        = 40;
    uint32_t clock_rate_mhz       = 1410;

    /// 从 nlohmann::json 的 "gpu_arch" 字段构造
    template <typename Json>
    static ArchitectureParams from_json(const Json& j) {
        ArchitectureParams p;
        if (j.contains("architecture"))
            p.architecture = j["architecture"].template get<std::string>();
        if (j.contains("device_name"))
            p.device_name = j["device_name"].template get<std::string>();
        if (j.contains("num_sms"))
            p.num_sms = j["num_sms"].template get<uint32_t>();
        if (j.contains("max_warps_per_sm"))
            p.max_warps_per_sm = j["max_warps_per_sm"].template get<uint32_t>();
        if (j.contains("max_threads_per_sm"))
            p.max_threads_per_sm = j["max_threads_per_sm"].template get<uint32_t>();
        if (j.contains("shared_mem_size_per_sm_kb"))
            p.shared_mem_kb_per_sm = j["shared_mem_size_per_sm_kb"].template get<uint32_t>();
        if (j.contains("registers_per_sm"))
            p.registers_per_sm = j["registers_per_sm"].template get<uint32_t>();
        if (j.contains("warp_size"))
            p.warp_size = j["warp_size"].template get<uint32_t>();
        if (j.contains("global_mem_size_gb"))
            p.global_mem_gb = j["global_mem_size_gb"].template get<uint32_t>();
        if (j.contains("clock_rate_mhz"))
            p.clock_rate_mhz = j["clock_rate_mhz"].template get<uint32_t>();
        return p;
    }
};

}  // namespace tlm

#endif
```

#### `src/tlm/gpu/kernel_launch_tlm.cc` (修改)

```cpp
// 在 tick() 的 tlm_objects_injected_ 块中，新增 gpu_arch 注入:

if (!tlm_objects_injected_) {
    uint32_t num_sms = driver_->num_sms();

    // Inject scoreboard/pipeline/tensor_core (Phase 2a)
    for (uint32_t sm_id = 0; sm_id < num_sms; ++sm_id) {
        driver_->inject_scoreboard(sm_id, std::make_unique<ScoreboardTLM>());
        driver_->inject_pipeline(sm_id, std::make_unique<PipelineTLM>());
        driver_->inject_tensor_core(sm_id, std::make_unique<TensorCoreTLM>());
    }

    // NEW: Send gpu_arch params to PTX-EMU
    if (!gpu_arch_params_.architecture.empty()) {
        if (auto* shim = get_shim_context()) {
            cpptlm_set_gpu_config(shim, &gpu_arch_params_);
        }
    }

    tlm_objects_injected_ = true;
}
```

#### `src/tlm/cluster/apu_soc.cc` (修改)

```cpp
// 在 simulate_instantiate() 中提取 gpu_arch:
if (!gpu_topology_.empty()) {
    auto tmpl = JsonIncluder::loadAndInclude(gpu_topology_);

    // NEW: Extract gpu_arch params
    if (tmpl.contains("gpu_arch")) {
        gpu_arch_params_ = ArchitectureParams::from_json(tmpl["gpu_arch"]);
    }

    wrap["modules"].push_back(wrap_template_as_module(tmpl, "gpu", "GpuCluster"));
}
```

### 4.2 PTX-EMU 侧

#### `include/cudart/cpptlm_bridge.h` (修改)

```cpp
// 新增: ArchitectureParams POD 结构体 (ABI-safe, 无虚函数)
struct GpuArchParams {
    uint32_t num_sms;
    uint32_t max_warps_per_sm;
    uint32_t max_threads_per_sm;
    uint32_t shared_mem_kb_per_sm;
    uint32_t registers_per_sm;
    uint32_t warp_size;
    uint32_t global_mem_gb;
    uint32_t clock_rate_mhz;
};

// 新增 ABI 入口 (弱符号)
extern "C" PTXEMU_BRIDGE_API void cpptlm_set_gpu_config(
    void* shim, const GpuArchParams* params);
```

#### `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` (修改)

```cpp
class PtxEmuDriverShim {
public:
    // ... existing methods ...

    /// Phase 2c: returns GPUConfig override from CppTLM, or nullptr
    const GpuArchParams* get_pending_gpu_config() const { return gpu_config_pending_.get(); }

    /// Phase 2c: receive GPU config from CppTLM (before init)
    void set_gpu_config(const GpuArchParams& params);

private:
    // ... existing members ...
    std::unique_ptr<GpuArchParams> gpu_config_pending_;
};
```

#### `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` (修改)

```cpp
void PtxEmuDriverShim::set_gpu_config(const GpuArchParams& params) {
    if (gpu_config_pending_) return;  // idempotent
    gpu_config_pending_ = std::make_unique<GpuArchParams>(params);
    PTX_INFO_EMU("PtxEmuDriverShim: received GPU config from CppTLM (SM=%u, %uMB shared, %uGB global)",
                 params.num_sms, params.shared_mem_kb_per_sm / 1024, params.global_mem_gb);
}
```

#### `src/cudart/cudart_sim.cpp` — `initialize_environment()` (修改)

```cpp
void initialize_environment() {
    // ... existing INI parsing ...

    // Step 2: Create GPUContext with JSON config
    g_gpu_context = std::make_unique<GPUContext>("configs/" + gpu_config_filename);

    // NEW: Override config from CppTLM (before init, so SMs use correct params)
    // PtxEmuDriverShim already exists at this point (created in Step 4 of a
    // prior environment init cycle). For first-time init, shim is created after,
    // but set_gpu_config is idempotent and can be called anytime before init().
    if (g_ptx_emu_driver_shim && g_ptx_emu_driver_shim->get_pending_gpu_config()) {
        gpu_context->apply_gpu_config(*g_ptx_emu_driver_shim->get_pending_gpu_config());
    }

    // Step 3: init() uses (possibly overridden) config to create SMs
    g_gpu_context->init();

    // ... existing interpreter + shim + bridge setup ...
}
```

#### `include/ptxsim/gpu_context.h` (修改)

```cpp
class GPUContext {
public:
    // ... existing methods ...

    // Phase 2c: Apply CppTLM GpuArchParams to override JSON-loaded config
    // Must be called BEFORE init().
    void apply_gpu_config(const GpuArchParams& params) {
        config.num_sms              = params.num_sms;
        config.max_warps_per_sm     = params.max_warps_per_sm;
        config.max_threads_per_sm   = params.max_threads_per_sm;
        config.shared_mem_size_per_sm = params.shared_mem_kb_per_sm * 1024;
        config.registers_per_sm     = params.registers_per_sm;
        config.warp_size            = params.warp_size;
        config.global_mem_size      = static_cast<uint64_t>(params.global_mem_gb) * 1024 * 1024 * 1024;
    }

    const GPUConfig& get_config() const { return config; }

    // ... existing members ...
};
```

---

## 5. 数据流总览

```
CppTLM 侧：
  JSON (gpu_soc_a100.json)
    └─ module_factory → KernelLaunchTLM::set_config(params)
      └─ on_config_loaded() → 提取 "gpu_arch" → ArchitectureParams
        └─ tick() first invocation:
          ├─ inject PipelineTLM/ScoreboardTLM/TensorCoreTLM (Phase 2a)
          └─ cpptlm_set_gpu_config(shim, &params)  ← 新增

PTX-EMU 侧：
  cpptlm_set_gpu_config() 弱符号
    └─ PtxEmuDriverShim::set_gpu_config()
      └─ store as gpu_config_pending_

  initialize_environment():
    ├─ GPUContext(json_path)   ← load_json_config
    ├─ if (shim->get_pending_gpu_config())
    │    gpu_context->apply_gpu_config(*params)    ← 覆盖 config
    ├─ gpu_context->init()     ← 使用覆盖后的 config 创建 SM
    └─ new PtxEmuDriverShim + cpptlm_set_driver
```
    → cpptlm_set_gpu_config(shim, &params)  [ABI]
      → PtxEmuDriverShim::set_gpu_config()
        → GPUContext::override_config()
        → GPUContext::reinit()
    → driver_->inject_scoreboard/pipeline/tensorcore  (Phase 2a)
  → advance() → exe_once() → execution with new config
```

---

## 6. 向后兼容

- `cpptlm_set_gpu_config` 在 PTX-EMU 侧是**弱符号**，无 CppTLM 时为空实现
- 不提供 `cpptlm_set_gpu_config` 调用时，GPUContext 使用 `config.ini` 指定的 JSON
- `GpuArchParams` 是 POD struct，跨 .so 边界安全
- `CPPTLMBRIDGE_VERSION` 不变（struct 在 ABI 兼容范围，方法签名未变）

---

## 7. 测试策略

| 测试 | 内容 |
|------|------|
| `unit_gpu_config_override` | `GPUContext::reinit()` 后 SM 数正确 |
| `integration_cpptlm_gpu_config` | CppTLM JSON → ArchitectureParams → PtxEmuDriverShim → 验证 config 值 |
| `e2e_cosim_a100_config` | EMU_COSIM=1 + a100 JSON → 验证执行正确 |
| `e2e_cosim_b200_config` | EMU_COSIM=1 + b200 JSON → TensorCore 延迟应不同 |
| 回归 | `EMU_COSIM=1 bash regression-cosim.sh` 零失败 |