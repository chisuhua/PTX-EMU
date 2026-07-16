# HSK-1 草稿：cpptlm_bridge.h ABI 真值源 commit hash 锁定

> **生成时间**: 2026-07-16  
> **来源**: PTX-EMU `cpptlm-d1-full` change（OpenSpec / ADR-0021 §D-PTX-1）  
> **目标**: CppTLM 团队 (`chisuhua/CppTLM` repo)  
> **上下文**: F12b-LD MemoryBridge 集成第一阶段交付

---

## 📧 Send-to (待用户填写)

- **C.C.**: CppTLM Adapter 团队
- **Channel**: GitHub issue `@chisuhua/CppTLM` + Slack `#cpptlm-ptxemu-bridge`
- **Subject**: `[CppTLM D1-Full] HSK-1: cpptlm_bridge.h ABI 真值源 — commit hash 锁定 + lockfile 更新指引`

---

## 📋 Message Body

### 1. 状态锁定

PTX-EMU 端已完成 `cpptlm-d1-full` change 全量实施（14 commits 已 push 到 main）：

- ADR-0021 状态：**Active**（2026-07-16 由 Proposed 转正）
- ABI 真值源 `include/cudart/cpptlm_bridge.h` 已就绪
- 端到端 205 测试 PASS（1 个 pre-existing failure `e2e_divergence` 与本次无关）

### 2. ABI 关键 hash

C++ 端 ABI 锁定在以下 commit hash（**C++ 端消费者请 rebase 这个 hash**）：

```
HEAD commit:        380a8b6a  (cpptlm-d1-full round 2 lessons)
ABI header commit:  603bd8bc  (cpptlm_bridge.h + PTXEMU_BRIDGE_API)
ABI impl commit:    de016f79  (cpptlm_attach_bridge/detach_bridge 实现)
Cross-validate SHA: 603bd8bc + 外部 ABI stub 实现 = ABI 真值源
```

**CppTLM 端消费方式**: `ExternalProject_Add(cpptlm ... GIT_TAG 380a8b6a ...)` 或 pin 到 `603bd8bc` for ABI-only 引入 + 通过 `cpptlm_bridge.h` 静态 include。

### 3. ABI 接口契约

#### 3.1 编译期版本断言

```cpp
// include/cudart/cpptlm_bridge.h (cpptlm_bridge.h:5-7)
#define CPPTLMBRIDGE_VERSION 1

// CppTLM 端必须镜像：
class CppTLMBridge {
public:
    virtual int version() const override { return 1; }   // 必须等于 CPPTLMBRIDGE_VERSION
};
```

`CPPTLMBRIDGE_VERSION` 修改时必须同步 bump，双方 commit hash 互引。

#### 3.2 头文件依赖（仅 3 个）

```cpp
// cpptlm_bridge.h:26-37
#include <cstddef>      // size_t 来源
#include <cstdint>      // uint64_t, uint32_t, uint8_t 来源
// cuda_runtime.h (条件编译)
#if defined(__CUDACC_RUNTIME_H__)
#include <cuda_runtime.h>
#elif !defined(CUDA_STREAM_T_DEFINED)
typedef void* cudaStream_t;
#define CUDA_STREAM_T_DEFINED
#endif
```

**强约束**: 未来 CppTLM 端不得向 cpptlm_bridge.h 添加其他 include（防 CppTLM 依赖反转）。

#### 3.3 双向 static_assert 拦截（12 端点）

从 ADR-0020 姊妹 change 引入的 enum 必须在双方编译期验证：

```cpp
// PTX-EMU 端在 include/ptxsim/pipeline_interface.h (FROM ADR-0020):
enum class PipelineId : uint32_t { P0_INT_FP32 = 0, V_SIMD = 1, P1_FP64 = 2, P2_SFU = 3, P3_LSU = 4, P4_TC = 5 };

// CppTLM 端 MemoryBridge::register_pipeline_provider 必须：
static_assert(static_cast<uint32_t>(CppTLM::PipelineId::P0_INT_FP32) == 0, "ABI drift");
static_assert(static_cast<uint32_t>(CppTLM::PipelineId::P4_TC) == 5, "ABI drift");

// 同样的 6 端点对 TcPrecision (FP4..TF32) 做镜像断言
```

#### 3.4 ABI 入口（5 虚方法 + 2 attach/detach）

```cpp
// cpptlm_bridge.h:69-132 CppTLMBridge 接口
class CppTLMBridge {
public:
    virtual ~CppTLMBridge() = default;
    virtual int    version() const = 0;
    virtual int    submit_kernel(uint64_t kernel_id, const char* kernel_name,
                                 uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                 uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                 const void** kernel_args, size_t args_count,
                                 size_t shared_mem, uint64_t stream_id) = 0;
    virtual uint64_t poll_kernel(uint64_t kernel_id) = 0;
    virtual int    synchronize_stream(uint64_t stream_id) = 0;
    virtual uint64_t global_access(uint64_t device_addr, uint64_t val, uint8_t type) = 0;
};

// cpptlm_bridge.h:161, 168 跨 so 入口（PTXEMU_BRIDGE_API 宏已定义）
extern "C" PTXEMU_BRIDGE_API void cpptlm_attach_bridge(CppTLMBridge* bridge);
extern "C" PTXEMU_BRIDGE_API void cpptlm_detach_bridge();
```

**调用语义**:

| 入口 | 时机 | CppTLM 实现要求 |
|------|------|----------------|
| `cpptlm_attach_bridge` | `libcpptlm_cudart.so` 加载后构造函数中调用 | 必须幂等，可被 PTX-EMU 多次调用（覆盖式语义）|
| `cpptlm_detach_bridge` | `libcpptlm_cudart.so` 卸载前析构函数 | 必须幂等，nullptr 状态下安全 |
| `version()` | 每次 PTX-EMU 启动时调用 | 必须返回 `CPPTLMBRIDGE_VERSION`（当前=1）|

### 4. 提交时的最终 manifest

| 文件 | 行数 | commit |
|------|------|--------|
| `include/cudart/cpptlm_bridge.h` | 175 | `603bd8bc` (+`a0be543b` 接续) |
| `src/cudart/cudart_sim.cpp` (ABI 实现) | 1199 | `de016f79` (新增 2 个 ABI 入口实现) |

### 5. CppTLM 端验证清单

```bash
# 1. 拉取 ABI 头文件
git -C external/cpptlm checkout 380a8b6a
cp external/cpptlm/include/cudart/cpptlm_bridge.h ./include/

# 2. 编译期版本断言（双向）
static_assert(CPPTLMBRIDGE_VERSION == 1, "ABI drift");
static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t), "ABI drift");

# 3. 12 端点 static_assert（参见 3.3 节）

# 4. ABI 符号导出验证（linker 级别）
nm -D build/lib/libcpptlm_cudart.so | grep -E "T cpptlm_(attach|detach)_bridge"
# 期望：T cpptlm_attach_bridge + T cpptlm_detach_bridge

# 5. ABI 协议冒烟测试
./build/bin/cpptlm_attach_smoke_test --bridge-version 1 --ptxemu-commit 380a8b6a
```

---

## 📎 交叉引用

- PTX-EMU 端 ADR-0021 §HSK 状态机: https://github.com/chisuhua/PTX-EMU/blob/380a8b6a/docs/adr/0021-cpptlm-d1-full-integration.md#hsk-状态机强制
- ABI 头文件: https://github.com/chisuhua/PTX-EMU/blob/380a8b6a/include/cudart/cpptlm_bridge.h
- CppTLM 协作同步: https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md §4
- 综合任务书: https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #1

---

## ⏱️ 等待 CppTLM 端的反馈

- **期望反馈类型**: PR → `chisuhua/CppTLM main` + cross-link 回 PTX-EMU issue
- **本 PR 应包含**:
  - CppTLM 端 MemoryBridge implements `Cpptlm_bridge.h` 5 虚方法
  - `cppTLMBridge_VERSION` 配套返回（= 1）
  - 12 端点双向 static_assert
  - `libcpptlm_cudart.so` 构建脚本（CMake ExternalProject_Add pattern）
- **不在本 PR 范围**:
  - 时序扩展（KernelLaunchTLM::tick） — 推迟到 F12c
  - async IAsyncCompletion — 推迟到 Phase 9+

---

**发送方**: PTX-EMU Architecture Team  
**ADR-0021 状态**: Active (2026-07-16)  
**本 HSK 版本**: HSK-1 v1 (initial)  
**签发**: ⏳ 待 PTX-EMU Architecture Team 发出（您审核后手动 send）
