# PTX-EMU HAL Backend Cross-Repo Defect Audit

> **Date**: 2026-08-13
> **Auditor**: Sisyphus（基于 Oracle 跨仓评审 + 直接文件验证）
> **Scope**: ADR-076 / tadr-307 / PTX-EMU ADR-0029 §D8 跨仓集成栈的端到端可执行性
> **Method**: 4 仓 (PTX-EMU / UsrLinuxEmu / TaskRunner / CppTLM) 文件精读 + ABI 签名对比 + Oracle 子代理评审 + 关键事实亲自验证
> **Audience**: UsrLinuxEmu Architecture Team + TaskRunner owner（评审输入）
> **Status**: 🔴 **3 stacked critical defects 阻止端到端 kernel 执行 — 必须先修才能 ship 任何真实 .so e2e 验证**

---

## 1. TL;DR

ADR-076（GPGPU Kernel Module IOCTL — PTX-EMU Image Executor HAL Backend）的代码实施在 UsrLinuxEmu `hal_user.cpp` 存在 **3 个堆叠 critical 缺陷**，按咬合顺序：

| # | 缺陷 | 严重度 | 后果 | 工作量 |
|---|------|--------|------|--------|
| **1** | `hal_user.cpp:702-708` dlsym typedef 签名与 `cpptlm_module.h:18-26` 真实 ABI 不匹配 | 🔴 阻塞 | `image_load` 100% 返回 `-EINVAL`（handle 错位） | Quick（改 2 个 typedef + 3 个 call site） |
| **2** | dlsym 路径下 `g_gpu_context` 永不被创建 | 🔴 阻塞 | `ptxemu_image_execute` 静默成功（kernel 不执行） | Quick-Short（懒初始化） |
| **3** | PTX-EMU `SimpleMemory` (4GB mmap) 与 UsrLinuxEmu HAL heap (256MB @ `0x100000000`) 不可桥接，且 ABI 无 memory 注册 API | 🟡 阻塞 | v1 范围内任何 dereference `cudaMalloc` buffer 的 kernel 崩溃或读错数据 | Medium（需 ADR-076 §D7 决策） |

**为什么 144/145 ctest PASS？** ADR-076 Migration Step 4（line 483-484）规定 e2e 用 **mock** `libptxemu_device.so` — mock 编码"谁写谁理解的契约"，所以契约漂移和 memory 域问题在测试里完全不可见。

**消费侧额外阻塞**：`UsrLinuxEmu/external/TaskRunner/src/umd/libcuda_shim/cu_module.cpp:135-138` — `cuModuleLoadData` 仍 `CUDA_ERROR_NOT_IMPLEMENTED`（tadr-307 在 submodule checkout 中尚未 ship）。

---

## 2. Investigation Method

### 2.1 已精读文件

**UsrLinuxEmu**（target host of `libptxemu_device.so`）：
- `docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md` — 553 行（canonical ADR）
- `plugins/gpu_driver/hal/hal_user.cpp:680-880` — dlsym + 3 个 fn-ptr 实现
- `plugins/gpu_driver/hal/hal_user.h:31-32` — `HAL_HEAP_BASE = 0x100000000`, `HAL_HEAP_SIZE = 256MB`
- `plugins/gpu_driver/hal/hal_user.cpp:74-89` — `user_mem_alloc` 走 `gpu_buddy`
- `plugins/gpu_driver/sim/vram_store.h` — `GpuVramStore` 256MB mmap pool + BAR 仿真
- `plugins/gpu_driver/hal/gpu_hal.h` — `struct gpu_hal_ops` #66/#67/#68

**PTX-EMU**（shipper of `libptxemu_device.so`）：
- `include/cudart/cpptlm_module.h:1-58` — 公共 ABI（VERSION=2）
- `src/cudart/cpptlm_module.cpp:1-274` — `PtxEmuImageExecutor` 全部方法
- `src/cudart/cuda_driver.cpp:1-129` — `CudaDriver` singleton + `SimpleMemoryAllocator` 调用
- `src/cudart/ptx_interpreter.cpp:71-180, 344-346` — `prepareKernelLaunchRequest` + `funcInterpreter` g_gpu_context 检查
- `src/memory/simple_memory.cpp:12-14, 30-50` — `direct_access` 地址归一化
- `src/cudart/cudart_sim.cpp:261-277, 1026-1040, 465-468` — `g_gpu_context` 创建点（仅 legacy LD_PRELOAD 路径）

**TaskRunner**（consumer）：
- `UsrLinuxEmu/external/TaskRunner/src/umd/libcuda_shim/cu_module.cpp:135-138` — `cuModuleLoadData` NOT_IMPLEMENTED

**CppTLM**（验证其与本路径正交）：
- `CppTLM/AGENTS.md:7` — TLM 2.0 NoC 仿真框架

### 2.2 验证协议

- Oracle 子代理（sonnet, session `ses_004c1da61ffeZdTs6hehkIEB2w`）— 4 hypothesis / 4 question 输出
- Oracle 关键声明**亲自 `Read` 验证**（`hal_user.cpp:701-708, 803-806` / `cpptlm_module.h:18-26` / `cu_module.cpp:135-138`）
- 所有 `file:line` 引用已 grep 确认存在

---

## 3. Defect 1: ABI Typedef 签名不匹配

### 3.1 问题

`hal_user.cpp` 声明的 dlsym typedef 与 PTX-EMU 真实 ABI 不一致（**也违反 ADR-076 §D4 自身定义的 v1 规范**）：

```c
// hal_user.cpp:701-708 (UsrLinuxEmu)
typedef int (*ptxemu_module_version_fn)(void);                                  // ✅ correct
typedef unsigned long (*ptxemu_image_load_fn)(const void*, unsigned long,
                                             unsigned long*);                  // ❌ 3 args, returns int
typedef int (*ptxemu_image_kernel_name_fn)(unsigned long, char*, size_t);      // ❌ signature wrong
typedef int (*ptxemu_image_execute_fn)(unsigned long,
                                       const uint32_t[3], const uint32_t[3],
                                       const void*, unsigned int, unsigned int); // ❌ 6 args
typedef int (*ptxemu_image_unload_fn)(unsigned long);                          // ❌ returns int, takes 1 arg
```

```c
// cpptlm_module.h:18-30 (PTX-EMU 真实 ABI)
uint64_t ptxemu_image_load(const uint8_t* image_bytes, size_t image_size);    // 2 args, returns handle
int ptxemu_image_kernel_name(uint64_t handle, char* buf, size_t buf_size);    // returns 0/-EINVAL
int ptxemu_image_execute(uint64_t handle,
                          uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                          uint32_t block_x, uint32_t block_y, uint32_t block_z,
                          size_t shared_mem_bytes,
                          void** kernel_args, size_t args_count);             // 10 args
int ptxemu_image_unload(uint64_t handle);                                     // 1 arg
int ptxemu_module_version(void);                                              // ✅
```

### 3.2 后果（x86-64 SysV ABI）

`user_kernel_module_load` at `hal_user.cpp:803-806`:
```c
unsigned long handle = 0;
int rc = g_ptxemu_abi.image_load(
    a->image_ptr, static_cast<unsigned long>(a->image_size), &handle);  // 3 args
if (rc != 0 || handle == 0) return -EINVAL;                              // ❌ 永远 fail
```

调用栈：
1. PTX-EMU `ptxemu_image_load(bytes, size)` 接收 2 args
2. `rdi=bytes, rsi=size`，`rdx` 第三个参数被忽略
3. 函数返回 `next_handle_{1}` ≥ 1 在 `rax`
4. 调用方把 `rax` 当 `int rc` 读 → `rc=1 ≠ 0` → return `-EINVAL`
5. 即使忽略 rc，`handle` 也仍是 0（out-param 从未被 PTX-EMU 写入）

**结果：100% `image_load` 失败**。`dlsym` resolution 成功（version 签名正确），但语义层完全错位。version 握手 **不能** 抓到这种 drift — 因为 `ptxemu_module_version` typedef (`hal_user.cpp:701`) 与 PTX-EMU 真实签名 (`cpptlm_module.h:30`) 一致。

### 3.3 为什么 144/145 ctest PASS

ADR-076 Migration Step 4（line 483-484）规定 e2e 用 **mock** `libptxemu_device.so`：
> `tests/e2e/test_ptxemu_kernel_module_e2e.cpp` (mock libptxemu_device.so + IGpuDriver load → launch → unload)

Mock `.so` 是按 `hal_user.cpp` typedef 编码的（mock 与调用方契约一致），所以 mock 测试通过。**但 mock 永远不能 catch 真 .so 的签名不一致**。这是 mock-based acceptance gate 的根本局限。

### 3.4 修复（Quick）

**文件**：`plugins/gpu_driver/hal/hal_user.cpp:701-708, 803-806, 808-813, 830-832, 842-843`

**改动**：
```c
// 修正后的 typedef（与 cpptlm_module.h 一致）
typedef uint64_t (*ptxemu_image_load_fn)(const uint8_t*, size_t);
typedef int (*ptxemu_image_kernel_name_fn)(uint64_t, char*, size_t);
typedef int (*ptxemu_image_execute_fn)(uint64_t,
                                       uint32_t, uint32_t, uint32_t,
                                       uint32_t, uint32_t, uint32_t,
                                       size_t, void**, size_t);
typedef int (*ptxemu_image_unload_fn)(uint64_t);
```

**Call site 修正（803-806）**：
```c
uint64_t handle = g_ptxemu_abi.image_load(
    reinterpret_cast<const uint8_t*>(a->image_ptr), a->image_size);
if (handle == 0) return -EINVAL;  // 0 表示 PTX-EMU 端 load 失败
```

类似的修正需应用到 `image_kernel_name` / `image_execute` / `image_unload` 的所有 call site。

**根本缓解**：在 PTX-EMU 仓加 **self-dlopen conformance test**：
- 在 `tests/unit/cudart/test_cpptlm_module_abi_conformance.cpp`
- 用 ADR-076 §D4 typedef dlopen 自己
- assert 所有 5 个符号成功解析
- 这能在 CI 早期抓到未来签名漂移

> **注**：PTX-EMU 仓已存在 `tests/integration/test_cpptlm_module_dlopen.cpp`（Metis 评审发现），它 dlopen `libptxemu_device.so` 并 resolve 5 个符号。但**只** resolve 不调用。推荐的 `test_cpptlm_module_abi_conformance.cpp` 是补充 test：除了 resolve 外，**实际调用** `ptxemu_image_load` / `image_kernel_name` / `image_execute_named` / `image_unload` 验证 round-trip 行为（用 cute_rmsnorm fixture），能在 unit 层抓 Defect 1 类的签名漂移。两个 test 互补，不重复。

---

## 4. Defect 2: `g_gpu_context` 在 dlsym 路径下永不被创建

### 4.1 问题

`g_gpu_context` 是 PTX-EMU 的全局 GPU 单例（`std::unique_ptr<GPUContext>`），所有 kernel 执行都通过它。但：

- `ptx_interpreter.cpp:150-152` — `if (!g_gpu_context) { return KernelLaunchRequest(); }` 早返回
- `ptx_interpreter.cpp:344-346` — `funcInterpreter` 仅在 `g_gpu_context` 存在时 submit kernel request
- `cpptlm_module.cpp` 全 274 行**从未**创建 `g_gpu_context`（亲自通读）
- 创建点仅在：
  - `cudart_sim.cpp:261-277`（legacy LD_PRELOAD 路径，依赖 `__cudaRegisterFatBinary` 拦截）
  - `tests/unit/cpptlm/test_cosim_smoke.cpp`（测试夹具）

### 4.2 后果

dlsym 路径下（`hal_user.cpp` dlsym 加载 `libptxemu_device.so`）的 `ptxemu_image_execute` 流程：

1. `ptxemu_image_execute(handle, ...)` → `PtxEmuImageExecutor::execute` (`cpptlm_module.cpp:89`)
2. `PtxInterpreter interpreter;` 局部构造
3. `interpreter.launchPtxInterpreter(ctx, ...)` → `prepareKernelLaunchRequest` (`ptx_interpreter.cpp:71`)
4. `setupConstantSymbols` / `setupKernelArguments` 调用 `CudaDriver::instance().malloc(...)` (line 446)
5. `CudaDriver::simple_memory_` 永未设置 → `get_global_pool()` 返回 `nullptr` (`cuda_driver.cpp:89-95`)
6. `dev_ptr = nullptr + offset` → `cuda_driver.cpp:44-45` 产生野指针（小地址如 0x1000）
7. `memset(param_space, ...)` (`ptx_interpreter.cpp:452` 等) 写小地址 → **segfault** 或 **silent corruption**
8. 或者：`prepareKernelLaunchRequest` 在 line 150-152 早返回（`!g_gpu_context` 检查）
9. `on_complete` callback 不注册，caller 拿到 default-constructed `KernelLaunchRequest`
10. `submit_kernel_request` 从未调用
11. `ptxemu_image_execute` 返回 0

**结果**：`ptxemu_image_execute` 看起来成功（rc=0）但 **kernel 静默不执行**。`cudaMemcpy(device_ptr, host_buf, ...)` 之后用户读 buffer，**数据全 0 或全 undefined**。

### 4.3 修复（Quick-Short）

**选项 A**（推荐）：在 `cpptlm_module.cpp::load_image` 懒初始化 GPU 状态

```cpp
uint64_t PtxEmuImageExecutor::load_image(const uint8_t* bytes, size_t size) {
    // ... 既有校验 ...
    
    // Lazy init: ensure GPUContext + CudaDriver backend ready
    // ⚠️ 必须显式调 init() —— constructor 不做 set_simple_memory
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        if (!g_gpu_context) {
            g_gpu_context = std::make_unique<GPUContext>("");  // 默认 config
            g_gpu_context->init();  // ← 关键: init() 调 set_simple_memory
                                       //   (gpu_context.cpp:34-46, not constructor)
        }
    });
    
    // ... 既有 handle 分配 ...
}
```

**重要**：`GPUContext` 的 constructor (`gpu_context.cpp:15-32`) 只初始化 `gpu_state` 和 `config`，**不**调 `set_simple_memory`。该调用在 `init()` 方法 (`gpu_context.cpp:34-46`)。漏调 `init()` 会让 `CudaDriver::simple_memory_` 仍为 `nullptr`，Defect 2 名义上修但实际未修。

**选项 B**：新增 `ptxemu_init()` ABI，hal_user.cpp 在首次 load 前调

- 需要 ADR-0029 修订
- 同步更新 ADR-076 §D
- 跨仓 ship 协议需要 canonical ADR (PTX-EMU ADR-0029) 先升 Accepted

**PTX-EMU 已知该问题**：`openspec/changes/archive/2026-08-13-fix-path2d-ptxir-execution-bugs/tasks.md` item 3.4 提议"Add explicit missing-context branch: if `g_gpu_context == nullptr`, return `-EINVAL`"。但该任务未勾选（archived change）。

---

## 5. Defect 3: Memory Pool 不可桥接

### 5.1 问题

PTX-EMU 和 UsrLinuxEmu 各有独立 mmap pool，**无 API 协调**：

| 实体 | 范围 | 分配器 | 来源 |
|------|------|--------|------|
| **UsrLinuxEmu HAL heap** | `0x100000000` 起，256MB | `gpu_buddy` over `hc->heap` | `hal_user.h:31-32` `HAL_HEAP_BASE=0x100000000`, `HAL_HEAP_SIZE=256MB` |
| **PTX-EMU SimpleMemory** | mmap-anonymous，默认 4GB | `SimpleMemoryAllocator` (bump+free-list) | `simple_memory.cpp:12-14` |
| **TaskRunner device ptr** | `cudaMalloc` → ioctl(0x10) → HAL heap → `0x1_0000_0000+off` | — | TaskRunner libcuda_shim |
| **PTX-EMU `CudaDriver` ptr** | `cudaMalloc` → `CudaDriver::malloc` → `SimpleMemory` 私有 mmap | — | `cudart_sim.cpp:1026-1040` |

### 5.2 后果：LD/ST.GLOBAL 必崩

`simple_memory.cpp:30-50` 的 `direct_access` 地址归一化逻辑：

```cpp
void SimpleMemory::direct_access(uint64_t address, void* data, size_t size, bool is_write) {
    bool in_range = (address >= (uint64_t)global_base_) &&
                    (address < ((uint64_t)global_base_ + global_size_));
    if (in_range) {
        address -= (uint64_t)global_base_;  // normalize to offset
    }
    // address NOT in mmap range → treated as raw offset
    
    if (!validate_offset(address, size)) {
        throw InvalidMemoryAccessException(...);
    }
    // ...
}
```

当 TaskRunner-allocated device ptr（如 `0x1_0000_0000+0x1000 = 0x1_0000_1000`）传给 `ld.global`：

1. `address = 0x1_0000_1000`（HAL heap VA）
2. `global_base_` 是 PTX-EMU mmap 的 host pointer（如 `0x7f...`） → `in_range == false`
3. 退化为 raw offset：`address = 0x1_0000_1000`
4. `validate_offset(0x1_0000_1000, ...)` → `offset < global_size_`（4GB）：`0x1_0000_1000 < 4GB` = true → 可能通过校验
5. **静默读错 PTX-EMU 私有 mmap 内偏移 4GB+16KB 处的随机数据**

如果 PTX-EMU pool 恰好等于 4GB（默认），`0x1_0000_0000 < 4GB` 严格 false → **`InvalidMemoryAccessException` 抛出**。

无论哪种情况，**v1 范围内任何 dereference TaskRunner-allocated device ptr 的 kernel 都不可能得到正确结果**。

### 5.3 根因

ABI 表面（`cpptlm_module.h:18-52`）**无 memory 注册 API**：
- 无 `ptxemu_set_memory_base(base, size)`
- 无 `ptxemu_register_external_region(base, size)`
- 无 `cudaMalloc` 替代
- 无 `MAP_FIXED` 共享 mmap 协调

kernel arg *值*（device ptr）从 TaskRunner memcpy 到 PTX-EMU `param_space`（`ptx_interpreter.cpp:484`），但 ptr 指向的 buffer **仍在 UsrLinuxEmu HAL heap**，PTX-EMU 看不见。

### 5.4 修复选项

#### 选项 (a) v1 scope limitation（最便宜，零代码）

**修法**：在 ADR-076 §D7 文档化 v1 限制：
> v1 HAL extension 仅支持**不 dereference device pointer**的 kernel（即 `arg_size == 0` 或纯标量参数）。任何需要 `ld.global`/`st.global` 访问 `cudaMalloc` 分配的 buffer 的 kernel **不在 v1 范围**。
>
> 适用场景：参数为 scalar（int / float）且 kernel 仅做寄存器操作的纯算法 kernel（如 reduction in registers）。
>
> v2 引入 memory coordination（见 §D7 后续工作）。

**Tradeoff**：诚实的范围缩减，立即可 ship。但实际 CUDA kernel 几乎都 dereference device buffer，所以 (a) 实质把 v1 缩到"理论正确但无实际用途"。

#### 选项 (b) 扩展 ABI + `ptxemu_mem_register`（Medium）

**修法**：
1. PTX-EMU ADR-0029 §D8 扩展：新增 `int ptxemu_mem_register(uint64_t base, size_t size);`
2. ADR-076 §D 新增 D7 描述协议：UsrLinuxEmu 在 init 时调 `ptxemu_mem_register(HAL_HEAP_BASE, HAL_HEAP_SIZE)`
3. PTX-EMU 内部把该区域**纳入** SimpleMemory 视野（用 shared mmap / MAP_FIXED / 或在 `direct_access` 加 VMA → backing mapping 表）
4. 跨仓 ABI bump：`CPPTLM_MODULE_VERSION` → 3

**Tradeoff**：需要两侧协调 commit，跨仓 ADR 同步（per ADR-035 §R5.1）。Medium 工作量。

#### 选项 (c) 共用 mmap region（Medium-Hard）

**修法**：
1. UsrLinuxEmu 不再用 256MB HAL heap，改用 `mmap(NULL, large_size, ..., MAP_SHARED|MAP_ANONYMOUS, -1, 0)`
2. PTX-EMU 通过 env var (`PTXEMU_SHM_BASE`) 接收同一 mmap
3. `SimpleMemory` 构造时使用 shared mapping

**Tradeoff**：破坏 UsrLinuxEmu 现有 buddy allocator（`gpu_buddy`），需要重写 mem pool 抽象。**不推荐** — 范围太大。

### 5.5 Oracle 推荐

**采用 (a) 先 ship，ADR-076 §D7 文档化 v1 限制；(b) 作为 follow-up ADR（ADR-076 v2）**。这样：
- ADR-076 v1 不会卡在 ABI 协调上
- 真实 e2e 测试在 v1 用 mock mem pool 验证
- 真实 .so 集成等 v2 ADR 落地

> ⚠️ **重要警告 (评审提醒)**: 选项 (a) 把 v1 缩到"理论正确但无实际用途"——任何 dereference `cudaMalloc`-allocated device buffer 的真实 CUDA kernel 都会失败。如果选 (a) ship v1，**必须**在 ADR-076 §D7 明确写"v1 仅供 mock 验证 / prototype 演示 / 单元测试，不可作为 production kernel 执行路径"。在 stakeholder 评审中需特别说明此限制，避免误读为"v1 已 ship 可用"。

---

## 6. ADR-076 更新建议（6 项）

| 段落 | 改动 | 工作量 | 阻塞当前 release？ |
|------|------|--------|---------------------|
| **§D 新 D7 Memory Domain Coordination** | 选 (a)/(b)/(c)，明文 v1 scope 限制 | Quick (a) / Medium (b) | 是 |
| **§D4 风险行** | 加"dlsym 无签名检查" row，引用本文 §3 | Quick | 是 |
| **§D 新注 GPUContext lifecycle** | 明确 `ptxemu_image_load` 懒初始化责任 | Quick | 是 |
| **§Migration Step 4** | 用**真 .so smoke gate** 替代 mock-only | Quick | 是 |
| **§Acceptance Gate** | 加 hard gate "real .so interop" + "tadr-307 landed" | Quick | 是 |
| **§Consequences 风险表** | 加 row"kernel args 引用 ALLOC_BO 内存不可解析（v1 范围）" | Quick | 是 |

### 6.1 关键修订草案

**§D7 (新)** — Memory Domain Coordination

> v1 HAL extension 不支持 dereference `cudaMalloc`-allocated device buffer 的 kernel。v1 适用场景限于：
> - Kernel arg 全为 scalar（int / float / pointer-to-host 已被 ioctl args 路径处理）
> - Kernel 仅做寄存器 / shared memory 操作
> - 任何 `ld.global`/`st.global` 访问 `GPU_IOCTL_ALLOC_BO` (0x10) 分配的 device buffer 在 v1 范围外
>
> v2 引入 `ptxemu_mem_register(base, size)` 协调协议（PTX-EMU ADR-0029 v2 amendment 同步），通过 `MAP_SHARED` mmap 或 explicit VMA mapping 桥接 HAL heap。

**§D4 风险 (新行)**

> **R-D4-3: ABI signature drift across dlsym boundary**
> 严重度：🟡 中
> 触发：UsrLinuxEmu 端 typedef 与 PTX-EMU `cpptlm_module.h` 漂移（实测发现 `image_load` 100% 失败）
> 缓解：
> 1. PTX-EMU 仓加 self-dlopen conformance test (`tests/unit/cudart/test_cpptlm_module_abi_conformance.cpp`)，CI 早期 trip
> 2. `kPtxemuAbiMinVersion` 握手基础上，**新加 typedef struct 描述符 ABI 校验**（v2）

**§Migration Step 4 修订**

> 旧：`tests/e2e/test_ptxemu_kernel_module_e2e.cpp` (mock libptxemu_device.so + IGpuDriver load → launch → unload)
> 新：`tests/e2e/test_ptxemu_kernel_module_real_so_e2e.cpp` (dlopen **真** `libptxemu_device.so` from `PTXEMU_ROOT` + IGpuDriver load → launch → unload + 输出 buffer 字节比对)
>
> Mock `.so` 测试降级为 `tests/unit/hal/test_hal_kernel_module_mock.cpp`（保留 contract test 价值，不再作为 acceptance gate）

---

## 7. 跨仓集成测试位置

**Oracle 推荐 + 共识**：UsrLinuxEmu `tests/e2e/`（**唯一能 catch 全部 3 个 defect**）

| 位置 | 所有权 | 能 catch Defect 1+2+3？ | CI 可行性 |
|------|--------|--------------------------|-----------|
| **PTX-EMU tests/** | 自有 .so + ABI 真相 | 部分（自 dlopen conformance test 抓 1；lazy-init 测试抓 2；不能抓 3） | Trivial（self-contained） |
| **UsrLinuxEmu tests/e2e/** ✅ | 拥有 ioctl→hal→dlsym seam + TaskRunner submodule | **全部 3 个** | 需 PTX-EMU artifact in CI（ADR-076 已假设 `PTXEMU_ROOT`） |
| TaskRunner tests/ | consumer-side | 不能（无 PTX-EMU 链接依赖） | 容易但低价值 |
| CppTLM | 与本路径正交 | 不能 | N/A |

**推荐双层策略**：
1. **PTX-EMU 仓 cheap self-dlopen test**（CI tripwire 防签名漂移）
2. **UsrLinuxEmu 仓真 .so e2e test**（CI 验证集成栈完整可执行）

---

## 8. CppTLM 角色澄清

**(b) co-simulation backplane**（TLM 2.0 周期精确 NoC 仿真框架）— 通过 `cpptlm_bridge.h` 5 虚方法做 timing-only 桥接。

**与本路径正交**：
- `cpptlm_module.cpp:27-28` assumption #3：`g_cpptlm_bridge` 在 image executor 路径下恒为 `nullptr`
- `CppTLM/AGENTS.md:7` 明确 TLM 2.0 NoC 仿真框架定位
- CppTLM 仅在外部部署显式 attach bridge 时进入 LD/ST 时序模拟

**结论**：CppTLM **不参与** ADR-076 路径，不需要为本文 3 个 defect 做任何修改。

---

## 9. 消费侧额外阻塞

| 阻塞 | 位置 | 影响 |
|------|------|------|
| `cuModuleLoadData` 仍 `CUDA_ERROR_NOT_IMPLEMENTED` | `UsrLinuxEmu/external/TaskRunner/src/umd/libcuda_shim/cu_module.cpp:135-138` | tadr-307 在 submodule checkout 中未 ship |
| `cuModuleLoadDataEx` 同样 | `cu_module.cpp:140-145` | 同上 |
| `cuModuleLoadFatBinary` 同样 | `cu_module.cpp:147-150` | 同上 |

**即使本文 3 个 defect 全修，端到端 TaskRunner → UsrLinuxEmu → PTX-EMU 链路仍未通**，因为 cu_module.cpp 整个 load family 仍是 stub。tadr-307 ship 顺序：

1. **canonical ADR 升 Accepted**（PTX-EMU ADR-0029 §D8 → Accepted，TaskRunner tadr-307 → Accepted）
2. **PTX-EMU 仓 Phase 1 修 Defect 2（lazy init）** + 修 cpptlm_module 已知 open issues
3. **TaskRunner 仓实施 IGpuDriver load/launch/unload 纯虚方法**（替换 cu_module.cpp:135-150 stub）
4. **UsrLinuxEmu 仓修 Defect 1（ABI typedef）** + Step 4 e2e 真 .so gate
5. **跨仓 ABI 版本对齐**：`CPPTLM_MODULE_VERSION` 当前 2；如选 5.4 (b) 需升 3

**跨仓 commit 顺序严格按 ADR-035 §R5.1** 串行 ship（per ADR-076 §Migration 4-step protocol）。

---

## 10. 总结

### 10.1 现状

3 个堆叠 critical defect 阻止 ADR-076 端到端可执行。Mock 测试 144/145 PASS 是**假阳性** — mock 不能验证真 .so 契约。

### 10.2 修复路径

| Step | 仓 | 工作量 | 验证 |
|------|------|--------|------|
| 1 | UsrLinuxEmu | Quick | 改 `hal_user.cpp:701-708, 803-806, 808-813, 830-832, 842-843` + CI 跑 ABI conformance |
| 2 | PTX-EMU | Quick-Short | `cpptlm_module.cpp::load_image` lazy-init `g_gpu_context` + 测试覆盖 |
| 3 | ADR-076 修订 | Quick（选 5.4 (a)）| 文档化 v1 scope limitation + 加 6 项修订 + 替换 mock gate 为真 .so gate |
| 4 | TaskRunner | Medium | tadr-307 实施 cuModuleLoadData 替换 stub + cu_launch 适配 |
| 5 | UsrLinuxEmu | Medium | 真 .so e2e 集成测试 |

### 10.3 ADR-076 是否需要更新？

**是，必须更新**。本文 §6 列出的 6 项修订全部为硬阻塞（缺一不可）：
- D7 Memory Domain Coordination：缺此决策 = v1 范围模糊
- D4 风险行：缺此行 = 未来漂移无法被 governance 抓住
- GPUContext lifecycle 注释：缺此 = 责任不清，无法 ship lazy-init 修复
- Step 4 真 .so gate：缺此 = Defect 1 还会再次发生
- Acceptance Gate 两项：缺此 = 跨仓 ship 顺序无强制
- 风险表新行：缺此 = 实施者无法理解 v1 限制

---

## 11. References

### 11.1 关键文件:行 引用

**PTX-EMU**：
- `include/cudart/cpptlm_module.h:1-58` — 公共 ABI（VERSION=2）
- `src/cudart/cpptlm_module.cpp:48-67, 89-132, 169-214, 227-272` — load/execute_named/extern "C" 包装
- `src/cudart/cuda_driver.cpp:32-52, 89-95` — malloc + simple_memory_ 检查
- `src/cudart/ptx_interpreter.cpp:150-152, 344-346, 446-484` — g_gpu_context 检查 + CudaDriver::malloc
- `src/memory/simple_memory.cpp:12-14, 30-50` — mmap + direct_access
- `src/cudart/cudart_sim.cpp:261-277, 465-468, 1026-1040` — g_gpu_context 创建点
- `openspec/changes/archive/2026-08-13-fix-path2d-ptxir-execution-bugs/tasks.md` — known open issues

**UsrLinuxEmu**：
- `docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md:1-553` — canonical ADR
- `plugins/gpu_driver/hal/hal_user.cpp:74-89, 680-880, 701-708, 797-846, 803-806, 830-832` — dlsym + 3 fn-ptr 实现
- `plugins/gpu_driver/hal/hal_user.h:31-32` — HAL_HEAP_BASE/SIZE
- `plugins/gpu_driver/sim/vram_store.h` — `GpuVramStore`
- `plugins/gpu_driver/hal/gpu_hal.h` — `struct gpu_hal_ops` #66/#67/#68

**TaskRunner**：
- `UsrLinuxEmu/external/TaskRunner/src/umd/libcuda_shim/cu_module.cpp:135-138, 140-145, 147-150` — load family 仍 stub

**CppTLM**：
- `CppTLM/AGENTS.md:7` — TLM 2.0 NoC 仿真框架定义

### 11.2 跨仓 ADR 引用

- PTX-EMU ADR-0029 §D8（CP 端集成约定 + HAL 扩展方案）
- UsrLinuxEmu ADR-076（canonical GPGPU Kernel Module IOCTL）
- UsrLinuxEmu ADR-023 §D4（HAL append-only 治理）
- UsrLinuxEmu ADR-035 §R5.1（跨仓 commit 同步协议）
- TaskRunner tadr-307（IGpuDriver kernel module extension — consumer-side）
- TaskRunner tadr-301（IGpuDriver 28→47 方法契约基线）

### 11.3 Oracle 评审 session

- 子代理 session: `ses_004c1da61ffeZdTs6hehkIEB2w`
- 4 hypothesis × 3-4 evidence each + 4 question 完整回答
- 所有引用已亲自 Read 验证

---

**审计结论**：ADR-076 实施存在 3 个堆叠 critical defect，**必须在文档化 + 修复后才能 ship 真 .so e2e 验证**。Mock-based acceptance gate 不足以防止此类漂移，必须改为 real-binary smoke gate（ADR-076 §Migration Step 4 修订）。

**建议 UsrLinuxEmu 仓 owner 行动**：
1. 评审本文 §3-§5 的 3 个 defect
2. 评审 §6 的 6 项 ADR-076 修订草案（按 §10.2 顺序 ship）
3. 同步 TaskRunner owner 评估 tadr-307 ship 顺序
4. 决策 memory domain 协调方案（§5.4 a/b/c 三选一）

— 审计人：Sisyphus，2026-08-13
