# PTX-EMU 代码质量评估与架构债务分析报告

> **日期:** 2026-05-07 | **分析范围:** 全项目 src/ + tests/ | **测试计数:** ~480 TEST_CASE

---

## 一、完整合并优先级矩阵

| Pri | # | 工作项 | 类型 | 风险 | 代码量 | 文件数 | 当前状态 |
|-----|---|--------|------|------|--------|--------|---------|
| **P0** | 1 | **ISSUE-004: active_mask 双源不一致** | bug | 低 | ~15 行 | 3 文件 | ✅ 方案就绪，测试就绪（J1-J5），可立即编码 |
| **P0** | 2 | **SHMEMADDR 共享可变静态变量** | debt | 高 | ~5 行 | 1 文件 | ⚠️ 多 CTA 场景竞态条件 |
| **P0** | 3 | **ptx_parser.cpp ~30 个裸 new 无 RAII** | debt | 高 | ~200 行 | 1 文件 | ⚠️ 内存泄漏风险，所有权不明 |
| **P1** | 4 | **Test 3 测试膨胀清理** | hygiene | 中 | ~-1800 行 | 11→5 文件 | 💣 3 个冗余集群，sanity.sh 已注释 |
| **P1** | 5 | **atom.global.add 实现** | feature | 低 | ~50 行 | 1 文件 | 🏗️ 管线完成，仅缺实现 |
| **P1** | 6 | **死代码 DEBUGINTE/LOGINTE 移除** | hygiene | 低 | ~-30 行 | 3 文件 | 🗑️ 从不定义的宏条件编译 |
| **P1** | 7 | **单例初始化顺序依赖** | debt | 中 | ~10 行 | 4 文件 | ⚠️ 4 单例 + 2 全局按序触发生命周期 |
| **P2** | 8 | **PTXIR Reader 保真度修复** | bug | 中 | ~100 行 | 1 文件 | ⚠️ 95+ 指令类型 default 分支静默丢弃 |
| **P2** | 9 | **ptx_visitor.cpp 包含源文件** | debt | 低 | ~0 行 | 1 文件 | 🐌 warp.cpp 被包含两次，无并行构建 |
| **P2** | 10 | **WMMA mma_sync 实现** | feature | 高 | ~200 行 | 2 文件 | 🏗️ 管线完成，tensor.cpp/wmma.cpp 重复 |
| **P2** | 11 | **注释代码块清理** | hygiene | 低 | ~-200 行 | 9 文件 | 📝 9 个文件共有 ~200 行注释代码 |
| **P3** | 12 | **多 fatbinary 支持** | feature | 高 | ~50 行 | 2 文件 | 🔍 当前单例覆盖，真正限制非 FIXME 死代码 |
| **P3** | 13 | **ADR-0011 Pipeline 架构** | arch | 高 | ~500 行 | 3+ 文件 | 📋 Proposed，依赖 ⑧ |
| **P3** | 14 | **Logger 拉取完整模拟栈** | debt | 低 | ~5 行 | 1 文件 | 📎 仅为 get_gpu_clock_from_context 引入 |
| **P3** | 15 | **已删除 CUDA 属性注释** | hygiene | 低 | ~-15 行 | 1 文件 | 💬 现代 CUDA 已废弃属性注释 |

---

## 二、架构债务详情

### 2.1 严重缺陷

#### #2 SHMEMADDR 静态变量（`thread_context.cpp:25`）
```cpp
static uint64_t SHMEMADDR = 0;  // 所有线程/CTA 共享
```
被 `initialize_shared_memory()` 写入，所有线程共享同一高 32 位地址假设。多 CTA 工作负载中为确定竞态条件。

#### #3 裸 new 泛滥（`ptx_parser/ptx_parser.cpp`）
- `new KernelContext()` / `new ParamContext()` × 多处
- `new StatementContext::XXX()` × ~9 处
- `new OperandContext::XXX()` × ~12 处
- **几乎无对应的 `delete`**——完全依赖 PtxContext 析构函数清理
- 通过 X-Macro `new StatementContext::opname()` × 宏展开
- `instruction_factory.cpp:15` `new opstr##Handler()` X-Macro 生成

#### #7 单例链
```
LoggerConfig (无依赖)
  → CudaDriver::instance() (依赖于 SimpleMemory 设置)
    → HardwareMemoryManager::instance() (依赖于 SimpleMemory 设置)
      → g_gpu_context (创建 SMs, 初始化 ResourceManager)
        → g_ptx_interpreter
```
+ `ResourceManager::instance()` 在 GPUContext 初始化期间被调用

### 2.2 设计问题

#### #6 DEBUGINTE/LOGINTE 死代码
```cpp
// cta_context.cpp:15-20
#ifdef DEBUGINTE
extern bool sync_thread;  // 从不定义 DEBUGINTE
#endif
#ifdef LOGINTE
extern bool IFLOG();     // 从不定义 LOGINTE
#endif
```
4 个 `extern` 声明 + 2 个代码块从不可达。`PTX_DEBUG_BARRIER` 在 `sm_context.cpp` 中有条件编译但未在构建系统中启用。

#### #9 ptx_visitor.cpp 包含 .cpp
`src/ptx_parser/ptx_visitor.cpp` 使用 `#include "ptx_visitor_xxx.cpp"` 包含 10+ 个 .cpp 文件，而非独立编译单元：
```cpp
#include "ptx_visitor_generic.cpp"   // 849
#include "ptx_visitor_atom.cpp"      // 852
#include "ptx_visitor_wmma.cpp"      // 858
#include "ptx_visitor_warp.cpp"      // 873
// ... 10 个包含 ...
#include "ptx_visitor_warp.cpp"      // 883 ← 重复包含（可能是 bug！）
```
阻止并行构建，隐藏文件间依赖关系，且 `warp.cpp` 被包含两次（可能是无意的）。

### 2.3 BUG-REPRODUCTION 测试膨胀

#### #4 冗余集群
| 集群 | 当前 | 目标 | 删除估计 |
|------|------|------|---------|
| CFG 分析 | 3 文件 (3 TEST_CASE) | 1 文件 | -2 文件 |
| 屏障执行重现 | 3 文件 (4 TEST_CASE) | 1 文件 | -2 文件 |
| 全执行环境 | 4 文件 (10 TEST_CASE) | 2 文件 | -2 文件 |
| **总计** | **11 文件 (~3800 行)** | **~5 文件 (~2000 行)** | **-6 文件 (-1800 行)** |

**保留的唯一测试：**
- `test3_reproduction.cpp`（18 TEST_CASE，最全面）
- `test_syncthreads_direction.cpp`（10 TEST_CASE，D3 shared memory+barrier 唯一）
- `test_syncthreads_test3_full.cpp`（1 TEST_CASE，SETP 谓词评估唯一）
- `test_specific_bugs_unit.cpp`（10 TEST_CASE，sanity.sh 活跃）
- `test_post_barrier_divergence.cpp`（5 TEST_CASE，不同 bug）

---

## 三、高优先级任务计划

已在以下路径创建详细实现计划：

| 优先级 | 计划文件 | 涉及文件 |
|--------|---------|---------|
| **P0** | `docs/superpowers/plans/2026-05-07-active-mask-consistency.md` | `warp_context.cpp`, `warp_context.h`, `thread_context.cpp` |
| **P1** | `docs/superpowers/plans/2026-05-07-test3-reproduction-cleanup.md` | 11 个测试文件 → 5 文件 |
| **P1** | `docs/superpowers/plans/2026-05-07-atomic-add-implementation.md` | `atomic.cpp`, 测试文件 |

---

## 四、关键发现摘要

1. **ISSUE-004 修复是最佳入手点**：仅 12 行代码修改、5 个已有测试用例、风险极低，但消除的是 core execution loop 中的根本不一致问题。

2. **Test 3 测试清理价值大但需谨慎**：11 个重叠文件中保留哪些唯一 TEST_CASE 需仔细核对。优先保 `test3_reproduction.cpp`（最全面）和 `test_specific_bugs_unit.cpp`（sanity.sh 活跃）。

3. **atom.global.add 实现管线已完整**：语法/解析/派发/管线全部就绪，唯一缺乏的是 `atomic.cpp` 中 ~50 行的实际实现。

4. **ptx_parser.cpp 是最大技术债务源**：~30 个裸 new 且缺少 RAII 清理，但由于 parser 路径在模拟器启动时运行一次后即释放，实际泄漏影响有限。

5. **SHMEMADDR 静态变量是真正的 P0 债务**：在多 CTA 场景下是明确竞态条件，修复简单但影响面广（涉及 shared memory 初始化路径）。
