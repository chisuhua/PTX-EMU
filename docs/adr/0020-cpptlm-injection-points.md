# ADR-0020: 接受 CppTLM Phase 8.B D1-Full 注入点（IScoreboard / IPipelineLatencyProvider / ITensorCoreTiming）

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-07-14 |
| **关联任务** | PTX-1 ~ PTX-6（见 `openspec/changes/cpptlm-phase8b-injection-points/`）|
| **关联 PR** | （待实施后填写）|
| **作者** | PTX-EMU Architecture Team |
| **审核人** | PTX-EMU Architecture Team（2026-07-16）|
| **关联 OpenSpec change** | `openspec/changes/cpptlm-phase8b-injection-points/`（待创建）|
| **关联 CppTLM 文档** | `CppTLM/docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md`<br>`CppTLM/docs/adr/ADR-NV-02-phase8b-d1-strategy.md`（Status Update 2026-07-14）<br>`CppTLM/docs/superpowers/specs/2026-07-03-ptxemu-phase8b-d1full-plan.md` |

---

## 上下文

PTX-EMU 当前 `SMContext` 仅暴露**一个**外部注入点：

```cpp
void SMContext::set_warp_scheduler(std::unique_ptr<WarpScheduler> scheduler);  // ✅ 已有
```

[CppTLM 任务书 §1.2] 审查发现，Scoreboard、Pipeline 延迟、TensorCore timing 的注入点**不存在**。当前 PTX-EMU 用以下隐式机制替代：

| 组件 | 现状 |
|------|------|
| Scoreboard | 无独立类；用 `WarpState::threads[lane].blocked_cycles_remaining` 隐式管理 |
| Pipeline 延迟 | `InstructionLatencyTable` 全局单例，通过 `load(InstructionLatencyConfig&)` JSON 覆盖 |
| TensorCore timing | 延迟来自 `tcgen05_handler.cpp` 内部硬编码或 JSON config |
| 寄存器信息 | 无 `dest_registers()` 暴露 API |
| 物理 warp ID | `WarpContext::get_physical_warp_id()` ✅ 已存在 |

**触发事件**：

1. **2026-07-03**：CppTLM 团队对 PTX-EMU 8 个关键头文件进行审查，确认仅 `WarpScheduler` 可注入
2. **2026-07-03**：CppTLM 发布任务书 `2026-07-03-ptxemu-modification-task.md`（803 行），请求 PTX-EMU 新增 3 个纯虚接口 + 3 个 SMContext setter + 修改 `exe_once()`
3. **2026-07-14**：CppTLM `ADR-NV-02` Status Update：D1-Lite → D1-Full 升级（4 组件全栈注入）
4. **2026-07-14**：CppTLM 协作同步文档追加 §13 D1-Full 双路径协作
5. **2026-07-14**：CppTLM 修订 OpenSpec change `2026-06-24-gpu-soc-phase8b-core` 设计/任务/规格

**当前问题**：

- PTX-EMU 缺乏可测试的扩展点，外部 timing 模型（CppTLM / gpgpu-sim / custom）无法替换内置实现
- `blocked_cycles_remaining` 仅 `S_LD` 指令使用，扩展至全指令需新增 per-warp 封装
- `exe_once()` 内部三段式注入窗口未被外部访问，限制了 timing 准确性

**技术约束**：

- PTX-EMU 头文件**不能依赖** CppTLM（避免循环依赖）
- 必须保持现有 `set_warp_scheduler` 接口向后兼容
- nullptr 注入 = 字节级回退到原行为

---

## 决策驱动因素

1. **跨团队协作需求**：CppTLM Phase 8.B D1-Full 集成是 PTX-EMU 已 commit 的未来方向（ADR-NV-02 v1.0 + Status Update）
2. **向后兼容**：现有 600+ 测试用例零回归；nullptr 注入路径行为字节级相同
3. **零依赖设计**：3 个接口头文件只依赖 `<cstdint>` + `<string>`，不引入 CppTLM 或任何外部库
4. **性能透明**：注入点仅在 setter 非 nullptr 时执行分支，零开销
5. **可测试性**：Mock 接口使 PTX-EMU 自身可独立测试注入行为（不依赖 CppTLM 真实实现）

---

## 考虑的替代方案

### 方案 A: 拒绝注入，保持 SMContext 全内置（❌ 拒绝）

**描述**：拒绝 CppTLM 注入请求，PTX-EMU 维持当前架构，timing 准确性受限于内置 `InstructionLatencyTable`。

**优点**：
- 无 API 变更，零迁移成本
- 现有测试零影响

**缺点**：
- 阻断 CppTLM Phase 8.B D1-Full 集成，破坏已签发的 `ADR-NV-02` 协作承诺
- PTX-EMU 错失与 gpgpu-sim 同级别 timing 精度的研究机会
- 单点集成路径（仅 `WarpScheduler`）长期被吐槽（2026-07-14 任务书 §1.2）

**拒绝理由**：跨团队协作已 commit 且有双向时间表（PTX-EMU 2.5 天 + CppTLM 2 周），拒绝会破坏已完成的工作。

---

### 方案 B: 直接 include CppTLM 头文件（❌ 拒绝）

**描述**：让 PTX-EMU 直接 `#include <tlm/gpu/...>`，使用 CppTLM 具体类型作为注入参数。

**优点**：
- 无 Adapter 层，调用直接
- 无类型转换开销

**缺点**：
- **循环依赖**：CppTLM 也需要 PTX-EMU 头文件（基类接口）
- **构建耦合**：PTX-EMU 必须能找到 CppTLM 头文件路径，CI 配置爆炸
- **测试瘫痪**：PTX-EMU 测试无法独立运行（必须 link CppTLM 库）
- **ANTLR4/Java/CUDA 依赖扩散**：违背现有 PTX-EMU 不依赖外部重组件原则

**拒绝理由**：违反"零依赖设计"核心约束。

---

### 方案 C: 纯虚接口 + Adapter 层（✅ 选中）

**描述**：PTX-EMU 定义 3 个纯虚基类（`IScoreboard` / `IPipelineLatencyProvider` / `ITensorCoreTiming`），零外部依赖；CppTLM 侧提供 Adapter 实现这些接口。SMContext 通过裸指针 setter 持有接口实例。

**优点**：
- ✅ PTX-EMU 零 CppTLM 依赖，CI/构建独立
- ✅ 现有 PTX-EMU 测试零影响（接口头文件不影响现有使用）
- ✅ Mock 测试可完全独立编写（不依赖 CppTLM）
- ✅ CppTLM Phase 8.B 模块也可独立测试（独立模式下不需 PTX-EMU）
- ✅ 接口变更只影响 Adapter 层，不影响核心模块
- ✅ nullptr = 禁用注入 = 字节级回退，**完全向后兼容**

**缺点**：
- 增加 3 个新头文件（~30 LOC × 3 = 90 LOC 总开销）
- 接口签名需双方维护（枚举值必须一致 → 编译期 `static_assert`）

**选择理由**：与"零依赖设计"和"向后兼容"两个核心约束完全对齐；是当前 PTX-EMU 与 CppTLM 协作的唯一可行路径。

---

## 决策内容

PTX-EMU 接受 CppTLM Phase 8.B D1-Full 注入请求，实施以下改造。

### 设计原则

1. **零外部依赖**：3 个接口头文件（`include/ptxsim/{scoreboard,pipeline,tensor_core}_interface.h`）只 include `<cstdint>` + `<string>`
2. **裸指针注入**：setter 使用裸指针（非 `unique_ptr`），所有权归外部（libcpptlm_cudart.so）；PTX-EMU 不负责释放
3. **nullptr = 禁用**：所有 setter 默认 nullptr，未注入时行为与改造前字节级相同
4. **per-warp 封装**：`WarpContext::set_blocked_cycles_for_active()` 替代当前 per-thread LD-only 路径
5. **src/dst 区分**：`RegisterAnalyzer::get_dest_registers_as_ids()` 新增，避免与 `analyze_registers()` 冲突

### 实现要点

#### A. 3 个纯虚接口头文件（新增）

**`include/ptxsim/scoreboard_interface.h`**：
```cpp
#ifndef PTXSIM_SCOREBOARD_INTERFACE_H
#define PTXSIM_SCOREBOARD_INTERFACE_H
#include <cstdint>

class IScoreboard {
public:
    virtual ~IScoreboard() = default;
    virtual bool has_free_entry() const = 0;
    virtual bool allocate(uint32_t reg_id, uint32_t warp_id) = 0;
    virtual bool release(uint32_t reg_id, uint32_t warp_id) = 0;
    virtual void tick() = 0;
};
#endif
```

**`include/ptxsim/pipeline_interface.h`**：
```cpp
#ifndef PTXSIM_PIPELINE_INTERFACE_H
#define PTXSIM_PIPELINE_INTERFACE_H
#include <cstdint>
#include <string>

enum class PipelineId : uint32_t {
    P0_INT_FP32 = 0, V_SIMD = 1, P1_FP64 = 2,
    P2_SFU = 3, P3_LSU = 4, P4_TC = 5
};

class IPipelineLatencyProvider {
public:
    virtual ~IPipelineLatencyProvider() = default;
    virtual double get_fractional_cycles(
        const std::string& instruction, PipelineId pipe_id) const = 0;
    virtual double get_fractional_cycles_by_type(
        int statement_type, PipelineId pipe_id) const = 0;
};
#endif
```

**`include/ptxsim/tensor_core_interface.h`**：
```cpp
#ifndef PTXSIM_TENSOR_CORE_INTERFACE_H
#define PTXSIM_TENSOR_CORE_INTERFACE_H
#include <cstdint>

enum class TcPrecision : uint32_t {
    FP4 = 0, FP6 = 1, FP8 = 2, FP16 = 3, BF16 = 4, TF32 = 5
};

class ITensorCoreTiming {
public:
    virtual ~ITensorCoreTiming() = default;
    virtual uint32_t get_latency(TcPrecision prec) const = 0;
    virtual uint32_t get_throughput_cycles(TcPrecision prec) const = 0;
    virtual uint32_t get_latency_mnk(
        TcPrecision prec, uint32_t M, uint32_t N, uint32_t K) const {
        return get_latency(prec);
    }
};
#endif
```

#### B. SMContext 接口扩展

**修改**：`include/ptxsim/sm_context.h`
- `#include` 3 个接口头文件
- 3 个 public setter（裸指针，默认 nullptr）
- 3 个 public getter
- 3 个 private 成员
- **不修改构造函数**

```cpp
class SMContext {
public:
    // 已有接口（不修改）
    void set_warp_scheduler(std::unique_ptr<WarpScheduler> scheduler);

    // 新增注入点
    void set_scoreboard(IScoreboard* scoreboard) { scoreboard_ = scoreboard; }
    void set_pipeline_latency_provider(IPipelineLatencyProvider* provider) {
        pipeline_provider_ = provider;
    }
    void set_tensor_core_timing(ITensorCoreTiming* tc) {
        tensor_core_timing_ = tc;
    }

    IScoreboard*              get_scoreboard()              const { return scoreboard_; }
    IPipelineLatencyProvider* get_pipeline_latency_provider() const { return pipeline_provider_; }
    ITensorCoreTiming*        get_tensor_core_timing()      const { return tensor_core_timing_; }

private:
    IScoreboard*              scoreboard_           = nullptr;
    IPipelineLatencyProvider* pipeline_provider_    = nullptr;
    ITensorCoreTiming*        tensor_core_timing_   = nullptr;
};
```

#### C. WarpContext 扩展

**修改**：`include/ptxsim/warp_context.h` + `.cpp`
- 新增 `set_blocked_cycles_for_active(uint32_t cycles)`
- 对 warp 内所有活跃线程（非阻塞状态）设置 `blocked_cycles_remaining = cycles; is_blocked = true`
- 替代当前 `LdHandler::processOperation()` per-thread LD-only 路径

```cpp
void WarpContext::set_blocked_cycles_for_active(uint32_t cycles) {
    for (auto& thread : warp_state_.threads) {
        if (thread.is_active && !thread.is_blocked) {
            thread.blocked_cycles_remaining = cycles;
            thread.is_blocked = true;
        }
    }
}
```

#### D. RegisterAnalyzer 扩展

**修改**：`src/ptxsim/register_analyzer.cpp` + `.h`
- 新增 `get_dest_registers_as_ids(const StatementContext&) -> vector<uint32_t>`
- **关键**：当前 `analyze_registers()` 提取所有操作数（不区分 src/dst），需新增方法而非修改原方法

#### E. exe_once() 三段式注入

**修改**：`src/ptxsim/core/sm_context.cpp`
- 在 `exe_once()` 内三处插入注入点（Scoreboard 检查 → 延迟查询 → Scoreboard 释放）
- nullptr 完全回退到原行为（**字节级相同**）
- 4 个辅助函数：`get_dest_registers()`, `map_instruction_to_pipeline()`, `is_tensor_core_instruction()`, `map_instruction_to_tc_precision()`

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/scoreboard_interface.h` | **新增** | IScoreboard 纯虚基类（~20 LOC）|
| `include/ptxsim/pipeline_interface.h` | **新增** | IPipelineLatencyProvider + PipelineId（~30 LOC）|
| `include/ptxsim/tensor_core_interface.h` | **新增** | ITensorCoreTiming + TcPrecision（~25 LOC）|
| `include/ptxsim/sm_context.h` | **修改** | +3 include + 3 setter + 3 getter + 3 私有成员 |
| `include/ptxsim/warp_context.h` + `.cpp` | **修改** | 新增 `set_blocked_cycles_for_active()` |
| `src/ptxsim/register_analyzer.h` + `.cpp` | **修改** | 新增 `get_dest_registers_as_ids()` |
| `src/ptxsim/core/sm_context.cpp` | **修改** | `exe_once()` 三段式注入 + 4 辅助函数 |
| `tests/unit/cpptlm/test_smcontext_injection.cpp` | **新增** | 7 个 Mock 测试用例（任务书 §5.2 完整移植）|
| `tests/integration/cpptlm/test_scoreboard_allocation.cpp` | **新增** | 真实 warp + Mock scoreboard 集成测试 |
| `tests/CMakeLists.txt` | **修改** | 注册 2 个新测试目标 |
| `docs/dev-process/lessons-learned.md` | **追加** | 新经验条目 |

**工时估算**：~2.5 天（参照 CppTLM 任务书 §4 PTX-1~PTX-6）

---

## 后果

### 正面影响

1. **跨团队协作落地**：CppTLM Phase 8.B D1-Full 可实施，Adapter 编译期验证通过
2. **可测试性提升**：Mock 接口使 PTX-EMU 自身可独立测试注入行为
3. **timing 准确性**：分数 cycle 延迟（FFMA 4.22 等）替代整数 cycle，逼近 gpgpu-sim baseline
4. **向后兼容**：nullptr 注入 = 字节级回退，现有 600+ 测试零回归
5. **架构清晰**：3 个独立接口清晰划分职责（hazard 检测 / 管线延迟 / TC timing）

### 负面影响

1. **API 表面扩展**：SMContext 新增 3 个 setter + 3 个 getter（6 个新 public 方法）
2. **代码量增加**：~100 LOC（3 接口 + SMContext 改动 + WarpContext 扩展 + RegisterAnalyzer 扩展 + exe_once 注入 + 测试）
3. **接口维护**：枚举值（PipelineId / TcPrecision / StatementType）必须与 CppTLM 同步，破坏需双方协调
4. **测试覆盖成本**：7 个 Mock 测试 + 集成测试 = ~0.8 天测试工时

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **R1**: blocked_cycles 扩展破坏现有 LD 处理 | 中 | 高 | 基线 worktree + Phase 5c 全回归测试 + 任务书 §6 向后兼容表 |
| **R2**: 4 注入点共存时性能开销 | 低 | 低 | nullptr 路径零开销；启用时仅简单分支 |
| **R3**: 枚举值与 CppTLM 不一致 | 中 | 高 | Phase 0 对齐会议（PTX-0.1~0.4）30 分钟强制项 |
| **R4**: get_dest_registers_as_ids 实现细节与 PTX-EMU 现有 StatementContext variant 不匹配 | 中 | 中 | Phase 3b 先 PoC 验证 std::visit 模式；备选：扩展 `RegisterAnalyzer` 提取逻辑 |
| **R5**: CppTLM 团队口头"确认"无 PTX-EMU 侧书面证据 | 高 | 中 | Phase 0 PTX-0.5 强制基线 + CppTLM 书面同步（`CppTLM/docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md §13`）|
| **R6**: 实施过程中与现有 3 个 active OpenSpec changes（cleanup-deprecated-barrier-apis / god-class-refactor-thread-context-phase3 / migrate-bar-warp-sync-to-barrier-module）冲突 | 低 | 中 | 序列化实施：本 change 在 barrier-related changes 归档后启动 |

---

## 实施纪律（PTX-EMU 经验沉淀强制项）

来自 `AGENTS.md` + `.opencode/skills/ptx-lessons-learned/SKILL.md`：

1. **基线 worktree**：实施前 1 分钟建立 `git worktree add ../ptxemu-baseline-2026-07-XX main`（Lessons Learned #4）
2. **分 Phase commit**：每个 PTX-X commit 独立可回退；任何测试回归立即 revert（Lessons Learned #3）
3. **OpenSpec artifacts 2-Phase commit**：artifacts（proposal/design/tasks/spec）必须先 `git add` + commit，再实施代码（Lessons Learned #6, Checklist E）
4. **跨模块状态翻译审计**：`exe_once()` 改造涉及 `blocked_cycles` 多处写入，必须 `grep -rn "blocked_cycles" src/ include/` 交叉引用（Lessons Learned #1, `.opencode/skills/state-modification-audit`）
5. **TDD 强制**：测试用例（PTX-7a/7b）必须先于 PTX-6 实现存在并失败（Red 阶段）；实施完成后 Green 阶段全绿
6. **ADR Status Update**：本 ADR 状态从 Proposed → Active 的转换必须在所有 PTX-X commit 完成 + Oracle 审查通过 + 测试基线 0 回归后执行

---

## 合规检查

后续相关开发应检查：

- [ ] 所有 `src/ptxsim/core/sm_context.cpp` 新增 setter/getter 路径无 `as any` / `[[deprecated]]`（AGENTS.md 禁止项）
- [ ] 3 个接口头文件 `grep -r '#include' include/ptxsim/{scoreboard,pipeline,tensor_core}_interface.h` 仅含 `<cstdint>` / `<string>`
- [ ] `nullptr` 注入时 `exe_once()` 输出与改造前**字节级相同**（通过 baseline worktree 对照测试）
- [ ] 现有 `[unit;memory]` `[unit;barrier]` `[integration;simt]` 测试基线 0 回归
- [ ] `clang-format -i` 对所有修改文件运行（`AGENTS.md` 提交前要求）
- [ ] `./scripts/sanity.sh` 全绿（含 7 个新 Mock 测试 + 集成测试）
- [ ] OpenSpec change `cpptlm-phase8b-injection-points` 归档后，本 ADR 状态更新为 Active
- [ ] `docs/dev-process/lessons-learned.md` 追加新经验条目（如发现新 bug 模式）

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-07-16 | Proposed → Accepted：3 接口完全指定（design.md）+ 59-task 姊妹 change 0 false positives + 11 实施 commits 已在 main + D-PTX-3 行号物理验证正确 → 满足 Checklist G | PTX-EMU Architecture Team |
| 2026-07-14 | 初始版本（Proposed） | PTX-EMU Architecture Team |

---

## 参考

### CppTLM 侧（触发本文档）

- [CppTLM 任务书 `2026-07-03-ptxemu-modification-task.md`](../../CppTLM/docs/superpowers/specs/2026-07-03-ptxemu-modification-task.md)（803 行）—— PTX-EMU 改造任务书 #0~#6
- [CppTLM 协同计划 `2026-07-03-ptxemu-phase8b-d1full-plan.md`](../../CppTLM/docs/superpowers/specs/2026-07-03-ptxemu-phase8b-d1full-plan.md)（440 行）—— 双方完整协同计划
- [CppTLM ADR `ADR-NV-02-phase8b-d1-strategy.md`](../../CppTLM/docs/adr/ADR-NV-02-phase8b-d1-strategy.md)（304 行）—— D1-Lite → D1-Full Status Update
- [CppTLM 协作同步 `2026-07-01-f12b-ld-ptxemu-collaboration-sync.md`](../../CppTLM/docs/superpowers/specs/2026-07-01-f12b-ld-ptxemu-collaboration-sync.md)（323 行）—— F12b-LD + §13 D1-Full 双路径协作
- [CppTLM OpenSpec change `2026-06-24-gpu-soc-phase8b-core/`](../../CppTLM/openspec/changes/2026-06-24-gpu-soc-phase8b-core/)—— D1-Full 设计 + Task 9-16

### PTX-EMU 内部

- [ADR-0009 X-Macro 指令分发模式](0009-xmacro-instruction-dispatch.md) —— StatementType 枚举来源
- [ADR-0019 PC management extraction](0019-pc-management-extraction.md) —— 同步活跃 change：`god-class-refactor-thread-context-phase3`
- [`docs/dev-process/lessons-learned.md`](../dev-process/lessons-learned.md) —— 16 章节经验沉淀（跨模块翻译、递归锁、分 Phase commit、基线 worktree 等）
- [`.opencode/skills/ptx-lessons-learned/SKILL.md`](../../.opencode/skills/ptx-lessons-learned/SKILL.md) —— 经验沉淀快速决策树 + 4 个 checklist + 失败模式速查表
- [`.opencode/skills/ptx-instruction-pipeline/SKILL.md`](../../.opencode/skills/ptx-instruction-pipeline/SKILL.md) —— 指令执行流水线（`exe_once()` 上下文）
- [`.opencode/skills/ptx-barrier-mechanism/SKILL.md`](../../.opencode/skills/ptx-barrier-mechanism/SKILL.md) —— 屏障机制（blocked_cycles 扩展的影响范围）
- [`.opencode/skills/state-modification-audit/SKILL.md`](../../.opencode/skills/state-modification-audit/SKILL.md) —— 状态修改交叉引用审计（验证 exe_once 改造不破坏现有 invariant）
- [`.opencode/skills/regression-bisect/SKILL.md`](../../.opencode/skills/regression-bisect/SKILL.md) —— 回归定位（如出现测试回归）
- [`.opencode/skills/using-superpowers/SKILL.md`](../../.opencode/skills/using-superpowers/SKILL.md) —— skills 使用规范

### 现有 PTX-EMU OpenSpec changes（序列化考虑）

- `cleanup-deprecated-barrier-apis` —— 屏障 API 清理（本 change 阻塞于此归档）
- `god-class-refactor-thread-context-phase3` —— ThreadContext 重构（本 change 需关注 `blocked_cycles` 字段迁移）
- `migrate-bar-warp-sync-to-barrier-module` —— 屏障迁移（barrier 后 PC 处理与 blocked_cycles 交互）

### 外部参考资料

- [PTX ISA 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
- [GPGPU-Sim](https://github.com/accel-sim/gpgpu-sim_distribution)
