# OpenSpec Change 审查报告

## `cleanup-deprecated-barrier-apis`

> **审查日期**: 2026-06-20
> **审查员**: Sisyphus agent
> **审查范围**: proposal.md + design.md + specs + tasks.md vs 当前代码事实
> **审查方法**: 直接代码审查 + Oracle 深度架构分析 + 假设验证门禁
> **建议总评**: ⚠️ **必须修改后再 proceed**,存在 1 个阻断性设计冲突 + 多个高风险遗漏

---

## 0. 执行摘要

`cleanup-deprecated-barrier-apis` 是 PTX-EMU 项目 Phase 6 清理工作,目标删除三套过时 barrier 实现(`Wbar`、`BsyncManager`、`SMContext::synchronize_barrier`)。审查发现:

1. **🚨 阻断性设计冲突**:design.md Decision 1 vs Decision 3 自相矛盾,"混合方案"在当前架构约束下不可行(`WarpBarrier` 是 `class` 不是 `struct`,且由 `BarrierModule` 拥有,不能嵌入 `WarpState`)
2. **🚨 关键文件遗漏**:Impact 表未列 `src/ptxsim/core/warp_context.cpp:283-296` BAR_SYNC fallback 路径(实际调用 `synchronize_barrier`),按当前 spec 直接执行会编译失败
3. **🚨 已知 BUG 测试保护缺失**:删除 `Wbar` 会让 `tests/integration/divergence/test_post_barrier_divergence.cpp`(记录 BUG-HISTORICAL 的回归测试)被强制重写或删除,失去回归保护
4. **⚠️ Oracle H3 部分证伪**:BAR_SYNC 状态不是 dead code,有 2 个生产 setter(`barrier.cpp:386` + `sm_context.cpp:703`),还有 1 个翻译器(`thread_context.cpp:749`),删除 `synchronize_barrier` 不会让 BAR_SYNC 失效
5. **⚠️ Lessons-learned Checklist 缺失 6 项**:基线 worktree 复用、AGENTS.md 同步、ADR 追加、commit message 独立 fix 编号等

**修订方案**: Option A+(8 文件修改,3 commit 拆分,6 section tasks,Wbar struct 保留到 Phase 5 独立 change)。实施工作量 2-4 小时。

---

## 1. 当前 change 内容速览

### 1.1 proposal.md 摘要

- **Why**: commit `12390b7` 后 `BarHandler` 已切到 `BarrierModule` API,但 `Wbar` + `BsyncManager` + `SMContext::synchronize_barrier` 仍在生产路径
- **What**: 删除 3 套旧 API;**不**触碰 `BarWarpSyncHandler` 主逻辑(留给独立 change `migrate-bar-warp-sync-to-barrier-module`)
- **前置验证**: `grep -rn "bsync_manager_\|synchronize_barrier"` 必须仅在 `barrier.cpp`(`BarWarpSyncHandler` 路径)匹配

### 1.2 design.md 摘要

- Decision 1: 删除 `BsyncManager`,但 `BarWarpSyncHandler` 仍用 `Wbar`
- Decision 2: 删除 SM 级全局 barrier 状态
- **Decision 3 (冲突)**: 删除 `warp_state.wbars[]` 字段 → **与 Decision 1 矛盾**
- "混合方案"(design.md:86-93): 把 `warp_state.wbars[0]` 字段类型改为 `WarpBarrier`
- Migration Plan: 6 Phases(审计 → WarpBarrier 迁移 → 同步 BarWarpSyncHandler → 删除 BsyncManager → 测试迁移 → 删 Wbar 头)

### 1.3 spec.md 摘要

- ADDED Requirement: Cleanup MUST be behavior-preserving
- REMOVED Requirement 1: `Wbar` struct MUST be removed
- REMOVED Requirement 2: `BsyncManager` class MUST be removed
- REMOVED Requirement 3: `SMContext::synchronize_barrier` MUST be removed

### 1.4 tasks.md 摘要

- 9 sections × 70+ atomic tasks
- §1 审计 → §2 WarpBarrier 字段迁移 → §3 同步 BarWarpSyncHandler → §4 验证 → §5 删除 BsyncManager + sm_context 字段 → §6 验证 → §7 测试迁移 → §8 删除 Wbar 头 + 最终验证 → §9 验证与发布

---

## 2. 🚨 阻断性问题:Decision 1 vs Decision 3 设计冲突

### 2.1 矛盾本质

proposal 自己承认存在决策冲突(design.md L86-93),并提出"**混合方案**":

> **"删除 `Wbar` 类的同时,把 `warp_state.wbars[]` 字段类型从 `Wbar` 改为 `WarpBarrier`"**

但这个方案在**当前架构约束下不可行**:

| 约束 | 冲突点 |
|------|--------|
| ADR-0008 §"2026-06-18 追加"决策 | "`BarrierModule` 由 `CTAContext` 独占持有" + "状态机统一" |
| `WarpBarrier` 是 `class`,字段 `private` | 不能直接作为 struct 嵌入 `WarpState`(语义破坏) |
| `WarpBarrier::init()` 4 参数 vs `Wbar::init()` 2 参数 | API 不兼容,无法做"字段替换" |
| `WarpBarrier` 依赖 `BarrierModule::MAX_WARP_BARRIERS=4` 数组管理 | 单实例字段化绕过 `BarrierModule` 抽象 |

### 2.2 当前事实(grep 验证)

```cpp
// include/ptxsim/barrier/warp_barrier.h:11  ← class 不是 struct
class WarpBarrier {
    State state_;  // private
    uint32_t participation_mask_;  // private
    ...
};

// include/ptxsim/barrier/barrier_module.h:74
std::array<WarpBarrier, MAX_WARP_BARRIERS> warp_barriers_;  // ← BarrierModule 才是 owner
```

### 2.3 proposal 边界声明 vs 实施步骤的真实矛盾

| proposal 声明 | tasks.md 实际步骤 | 一致性 |
|------------|-----------------|--------|
| "**不**触碰 `BarWarpSyncHandler` 主逻辑" | Phase 3 标题:"同步 BarWarpSyncHandler" | ❌ 矛盾 |
| "BarWarpSyncHandler 仍用 `warp_state.wbars[0]`" | Phase 2.1 删除该字段、改为 `barrier` | ❌ 矛盾 |
| "Phase 5 单独建 change 处理" | tasks 7.2 实施测试迁移(已涉及 handler 语义) | ⚠ 边界模糊 |

---

## 3. 🚨 高风险遗漏:`warp_context.cpp:283-296` BAR_SYNC 回退路径

### 3.1 实际代码(proposal 完全遗漏)

```cpp
// src/ptxsim/core/warp_context.cpp:283-296  ← proposal 未提及
if (thread->get_state() == BAR_SYNC) {
    if (sm_context_ != nullptr) {
        bool is_warp_barrier = (warp_state.current_wbar_id >= 0);
        bool warp_barrier_complete = is_warp_barrier &&
            warp_state.wbars[warp_state.current_wbar_id].is_complete();

        if (!warp_barrier_complete) {
            PTX_WARN_EMU("Fallback CTA sync: lane %d, wbar %d incomplete", ...);
            sm_context_->synchronize_barrier(thread->bar_id, thread);
        }
    }
    thread->sync_to_warp_state();
    continue;
}
```

### 3.2 删除 `Wbar` + `synchronize_barrier` 后后果

- **编译失败**(直接 break build)
- 这是 BAR_SYNC 状态机的关键 fallback,**绕过它会导致分裂场景死锁**
- 必须**先决定**怎么处理(转 `BarrierModule::arrive_at_cta_barrier` 或删除整段回退)

### 3.3 `warp_context.h:212-217` 公共 API 同样被遗漏

```cpp
// include/ptxsim/warp_context.h:212-217
ptxsim::Wbar& get_wbar(int wbar_id) {  // ← 公共 API
    if (wbar_id >= 0 && wbar_id < 4) {
        return warp_state.wbars[wbar_id];
    }
    return warp_state.wbars[0];
}
```

被 19 个测试文件使用(grep 验证):
- `tests/integration/barrier/test_barrier_module_integrated.cpp`
- `tests/integration/barrier/test_barrier_verification_integrated.cpp`
- `tests/integration/barrier/test_warp_barrier_integrated.cpp`

---

## 4. 🚨 已知 BUG 测试保护缺失

### 4.1 重要事实

`tests/integration/divergence/test_post_barrier_divergence.cpp` **记录了一个已知 BUG**(`src/ptxsim/core/AGENTS.md:85`):

```cpp
// KNOWN ISSUE: synchronize_barrier() does not call update_active_mask() (AGENTS.md#48)
```

### 4.2 删除 Wbar 后的影响

- 该测试**直接构造 Wbar 实例**(`warp.get_warp_state().wbars[0]`)并调用 Wbar API
- 删除 `Wbar` 后,该测试**必须删除或重写**
- **重写会失去原 BUG 的回归保护** - lessons-learned §6 强调"集成测试是发现跨模块状态翻译的唯一手段"
- **删除会让一个已知 bug 在未来'悄悄自愈'(其实是测试不存在)**

---

## 5. ⚠️ Oracle 深度分析

### 5.1 Oracle Hypotheses

| # | 假设 | 置信度 | 验证结果 |
|---|------|--------|---------|
| H1 | Impact 表遗漏 `warp_context.cpp` | High | ✅ **确认** |
| H2 | BsyncManager 删除安全(无下游消费者) | High | ✅ **确认** |
| H3 | BAR_SYNC 仅由 `synchronize_barrier` 设置,fallback 是 dead code | Medium | ❌ **证伪**:2 个生产 setter(`barrier.cpp:386` + `sm_context.cpp:703`)+ 1 个翻译器(`thread_context.cpp:749`) |
| H4 | `barrier_mutex_` 唯一消费者是周期检查 + synchronize_barrier | High | ✅ **确认**(仅 2 个 lock 点) |
| H5 | 保留 Wbar 与 ADR-0008 "Phase 5 推迟" 决策一致 | Medium | ✅ **确认**(ADR-0008 §"2026-06-18 Postmortem" 明确 Phase 5 deferred) |

### 5.2 验证命令(实际执行结果)

```bash
# 1. BAR_SYNC setter 验证(证伪 Oracle H3)
$ grep -rn "set_state(BAR_SYNC)\|state_ = BAR_SYNC\|state = BAR_SYNC\|set_state.*BAR_SYNC" src/ include/
src/ptxsim/core/sm_context.cpp:703:    thread->set_state(BAR_SYNC);        # ← synchronize_barrier(将删除)
src/ptxsim/core/thread_context.cpp:749:    state = BAR_SYNC;                  # ← sync_from_warp_state 翻译器(被动)
src/ptxsim/instructions/barrier.cpp:386:    context->set_state(BAR_SYNC);      # ← BarHandler(commit f033312 修复后保留!)

# 2. barrier_mutex_ 消费者验证(确认 Oracle H4)
$ grep -rn "barrier_mutex_" src/ include/
src/ptxsim/core/sm_context.cpp:204:    std::lock_guard<std::mutex> lock(barrier_mutex_);  # 周期检查
src/ptxsim/core/sm_context.cpp:608:    std::lock_guard<std::mutex> lock(barrier_mutex_);  # synchronize_barrier
include/ptxsim/sm_context.h:192:    mutable std::mutex barrier_mutex_;                  # 字段声明

# 3. BsyncManager getter 消费者验证(确认 Oracle H2)
$ grep -rn "is_waiting\|get_waiting_mask\|get_state\|bssy" src/ include/ tests/ \
    | grep -v "bsync_state.cpp\|bsync_state.h" | grep -v "/test_"
# 全部输出是 EXE_STATE / GPUState / CTAState / WarpBarrier::State 等其他类,
# 没有任何 BsnycManager 的 getter 调用 — 删除安全

# 4. bsync_state.h 文件状态验证
$ git ls-files src/ptxsim/core/bsync_state.h include/ptxsim/bsync_state.h
include/ptxsim/bsync_state.h    # ← 真实路径是 include/,不是 src/

# 5. 当前构建状态
$ cmake --build build --target ptxsim
[ 14%] Built target ptx_ir
[100%] Built target ptxsim   # ← 编译通过
```

### 5.3 关键事实更正

- **`include/ptxsim/bsync_state.h` 存在**(proposal 路径错误,在 `include/` 不在 `src/`)
- **`thread_context.cpp:749` 不是 setter**:`sync_from_warp_state` 的 `Blocked → BAR_SYNC` 翻译器(lessons-learned §1 关键经验)
- **BAR_SYNC 仍有 2 个活跃 setter**:删除 `synchronize_barrier` 不会让 BAR_SYNC 失效
- **`warp_context.cpp:283-296` fallback 不是 dead code** - 必须保留并替换
- **当前 main 分支构建成功**(`ptxsim` target built 100%)
- **`.worktrees/fix-pre-p0-baseline` 已存在**(可复用为 baseline)

---

## 6. 📋 实施清单遗漏(Impact 表 vs 实际差异)

### 6.1 proposal Impact 表列出的文件(7 项)

```
include/ptxsim/wbar.h                       ← 删除
src/ptxsim/core/bsync_state.{h,cpp}         ← 部分: .h 已不存在
include/ptxsim/warp_state.h                  ← 改
src/ptxsim/core/sm_context.{h,cpp}           ← 改
src/ptxsim/instructions/barrier.cpp          ← 改
src/CMakeLists.txt                           ← 改
tests/unit/barrier/                          ← 可能
```

### 6.2 实际需要触碰的文件(**遗漏** ≥14 项)

| 文件 | 实际改动 | proposal 状态 |
|------|---------|-------------|
| `include/ptxsim/warp_context.h:212-217` | 删除 `get_wbar()` | ❌ 遗漏 |
| `src/ptxsim/core/warp_context.cpp:283-296` | 重写 BAR_SYNC 回退 | ❌ 遗漏 |
| `src/ptxsim/core/thread_context.cpp:774` | 更新注释(`synchronize_barrier` 引用) | ❌ 遗漏 |
| `src/ptxsim/AGENTS.md:42, 48` | 删除 "DO NOT add new uses of Wbar"(Wbar 删除后无对象) | ❌ 遗漏 |
| `src/ptxsim/core/AGENTS.md:22, 85` | 删除 `synchronize_barrier()` 行 + KNOWN ISSUE | ❌ 遗漏 |
| `tests/AGENTS.md:15` | 更新 "barrier/Wbar 数据结构" 描述 | ❌ 遗漏 |
| `src/ptxsim/instructions/barrier.cpp:385` | 更新过期注释(synchronize_barrier 在 :605 而非 :703) | ❌ 遗漏 |
| `src/ptxsim/instructions/AGENTS.md` | 更新 barrier 路由规则 | ❌ 遗漏 |
| `tests/integration/divergence/test_post_barrier_divergence.cpp` | 迁移到 WarpBarrier **保留 BUG 注释** | ❌ 遗漏 |
| `tests/integration/barrier/*.cpp` (4 个) | 迁移 + 重新接口 | ⚠ 提及但没列具体 |
| `tests/unit/barrier/*.cpp` (3 个) | 迁移 | ⚠ 提及但没列具体 |
| `tests/integration/{exec,pc,simt,sync}/*.cpp` (4 个) | 删除 `#include "ptxsim/wbar.h"`(可能仅 include) | ❌ 遗漏 |
| `tests/unit/{exec,simt,common}/*.cpp` (4 个) | 删除 `#include "ptxsim/wbar.h"`(可能仅 include) | ❌ 遗漏 |
| `docs/adr/ADR-0008-barrier-semantics.md` | 追加 "Phase 6 清理完成" postmortem | ❌ 遗漏 |

**19 个测试文件 include `ptxsim/wbar.h`**(grep 已验证)

---

## 7. ⚠️ ADR-0008 合规性问题

### 7.1 adr-compliance-check skill 的关键检查项

| ADR-0008 检查项 | 现状 | 本 change 后预期 | 合规性 |
|----------------|------|----------------|--------|
| `BarHandler` 走 `BarrierModule` API | ✅ 已完成 | 维持 | ✓ |
| `BarWarpSyncHandler` 走 `BarrierModule` API | ❌ 仍直接操作 `wbar` | 本 change 声明"不修改" | ⚠ 维持现状 |
| `CTAContext` 持有 `BarrierModule` | ✅ 已完成 | 维持 | ✓ |
| `release_warp_barrier` 用 OR 语义 | ✅ 已完成 | 维持 | ✓ |
| `WarpBarrier::init` re-init 保留 arrived_mask | ✅ 已完成 | 维持 | ✓ |
| Wbar 完全删除 | ⚠ 仍存在 | 本 change 目标 | △ 待完成 |
| `set_active_mask` 全局语义未改 | ✅ 已完成 | 维持 | ✓ |

### 7.2 缺失的 ADR 追加

design.md 引用 ADR-0008 但**没要求追加段落**。lessons-learned §11 明确:
> 整体架构变更 → ADR 追加段落

本 change 完成后应追加:"2026-06-XX Phase 6 partial cleanup - BsyncManager + SM 级 barrier 状态删除,Wbar 保留到 Phase 5"

---

## 8. ⚠️ Lessons-Learned Checklist 应用检查

### 8.1 Checklist A: 函数迁移

```
□ 列出 baseline 函数中所有的 set_* / commit_* / force_* 调用
□ 列出所有 mutex_ / lock_guard / unique_lock 使用
□ 对每个 set_*，grep 其值的下游消费者
□ 对每个锁点，确认"持锁方法调用的所有其他方法"也持同一锁
□ 比对行级 diff（不只比对主要逻辑）
```

**部分完成**:
- ✅ task 1.1 列了 grep 命令
- ❌ task 1.2 列出 `bsync_state.cpp` 10 个方法引用点 → 未在 tasks.md 体现
- ❌ task 1.3 `sm_context.cpp:200-260` 周期检查 → 未审计实际代码已包含 `barrier_mutex_` (sm_context.cpp:204)

### 8.2 Checklist B: 重构前

```
□ 建立基线 worktree
□ 列出本 change 的所有 Phase，决定 commit 粒度
□ 决定哪些 Phase 需要基线对比
□ 准备 revert 策略
```

**部分完成**:
- ✅ tasks.md 1.4: `git worktree add ... -b refactor/cleanup-deprecated-barrier-apis`
- ❌ **未要求建立基线 worktree** 用于对比(lessons-learned §5 强烈推荐)
- ✅ design.md Rollback Strategy 存在
- ⚠ Rollback Strategy 描述不准确:"Phase 2-3 失败:git revert 到 Phase 1 完成点" → Phase 1 是审计无代码变更,应该是"git reset 到 Phase 1 起点"

### 8.3 Checklist D: Commit 前

```
□ 跑过 baseline worktree 对比
□ AGENTS.md 是否需要同步
□ ADR 是否需要追加
□ OpenSpec tasks.md 是否需要更新
□ commit message 列出独立的 fix 编号
```

**未完成**:
- tasks.md 9.1 commit message 计划:**只引用 BUG 编号**,没列 AGENTS.md/ADR/tasks.md 同步
- ❌ 缺 5 项中的 3 项

### 8.4 失败模式预测

| 经验 | 风险 | 缓解 |
|------|------|------|
| §1 跨模块状态翻译 | `warp_context.cpp:283` fallback 处理必须保留 - **不能直接删除**,因为 `barrier.cpp:386` 仍设置 BAR_SYNC | 必须替换 fallback 而非删除 |
| §2 递归锁 | `barrier_mutex_` 唯一消费者是 sm_context.cpp:204/608,无嵌套锁风险 | OK |
| §6 测试类型选择 | `warp_context.cpp:283` fallback 修改需要类型二/三测试覆盖 | 需新增集成测试覆盖替换路径 |

---

## 9. ⚠️ 边界场景与编号问题

### 9.1 tasks.md 编号混乱

| tasks.md § | 实际内容 | design.md 对应 Phase | 是否一致 |
|-----------|---------|---------------------|---------|
| § 1 | 审计与准备(1.1-1.5) | Phase 1 | ✓ |
| § 2 | WarpBarrier 字段迁移(2.1-2.5) | Phase 2 | ✓ |
| § 3 | 同步 BarWarpSyncHandler(3.1-3.6) | Phase 3 | ⚠ Phase 3 在 design.md 是"删除 BsyncManager"! |
| § 4 | 验证(4.1-4.5) | (无对应) | ❌ design.md Phase 3 后直接 Phase 4 |
| § 5 | 删除 BsyncManager + sm_context 字段(5.1-5.7) | Phase 4 | ⚠ Phase 4 vs Phase 3 不对应 |
| § 6 | 验证(6.1-6.4) | (无对应) | ❌ |
| § 7 | 测试迁移(7.1-7.3) | Phase 5 | ✓ |
| § 8 | 删除 Wbar 头 + 最终验证(8.1-8.6) | Phase 6 | ✓ |
| § 9 | 验证与发布(9.1-9.3) | (无对应) | ❌ design.md 没有 "发布" 阶段 |

**问题**: `§ 4` 和 `§ 6` 都是"验证",但 design.md 没有显式验证阶段,且 § 9 "验证与发布"在 design.md 没有对应

### 9.2 `barrier.cpp:385` 注释行号引用已过期

```cpp
// (Mirrors legacy SMContext::synchronize_barrier at sm_context.cpp:703.)  ← 错!
```

实际文件位置:`sm_context.cpp:605-706`(synchronize_barrier 方法从 line 605 开始)。这是细节管理问题。

### 9.3 BAR_SYNC 翻译(lessons-learned §1 的幽灵)

```cpp
// warp_context.cpp:283
if (thread->get_state() == BAR_SYNC) {
```

`state == BAR_SYNC` 的写入者是 `BarHandler::executeBarrier`(commit f033312 修复过)。如果本 change 修改 `BarHandler`,**必须行级 Diff 比对**(lessons-learned §1)

---

## 10. 🎯 修订后的修复方案:Option A+(实际可行)

### 10.1 用户意图对齐

- ✅ 删除 BsyncManager + synchronize_barrier(用户选项 A)
- ✅ 删除 Wbar 相关 spec(用户第二个选择)
- ✅ 保留 `Wbar` struct + `warp_state.wbars[]` 字段(隐含选项 A 方向)

### 10.2 文件清单(8 项,vs proposal 7 项)

| # | 文件 | 动作 | 备注 |
|---|------|------|------|
| 1 | `include/ptxsim/bsync_state.h` | **DELETE** | 与 .cpp 一并删除 |
| 2 | `src/ptxsim/core/bsync_state.cpp` | **DELETE** | 唯一生产调用点是 `barrier.cpp:189,240,249` |
| 3 | `include/ptxsim/core/sm_context.h` | **MODIFY** | 删除 `BsyncManager bsync_manager_` (line 195) + `barrier_waiting_threads/counts/mutex_` (lines 189-192) + `synchronize_barrier` 声明 (line 114) + `bsync_state.h` include |
| 4 | `src/ptxsim/core/sm_context.cpp` | **MODIFY** | 删除 lines 200-260 周期检查 + lines 605-706 `synchronize_barrier()` 方法体 |
| 5 | `src/ptxsim/instructions/barrier.cpp` | **MODIFY** | 删除 lines 189, 240, 249 的 `bsync_manager_.bsync/release` 调用;**保留** `warp_state.wbars[0]` 操作(用户方向) |
| 6 | `src/ptxsim/core/warp_context.cpp` | **MODIFY** | **关键**:替换 line 292 `synchronize_barrier` 调用为 `cta_context_->get_barrier_module()->arrive_at_cta_barrier(...)` |
| 7 | `src/ptxsim/core/warp_scheduler.cpp` | **MODIFY** | 删除 line 2 `#include "ptxsim/bsync_state.h"`(验证无 BsyncManager 使用后) |
| 8 | `src/CMakeLists.txt` | **MODIFY** | 移除 `ptxsim/core/bsync_state.cpp` 条目 |

### 10.3 文档同步(必须,proposal 遗漏)

| 文件 | 动作 | 内容 |
|------|------|------|
| `docs/adr/ADR-0008-barrier-semantics.md` | 追加段 | "2026-06-20:Phase 6 partial cleanup - BsyncManager 与 SM 级 barrier 状态删除;Wbar struct 保留到独立 change `migrate-bar-warp-sync-to-barrier-module`(Phase 5);barrier.cpp:386 BAR_SYNC 设置(commit f033312 修复)保持" |
| `src/ptxsim/core/AGENTS.md` | MODIFY | 删除 "Barrier sync \| `sm_context.cpp` \| `synchronize_barrier()`" 行 + KNOWN ISSUES 中 `synchronize_barrier() may not update active_mask` 注释(改为指向 `test_post_barrier_divergence.cpp` 作为 BUG 文档) |
| `src/ptxsim/AGENTS.md` | MODIFY | line 42 注释更新:"`BarHandler` MUST route through `BarrierModule` API — `BarWarpSyncHandler` still uses `Wbar` (Phase 5 deferred)" |
| `tests/AGENTS.md` | MODIFY | line 15 "barrier/Wbar 数据结构" 描述保持(Wbar 仍存在);`bsync` 描述移到 `archive/` |

### 10.4 3-Commit 拆分策略

```
Commit 1: refactor(barrier): remove BsyncManager dead code
  - rm bsync_state.{h,cpp}
  - 移除 sm_context.h BsyncManager 字段 + sm_context.h bsync_state.h include
  - 移除 barrier.cpp:189,240,249 调用点
  - 移除 warp_scheduler.cpp bsync_state.h include
  - 移除 CMakeLists.txt bsync_state.cpp 条目
  - 验证: ctest -L 'barrier;warp' all pass
  - 独立可 revert:仅 BsyncManager 类 + 调用点;Wbar/WarpState/synchronize_barrier 完全不动

Commit 2: refactor(barrier): remove SM-level barrier state and migrate warp_context fallback
  - sm_context.h: 删除 barrier_waiting_threads/counts/mutex_ 字段 + synchronize_barrier 声明
  - sm_context.cpp: 删除 lines 200-260 周期检查 + lines 605-706 synchronize_barrier body
  - warp_context.cpp:283-296 替换 fallback 为 cta_context_->get_barrier_module()->arrive_at_cta_barrier
  - 验证: ctest -L 'barrier' all pass; sanity.sh --quick pass
  - 独立可 revert:Phase 1 已 landed;Phase 2 revert 后,BsyncManager 仍为删除态,但 synchronize_barrier 恢复 → 编译仍通过

Commit 3: docs(barrier): update ADR/AGENTS/OpenSpec for partial cleanup
  - ADR-0008: add 2026-06-20 paragraph
  - src/ptxsim/AGENTS.md + src/ptxsim/core/AGENTS.md + tests/AGENTS.md
  - spec.md + design.md + tasks.md (本审查产出)
```

---

## 11. 📊 综合评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **设计完整性** | ⭐⭐☆☆☆ | Decision 1+3 冲突未解决,边界声明与实施步骤矛盾 |
| **Spec 合规** | ⭐⭐⭐☆☆ | spec.md 本身清晰,但与 proposal "混合方案" 不一致 |
| **Tasks 可执行性** | ⭐⭐☆☆☆ | 编号混乱,多个文件遗漏,基线 worktree 缺失 |
| **Lessons learned 应用** | ⭐⭐☆☆☆ | Checklist A 80%、B 60%、D 40% |
| **风险评估** | ⭐⭐☆☆☆ | Rollback 描述不准,边界场景未识别 |
| **测试覆盖规划** | ⭐⭐☆☆☆ | 没要求新增 WarpBarrier 字段访问测试,已知 BUG 测试保护缺失 |
| **ADR 同步规划** | ⭐☆☆☆☆ | 未要求追加 ADR 段落 |

**总评**: ⚠️ **必须修改后再 proceed**

---

## 12. 🚦 给用户的最终建议

1. **采纳 Option A+ 方案**(8 文件修改,3 commit 拆分,6 section tasks)
2. **修改 5 个 OpenSpec artifacts**(spec.md + design.md + tasks.md + proposal.md + README.md)
3. **复用既有 `fix-pre-p0-baseline` worktree** 作为 baseline(避免 lessons-learned §5 的 15-20 分钟基线 build)
4. **保留 Wbar struct**(`include/ptxsim/wbar.h` + `warp_state.wbars[]` 不动)
5. **保留 BAR_SYNC fallback 但替换调用**(`warp_context.cpp:292` 替换为 `BarrierModule::arrive_at_cta_barrier`)
6. **保留 19 个 Wbar 测试**(因 Wbar 仍存在)
7. **删除 `tests/unit/sync/test_bsync_state.cpp`**(因 BsyncManager 删除)

**最大风险**: `warp_context.cpp:283-296` fallback 修改后的正确性,需要类型二/三测试覆盖(集成测试驱动调度器)。

**实施工作量**: 2-4 小时(2 code commit + 1 docs commit),每 commit 独立可 revert。

---

## 13. 元信息

- **审查员**: Sisyphus agent (主对话)
- **Oracle 子代理**: `qwen3.7-plus`(通过 minimax-cn-coding-plan routing)
- **审查时长**: ~5 分钟(主对话)+ ~3 分钟(Oracle 后台)
- **关联文档**:
  - `docs/dev-process/lessons-learned.md`(项目经验沉淀,强制引用)
  - `docs/adr/ADR-0008-barrier-semantics.md`(Barrier 语义 ADR)
  - `.opencode/skills/ptx-barrier-mechanism/`(屏障机制全解)
  - `.opencode/skills/ptx-lessons-learned/`(项目经验沉淀)
  - `.opencode/skills/adr-compliance-check/`(ADR 合规检查)
  - `.opencode/skills/oracle-prompting/`(Oracle 防幻觉规则)

**审查报告生成时间**: 2026-06-20
