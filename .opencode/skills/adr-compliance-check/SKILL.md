---
name: "adr-compliance-check"
description: "ADR 合规检查 — 开发完成后对照 ADR 检查清单验证实现是否符合架构决策"
when_to_use: |
  PTX-EMU 项目中完成以下任务后：
  - 实现了新的 ADR 中描述的功能
  - 修改了 SIMT Stack、Barrier、CFG、Thread PC 等核心组件
  - 添加了新的指令处理器或修改了指令执行流程
  - 代码审查前验证实现是否符合 ADR

  触发场景：
  - "检查是否符合 ADR"
  - "合规检查", "compliance check"
  - "ADR 清单", "ADR checklist"
  - "实现是否符合架构决策"
skills_required: []
---

# ADR 合规检查清单

## 概述

ADR（Architecture Decision Records）是 PTX-EMU 项目的架构决策记录。开发完成后，应对照相关 ADR 的合规检查清单验证实现是否符合架构决策。

---

## 快速检查

### ADR-0006: SIMT Stack 管理

```bash
# 验证 SIMT Stack 实现的关键文件
ls -la src/ptxsim/core/simt_stack.cpp
ls -la include/ptxsim/simt_stack.h
grep -n "check_reconvergence" src/ptxsim/core/warp_context.cpp
grep -n "pc != current_inst_pc" src/ptxsim/core/warp_context.cpp
```

| 检查项 | 验证方法 | 文件位置 |
|--------|---------|---------|
| 分支指令执行时正确 push SIMT 栈 | `grep -n "simt_stack.push" src/ptxsim/core/warp_context.cpp` | `warp_context.cpp` |
| reconvergence 时正确 pop SIMT 栈 | `grep -n "check_reconvergence\|pop" src/ptxsim/core/simt_stack.cpp` | `simt_stack.cpp` |
| check_reconvergence 跳过退出线程 | 查看 `is_exited` 检查逻辑 | `simt_stack.cpp:is_converged()` |
| 栈深度不超过 MAX_DEPTH | `grep -n "MAX_DEPTH\|overflow" include/ptxsim/simt_stack.h` | `simt_stack.h` |
| barrier 后使用 while 循环处理所有收敛条目 | `grep -n "while.*check_reconvergence" src/ptxsim/core/sm_context.cpp` | `sm_context.cpp` |
| check_reconvergence 返回 bool | `grep -n "bool.*check_reconvergence" include/ptxsim/warp_context.h` | `warp_context.h` |
| handle_branch 中使用 PC 过滤 | `grep -n "pc != current_inst_pc" src/ptxsim/core/warp_context.cpp` | `warp_context.cpp:20` |

### ADR-0007: CFG Post-Dominator

| 检查项 | 验证方法 | 文件位置 |
|--------|---------|---------|
| 分支指令的 reconvergence_pc 来自 CFG | `grep -n "reconvergence_pc\|postDominator" src/ptx_parser/cfg_builder.cpp` | `cfg_builder.cpp` |
| 不硬编码 reconvergence 规则 | `grep -n "i + 1\|fallback" src/ptxsim/instructions/barrier.cpp` | `barrier.cpp` |
| CFG 分析在 kernel 加载时执行一次 | `grep -n "computePostDominators\|CFGBuilder" src/ptx_parser/ptx_visitor.cpp` | `ptx_visitor.cpp` |
| Fallback 到 i+1 时在日志中说明原因 | `grep -n "FALLBACK\|PTX_DEBUG_EMU" src/ptxsim/instructions/barrier.cpp` | `barrier.cpp` |

### ADR-0008: Barrier 语义

| 检查项 | 验证方法 | 文件位置 |
|--------|---------|---------|
| barrier 阻塞线程设置 pc_overridden_ | `grep -n "set_pc_overridden.*true" src/ptxsim/instructions/barrier.cpp` | `barrier.cpp:198` |
| barrier 完成后不重置 exec_mask/active_mask | 检查 `barrier.cpp` 是否有 `exec_mask =` 或 `active_mask =` 赋值 | `barrier.cpp` |
| barrier 后调用 check_reconvergence | `grep -n "check_reconvergence" src/ptxsim/core/sm_context.cpp` | `sm_context.cpp` |
| Wbar 使用 is_initialized + memory_fence_verification_enabled | `grep -n "is_initialized\|memory_fence_verification" include/ptxsim/wbar.h` | `wbar.h` |

### ADR-0002: PC 权威源

| 检查项 | 验证方法 | 文件位置 |
|--------|---------|---------|
| PC 从 warp_state.threads[lane].pc 读取 | `grep -n "get_pc\|warp_state.threads.*pc" src/ptxsim/core/warp_context.cpp` | `warp_context.cpp` |
| 不使用 ThreadContext::pc 作为权威源 | 检查是否有 `context->pc` 赋值 | `thread_context.cpp` |

---

## 详细检查清单

### ADR-0006: SIMT Stack 显式控制流管理

#### 合规检查项

- [ ] **分支指令执行时正确 push SIMT 栈**
  ```cpp
  // 在 handle_branch() 的 is_divergent 分支中
  simt_stack.push(entry);  // entry 包含 branch_pc, reconvergence_pc, active_mask, return_mask
  ```

- [ ] **reconvergence 时正确 pop SIMT 栈**
  ```cpp
  // check_reconvergence() 应在 SIMTStack::check_reconvergence() 后 pop
  bool WarpContext::check_reconvergence() {
      if (simt_stack.empty()) return false;
      size_t depth_before = simt_stack.depth();
      simt_stack.check_reconvergence(warp_state.threads);
      if (simt_stack.depth() < depth_before) {
          // An entry was popped
          return true;
      }
      return false;
  }
  ```

- [ ] **check_reconvergence 跳过退出线程**
  ```cpp
  // SIMTStack::is_converged() 中
  if (threads[i].is_exited || !threads[i].is_active) {
      continue;  // 跳过退出/非活跃线程
  }
  ```

- [ ] **栈深度不超过 MAX_DEPTH**
  ```cpp
  // simt_stack.push() 中
  if (entries_.size() >= MAX_DEPTH) {
      throw ExecutionStateException("SIMT stack overflow");
  }
  ```

- [ ] **barrier 后使用 while 循环处理所有收敛条目**
  ```cpp
  // sm_context.cpp 中
  while (next_warp->check_reconvergence()) {
      // Keep popping until no more convergent entries
  }
  ```

- [ ] **check_reconvergence 返回 bool 表示是否有条目被 pop**
  ```cpp
  // 返回 true 表示有条目被 pop，可继续循环
  // 返回 false 表示无条目被 pop，循环结束
  ```

- [ ] **handle_branch 中使用 PC 过滤防止 stale PC 影响分支决策**
  ```cpp
  // warp_context.cpp handle_branch()
  for (int i = 0; i < 32; i++) {
      if (!warp_state.threads[i].is_active) continue;
      if (warp_state.threads[i].pc != current_inst_pc) continue;  // PC 过滤
      // ...
  }
  ```

### ADR-0007: CFG Post-Dominator 收敛分析

#### 合规检查项

- [ ] **分支指令的 reconvergence_pc 来自 CFG post-dominator**
  - CFGBuilder::computePostDominators() 在 kernel 加载时计算
  - bra 指令的 reconvergence_pc 从 postDomMap[branch_pc] 获取

- [ ] **不硬编码 reconvergence 规则（bar.warp.sync 除外）**
  - `bar.warp.sync` 总是使用 `i+1`（warp barrier 后立即收敛）
  - 其他 barrier 使用 CFG 分析结果

- [ ] **CFG 分析在 kernel 加载时执行一次**
  - PtxVisitor 在解析完成后调用 CFGBuilder
  - 结果存储在 kernel 的 CFG 结构中

- [ ] **单元测试覆盖嵌套分支、循环内分支等复杂场景**
  - 检查 tests/ 目录下相关测试

- [ ] **Fallback 到 i+1 时在日志中说明原因**
  ```cpp
  PTX_DEBUG_EMU("CFG[PC=%d]: S_BAR FALLBACK - new_reconvergence_pc=%d (no post-dominator, likely in loop)", i, i + 1);
  ```

- [ ] **S_BAR 和 S_BAR_WARP_SYNC 都正确处理 reconvergence_pc**

### ADR-0008: Barrier 语义增强

#### 合规检查项

- [ ] **barrier 指令正确初始化 participation_mask**
  - 首次到达的线程构建动态掩码
  - 只包含活跃且 PC 匹配的线程

- [ ] **所有参与线程都调用 arrive()**
  - arrive() 更新 arrived_mask

- [ ] **barrier 完成后验证 is_complete()**
  ```cpp
  if (wbar.is_complete()) {
      // 释放所有线程
  }
  ```

- [ ] **barrier 后线程恢复到正确的 reconvergence_pc**
  - 使用 `force_set_pc(reconvergence_pc)` 设置当前线程
  - 使用 `set_thread_pc(i, reconvergence_pc)` 设置其他线程

- [ ] **Debug 模式下 memory fence 验证启用**
  ```cpp
  #ifdef PTX_DEBUG
  wbar.verify_memory_fence();
  #endif
  ```

- [ ] **barrier 阻塞线程设置 pc_overridden_ 保护 PC**
  ```cpp
  // barrier.cpp
  set_pc_overridden(true);  // 阻止 next_pc 被覆盖
  warp_state.threads[lane_id].is_blocked = true;
  ```

- [ ] **barrier 完成后不重置 exec_mask/active_mask**
  - 保留 divergence 状态直到显式收敛

- [ ] **barrier 后调用 check_reconvergence 检查 SIMT 栈收敛**
  ```cpp
  // sm_context.cpp
  while (next_warp->check_reconvergence()) {
      // 处理所有收敛条目
  }
  ```

### ADR-0002: PC 权威源统一

#### 合规检查项

- [ ] **PC 从 warp_state.threads[lane].pc 读取（单一权威源）**
  - `context->get_pc()` 内部调用 `warp_state.threads[lane_id].pc`

- [ ] **set_thread_pc() 同时设置 pc 和 next_pc**
  ```cpp
  warp_state.threads[lane_id].pc = pc;
  warp_state.threads[lane_id].next_pc = pc;
  ```

- [ ] **不使用 ThreadContext::pc 作为权威源**

---

## 检查流程

### 1. 识别相关 ADR

根据修改的组件确定相关 ADR：

| 组件 | 相关 ADR |
|------|---------|
| SIMT Stack / handle_branch | ADR-0006 |
| CFG / Post-Dominator | ADR-0007 |
| Barrier / Wbar | ADR-0008 |
| PC 管理 | ADR-0002, ADR-0003 |

### 2. 执行合规检查

```bash
# 1. 读取相关 ADR 的合规检查清单
cat docs/adr/ADR-0006-simt-stack-management.md | grep -A 20 "合规检查"

# 2. 对照检查代码实现
# 3. 标记检查项为完成或问题
```

### 3. 报告结果

```
## ADR 合规检查报告

### ADR-0006: SIMT Stack 管理
- [x] 分支指令执行时正确 push SIMT 栈
- [x] reconvergence 时正确 pop SIMT 栈
- [ ] 问题：MAX_DEPTH 检查未实现 ← 需要修复

### ADR-0008: Barrier 语义
- [x] barrier 阻塞线程设置 pc_overridden_
- [ ] 问题：未调用 check_reconvergence ← 需要修复
```

---

## 相关文档

- [ADR-0006: SIMT Stack 管理](../../adr/0006-simt-stack-management.md)
- [ADR-0007: CFG Post-Dominator](../../adr/0007-cfg-post-dominator.md)
- [ADR-0008: Barrier 语义](../../adr/0008-barrier-semantics.md)
- [ADR-0002: PC 权威源统一](../../adr/0002-pc-unification.md)
- [SIMT 架构文档](../../architecture/SIMT-ARCHITECTURE-V2.md)

---

## 🔍 与 Lessons-Learned 的 Cross-Check (强制)

> **加载 skill**: [`ptx-lessons-learned`](../ptx-lessons-learned/)
> **来源**: 2026-06-18 commit `f033312` 实战经验沉淀
> **目的**: 把 ADR 合规检查从"是否符合设计决策"扩展到"是否违反已知的失败模式"

### 必查项（每个 ADR 合规检查时）

在完成标准 ADR 合规检查清单后，**必须**额外 cross-check 以下 lessons-learned 失败模式（确保实现没有违反）：

#### A. 跨模块状态翻译（Lessons 1）

**触发条件**: 涉及 `ThreadContext` / `WarpState` / `set_state` / `sync_to_warp_state` 的改动

**检查命令**：
```bash
# 列出修改的代码中所有 set_state 调用
grep -n "set_state" <modified-files>
# 列出所有消费该 state 的位置
grep -rn "state == BAR_SYNC\|state == RUN\|state == EXIT\|is_at_barrier" src/ptxsim/
# 比对：每个 set_state 必须有对应的消费者，没有遗漏
```

**判定标准**：
- ❌ 有 `set_state(X)` 但找不到消费者 → 可能翻译不完整
- ❌ 有消费者但上游代码没设 → 中间路径漏翻译（本次 bug 的模式）
- ✅ 必须有完整的"set_state → consumer"对应关系

**关联 invariant**: `src/ptxsim/core/AGENTS.md` DUAL STATE MECHANISM 章节

#### B. 递归锁死锁（Lessons 2）

**触发条件**: 涉及互斥量、`lock_guard`、新的 `mutex_` 字段

**检查命令**：
```bash
# 列出所有锁点
grep -n "lock_guard\|unique_lock\|lock()\|unlock()" <modified-files>
# 检查是否有"持锁方法调用其他持锁方法"的模式
# - 同一 mutex 上的多个 lock_guard 在不同方法中
# - 一个方法内 lock 后调用同锁的另一方法
```

**判定标准**：
- ❌ 持锁方法调用同锁的另一 public 方法 → 死锁风险
- ❌ 新加的 mutex 字段没有 unique_ptr 包装 → 拷贝风险
- ✅ 应使用 "internal unsafe" 模式或 `std::recursive_mutex`（明确标注）

**关联修复模式**: lessons-learned.md §3

#### C. set_active_mask 全局语义（Lessons 1 关联）

**触发条件**: 涉及 barrier 释放、`release_warp_barrier`、`release_cta_barrier`、active_mask 修改

**检查命令**：
```bash
# 列出所有 set_active_mask 调用
grep -rn "set_active_mask" src/ptxsim/
# 验证是否使用 OR 语义（而非 overwrite）
grep -B2 "set_active_mask" src/ptxsim/barrier/ | grep "get_active_mask() |"
```

**判定标准**：
- ❌ 在 barrier 释放路径用 `set_active_mask(arrived_mask)` 而非 OR → BUG-POSTBARRIER-TWOHALVES 回归
- ✅ 必须用 `set_active_mask(get_active_mask() | arrived_mask)`（caller 层 OR）
- ✅ ret handler 仍可用 `set_active_mask(0u)`（overwrite 是合法的，这里只查 barrier 路径）

**关联 invariant**: `src/ptxsim/core/AGENTS.md` BUG-POSTBARRIER-TWOHALVES 章节

#### D. Barrier 释放后状态恢复（Lessons 1 关联）

**触发条件**: 涉及 `release_warp_barrier` / `release_cta_barrier`

**检查命令**：
```bash
# 验证 release 后设置 state = RUN
grep -A5 "release_warp_barrier\|release_cta_barrier" src/ptxsim/barrier/barrier_module.cpp | grep "set_state"
# 验证 is_blocked 被显式清除
grep -A10 "release_warp_barrier\|release_cta_barrier" src/ptxsim/barrier/barrier_module.cpp | grep "is_blocked"
```

**判定标准**：
- ❌ release 路径只 set_state(RUN) 但没清除 is_blocked → `sync_to_warp_state()` 的 guard 会阻止 is_blocked 被清除
- ✅ release 路径必须：set_state(RUN) + 显式清 is_blocked
- ✅ release 路径必须调用 advance_thread_pc 推进 PC

**关联 invariant**: `src/ptxsim/core/AGENTS.md` DUAL STATE MECHANISM 章节

#### E. Wbar 废弃状态（Lessons 6 关联）

**触发条件**: 涉及 `Wbar` 引用、bar.warp.sync 路径、`warp_state.wbars[]` 字段

**检查命令**：
```bash
# 列出所有 Wbar 引用
grep -rn "Wbar\b\|wbar\.\|warp_state.wbars\[" src/ include/ | grep -v "deprecated"
# 检查是否被标记为 deprecated
grep "deprecated" include/ptxsim/wbar.h
```

**判定标准**：
- ❌ 新代码使用 `Wbar` 而非 `BarrierModule` → 违反废弃决策
- ✅ 新代码应使用 `BarrierModule` / `WarpBarrier` / `CTABarrier`
- ✅ 旧代码引用应标注 `// TODO: migrate to BarrierModule`

**关联决策**: `docs/adr/ADR-0008-barrier-semantics.md` §"2026-06-18 Postmortem"（Phase 5 推迟决策）

### 集成到 ADR 合规报告

ADR 合规报告应包含 **Lessons-Learned Cross-Check** 章节：

```markdown
## Lessons-Learned Cross-Check

- [ ] A. 跨模块状态翻译完整性
- [ ] B. 递归锁死锁风险
- [ ] C. set_active_mask 全局语义（barrier 路径）
- [ ] D. Barrier 释放后状态恢复
- [ ] E. Wbar 废弃状态（如涉及）

<每个项的检查结果 + 证据>
```

如果任何项 ❌，**必须**在报告中标出，并在 commit 中修复（参考 lessons-learned.md 对应章节的修复模式）。

### 与其他 skill 的关系

- **`ptx-lessons-learned`**: 提供所有 cross-check 项的源
- **`state-modification-audit`**: 增强 cross-check A 和 D 的检查（当怀疑状态被意外修改时）
- **`regression-bisect`**: 增强 cross-check A 的诊断（当 `set_state` 漏掉导致测试回归时）
