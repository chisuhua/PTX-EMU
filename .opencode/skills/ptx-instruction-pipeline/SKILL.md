---
name: "ptx-instruction-pipeline"
description: "PTX-EMU 指令执行流水线全貌 — 从 Warp 调度到 Thread 执行、Handler 分发、屏障处理、PC 同步的完整调用链与危险区地图"
when_to_use: |
  触发关键词——PTX-EMU 项目中出现：
  - "指令执行流程", "ExecPipe", "handler", "PipelineHandler"
  - "屏障", "barrier", "bar.warp.sync", "bar.sync", "Wbar"
  - "PC 不对", "PC 被覆盖", "next_pc 错误", "跳过了指令"
  - "sync_from_warp_state", "sync_to_warp_state", "set_thread_pc"
  - "execute_warp_instruction", "execute_thread_instruction", "_execute_once"
  - "指令处理器怎么工作的", "为什么指令没执行"
skills_required: ["cpp-debug"]
---

# PTX-EMU 指令执行流水线

## 总览

```
SMContext::check_barriers / schedule_next
  │
  ▼
WarpContext::execute_warp_instruction(target_pc)  ← warp_context.cpp:146
  │
  ├─ for each lane i:
  │   ├─ is_lane_active(i) ?                          ← 检查 active_mask
  │   ├─ warp_state.threads[i].pc == target_pc ?      ← PC 必须匹配
  │   ├─ thread->sync_from_warp_state()               ← 从 warp_state 加载状态
  │   ├─ BAR_SYNC? → fallback CTA sync                ← 屏障回退路径
  │   ├─ thread->execute_thread_instruction()          ← 核心执行
  │   └─ thread->sync_to_warp_state()                 ← 写回 warp_state
  │
  └─ update_active_mask()                              ← 根据 is_blocked 刷新掩码
```

---

## 第一层：Handler 分发

`_execute_once()` 根据 `statement.type` 选择处理器：

| StatementType | Handler 类 | ExecPipe 行为 |
|---|---|---|
| `S_*` (declaration) | `DeclarationHandler` | `set_next_pc(pc+1)` — 声明不执行，直接跳过 |
| `S_BRA` 等分支 | `BranchHandler` | 调用 `executeBranch()`，**处理器内部修改 next_pc** |
| `S_BAR` (bar.sync) | `BarrierHandler` | 调用 `executeBarrier()` → `sm_context->synchronize_barrier()` |
| `S_BAR_WARP_SYNC` | `BarWarpSyncHandler` | **继承 PipelineHandler**（见下文） |
| 其他 PTX 指令 | `GenericPipelineHandler` 子类 | **继承 PipelineHandler**（见下文） |

---

## 第二层：PipelineHandler::ExecPipe（⚠️ 本次回归根因所在）

```cpp
// instruction_base.cpp:78
void PipelineHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    int saved_pc = context->get_pc();  // ✅ 修复：保存执行前 PC

    if (!prepareOperands(context, stmt)) return;      // 阶段 1
    if (!executeOperation(context, stmt)) return;     // 阶段 2 ──→ processOperation()
    if (!commitResults(context, stmt)) return;         // 阶段 3

    context->set_next_pc(saved_pc + 1);                // ⚠️ 危险区：覆盖 next_pc
    // 注意：如果 processOperation() 内部通过 set_thread_pc() 修改了 PC，
    // 此行必须使用 saved_pc 而非 get_pc()，否则 next_pc 会多 +1
}
```

**子类覆盖关系**：
```
InstructionHandler (virtual ExecPipe)
  └─ PipelineHandler (ExecPipe — 标准三阶段流水线)
       └─ GenericPipelineHandler (prepare/execute/commit 默认实现)
            └─ BarWarpSyncHandler (覆盖 prepare/execute/commit)
```

`BarWarpSyncHandler` 虽覆盖了三阶段方法，但 **ExecPipe 仍继承自 PipelineHandler**，所以 `set_next_pc(saved_pc+1)` 仍会执行。

---

## 第三层：_execute_once 的 PC 生命周期

```cpp
// thread_context.cpp:74
void ThreadContext::_execute_once() {
    int current_pc = get_pc();                    // ① 读当前 PC
    StatementContext &stmt = (*statements)[current_pc];
    set_next_pc(current_pc + 1);                  // ② 默认 next = pc+1

    handler->ExecPipe(this, stmt);                // ③ 处理器可能修改 PC/next_pc
       // └─ BarWarpSyncHandler → processOperation → set_thread_pc(全部, reconv_pc)
       //    └─ PipelineHandler::ExecPipe → set_next_pc(saved_pc+1)
       //    └─ BranchHandler → set_next_pc(branch_target)

    set_pc(get_next_pc());                        // ④ 提交 next_pc → pc
       // ⚠️ 危险：get_next_pc() 读到的值可能已被 ③ 中多个路径修改
}
```

**PC 推进的四种情况**：

| 指令类型 | ③ 中发生什么 | ④ set_pc(?) 结果 |
|---------|-------------|-----------------|
| 普通指令 | PipelineHandler → set_next_pc(pc+1) | pc = pc+1 ✓ |
| 分支指令 | BranchHandler → set_next_pc(target) | pc = target ✓ |
| 屏障（未完成） | Wbar.arrive + is_blocked=true | pc = pc+1，下次重试屏障 |
| **屏障（完成，旧代码 bug）** | set_thread_pc→pc=5, ExecPipe→next_pc=get_pc()+1=6 | **pc = 6 ✗**（跳过了 PC=5） |

---

## 第四层：屏障处理的双路径

### 路径 A：CTA 级屏障（S_BAR → sm_context.cpp）

```
BarHandler::executeBarrier
  → sm_context->synchronize_barrier(barId, context)
     → 等待所有 CTA 线程到达
     → set_next_pc(get_pc() + 1)  释放 → 从下一条指令继续
     → set_next_pc(get_pc())      等待 → 原地重试
```

### 路径 B：Warp 级屏障（S_BAR_WARP_SYNC → barrier.cpp）

```
BarWarpSyncHandler::processOperation
  → wbar.arrive(lane_id)                   // 标记当前线程到达
  → if wbar.is_complete():
       for each arrived lane:
         set_thread_pc(i, reconv_pc)        // 写 warp_state.threads[i].pc = reconv_pc
                                            //      warp_state.threads[i].next_pc = reconv_pc
         is_blocked = false
         status = Active
       set_exec_mask / set_active_mask
       wbar.reset()
  → else:
       is_blocked = true                    // 阻塞等待
```

**bar.sync → bar.warp.sync 翻译**（ptx_visitor_barrier.cpp:43）：
- 条件：`isWarpLevelBarrier()` → CTA 线程数 ≤ 32（单 warp）
- 翻译后：`bar.warp.sync mask, reconvergence_pc`，mask=所有线程的位掩码

### 屏障执行流程中的 execute_warp_instruction

```cpp
// warp_context.cpp:146
for each lane:
    if (!lane_active && !BAR_SYNC) continue;     // 跳过非活跃线程
    if (warp_state.threads[i].pc != target_pc) continue;  // PC 不匹配 → 跳过

    sync_from_warp_state();
    if (BAR_SYNC) {                              // CTA 屏障回退
        sm_context_->synchronize_barrier(...);
        continue;
    }
    execute_thread_instruction();                 // → _execute_once → handler → 屏障逻辑
    sync_to_warp_state();

update_active_mask();  // 根据 is_blocked 刷新 → 屏障阻塞的线程变为 inactive
```

---

## 第五层：PC 同步机制

### sync_from_warp_state（warp_state → ThreadContext）

```cpp
// thread_context.cpp:695
auto& ref = warp_state.threads[lane_id];  // ⚠️ 别名引用
set_pc(ref.pc);           // 写 warp_state → ref 已被"污染"
set_next_pc(ref.next_pc); // 读的是被 set_pc 修改后的 ref
// 修复：set_pc 不应修改 next_pc（已于 92f7585 中修复）
```

### sync_to_warp_state（ThreadContext → warp_state）

```cpp
// thread_context.cpp:724
int current_pc = get_pc();          // 从 warp_state 读
if (thread_state.pc > current_pc)   // 如果处理器已推进 PC → 保留
    /* keep */
else
    thread_state.pc = current_pc;   // 否则写入当前值

thread_state.next_pc = get_next_pc();
```

### set_thread_pc（屏障专用，warp_context.h:69）

```cpp
void set_thread_pc(int lane_id, uint32_t new_pc) {
    warp_state.threads[lane_id].pc = new_pc;
    warp_state.threads[lane_id].next_pc = new_pc;  // ⚠️ 同时设置 next_pc
}
```



## ⚠️ 危险区地图

| 位置 | 操作 | 风险 | 修复 |
|------|------|------|------|
| **PipelineHandler::ExecPipe:102** | `set_next_pc(get_pc()+1)` | 屏障后 `get_pc()` 返回已修改的值 | 用 `saved_pc` 代替 `get_pc()` |
| **BarWarpSyncHandler:158** | `set_thread_pc(i, reconv_pc)` | 修改 warp_state 影响后续 `get_pc()` 调用 | 这是正确行为，但需配合 ExecPipe 的 saved_pc 修复 |
| **_execute_once:116** | `set_pc(get_next_pc())` | 如果 next_pc 被错误修改，pc 也被污染 | 依赖 ExecPipe 修复 |
| **sync_from_warp_state:700** | `set_pc(ref.pc)` + alias | `ref` 引用被 `set_pc` 副作用污染 | 确保 `set_pc` 只修改 pc |
| **execute_warp_instruction:164** | `warp_state.pc != target_pc → skip` | 屏障后 PC 变更 → target_pc 不匹配 → 线程被跳过 | 下个周期重新调度即可 |
| **update_active_mask:202** | `if (is_blocked) active=false` | 屏障未完成时线程被标记 inactive | 屏障完成时 `is_blocked=false` 恢复正常 |


## 快速诊断命令

```bash
# 追踪某个指令从解析到执行的全路径
grep -rn "S_BAR_WARP_SYNC\|BarWarpSync\|bar.warp" src/ --include="*.cpp"

# 查看所有修改 warp_state.threads[lane].pc 的位置
grep -rn "\.pc\s*=\|set_pc\|set_thread_pc" src/ptxsim/ --include="*.cpp"

# 查看 ExecPipe 调用链
grep -rn "ExecPipe\|execute_thread_instruction\|_execute_once" src/ptxsim/ --include="*.cpp"

# 查看屏障处理
grep -rn "synchronize_barrier\|wbar\|is_complete\|set_thread_pc" src/ptxsim/ --include="*.cpp"
```

---

## 调试时的关键断点位置

| 文件 | 行 | 用途 |
|------|----|------|
| `thread_context.cpp` | 94 | `_execute_once` 入口 — 打印当前 PC 和 statement type |
| `instruction_base.cpp` | 78 | `PipelineHandler::ExecPipe` 入口 — 打印 saved_pc |
| `barrier.cpp` | 143 | `wbar.is_complete()` — 屏障何时完成 |
| `barrier.cpp` | 158 | `set_thread_pc(i, reconv_pc)` — 记录被设置的 PC |
| `warp_context.cpp` | 164 | warp_state.pc vs target_pc — 哪些线程被跳过 |
| `thread_context.cpp` | 748 | `sync_to_warp_state` — PC 最终写回 warp_state 的值 |
