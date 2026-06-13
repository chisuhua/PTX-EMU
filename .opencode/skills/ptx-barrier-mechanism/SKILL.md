---
name: "ptx-barrier-mechanism"
description: "PTX-EMU 屏障机制全解 — S_BAR vs S_BAR_WARP_SYNC 双路径、bar.sync 翻译、Wbar 生命周期、PC 管理、多层覆写风险"
when_to_use: |
  PTX-EMU 项目中出现：
  - "屏障", "barrier", "bar.sync", "bar.warp.sync", "__syncthreads"
  - "屏障死锁", "barrier hang", "线程卡在屏障"
  - "barrier 之后结果不对", "同步后数据错误"
  - "Wbar", "is_blocked", "set_thread_pc", "reconvergence_pc"
  - "bar.sync 翻译", "bar.warp.sync 翻译", "barrier translation"
skills_required: ["ptx-instruction-pipeline", "cpp-debug"]
---

# PTX-EMU 屏障机制

## 总览：两条路径

```
                     bar.sync 0
                         │
                         ▼
              ┌──────────────────────┐
              │ isWarpLevelBarrier?  │  ← CTA 线程数 ≤ 32？
              │ (ptx_visitor_barrier │
              │  .cpp:43)            │
              └──────┬───────────────┘
                     │
          ┌──────────┴──────────┐
          │ YES (单 warp CTA)   │ NO (多 warp CTA)
          ▼                     ▼
   bar.sync 翻译为          保持 S_BAR
   S_BAR_WARP_SYNC               │
          │                      ▼
          ▼              BarHandler::executeBarrier
   BarWarpSyncHandler        → SMContext::synchronize_barrier
   使用 Wbar 机制                 CTA 级线程计数
   warp 级快速同步              
```

---

## 路径 A：Warp 级屏障（S_BAR_WARP_SYNC）

### 翻译规则（ptx_visitor_barrier.cpp:38-63）

```cpp
// 触发条件：CTA 总线程数 ≤ 32（只含 1 个 warp）
if (openum == S_BAR && isWarpLevelBarrier(currentKernel)) {
    // 生成参与掩码：所有线程的位掩码
    uint32_t mask = (1u << total_threads) - 1;  // 32线程 → 0xFFFFFFFF

    // 生成 bar.warp.sync 指令
    stmtCtx.type = S_BAR_WARP_SYNC;
    // reconvergence_pc = barrier 的 PC + 1（解析时 kernelStatements.size() + 1）
    int reconvergence_pc = kernelStatements.size() + 1;  // 注：代码中实际为 size()
                                                          // CFG pass 会修正
}
```

**重要**：翻译发生在**解析时**，不是启动时。`reconvergence_pc` 可能被 CFG pass 修正（cf. `CFG[PC=4]: S_BAR_WARP_SYNC updated - new_reconvergence_pc=5`）。

### Wbar 数据结构

```
Wbar {
    is_initialized : bool           // 是否已初始化
    participation_mask : uint32_t   // 需参与的线程掩码
    arrived_mask : uint32_t         // 已到达的线程掩码
    reconvergence_pc : int          // 屏障释放后跳转的 PC

    init(mask, reconv_pc)           // 初始化
    arrive(lane_id)                 // 标记到达 → arrived_mask |= (1<<lane)
    is_complete()                   // arrived_mask == participation_mask？
    count_participants()            // participation_mask 中置位的线程数
    reset()                         // 重置为未初始化
}
```

### Wbar 生命周期

```
线程 0 到达          线程 1..30 到达        线程 31 到达
    │                     │                     │
    ├─ wbar 未初始化       ├─ wbar 已初始化       ├─ wbar 已初始化
    ├─ 构建动态掩码        ├─ arrive(1..30)       ├─ arrive(31)
    ├─ wbar.init(mask,5)   ├─ !is_complete       ├─ is_complete() → true
    ├─ arrive(0)           ├─ is_blocked=true     ├─ 释放所有线程：
    ├─ !is_complete        │                     │   set_thread_pc(i, 5)
    └─ is_blocked=true     │                     │   is_blocked=false
                           │                     │   status=Active
                           │                     │   set_active_mask
                           │                     │   wbar.reset()
                           │                     │
                           ▼                     ▼
                      等待中...              全部继续执行 PC=5
```

### 动态掩码构建（barrier.cpp:117-126）

```cpp
if (warp_state.current_wbar_id < 0) {
    // 首次到达的线程负责构建参与掩码
    uint32_t current_pc = warp_state.threads[lane_id].pc;
    for (int i = 0; i < 32; i++) {
        if (!warp_state.threads[i].is_active) continue;
        if (warp_state.threads[i].is_exited) continue;
        // ⚠️ 只包含 PC 匹配的线程——处理发散执行
        if (warp_state.threads[i].pc == current_pc ||
            warp_state.threads[i].next_pc == current_pc) {
            dynamic_mask |= (1u << i);
        }
    }
}
```

**关键**：只有**活跃且 PC 匹配**的线程被包含。如果线程因分支发散导致 PC 不同，不会被纳入屏障。

---

## 路径 B：CTA 级屏障（S_BAR）

### BarHandler::executeBarrier（barrier.cpp:232）

```cpp
void BarHandler::executeBarrier(ThreadContext* context, const BarrierInstr& instr) {
    int barId = instr.barId.value_or(0);
    bool sync_complete = sm_context->synchronize_barrier(barId, context);

    if (sync_complete) {
        context->set_next_pc(context->get_pc() + 1);  // 释放 → 下一条指令
    } else {
        context->set_next_pc(context->get_pc());      // 等待 → 原地重试
        context->set_state(BAR_SYNC);                  // 标记为屏障等待状态
    }
}
```

### SMContext::synchronize_barrier（sm_context.cpp:412）

**两种释放路径**：

```cpp
// 路径 1：check_barriers() 批量释放（sm_context.cpp:140-175）
for (auto& [barId, waiting_threads] : barrier_waiting_threads) {
    if (waiting_threads.size() >= total_threads_in_block) {
        for (auto t : waiting_threads) {
            t->set_state(RUN);
            t->set_next_pc(t->get_pc() + 1);     // ⚠️ 用当前 PC + 1
        }
        waiting_threads.clear();
    }
}

// 路径 2：单个线程同步调用返回（sm_context.cpp:487-494）
bool SMContext::synchronize_barrier(int barId, ThreadContext *thread) {
    // ... 将线程加入等待队列 ...
    if (已全部到达) {
        thread->set_state(RUN);
        thread->set_next_pc(thread->get_pc() + 1);  // ⚠️ 用当前 PC + 1
        return true;
    }
    thread->set_state(BAR_SYNC);
    return false;
}
```

### execute_warp_instruction 中的屏障回退（warp_context.cpp:170-184）

```cpp
if (thread->get_state() == BAR_SYNC) {
    // ⚠️ warp 级屏障可能已部分完成，但仍有线程在等待
    if (warp_state.current_wbar_id >= 0) {
        bool complete = warp_state.wbars[wbar_id].is_complete();
        if (!complete) {
            // 回退到 CTA 级同步
            sm_context_->synchronize_barrier(thread->bar_id, thread);
        }
    }
    thread->sync_to_warp_state();
    continue;  // BAR_SYNC 线程不执行指令
}
```

---

## PC 管理：多层覆写链

屏障释放时，PC 经过**多个层次**的设置，每一层都可能覆盖上一层的值：

```
层次 1: set_thread_pc(i, reconv_pc)
        → warp_state.threads[i].pc    = reconv_pc    (正确值)
        → warp_state.threads[i].next_pc = reconv_pc  (也会被写)

层次 2: PipelineHandler::ExecPipe → set_next_pc(saved_pc + 1)
        → 如果 saved_pc 正确：next_pc = barrier_pc + 1 = reconv_pc ✓
        → 如果用了 get_pc()：   next_pc = reconv_pc + 1          ✗

层次 3: _execute_once → set_pc(get_next_pc())
        → 如果层次 2 正确：pc = reconv_pc ✓
        → 如果层次 2 错误：pc = reconv_pc + 1 ✗ (跳过一条指令)

层次 4: sync_to_warp_state → thread_state.pc = current_pc
        → 如果 get_pc() 返回 reconv_pc+1，写入 warp_state
        → 下个周期线程从错误 PC 继续执行
```

**结论**：层次 2 是本次回归的根因。`PipelineHandler::ExecPipe` 必须用 `saved_pc` 而非 `get_pc()`。

### 为什么旧代码没问题？

```
旧代码：context->next_pc = context->pc + 1
       → context->pc 是 ThreadContext 的本地字段
       → set_thread_pc() 写的是 warp_state，不影响 context->pc
       → context->pc 仍为 barrier_pc → next_pc = barrier_pc + 1 ✓

新代码：context->set_next_pc(context->get_pc() + 1)
       → get_pc() 从 warp_state 读取
       → set_thread_pc() 已修改 warp_state
       → get_pc() 返回 reconv_pc → set_next_pc(reconv_pc + 1) ✗
```



## is_blocked / active_mask / status 生命周期

```
屏障前                    屏障到达（未完成）         屏障完成
─────────────────────────────────────────────────────────────────
state = RUN              state = RUN              state = RUN
is_blocked = false       is_blocked = true        is_blocked = false
status = Active          status = Active          status = Active
active_mask[i] = true    active_mask[i] = false   active_mask[i] = true
                         (update_active_mask
                          检测到 is_blocked)

执行指令 ✓               被 execute_warp_         执行下一条指令 ✓
                         instruction 跳过
```

**关键**：BarWarpSyncHandler **不设置** `thread->state = BAR_SYNC`。线程被阻塞的**唯一标记**是 `warp_state.threads[i].is_blocked = true`。`update_active_mask()` 据此将线程标记为 inactive。

这与 CTA 级屏障不同——后者会设置 `thread->set_state(BAR_SYNC)`，从而触发 `execute_warp_instruction` 中的 BAR_SYNC 回退路径。

---

## CFG 分析对屏障的影响

CFG pass 会扫描所有 `S_BAR_WARP_SYNC` 指令并可能**修正** `reconvergence_pc`：

```
CFG[PC=4]: S_BAR_WARP_SYNC updated
  old_reconvergence=4, new_reconvergence_pc=5
```

**日志解读**：
- `CFG analysis: 0 barriers` → 该 kernel 无屏障（CFG 未找到需更新的屏障）
- `CFG analysis: 1 barriers (0 fallback)` → 找到 1 个屏障，0 个需要回退
- `old_reconvergence = ???` → 原始值乱码 → 可能未被正确初始化



## 诊断命令速查

```bash
# 1. 查看运行时日志中屏障相关信息
ctest -R <test> -V 2>&1 | grep -E "bar\.warp|Released lane|wbar|CFG.*barrier"

# 2. 检查 kernel PTX 中有几个 bar.sync
grep "bar.sync\|bar.warp" build/tests/*.ptx

# 3. 确认翻译是否生效（S_BAR vs S_BAR_WARP_SYNC）
grep "S_BAR\|S_BAR_WARP_SYNC" src/ptx_parser/ptx_visitor_barrier.cpp

# 4. 追踪 isWarpLevelBarrier 的返回值
# 添加 PTX_WARN_EMU("isWarpLevelBarrier: %d threads, result=%d", ...)
```

## 常见 bug 模式

| 症状 | 可能原因 | 检查点 |
|------|---------|--------|
| 屏障后 PC 跳过一条指令 | `PipelineHandler::ExecPipe` 用了 `get_pc()` 而非 `saved_pc` | `instruction_base.cpp:102` |
| 单 warp CTA 屏障 hang | `isWarpLevelBarrier` 返回了错误值 | `ptx_visitor_barrier.cpp:30` |
| 发散线程未参与屏障 | 动态掩码构建排除了它们 | `barrier.cpp:117-126` |
| reconvergence_pc 错误 | CFG pass 未修正或 visitor 计算有误 | `ptx_visitor_barrier.cpp:62` / CFG pass |
| 屏障后 active_mask 不正确 | `set_active_mask` 在 `update_active_mask` 之前被调用 | `barrier.cpp:167` vs `warp_context.cpp:195` |
| **divergent warp 两半分别到达 barrier 后第一半 lane 丢失** | **第二次 barrier 释放 `set_active_mask(arrived_mask)` 覆写而非合并** | **`barrier.cpp:176` 和 `barrier.cpp:244`** |

### 两半 barrier 分歧模式（BUG-POSTBARRIER-TWOHALVES）

**触发条件**：divergent warp 的两条路径在不同 cycle 到达同一 `bar.warp.sync`（典型场景：CTA 内两条 divergent 路径执行不同长度的循环后汇合于 `__syncthreads()`）。

**调用序列**：
1. 路径 A 到达 barrier PC → `force_reconvergence` 路径 → `wbar.init(arrived_mask=路径A)` → 所有路径 A lane 到达 → `set_active_mask(路径A)` → `current_wbar_id = -1`
2. 路径 B 到达 barrier PC → `force_reconvergence` 路径再次进入（`current_wbar_id < 0`）→ `wbar.init(arrived_mask=路径B)` → 所有路径 B lane 到达 → `set_active_mask(路径B)` ← **覆写！丢失路径 A 的 lane**

**症状**：路径 A 的 lane 不再在 `active_mask[]` 中，调度器永不调度它们。但 `update_active_mask()` 会在下一条指令从 `warp_state.threads[i].is_active` 重新计算 `active_mask[]`，**自愈了** lost lanes — 这就是为什么集成测试可能 PASS 而单元测试能 RED。

**修复**：在 `barrier.cpp:176` 和 `barrier.cpp:244`（两处 barrier 完成点），将
```cpp
warp_ctx->set_active_mask(arrived_mask);
```
改为
```cpp
warp_ctx->set_active_mask(warp_ctx->get_active_mask() | arrived_mask);
```

**禁忌**：不要改 `set_active_mask` 全局语义为 OR！ret handler 用 `set_active_mask(0u)` 显式清零，OR 语义会破坏。OR 逻辑必须放在 caller。

**诊断命令**：
```bash
# 单测直接测 set_active_mask 行为（捕获 bug）
ctest -R "post_barrier_two_halves" -V

# 集成测试：仅看 warp 是否完成（会被 update_active_mask 自愈骗过）
ctest -R "integration_post_barrier_two_halves" -V
# 真实症状检查：simpleGEMM-{int,float,double} 是否不再卡住
for t in int float double; do timeout 5 ./build/bin/simpleGEMM-$t 2>&1 | tail -1; done
```

**回归测试**：
- `tests/unit/barrier/test_post_barrier_two_halves.cpp`（单元，52 assertions RED→GREEN）
- `tests/integration/divergence/test_post_barrier_two_halves.cpp`（集成，smoke test）
