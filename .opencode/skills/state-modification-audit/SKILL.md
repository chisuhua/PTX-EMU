---
name: "state-modification-audit"
description: "对指定状态变量执行全项目读写交叉引用审计，识别并发/非预期的修改路径"
when_to_use: |
  触发关键词：
  - "为什么这个值不对", "谁改了这个变量", "值被覆盖了"
  - "并发修改", "race condition", "非预期修改"
  - "get_pc 返回了错误的值", "读到了旧值/新值"
  - 任何"读出来的值不符合预期"的场景
skills_required: []
---

# 状态修改交叉引用审计

## 核心原则

当 `get_X()` 返回的值不符合预期时，**不要只看调用点**。找到**所有**修改 `X` 的代码路径，检查它们是否在你不知道的时候被触发了。

## 流程

### 步骤 1：定义审计目标

明确你要追踪的状态变量：
- 变量名（如 `warp_state.threads[lane].pc`）
- 读访问器（如 `get_pc()`, `thread_state.pc`）
- 写访问器（如 `set_pc()`, `set_thread_pc()`, `thread_state.pc =`）

### 步骤 2：搜索所有写入点

```bash
# 搜索直接字段赋值
grep -rn "\.pc\s*=" src/ --include="*.cpp" --include="*.h"

# 搜索 setter 方法调用
grep -rn "set_pc\|set_thread_pc" src/ --include="*.cpp" --include="*.h"

# 搜索通过引用/别名的间接写入
# 重点关注 auto& 和返回引用的函数
grep -rn "auto&\|\.get_warp_state()" src/ --include="*.cpp"
```

### 步骤 3：搜索所有读取点

```bash
grep -rn "get_pc()\|\.pc\b" src/ --include="*.cpp" --include="*.h" | grep -v "\.pc "
```

### 步骤 4：交叉引用分类

将每个写点与你的目标读点配对，回答：
- 写操作**何时**触发？（初始化？每个周期？条件触发？）
- 写操作和读操作之间有没有**时序关系**？（写在前还是读在前？）
- 写操作是否在**同一调用栈**中？（直接影响读点）还是**异步/间接**触发？

### 步骤 5：绘制修改时序图

```
时间 ──────────────────────────────────────────►

  写点A              写点B          你的读点
  (初始化 pc=0)      (屏障释放       (ExecPipe
                      set_pc=5)       get_pc → ?)
```

如果写点B在你的读点之前发生，你的 `get_X()` 读到的就是被修改后的值。

---

## 输出格式

```markdown
## 审计报告：warp_state.threads[lane].pc

### 写入点（5处）
| 位置 | 触发时机 | 影响 |
|------|---------|------|
| thread_context.cpp:782 set_pc() | 每指令结束 | 正常 PC 推进 |
| warp_context.h:71 set_thread_pc() | 屏障完成 | ⚠️ 修改所有线程 PC |
| thread_context.cpp:700 sync_from_warp_state() | Warp 调度 | 同步 warp_state → ThreadContext |
| thread_context.cpp:748 sync_to_warp_state() | 指令执行后 | 同步 ThreadContext → warp_state |
| warp_context.cpp:193 update_active_mask() | 每周期 | 更新活跃掩码 |

### 你的读点
- instruction_base.cpp:102 `context->get_pc()` 在 PipelineHandler::ExecPipe

### 冲突分析
- `set_thread_pc()` (写点2) 在屏障处理器中调用
- `get_pc()` (你的读点) 在 PipelineHandler::ExecPipe 中被调用
- 调用顺序：屏障处理器 → PipelineHandler::ExecPipe
- **结论**：读点读到的值已被写点2修改 → 值"漂移"
```

---

## 快速命令模板

```bash
# 对变量 X 执行完整审计（替换 X 为实际变量名）
echo "=== 写入点 ==="
grep -rn "set_X\|X\s*=\|\.X\s*=" src/ --include="*.cpp" --include="*.h"

echo "=== 读取点 ==="
grep -rn "get_X\|\.X\b" src/ --include="*.cpp" --include="*.h" | grep -v "\.X "
```

---

## 典型陷阱

### 陷阱 1：隐藏的别名修改
```cpp
auto& ref = obj.get_state().field;  // ref 是内部数据的引用
set_Y(ref.subfield);  // set_Y 可能修改 ref 的其他字段
```

### 陷阱 2：getter 内部有副作用
```cpp
int get_pc() const {
    if (!warp_context_) return 0;  // 空指针返回 0，但调用者不知道
    return warp_context_->get_warp_state().threads[lane].pc;
}
```

### 陷阱 3：sync 函数改写值
```cpp
void sync_to_warp_state() {
    int current_pc = get_pc();  // 从 warp_state 读
    thread_state.pc = current_pc;  // 写回 warp_state —— 看似无操作但有副作用
}
```
