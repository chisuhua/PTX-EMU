---
name: "regression-bisect"
description: "重构后测试回归的系统调试方法论 — git bisect 定位 + 新旧代码语义对比 + 状态修改交叉引用"
when_to_use: |
  触发关键词：
  - "回归", "regression", "之前好的现在坏了", "重构后测试失败"
  - "bisect", "哪个commit引入的"
  - "refactor后挂了", "改完之后测试不过"
  - 任何"测试从通过变成失败"的场景
skills_required: ["git-master"]
---

# 回归定位技能

## 核心原则

**不猜测根因，不随机尝试修复。** 先定位回归 commit，再逐行对比新旧代码语义差异，最后查找交叉修改路径。

## 流程（强制执行）

### 阶段 1：定位回归 commit

```
□ 1. 确认当前 HEAD 失败，已知 good commit 通过
□ 2. git diff <good>..<bad> --stat 了解变更范围
□ 3. 如果 diff 跨度大，git bisect 二分定位到单个 commit
□ 4. git diff <good>..<bad> 阅读每个文件的 diff
```

**关键命令**：
```bash
git diff <good>..<bad> --stat
git diff <good>..<bad> -- src/path/to/module/
git show <good>:src/file.cpp | head -120   # 查看旧版本文件
```

### 阶段 2：识别语义变化

对 diff 中**每一个变更的方法/访问器**，填写语义对比表：

| 方法 | 旧语义（读什么/写什么） | 新语义（读什么/写什么） | 谁还会修改同一状态？ |
|------|----------------------|----------------------|---------------------|
| `get_pc()` | 读 `this->pc`（本地字段） | 读 `warp_state.threads[lane].pc` | 屏障处理器的 `set_thread_pc()` |
| `set_pc(x)` | 写 `this->pc = x` | 写 `warp_state.threads[lane].pc = x` | 同上 |
| `next_pc = x` | 写 `this->next_pc` | `set_next_pc(x)` → 写 warp_state | `PipelineHandler::ExecPipe` |

**核心洞察**：当字段从本地迁移到共享状态时，每个读操作都可能读到**被其他代码路径修改过**的值。这就是回归的常见根因。

### 阶段 3：状态修改交叉引用

对阶段 2 中标记为"共享状态"的变量，执行交叉引用审计：

```bash
# 找到所有修改该状态的位置
grep -rn "set_thread_pc\|\.pc\s*=\|warp_state.*pc" src/ --include="*.cpp"

# 找到所有读取该状态的位置
grep -rn "get_pc()\|\.pc\b" src/ --include="*.cpp"
```

**绘制修改时序图**：
```
屏障处理器                         PipelineHandler::ExecPipe
    │                                      │
    ├─ set_thread_pc(i, 5) ──────────────►│ warp_state.pc[i] = 5
    │                                      │
    │                                      ├─ get_pc() → 5（刚被改过！）
    │                                      ├─ set_next_pc(5+1) → next_pc = 6 ❌
    │                                      │
    ▼                                      ▼
```

### 阶段 4：验证与修复

- [ ] 确认根因：修改后测试通过
- [ ] 修复必须**最小化**（通常 1-3 行）
- [ ] 运行**全部相关测试**，确认无新增回归
- [ ] 如果修复失败：回到阶段 2，检查是否遗漏了其他语义变化的路径

---

## 典型错误模式

### 模式 A：访问器间接层导致的"值漂移"

```
旧：局部字段，不受外部修改影响
新：访问器 → 共享状态，已被外部修改

→ PipelineHandler::ExecPipe 读取 get_pc() 拿到的是屏障处理器刚改过的值
```

### 模式 B：管道/回调链中的重复写入

```
_setup:    set_next_pc(pc + 1)     → next_pc = 5 ✓
_handler:  set_thread_pc(所有, 5)   → 修改 warp_state
_cleanup:  set_next_pc(get_pc()+1) → get_pc()=5, next_pc=6 ✗
_commit:   set_pc(get_next_pc())   → pc = 6 ✗
```

### 模式 C：别名引用污染

```cpp
auto& ref = shared_state.threads[lane];  // 别名引用
set_pc(ref.pc);       // set_pc 内部可能修改 ref 的其他字段
set_next_pc(ref.next_pc);  // ref.next_pc 可能已被 set_pc 修改
```

---

## 诊断信号速查

| 信号 | 含义 | 下一步 |
|------|------|--------|
| 单线程通过，多线程失败 | 同步/并发问题 | 检查屏障、锁、共享状态 |
| 结果为 0（未初始化值） | 写入指令被跳过 | 检查 PC 推进逻辑 |
| CFG 正确但执行错误 | 执行阶段 bug，非解析阶段 | 追踪指令管道 |
| "Released lane: PC=X → X" | reconvergence_pc == 当前 PC | 检查 `_execute_once` 和 `ExecPipe` 的 PC 覆写 |
| 部分线程结果正确 | 发散执行问题 | 检查 `get_lanes_by_pc` 和 active mask |

---

## 禁止行为

- ❌ 未确认根因就修改代码
- ❌ 同时修改多个"可能相关"的地方（shotgun debugging）
- ❌ 相信大模型代理的结论而不自己阅读 diff
- ❌ 修复一个 bug 的同时"顺便"重构（保持最小改动）
- ❌ 不运行全部测试就声称修复完成

---

## 参考案例

详见项目文档：`docs/developer-guide/REGRESSION-DEBUGGING-GUIDE.md`

本技能基于 ThreadContext PC 重构回归（Commit 92f7585）的调试实战总结。
