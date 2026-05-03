---
name: "oracle-prompting"
description: "防止 Oracle 子代理虚构文件/代码的提示词构造规则 — 预加载索引 + 强制引用 + 假设框架 + 验证门禁"
when_to_use: |
  任何需要咨询 Oracle 的场景——尤其是：
  - "分析根因", "为什么这段代码不工作"
  - "帮我理解这个架构", "这个 bug 可能在哪里"
  - Oracle 之前给出过可疑结论（引用不存在的文件等）
skills_required: []
---

# Oracle 防幻觉提示词规则

## 问题

Oracle 子代理在缺乏文件清单时，会基于"合理推断"虚构不存在的文件名（如 `ptx_interpreter.cpp`）。它给出看似合理的推理链，但基础是错的。

## 规则（每次调用 Oracle 必须遵守）

### 规则 1：预加载文件索引

在提示词开头列出 Oracle 需要了解的所有相关文件及其用途：

```markdown
RELEVANT FILES (verified to exist):
- src/ptxsim/core/thread_context.cpp — PC accessor implementations, _execute_once
- src/ptxsim/instructions/barrier.cpp — BarWarpSyncHandler, BarHandler
- src/ptxsim/core/sm_context.cpp — synchronize_barrier, check_barriers
- include/ptxsim/warp_context.h — set_thread_pc definition

Only reference files in this list. If you need to reference a file not in
the list, explicitly state that you are HYPOTHESIZING its existence.
```

### 规则 2：要求引用格式

明确要求 Oracle 用 `file:line` 格式引用代码：

```markdown
For every claim about code behavior, cite the exact source using the format:
  thread_context.cpp:116 — "set_pc(get_next_pc())" overwrites pc
If you cannot find the code, say "NOT FOUND" instead of guessing.
```

### 规则 3：假设框架

让 Oracle 输出多个假设而非单一结论：

```markdown
Output format:
## Hypothesis 1: [description] (confidence: high/medium/low)
  Evidence: file:line — "code snippet"
  Test: how to verify or falsify

## Hypothesis 2: [description] (confidence: high/medium/low)
  Evidence: ...
  Test: ...

## Most likely: Hypothesis N (because ...)
```

**为什么这有效**：多个假设迫使 Oracle 为每个假设寻找证据，暴露证据薄弱的假设。

### 规则 4：验证门禁

**拿到 Oracle 输出后，必须先验证再行动**：

```bash
# 1. 验证所有引用的文件存在
grep -l "ptx_interpreter" $(find . -name "*.cpp")   # 如果没有输出，文件不存在

# 2. 验证所有引用的行号存在且有预期的代码
sed -n '638p' src/ptx_parser/ptx_interpreter.cpp 2>/dev/null || echo "FILE NOT FOUND"

# 3. 如果任何引用验证失败 → 丢弃该假设，回到规则 1 重新提问
```

---

## 完整模板

```markdown
I'm debugging [问题描述].

RELEVANT FILES (verified to exist):
- path/to/file1.cpp — [用途]
- path/to/file2.h   — [用途]

RULES:
1. Only reference files and line numbers from the list above.
2. Use format: "file.cpp:NN — code snippet" for every claim.
3. Output 2-3 hypotheses with confidence levels, evidence, and falsification tests.
4. If any claim requires a file not in the list, mark it as SPECULATIVE.

QUESTION: [具体问题]
```

---

## 反例：本次调试中的失败案例

```markdown
# ❌ 当时的提示词（导致虚构文件）：
"I'm debugging a regression... QUESTION: What is the root cause?"

# ✅ 应该用的提示词：
"RELEVANT FILES:
- src/ptxsim/core/thread_context.cpp — _execute_once, sync, PC accessors
- src/ptxsim/instructions/barrier.cpp — BarWarpSyncHandler
- src/ptxsim/core/sm_context.cpp — synchronize_barrier
- src/ptx_parser/ptx_visitor_barrier.cpp — bar.sync translation
- include/ptxsim/warp_context.h — set_thread_pc
RULES: Only reference these files. Cite file:line for every claim.
QUESTION: Why does set_pc(get_next_pc()) in _execute_once overwrite the barrier handler's reconvergence_pc?"
```

后者会让 Oracle 聚焦在真实文件上，而非凭空推断一个 `ptx_interpreter.cpp`。

---

## 与 regression-bisect 的集成

在 `regression-bisect` 技能中：
- **阶段 2（语义变化表）** 完成后 → 此时已掌握所有相关文件列表
- **阶段 3（交叉引用）** 之前 → 用规则模板向 Oracle 提问
- **拿到结果后** → 规则 4 验证门禁 → 通过后才进入修复阶段
