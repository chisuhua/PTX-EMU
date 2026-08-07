---
name: "ptx-lessons-learned"
description: "PTX-EMU 项目经验沉淀 — 跨模块状态翻译、递归锁、分 Phase commit、基线 worktree、qualifier 类型判断等具体失败模式与可复用 checklist。来自 2026-06 barrier module 迁移 + 2026-07 cute_rmsnorm float 类型判断实战"
when_to_use: |
  PTX-EMU 项目中遇到以下场景时强制使用：

  实施新功能/重构前：
  - "迁移/重构现有函数"、"API 切换"、"重写 handler"
  - 涉及 ThreadContext / WarpState / Barrier / SIMT 栈的状态修改
  - 涉及互斥量、锁、并发原语
  - 涉及多 Phase 推进（OpenSpec change 实施）
  - 涉及类型系统修改（qualifier 解析、is_float/is_signed 判断）

  调试失败时：
  - "测试在断言 N+1 处挂起" → 递归锁死锁
  - "集成测试 is_blocked 失败" → 跨模块状态翻译缺失
  - "测试套件整体超时" → 用 per-test timeout 定位
  - "Phase N 通过但 Phase N+1 失败" → 分 Phase commit 纪律
  - "E2E 测试输出非确定性但单元测试通过" → qualifier.back() 类型判断 bug

  实施完成后：
  - 提交前必查：checklist D（AGENTS.md 同步、ADR 追加、commit message 列出 fix 编号）

skills_required: []
---

# PTX-EMU 经验沉淀

> **来源**: 2026-06-18 commit `f033312` (integrate-barrier-module-cta-warp) 实战
> **完整文档**: [`docs/dev-process/lessons-learned.md`](../../docs/dev-process/lessons-learned.md)
> **互补文档**: [`docs/dev-process/debugging-strategy.md`](../../docs/dev-process/debugging-strategy.md)（问题分类与快速验证）、[`.opencode/skills/ptx-barrier-mechanism/`](./ptx-barrier-mechanism/)（领域知识）

## ⚡ 快速决策树

```
任务是什么？
├─ 迁移/重构函数 → 查"Checklist A: 函数迁移"
├─ 重大重构前   → 查"Checklist B: 重构前"
├─ 写注释      → 查"Checklist C: 写注释"
├─ Commit 前   → 查"Checklist D: Commit 前"
└─ 调试失败    → 查"失败模式速查表"
```

---

## 📚 核心经验（最常用 13 条）

### 1. 跨模块间接状态翻译（最隐蔽的 bug 来源）

**问题模式**: 迁移函数时漏掉看似冗余的 `set_state(BAR_SYNC)`，因为下一模块的 `sync_to_warp_state()` 才把它翻译为 `is_blocked = true`。没有这行，调度器死循环。

**关键经验**：
- 跨函数的"间接翻译"是最隐蔽的 bug 来源
- 迁移函数必须做到"行级 Diff"，不只比对主要逻辑
- `set_state` 看似冗余（next_pc 已设），但它是另一模块的 API 契约

**诊断命令**：
```bash
# 找出所有 set_state 调用点
grep -rn "set_state(" src/ptxsim/ include/ptxsim/ | grep -v "test"
# 找出所有读取 state 的位置
grep -rn "state == BAR_SYNC\|get_state()\|is_at_barrier" src/ptxsim/
# 比对：每个 set_state 应该有对应的翻译规则
```

**修复模板**（详见 `debugging-strategy.md` §"跨模块状态翻译检查"）：
```cpp
} else {
    // Mark thread as waiting at <event> so the executor (<file>:<line>)
    // recognizes <STATE> and skips re-execution. Without this,
    // sync_to_warp_state() keeps is_<field>=false and the scheduler
    // spins on the instruction. (Mirrors legacy <old-code> at <file>:<line>.)
    context->set_state(<STATE>);
    context->set_next_pc(context->get_pc());
}
```

### 2. 递归锁死锁（互斥量需要集中审计）

**问题模式**: `arrive()` 持 `mutex_` 后调用 `is_complete()`，后者再次 `lock_guard(mutex_)`。`std::mutex` 不可重入 → 死锁。

**关键经验**：
- 可重入性问题通常被单元测试忽略
- 死锁信号模式："测试在断言 N+1 处挂起，N+1 通常是循环的第一次调用"
- public 方法不应该再锁（应提供 internal unsafe 版本）

**诊断命令**：
```bash
grep -n "lock_guard\|unique_lock" <file>           # 列出所有锁点
grep -B2 "lock_guard" src/ | grep -A1 "lock_guard"  # 找出嵌套锁
```

**修复模式**：
```cpp
// 方案 A：拆分两个版本
class Foo {
    bool is_complete_unsafe() const;  // 假设已持锁
    bool is_complete() const {        // public 持锁
        std::lock_guard lock(mutex_);
        return is_complete_unsafe();
    }
};

// 方案 B：内联（最小改动）
bool arrive() {
    std::lock_guard lock(mutex_);
    arrived_threads_.insert(thread);
    bool complete = arrived_threads_.size() >= expected_threads_;  // 直接比较
}
```

### 3. 复杂迁移必须分 Phase commit

**关键经验**：
- 每个 Phase 独立 commit、独立验证，失败立即 revert
- 涉及控制流/分歧/同步的迁移，必须有"分歧场景"测试覆盖
- 单 PC 测试通过 ≠ 全部分歧场景通过

**判定"Phase 完成"的标准**：
- ✅ 所有原通过的测试仍然通过
- ✅ 新增的测试通过
- ✅ 没有测试"意外"变快/变慢
- ❌ 任何已有测试回归 → **立即 revert 该 Phase**

### 4. 基线 worktree 是最低成本保险

**关键经验**：
- 任何重大重构前花 1 分钟建立基线 worktree
- 节省数小时的"这个失败是基线的还是我的"争论
- 保留直到 change 完整合并 + 验证通过

**实测验证**: worktree 里 nvcc + CUDA 编译完全正常（实测 baseline worktree 中 `e2e_barrier_warp_sync` PASS 0.53s）。`env.sh` 是路径无关的（`NVCC_PATH=$(which nvcc)`），不依赖固定路径。

**标准操作**：
```bash
# Step 1: 建立 baseline（全量 build 预算 15-20 分钟）
git worktree add .worktrees/baseline-check <baseline-commit>
cd .worktrees/baseline-check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)  # 必须全量！否则部分 target 找不到

# Step 2: 验证 baseline 通过
cd build && ctest -L <label> --output-on-failure

# Step 3: 对比 main
cd <main-build> && ctest -L <label> --output-on-failure

# Step 4: 清理（实施完成且验证通过后）
git worktree remove .worktrees/baseline-check
```

**⚠️ 三个常见陷阱**:
1. baseline commit 必须**包含** main 上所有测试（不能选任意旧 commit）
2. baseline 首次 build 必须**全量**（partial build 导致 ctest 报"找不到可执行文件"）
3. worktree 共享 `.git` 但 build 独立（同一时刻不能在两个 worktree 切到同一分支）

**时间预算**:
- baseline 首次 build：15-20 分钟
- 增量 build：几秒-几十秒
- single target：5-30 秒

### 5. 类型判断依赖 "最后一个 qualifier" 导致 float 指令被当作整数

**问题模式**: `TypeUtils::is_float_type()` 只检查 `qualifiers.back()`。PTX 解析器生成的 qualifier 列表中 Q_F32 不在末尾（如 `{Q_U32, Q_S32}`），导致 `mul.f32` 等所有 float 指令被当作整数处理 → 64 位整数乘积写入 32 位浮点寄存器 → 非确定性垃圾输出。

**关键经验**：
- `qualifiers.back()` 是脆弱的类型判断 — 最后一个元素可能是数据类型、内存空间或修饰符
- 必须遍历**整个列表**检查目标类型
- 单元测试通过 ≠ handler 正确：单元测试直接构造 `{Q_F32}`，PTX 解析器生成的是多元素列表
- 非确定性输出 + handler 单元测试通过 = 类型判断层 bug 的强信号

**诊断命令**：
```bash
# handler 入口注入临时代码确认 is_float 标志
# 在 executeOperation 中打印 qualifiers 列表
grep -rn "is_float_type" src/ptxsim/  # 列出所有受影响 handler
grep -rn "qualifiers.back()\|\\.back()" src/  # 检查其他脆弱的 back() 使用
```

**修复模板**：
```cpp
// BEFORE (buggy): 只检查最后一个
bool is_float_type(const auto &qualifiers) {
    return !qualifiers.empty() && 
           (qualifiers.back() == Q_F32 || qualifiers.back() == Q_F64);
}

// AFTER (correct): 遍历所有 qualifier
bool is_float_type(const auto &qualifiers) {
    for (const auto &q : qualifiers) {
        if (q == Q_F32 || q == Q_F64 || q == Q_F16 || q == Q_BF16)
            return true;
    }
    return false;
}
```

---

### 6. OpenSpec artifacts 提交遗漏 + Debt audit 必须 git verify（2026-07 新增）

**问题模式**: 实施 OpenSpec change 时，工作区修改了 `openspec/changes/<name>/{design.md,tasks.md,spec.md}` 反映实施调整，但**这些 artifacts 修改从未 `git add`** — 仅源码 + commit message 描述了改动。fast-forward merge 后，OpenSpec 状态与代码不一致，需 commit `reconstruct` 补救。

**触发场景**: `cleanup-deprecated-barrier-apis` (2026-06-20) — 3 个实施 commits (`8a5573d`/`7914764`/`6ec8efd`) 合并时未追踪 artifacts；12 天后 (`barrier-migration-amendment` 2026-07-02) 基于 untracked reconstructed artifacts 误判 4 条 P0-A 为 active debt。

**关键经验**：
- **实施 OpenSpec change 必须 2-Phase commit**：
  1. Phase 0：`git add openspec/changes/<name>/` + commit "docs(openspec): <name> design adjustments" (artifacts FIRST)
  2. Phase 1+：实施代码 + commit
- **Debt audit 必须满足 2 个先决条件**：
  1. 当前 git HEAD 状态（不是 working tree）— `git status` + `git log -- <path>` 验证
  2. 引用 commit hash 而非文件路径
- **OpenSpec archive = change 终态**：归档后任何修补需求应新建 `fix-*`/`refactor-*` change + `Ref: archive/<date>-<name>/`，不要 amend 已归档 change

**诊断命令**：
```bash
# 验证 change 是否已归档
git log --all --oneline -- "openspec/changes/<change-name>/"
# 应包含 archive commit（如 ded4f96 chore(openspec): archive ...）

# 验证 artifacts 是否 tracked（实施后必须）
git ls-files openspec/changes/<change-name>/
# 不应为空

# 审计前自检
git status openspec/changes/  # 未提交修改警告
```

**真实案例**:
- `barrier-migration-amendment` (2026-07-02) — 试图 amend 已于 2026-06-20 归档的 `cleanup-deprecated-barrier-apis`
- `.opencode/notes/debt-audit-2026-07-02.md` §1.1 P0-A1~A4 误标为 active debt — 实际已通过 commits `8a5573d`/`7914764`/`6ec8efd` 解决

**修复模板**：
```bash
# 1. 删除 obsolete untracked dirs
rm -rf openspec/changes/<obsolete-amendment>/ openspec/changes/<already-archived-as-reconstructed>/

# 2. 更新 debt audit 标记为 RESOLVED
# docs/audits/<audit>.md §1.1 添加"状态"列 + 引用 commits

# 3. 沉淀 lesson
# docs/dev-process/lessons-learned.md 添加新条目（按现象/教训/检查工具/真实案例结构）
# .opencode/skills/ptx-lessons-learned/SKILL.md 添加 §核心经验 + Checklists E/F/G
```

### 7. Pre-implementation Review：实施 OpenSpec change 前必须跑 Metis 审计（2026-07 新增）

**真实案例**: `fix-cvt-strategy-actual-split`（commits `e8db807`/`f3ef891`/`43edf55`，2026-07-05）原 6-Phase 计划基于错误前提：
- ❌ 假设 919 行 switch 块未拆分 → 实证：4 个 Strategy 类已部署（`fc3c352`/`9837d44`/`d6123e0`）
- ❌ 假设 `.worktrees/fix-pre-p0-baseline` 可复用 → 实证：worktree 目录为空
- ❌ 假设 94 个 integration 测试为 oracle → 实证：仅 14 个
- ❌ 假设 `select_strategy()` 返回 `unique_ptr` → 实证：返回 `const ConversionStrategy&`

**关键经验**：
- **OpenSpec proposal 必须基于实证**：`git log -- <file>` + `grep <api>` + `wc -l <file>`，**禁止用"存在/未存在"推断状态**
- **必须区分"已实施但未清理"与"未实施"**：4 类表象对照见 `docs/dev-process/lessons-learned.md §20`
- **实施 OpenSpec change 之前必须跑 Metis pre-implementation review**：本 case 的 5 项 MUST-RESOLVE 全是实施前的隐形炸弹（scope + 接口 + 测试 + worktree + 路径），由 Metis 一次 review 全部揭示
- **archive "已归档"状态不可 amend**：任何修补必须新建 `fix-*` change + `Ref: archive/<date>-<name>/`（与 §6 互补）

**诊断命令**：
```bash
# 1. 验证 proposal 引用的 API 真实存在
grep -rn "<symbol>" src/ include/ tests/

# 2. 验证 oracle 测试数量真实
ctest -N -L "<label>" 2>&1 | tail -5

# 3. 验证提到的工作目录/路径/工具真存在
ls <worktree-path> 2>/dev/null
test -f <path> && echo exists || echo missing

# 4. 列出 baseline 文件的关键状态
wc -l <file>
git log --oneline -10 -- <file>
git log --all --oneline -- "<change-dir>"
```

**修复模板**：
1. 立即调用 Metis - Plan Consultant 子代理审计 4 个 OpenSpec artifacts
2. 检查 Metis 输出 ⚠️ CONDITIONAL 决策的 MUST-RESOLVE 列表（≥3 项阻塞 apply）
3. 重写 proposal/design/tasks.md 反映真实 scope（示例：6 Phase 拆分 → 3 Phase 死代码清理）
4. 在每个 artifact §Ref 段加 "Metis pre-implementation review" 链接
5. 沉淀到 lessons-learned.md §20 + ADR-0015 §2026-07 Fix 段

### 8. 重大功能交付必须同步根 README（2026-07 新增）

**问题模式**: 实施 `feat-*/implement-*` change 并归档后，根 `README.md` 仍描述 stale 状态（如 "WMMA 是 stub"）。新开发者读根 README 找方向，错误描述会立即误导。

**关键经验**：
- "重大功能交付" = 代码 + 单元测试 + e2e + **README 同步**（4 项缺一不可）
- 根 README 是"对外第一印象" — 修复越晚，误导人数越多
- 任何 archive commit 前必跑 grep 验证（`stub / TODO / FIXME / 硬编码百分比` 应为空）

**诊断命令**：
```bash
# Archive commit 前必跑
grep -n "stub\|TODO\|FIXME\|不实现\|未完成" README.md
grep -nE "[0-9]+%|第[一二三]" README.md  # 硬编码数字
grep -n "进行中\|完成" README.md  # 状态描述
```

**修复模板**：
```bash
# 新建 sync-* change（不 amend 已归档 change）
git checkout -b docs/sync-readme-after-<feature>
# 4 Phase: artifacts → 修订 → 3 README Fix #1-#3 → 归档
# 详见 docs/dev-process/lessons-learned.md §21 完整案例
```

**真实案例**:
- `sync-readme-after-tcgen05`（commits `8427829`/`80271cd`/`91aeef2`/`4b8cb6b`/`746d083`/`cee527f`，2026-07-05）
- 延迟: implement-wmma-tensor-core-tcgen05 (2026-07-04) → sync-readme (2026-07-05) = **1 天延迟**
- 修复量: README.md +15/-5 行（5 components, 3 commits, 5-step Phase 0-4 流程）
- Lessons-learned 集成: §6 (artifacts-first) + §19 (跨模块) + §20 (pre-impl review) 三者协同

### 9. ANTLR4 lexer 禁止定义 bare string token 与 ID 规则冲突（2026-07 新增）

**问题模式**: 在 `src/grammar/ptxLexer.g4` 新增 `TOKEN : 'bare_string'` 时，如果 `bare_string` 字符串只含 `[a-zA-Z_0-9$]`（即匹配 `ID : [a-zA-Z_$][a-zA-Z_0-9$]*`），ANTLR4 lexer 平局规则（first-defined-wins）会让 TOKEN 抢占 ID。所有以 `bare_string` 为寄存器名/变量名的 PTX 全部解析失败。

**关键经验**：
- **bare string token 是反模式**：能用 `.foo`（带点前缀）就用点前缀；能用 lexer mode 隔离就用 lexer mode；能不定义新 token 就不定义
- **声称 "X/X PASS" 必须用真实 kernel PTX 验证**：lexer 修改后必须 `cp bench/cute/*.ptx tests/ptx/regression_*.ptx` 跑 `./tests/ptx/test_all_ptx.sh`
- **一个 lexer 修复可同时解决 Kleene star 预测冲突**：root cause 错位会导致多个看似独立的 bug 共存

**诊断命令**：
```bash
# 1. 列出所有 bare string lexer tokens
grep -nE "^\w+\s*:\s*'[a-zA-Z]" src/grammar/ptxLexer.g4

# 2. 对每个 bare token，验证字符串模式与 ID 规则冲突
grep -A1 "^[A-Z_]\+\s*$" src/grammar/ptxLexer.g4 | grep -E ":\s*'[a-zA-Z_][a-zA-Z_0-9_]*'"

# 3. lexer 修改后用真实 kernel PTX 验证
cp bench/cute/cute_rmsnorm.ptx tests/ptx/regression_cute_rmsnorm_f16_register.ptx
bash ./tests/ptx/test_all_ptx.sh
```

**真实案例**:
- `commit ad808e3`（fix(grammar): resolve tcgen05 LL(*) prediction conflict）引入 `TCGEN_F16 : 'f16'` + `TCGEN_BF16 : 'bf16'` 抢占 ID → 5 ctest 失败（cute_rmsnorm/simpleGEMM 等）+ 7 tcgen05 fixture LL(*) 失败
- 修复（commit `55e216a`）：5 行 lexer/parser diff — 删除 bare tokens + parser `tcgen05Qual` 加 `ID` fallback + `tcgen05Dtype` 用 dot-prefixed `F16/BF16`
- 测试（commit `e92f1c1`）：`tests/ptx/regression_cute_rmsnorm_f16_register.ptx`（含 8 个 `%f1N` 寄存器名）→ 47/47 PASS
- 沉淀：§25 + Checklist L + 失败模式速查表新行

### 10. Helper 累加器 single-warp 假设 (2026-07-11)

**问题模式**: `tcgen05_fragment_mma_f16(Tmem&)` helper 假设调用方保证 single-warp 执行。FlashAttention 多 warp 协作时 `c_slot = 64 + lane_id` 让 warp 0 和 warp 1 都写 slot 64 → 数据竞争。

**关键经验**：
- "Currently safe because SM scheduler runs one warp at a time" 注释是**已知 debt** 的标记，必须显式标注 `[SINGLE-WARP ASSUMPTION]`
- 新增累加路径时必须同时考虑多 warp 影响
- 单元测试用 `SMContext(1 warp, 32, 1 cta)` 是 single-warp 测试，多 warp 必须独立测试

**诊断命令**：
```bash
grep -rn "single-warp\|one warp at a time\|sequential execution" src/ include/
```

**真实案例**: `fix-tcgen05-mma-accumulator-and-f32-storage` Oracle 2026-07-11 审计 C4 BLOCKER

### 11. TcQueue wait() commit_group_counter 检查顺序 (2026-07-11)

**问题模式**: `TcQueue::wait()` 先 push 到 `pending_waiters_` 再检查 counter，导致 commit→wait 序列后 `pending_count()` 返回 1（waiter 仍在 list 中）。

**关键经验**：
- TcQueue 状态机: commit bumps counter, wait 必须**先** check counter 再 push
- Integration test 第一次跑 `pending_count() == 0` 断言暴露此问题

**诊断命令**：
```bash
grep -n "wait\|pending_waiters_" src/ptxsim/async/tc_queue.cpp
```

**真实案例**: `fix-tcgen05-mma-accumulator-and-f32-storage` Phase 1 B2 test 暴露

### 12. PTX §9.7.16 f16×f16→f32 storage 对齐 (2026-07-11)

**问题模式**: Helper 改 `c_frag` 为 `float` 后 readback 站点未同步迁移，f32 bits 被当 f16 bits 读 → 垃圾值（silent corruption）。

**关键经验**：
- Helper 输出 dtype 是 hardware contract，golden header 必须声明 storage format
- `grep "c_buf[idx * 2]" tests/` 是 readback 残留快速检测
- `Catch::Approx` 默认 epsilon 对 storage format 错误不敏感

**诊断命令**：
```bash
grep -rn "f16_to_f32\|c_buf\[idx \* 2\]" tests/integration/tcgen05/
grep -n "f32_to_f16" src/ptxsim/instructions/tcgen05_helpers.cpp
```

**真实案例**: `fix-tcgen05-mma-accumulator-and-f32-storage` Phase 2 readback 漏改

### 13. ANTLR extractQualifiersFromContext 丢失 IMMEDIATE 值 (2026-07-11)

**问题模式**: `extractQualifiersFromContext` 只映射 terminal token 到 `Qualifier` enum，`IMMEDIATE` 节点被 `tokenToQualifier` 返回 `Q_UNKNOWN` 后静默丢弃。`instr.cta_group` 永远 defaults to 1。

**关键经验**：
- 被 **19 个 call sites** 调用 - 改返回类型破坏所有 caller
- 需要 IMMEDIATE 的 caller（commit/wait/lane_id 等）必须**单独 walk parse tree**
- 这种"通用 helper 丢失上下文信息"模式是 ANTLR visitor 常见 trap

**诊断命令**：
```bash
grep -n "Q_UNKNOWN" src/ptx_parser/ptx_visitor.cpp
grep -rn "extractQualifiersFromContext" src/
```

**真实案例**: `fix-tcgen05-mma-accumulator-and-f32-storage` Oracle 2026-07-11 审计 C3 BLOCKER

### 14. "byte-identical fallback" 契约必须由测试锁定 (2026-07-17)

**问题模式**: `step_b_set_blocked_cycles` 注释承诺 "All three nullptr = use existing InstructionLatencyTable only"，但实现中 nullptr 路径**fallthrough 到 `getLatency()` + `set_blocked_cycles_for_active()`**--这是 `exe_once()` 历史上从未有过的行为（`blocked_cycles` 设置此前仅 LdHandler 路径）。注释承诺 no-op，实现未兑现。

**关键经验**：
- "byte-identical fallback" / "nullptr = no-op" 契约**必须由直接单元测试锁定**，不能仅靠注释
- fallthrough 也是调用--"代码看起来没显式调用" ≠ no-op
- 注释承诺 no-op 时，必须明确列出"pre-change 路径中该状态变量由谁设置"（否则注释本身就是错的）
- 匿名命名空间内的 file-local 函数无法直接测试 -> 需提取为 public static 方法（与 `is_tensor_core_instruction` 等 helper 一致）

**诊断命令**：
```bash
# 1. 查找所有 "byte-identical" / "no-op" 契约注释
grep -rn "byte-identical\|no-op\|nullptr = " src/ptxsim/ include/ptxsim/

# 2. 对每个 no-op 契约，验证是否有测试锁定
grep -rn "no-op\|byte-identical" tests/

# 3. 验证 file-local 函数可测试性（匿名命名空间 = 不可直接测试）
grep -B2 "namespace {" src/ptxsim/core/*.cpp
```

**修复模板**：
```cpp
// BEFORE (buggy): 注释承诺 no-op，但 nullptr 路径 fallthrough
inline void step_b(IPipeline* p, ITc* tc, Warp* w, const Stmt& s) {
    // nullptr = byte-identical fallback  <-- 注释承诺
    uint32_t latency = 0;
    if (p) { /* ... */ }
    if (latency == 0 && tc) { /* ... */ }
    if (latency == 0) latency = getLatency(s);  // <-- nullptr 时也执行！
    if (latency > 0) w->set_blocked_cycles(latency);
}

// AFTER (fixed): 显式 early return + 测试锁定
inline void step_b(IPipeline* p, ITc* tc, Warp* w, const Stmt& s) {
    if (!p && !tc) return;  // both nullptr = no-op (TESTED)
    // ... priority chain ...
}
// + 4 个单元测试覆盖 4 条分支（both nullptr / pipeline / tc / fallback）
```

**真实案例**:
- `commit 367fd6a5` (feat) 引入 `step_b_set_blocked_cycles`，声称 "0 regression"
- `commit 5b292a91` (fix) 修复回归：`unit_simt_integration` 2 个断言失败（`pc_groups_after == 1 got 0`, `threads[0].pc == 1 got 0`）
- 回归在 feat 提交后**下一个提交**即被发现，说明 TDD 纪律有效，但 feat 提交的 "0 regression" 声明不准确
- 修复后补 `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4 个 case 直接锁定 4 条分支

---

## ✅ 可复用 Checklist

### Checklist A: 函数迁移

```
□ 列出 baseline 函数中所有的 set_* / commit_* / force_* 调用
□ 列出所有 mutex_ / lock_guard / unique_lock 使用
□ 对每个 set_*，grep 其值的下游消费者
□ 对每个锁点，确认"持锁方法调用的所有其他方法"也持同一锁，或重写为无锁版本
□ 比对行级 diff（不只比对主要逻辑）
□ 对所有 GenericPipelineHandler 子类，验证 is_float/is_signed 遍历全部 qualifier 而非只看 back()
□ 比较 helper 输出 dtype 与所有 readback 模式（f16 storage → f16 readback；f32 storage → memcpy readback）
□ 检查 helper body 内部是否残留旧 dtype 转换（f32_to_f16 / f16_to_f32）
```

### Checklist B: 重构前

```
□ 建立基线 worktree
□ 列出本 change 的所有 Phase，决定 commit 粒度
□ 决定哪些 Phase 需要基线对比（涉及 invariant 的一定要）
□ 准备 revert 策略：每个 Phase 独立 commit，失败立即 revert
□ 实施前 grep "f16\|f32" 在 helper 和所有 readback 站点，记录迁移清单
```

### Checklist C: 写注释

```
□ 这条注释能否让 3 个月后的陌生人避免犯同样的错？
□ 是 → 写
□ 否 → 不写
□ 例外：警告"不要做 X"必须写
```

### Checklist D: Commit 前

```
□ 跑过 baseline worktree 对比
□ AGENTS.md 是否需要同步
□ ADR 是否需要追加
□ OpenSpec tasks.md 是否需要更新
□ commit message 列出独立的 fix 编号
```

### Checklist E: OpenSpec change 实施后（2026-07 新增）

```
□ 所有 OpenSpec artifacts (design.md / tasks.md / spec.md / proposal.md) 已 git-tracked
  - 验证：git ls-files openspec/changes/<name>/ 不应为空
□ commit message 列出独立 fix 编号（如 Fix #1, Fix #2）
□ 每个 commit 独立可 revert（git revert HEAD 后编译通过）
□ 实施 commits 合并后立即 git-tracked artifacts（避免 working tree 遗漏）
□ 归档前 grep 验证 artifacts 与代码一致（无过期 task 编号）
```

### Checklist F: Debt audit 撰写（2026-07 新增）

```
□ 审计前 git log --since=<audit-date-1> -- <path> 验证所有引用 change 的实施状态
  - 验证：git log --all --oneline -- openspec/changes/<change-name>/ 不应仅含 archive commit
□ 引用 commit hash 而非文件路径
  - 示例："P0-A1 RESOLVED by commit 8a5573d" 而非 "P0-A1 当前 design.md 已修复"
□ 区分 "active debt"（影响实施）vs "stale debt"（已解决但审计未更新）
□ 每次审计标注 "基于 HEAD <hash>" 而非 "当前状态"
□ 审计撰写时若 working tree 与 git HEAD 不一致，必须明确标注：
  - "基于 working tree 状态，可能与 HEAD 不一致"
```

### Checklist G: OpenSpec lifecycle 约束（2026-07 新增）

```
□ Proposed: 未实施，artifacts 可修改
□ Accepted: 已批准实施，artifacts 可修改但需说明理由
□ Active: 实施中，artifacts 可修改（带 progress 标记）
□ Archived: 终态，artifacts 不可修改
  - 若需修补 → 新建 fix-* / refactor-* change
  - 引用方式：Ref: archive/<date>-<change-name>/
  - 禁止 amend 已归档 change（违反 OpenSpec 生命周期）
```

### Checklist H: Pre-implementation Review 强制项（2026-07 新增）

```
□ 实施 OpenSpec change 前：调用 Metis - Plan Consultant 子代理
  - 提供 4 个 artifacts 路径 + 真实文件路径/行号引用要求
  - 要求输出：Hidden Intentions / Ambiguities / AI Failure Points / Missing Context
  - 要求给出 GO / ⚠️ CONDITIONAL / ❌ NO-GO 决策
□ Metis 输出 ⚠️ CONDITIONAL 时：5 项 MUST-RESOLVE 全部完成才能 apply
□ 验证 proposal 的关键假设（实证）：
  - wc -l 验证声称的"X 行"真实存在
  - git log 验证 archive 中的"已实施 commits"真存在
  - grep 验证引用的 API 真存在（0 matches = 假设错误）
  - ctest -N 验证 oracle 数量（如 94 → 14 是常见偏差）
  - ls 验证 worktree/路径真存在（空目录 ≠ "可复用现有"）
□ 区分"已实施但未清理" vs "未实施"（4 类表象对照 §20）
```

### Checklist I: 重大功能交付清单（2026-07 新增）

```
□ "重大功能交付" = 代码 + 单元测试 + e2e + 根 README 同步（4 项缺一不可）
□ 实施阶段：根 README.md "状态" / "已知限制" 章节随代码同步更新（不延后到 archive）
□ Archive commit 前 grep 验证：
  - grep -n "stub\|TODO\|FIXME" README.md 应为空（或有明确 TODO + 修复 plan）
  - grep -n "进行中\|完成" README.md 应与 docs/README.md Phase 表格一致
  - grep -nE "[0-9]+%|硬编码" README.md 应替换为自动统计链接
□ 任何 feat-*/implement-* change 归档前必跑本 checklist
□ 新 sync-* / fix-* change 处理已归档案例：通过 Ref 链接 + 不 amend（per Checklist G）
□ postmortem 沉淀：追加 lessons-learned.md §N（本 checklist 是 §21 模板）
```

### Checklist J: OpenSpec artifacts 内部一致性强制检查（2026-07 新增）

```
□ 4 个 artifacts 范围数字对齐（proposal/design/tasks/spec 同一债务项的对象列表一致）
  - 示例：D-5 删除范围 = 3 个活跃 + 1 个已禁用副本（4 个）必须 4 个 artifact 同时出现
□ design Decision 路径示例与 spec Scenario 路径描述一致
  - design 写 "archive/<name>.design.md"（与子目录并列）→ spec 写"同级创建" → 一致 ✓
  - design 写 "archive/<name>/design.md"（在子目录内）→ spec 写"git hash 不变" → 冲突 ❌
□ design Decision 操作与 spec Scenario 操作描述一致
  - design 写"添加 README.md 段落" → spec 写"任何文件 git hash 不变" → 冲突 ❌
  - design 写"禁止修改归档" → spec 写"git hash 不变" → 一致 ✓
□ tasks 验证命令路径 = design 路径示例
  - tasks Phase 2.6 写 `test -f $d/design.md` 但 design 写 `$d.design.md` → 不一致 ❌
□ tasks 中"验证归档未变"任务存在（git status openspec/changes/archive/<name>/）
□ 范围模糊时优先严格约束（"完全不修改归档" > "修改归档 README"）
□ 审查产物登记到 `.opencode/notes/<name>-review-report.md`（≥30 commits 影响范围）
□ 沉淀到 lessons-learned.md §23（§23 是本 checklist 的真实案例模板）
```

### Checklist K: docs-* change 实施侧 3 陷阱（2026-07 新增）

```
□ Retroactive artifact 合成：≥3 共享模板 → 1 个 subagent（共享上下文 + 模板一致性）
  - 反例：N 个并行 subagent 各自维护模板 + commit hash 验证重复 N 次
□ Inline edit 策略：每个 ERRATA 项 1 个 edit，保留原文不变 + inline 注释
  - 每个 edit 前 Read 验证精确字符串 → Edit 仅追加 `[勘误 ...]` → 不修改原文
  - 验证：diff <(git show HEAD~1:file | grep -v '勘误') <(git show HEAD:file | grep -v '勘误')
□ git rm + .gitignore 盲区：tracked 删除后 untracked 子目录残留
  - 流程：git rm -r <dir>/（删 tracked）→ find <dir> -type d 检查残留 → rm -rf <untracked-subdir>
  - 替代：git clean -fdn <dir>/（dry-run）→ git clean -fd <dir>/（不可逆，慎用）
```

### Checklist L: ANTLR grammar modification（2026-07 新增）

```
□ 修改前：
  □ 列出 lexer 中所有 bare string tokens：grep -nE "^\w+\s*:\s*'[a-zA-Z]" src/grammar/ptxLexer.g4
  □ 验证每个 bare token 字符串不与 ID 规则冲突（不能只含 [a-zA-Z_0-9$]）
  □ 如有冲突，必选其一：点前缀 / lexer mode / 删除冗余 token
□ 修改后必跑 TDD 流程（per ptx-grammar-modification skill）：
  □ RED：建立 baseline（git bisect / 5 个 ctest 失败列表）
  □ 复制 bench/cute/*.ptx → tests/ptx/regression_*.ptx 真实 kernel guard
  □ GREEN：实施 lexer/parser 修改 + cmake --build build --target GenerateParser
  □ REFACTOR：./tests/ptx/test_all_ptx.sh 47/47 + ctest 全绿
□ Commit 顺序：fix(grammar) → test(ptx) regression guard → docs(dev-process) lesson
□ Commit message 引用 ad808e3（引入回归 commit）+ ADR-0016（架构依据）+ §25（lesson）
```

### Checklist M: container-erase-index-trap（2026-08 新增）

```
□ 任何维护外部索引成员（如 `size_t current_idx`）的容器操作：
  □ forward 遍历 + `erase(it)` 时，显式 clamp：
    if (removed_idx < current_idx) --current_idx;
    if (current_idx >= container.size() && !container.empty()) current_idx = 0;
  □ `schedule_next` / `get_next` 等入口加防御性 guard：if (idx >= size) idx = 0;
  □ 考虑反向遍历（`it--`）或换 `std::deque`/`std::list`（插入/删除不失效）
□ 检测 layout-sensitive UAF：
  □ 基线 binary PASS，clean rebuild 后 FAIL → 几乎必是此 trap
  □ 基线 ASLR off vs on 表现不一致 → 强信号
  □ 栈顶 `T::method(this=heap_capacity_addr)` 或 `T*` 越界 → 立即 grep 关联索引成员
□ 诊断命令：
  grep -rn "current_warp_idx\|current_idx" src/ptxsim/core/warp_scheduler.cpp
  grep -rn "std::find.*->erase\|erase(it)" src/ptxsim/ src/cudart/
□ 修复后必跑 clean rebuild ctest（不能只跑 incremental）确认 100% pass
```

---

## 🔍 失败模式速查表

| 症状 | 最可能原因 | 诊断命令 |
|------|-----------|---------|
| 集成测试断言 N+1 处挂起 | 递归锁死锁 | `grep -n "lock_guard\|unique_lock" <file>` |
| `CHECK(!is_blocked)` 类似断言失败 | 跨模块状态翻译缺失 | `grep "set_state.*BAR_SYNC" src/` 对比 `grep "state == BAR_SYNC" src/` |
| 调度器在 barrier 指令处死循环 | `set_state(BAR_SYNC)` 漏掉 | `grep "set_state" src/ptxsim/instructions/barrier.cpp` |
| 分歧场景一半 lanes 卡住 | 屏障释放逻辑覆盖了已释放 lanes | 检查 `set_active_mask` 是否 OR 而非 overwrite |
| Phase N 通过但 Phase N+1 失败 | 跨 Phase invariant 冲突 | 用基线 worktree 隔离每个 Phase |
| `git revert` 后 `git status` 异常 | stash/pop 改变了 staged 状态 | `git status` 验证 + 必要时 `git reset` |
| 注释"看似冗余" | 跨模块间接状态翻译 | 查"经验 1"，看是否需要保留 |
| `ctest -L <label>` 整体超时 | 某个测试死锁 | 切到 single-test + per-test timeout |
| **E2E 测试输出非确定性（每次不同值），但 handler 单元测试通过** | **`is_float_type()` 只看 `qualifiers.back()`，Q_F32 不在末尾时被误判为整数** | **handler 入口加 printf: `is_float` 标志；`grep "is_float_type" src/ptxsim/` 列出受影响 handler** |
| **OpenSpec 4 个 artifacts 单独看 OK 但 apply 后行为不一致** | **proposal/design/spec/tasks 间存在内部冲突（范围数字 / 路径策略 / 操作语义）** | **运行 Checklist J：4 个 artifacts 同债务项范围对齐 + design Decision 路径 vs spec Scenario 路径一致 + tasks 验证命令 = design 路径示例** |
| **Retroactive artifact 模板不一致 / commit hash 漏掉** | **5 个并行 subagent 各自维护模板 + commit 验证重复** | **N≥3 共享模板 → 1 个 writing subagent 一次性合成（共享上下文 + 模板一致性自然保证）** |
| **Inline 标记后原文被意外修改 / ERRATA 合并破坏快照** | **Edit 操作未用 Read 验证精确字符串** | **每个 Edit 前 Read 验证目标段落精确内容 → Edit 仅追加 inline → diff `grep -v '勘误'` 验证原文未变** |
| **`git rm -r` 后 untracked 子目录残留文件系统上** | **`.gitignore` 规则使子目录 untracked，git rm 不处理** | **`git rm -r <dir>/` 后 `find <dir> -type d` 检查残留 → `rm -rf <untracked-subdir>`；或 dry-run `git clean -fdn` 后 `git clean -fd`** |
| **ANTLR 解析错误：`mismatched input 'f16' expecting ID`** | **lexer 中 bare string token（如 `TCGEN_F16 : 'f16'`）抢占 ID 规则** | **`grep -nE "^\w+\s*:\s*'[a-zA-Z]" src/grammar/ptxLexer.g4` 列出 bare tokens；用点前缀（`.f16`）或 lexer mode 隔离** |
| **声称 "X/X PASS" 但 ctest 失败（如 5 个 cute_rmsnorm/simpleGEMM 等）** | **grammar 修改未用真实 kernel PTX 验证**（"自证"测试漏掉真实场景） | **修改后必跑 `./tests/ptx/test_all_ptx.sh` + 复制 `bench/cute/*.ptx` 到 `tests/ptx/regression_*.ptx`** |
| **Kleene star 预测冲突 + 寄存器解析失败同时发生** | **lexer 错位 — bare token 抢占 ID rule 同时影响 qualifier 与 register 解析** | **优先检查 lexer 是否有 bare token；删除/加前缀一次性解决多类问题** |
| **多线程单元测试 `th.join()` 后 REQUIRE 未触发** | **deadlock 时 `join()` 永久阻塞；`elapsed` 测量 post-join，bug 实际是 60s 软超时而非 30s 检测** | **`std::async(std::launch::async, ...)` + `future.wait_for(30s)` 返回 `future_status::timeout` 时主动 `REQUIRE(false, "deadlock")`** |
| **`TmemAllocator` read-only methods (`is_allocated_start`/`is_allocated`/`active_allocation_count`/`total_allocated_slots`) 声称 "safe under concurrent erase" 但实际是 UB** | **`std::map::find` 与 `std::bitset::test` 在并发 `erase`/`set` 下 UB（C++17 只保证迭代器不失效，不保证并发安全）** | **所有 public methods 一致加 `lock_guard(mu_)`；或用 `static_assert` 强制设计时一致性** |
| **`Tcgen05OpKind::MMA_WS` dispatch branch 写好但真实 PTX 永远不进** | **grammar 把 `.ws` 当作 `Q_TCGEN_WS` qualifier 在 MMA sub-op 上（不是独立 `MMA_WS` sub-op），所以真实 PTX 始终 `op_kind=MMA + qualifiers={Q_TCGEN_WS, ...}`** | **写新 dispatch 前 grep grammar（`ptxInstructions.g4:tcgen05SubOp`）确认 sub-op 真存在；否则在 handler 内部做 qualifier scan + 路由** |
| **Spec/Design 用了 `.warpspecialized::1` 词汇但 grammar 实际只有 `.ws`（裸 token）** | **PTX spec 用了修饰符语法（`.warpspecialized::N`）vs grammar 简化为裸 token（`.ws`），两者词汇脱节** | **设计阶段必跑 `grep -nE "warpspecialized|TCGEN_WARPSPECIALIZED" src/grammar/` 验证词汇对齐；或在 spec.md 加注 "grammar 简化" 说明** |
| **`Tcgen05Instr` 便捷字段（`cta_group`/`dtype`/`num_regs`/`has_block_scale`）全是默认值** | **visitor `visitTcgen05Inst` 只填 `op_kind`/`qualifiers`/`operands`/`instructionText`，这些字段从不被赋值** | **handler 检查前 `grep -n "Tcgen05Instr" include/ptx_ir/statement_context.h` + grep visitor 验证 visitor 是否真的提取这些字段；否则改用 `instr.qualifiers` 扫描对应 qualifier token（如 `Q_TCGEN_CTA_GROUP`/`Q_F16`）** |
| mma 累加后 C slot 是 1× 而不是 2× golden | helper `sum=0` 从不读取 c_slot | `grep "sum = 0\|sum=0" src/ptxsim/instructions/tcgen05_helpers.cpp` |
| `tc_queue().pending_count() == 0` 在 commit→wait 后失败 | wait() push 顺序问题 | `grep -n "pending_waiters_.push" src/ptxsim/async/tc_queue.cpp` |
| Helper 改 f32 storage 后 readback 返回 garbage | readback 仍是 f16 pattern | `grep "f16_to_f32\|c_buf\[idx \* 2\]" tests/` |
| `instr.cta_group` 永远是 default 1 | visitor 不提取 IMMEDIATE | `grep "extractQualifiersFromContext" src/` |
| **基线 e2e PASS，clean rebuild 后 e2e SEGFAULT，栈顶 `WarpContext::is_active(this=heap_capacity_addr)` 或类似越界 `T*` 指针** | **`std::vector<T*>::erase(it)` 不会调整维护的 `current_idx` 索引成员；rebuild 改变 heap layout 后野指针越界进入 guard page → segfault**（基线 binary 的 capacity 区落在合法映射内 silently 通过） | **`grep -rn "current_warp_idx\|current_idx" src/ptxsim/core/warp_scheduler.cpp`** → forward 遍历且 `remove_warp` 未 clamp 索引；按 Checklist M 修复（erase 后 clamp `current_idx`，schedule_next 防御性 guard）；参考 [`docs/dev-process/lessons-learned.md` §43](../../docs/dev-process/lessons-learned.md) |

---

## 🔗 与 OpenSpec 流程的集成点

| OpenSpec 阶段 | 使用此 skill 的方式 |
|---------------|------------------|
| `openspec-propose`（设计阶段）| 强制调用本 skill 的 "Checklist A" 验证函数迁移完整性 |
| `openspec-apply-change`（实施阶段）| 强制调用本 skill 的 "Checklist D" 提交前验证 |
| `openspec-archive-change`（归档阶段）| 自动 prompt 询问"是否生成 postmortem"，引用 ADR 追加段落 |
| `adr-compliance-check`（合规检查）| 自动 cross-check 本 skill 的失败模式速查表 |

---

## 📊 经验沉淀的元规则

> **新经验的加入流程**:
> 1. 在实施中发现 bug 模式
> 2. 修复后，写入本 skill + `lessons-learned.md` + 相关 ADR postmortem
> 3. 同步更新相关 skill（如 `ptx-barrier-mechanism` 涉及屏障问题时）
> 4. 在 `skills/README.md` 更新调用关系图

> **本 skill 与 `lessons-learned.md` 的关系**:
> - **本 skill**: agent 主动加载，提供快速决策树 + checklist + 速查表
> - **lessons-learned.md**: 完整文档（具体案例、代码片段、长篇解释）
> - **互补关系**: 加载 skill 后能快速判断"我遇到了哪类问题"，再 deep-dive 到 lessons-learned.md 看完整案例

---

## 🔗 相关资源

- **完整经验文档**: [`docs/dev-process/lessons-learned.md`](../../docs/dev-process/lessons-learned.md)（16 章节，~600 行）
- **调试策略**: [`docs/dev-process/debugging-strategy.md`](../../docs/dev-process/debugging-strategy.md)（含"跨模块状态翻译检查"章节）
- **Postmortem 案例**: [`docs/adr/ADR-0008-barrier-semantics.md`](../../docs/adr/ADR-0008-barrier-semantics.md) §2026-06-18 Postmortem
- **原 change 计划**: [`docs/superpowers/plans/2026-06-18-integrate-barrier-module-cta-warp-fix.md`](../../docs/superpowers/plans/2026-06-18-integrate-barrier-module-cta-warp-fix.md)
- **领域知识**: [`.opencode/skills/ptx-barrier-mechanism/`](./ptx-barrier-mechanism/)、[`.opencode/skills/ptx-instruction-pipeline/`](./ptx-instruction-pipeline/)
- **状态审计**: [`.opencode/skills/state-modification-audit/`](./state-modification-audit/)（与本 skill 经验 1 配套使用）
- **回归定位**: [`.opencode/skills/regression-bisect/`](./regression-bisect/)（与本 skill 经验 3-4 配套使用）
