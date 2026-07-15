# Internal Implementation Plan: cpptlm-phase8b-injection-points

> **Status**: Draft (companion to OpenSpec change `openspec/changes/cpptlm-phase8b-injection-points/`)
> **Audience**: PTX-EMU 团队工程师（含未来 6 个月后的自己）
> **From**: PTX-EMU Architecture Team
> **关联**: [proposal.md](./proposal.md) + [design.md](./design.md) + [specs/cpptlm-injection-points/spec.md](./specs/cpptlm-injection-points/spec.md) + [tasks.md](./tasks.md) + [ADR-0020](../../../docs/adr/0020-cpptlm-injection-points.md)

---

## 0. 这份文档是什么

OpenSpec artifacts（proposal/design/spec/tasks）是**对外契约**——给 CppTLM 团队审阅，给未来审计追溯。
本 internal-plan 是 PTX-EMU 团队**自用**的完整实施手册：

- 实施时**实际**打开的文件路径、命令、行号
- 经验沉淀（来自 `docs/dev-process/lessons-learned.md` 的 16 章）
- 失败模式速查（发现 bug 时先翻这里）
- 6 Phase commit 节奏 + 每步验证命令
- 与姊妹 change `cpptlm-d1-full` 的协调

---

## 1. 完整实施路径（6 Phase commit 节奏）

> ⚠️ **Lessons Learned #3 强制**: 每个 Phase 独立 commit + 独立可回退。已有测试回归 → 立即 revert 该 Phase。

### Internal Phase ↔ tasks.md Phase 映射

| Internal Phase | tasks.md Phase | 说明 |
|:---:|:---:|---|
| A | PTX-1 | 3 个纯虚接口头文件（IScoreboard/IPipeline/ITensorCore）|
| B | PTX-2 | SMContext 扩展（3 setter + 3 getter）|
| C | PTX-3 + PTX-4 | WarpContext::set_blocked_cycles_for_active() + RegisterAnalyzer::get_dest_registers_as_ids() |
| D | PTX-5 | exe_once() 三段式注入（scoreboard 检查 → 延迟查询 → scoreboard 释放）|
| E | PTX-6 | 全量回归 + 测试覆盖 |
| — | PTX-0 | 基线 worktree + 枚举对齐（前置步骤）|
| — | PTX-7 | 文档同步 + postmortem |

### Phase A: 3 个纯虚接口头文件（~0.3d）

**目标**：`include/ptxsim/{scoreboard,pipeline,tensor_core}_interface.h` 可独立编译

**关键文件**：
- `include/ptxsim/scoreboard_interface.h`（新建，~20 LOC）
- `include/ptxsim/pipeline_interface.h`（新建，~30 LOC）
- `include/ptxsim/tensor_core_interface.h`（新建，~25 LOC）

**关键约束**：
- 零外部依赖：仅 `<cstdint>` + `<string>`
- PipelineId 0-5 枚举值与 CppTLM 一致（Phase 0 对齐确认）
- TcPrecision 0-5 枚举值与 CppTLM 一致
- 所有析构函数 `virtual ~Interface() = default`

### Phase B: SMContext 扩展（~0.3d）

**目标**：`SMContext` 接受 3 个 setter/getter

**关键文件**：
- `include/ptxsim/sm_context.h`（修改 — 新增 includes + setter/getter + 私有成员）

```cpp
void set_scoreboard(IScoreboard* sb) { scoreboard_ = sb; }
void set_pipeline_latency_provider(IPipelineLatencyProvider* p) { pipeline_provider_ = p; }
void set_tensor_core_timing(ITensorCoreTiming* tc) { tensor_core_timing_ = tc; }
```

**关键约束**：nullptr 默认值 → 字节级向后兼容

### Phase C: WarpContext + RegisterAnalyzer 扩展（~0.4d）

**WarpContext**：`set_blocked_cycles_for_active(uint32_t cycles)` — 对活跃线程设置阻塞周期

**RegisterAnalyzer**：`get_dest_registers_as_ids()` — 提取目标寄存器 ID（区分 src/dst）

### Phase D: exe_once() 三段式注入（~0.5d）

**目标**：sm_context.cpp 三处注入点（A/B/C）

**关键文件**：
- `src/ptxsim/core/sm_context.cpp`（修改：222 后 + 253/338 旁）

**注入点**：
| Step | 位置 | 功能 | nullptr 行为 |
|------|------|------|-------------|
| A | sm_context.cpp:222 后 | Scoreboard 分配检查 | 跳过 |
| B | :253 或 :338 前（单 PC／多 PC 分支）| 延迟查询 | InstructionLatencyTable fallback |
| C | :253 或 :338 后 | Scoreboard 释放 | 跳过 |

**关键约束**（Lessons Learned #1 强制）：双分支都不能遗漏

### Phase E: 测试覆盖 + 全量回归

Mock 测试 + 集成测试 → `ctest -L "unit;integration;e2e"` 全绿

---

## 2. 经验沉淀检查清单

### Checklist A: 函数迁移完整性

- [ ] 列出 baseline `exe_once()` 中所有 `set_*` / `block_*` 调用点
- [ ] 新增注入点不破坏现有 barrier 路径（barrier 后 check_reconvergence 必须保留）
- [ ] `set_blocked_cycles_for_active()` 不影响已阻塞线程

### Checklist B: 重构前

- [ ] 建立基线 worktree
- [ ] 列出本 change 的 6 个 Phase
- [ ] revert 策略：每个 Phase 独立 commit，失败 `git revert HEAD`

### Checklist H: Pre-implementation Review

- [ ] 实施前调用 Metis 审计
- [ ] 验证：3 接口头文件 `grep '#include'` 仅含 `<cstdint>` / `<string>`
- [ ] 验证：`exe_once()` 行号 222/253/338 在当前代码中存在

---

## 3. 与姊妹 change `cpptlm-d1-full` 的协调

### 不冲突区域（可并行）

| 文件 | 本 change | 姊妹 change |
|------|-----------|-------------|
| `sm_context.cpp` | ✅ 修改 | ❌ 不触碰 |
| `warp_context.cpp/h` | ✅ 修改 | ❌ 不触碰 |
| `scoreboard/pipeline/tensor_core_interface.h` | ✅ 新增 | ❌ 不触碰 |
| `cudart_sim.cpp` | ❌ 不触碰 | ✅ 修改 |
| `memory.cpp` | ❌ 不触碰 | ✅ 修改 |
| `cpptlm_bridge.h` | ❌ 不触碰 | ✅ 新增 |

### 潜在关联区域

- **blocked_cycles 双重 timing**：桥接 global_access() 和 pipeline_provider_ 都设置 blocked_cycles → 取 max-of-two（与 design.md §11.3 一致）

---

## 4. 失败时回退策略

```bash
# Phase N 失败 → 立即 revert
git revert HEAD
cmake --build build
ctest -L "unit;integration;e2e" --output-on-failure
```

---

**最后更新**: 2026-07-16（v1.0 草案，pre-flight B.1 补建）
**下次 review**: ADR-0020 Accepted + OpenSpec change → Active 后启动实施
