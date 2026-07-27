# refactor-warp-context — Design

## Context

**当前状态**：`src/ptxsim/core/warp_context.cpp` 实测 558 行（roadmap 537 行已过期），最近 30 commits 触碰 6 次（高 churn）。该文件核心职责是 WarpContext 的执行/状态/同步管理，但存在 3 个明显的耦合问题：

1. **指令分发**（`execute_warp_instruction`）嵌入在主类中，每次新增指令需修改 switch/if-else
2. **active mask 操作**（`set_active_mask`）混合 overwrite/OR 语义，ret handler 强依赖 overwrite
3. **SIMT 编排**（`push/pop/check_reconvergence` 调用，warp_context.cpp:64-143）嵌在主类中，与 `simt_stack.cpp` 数据结构（已抽离）的边界模糊

**§1 跨模块状态翻译实证**：warp_context.cpp:337/:345/:370/:375 共 4 处 `thread->sync_to_warp_state()` 调用——这些是 thread→warp 状态翻译的关键站点（lessons-learned §1，SKILL.md:48-77）。迁移时漏任一处即调度器死循环（失败模式速查表）。

**API 冻结消费方**：sm_context.cpp:379 调 `update_active_mask()`、:468/:590 调 `check_reconvergence()`、:461/:583 调 `get_simt_stack()`——WarpContext public API 签名是本 change 与 C-2 的共享边界。

## Goals / Non-Goals

**Goals:**
- 提取 3 个职责单一子模块（dispatch / active_mask / simt）
- API 冻结：WarpContext public 签名不变（C-2 依赖）
- §1 强制项：4 处 sync_to_warp_state() 行级保留
- 拆分主文件至 < 300 行
- 保持执行/同步/汇聚语义完全一致

**Non-Goals:**
- 重写 execute_warp_instruction 主循环
- 改变 set_active_mask overwrite 语义
- 重抽 simt_stack 数据结构（已独立）
- 改变 BarrierModule / ThreadContext 行为
- 性能优化（保持语义优先）

## Decisions

### 决策 1: 提取顺序（ActiveMask → Simt → Dispatch）

**选择**：先 active mask helper（最小依赖），再 SIMT 编排，最后分发策略表

**理由**：
- active mask 与 set_active_mask overwrite 语义强相关，独立性强，先做风险最低
- SIMT 编排依赖 simt_stack.cpp，编排逻辑相对独立
- 分发策略表是最后一步，依赖前述 API 稳定

**替代方案**：
- A. 一次性大提取 → 风险高
- B. 倒序 → 一样可达目标
- C. **采用**：分 3 commit 顺序提取（Checklist B）

### 决策 2: API 冻结范围

**选择**：4 个 WarpContext public 方法签名完全冻结
- `update_active_mask()`
- `check_reconvergence()`
- `get_simt_stack()`
- `get_lanes_by_pc()`

**理由**：
- sm_context.cpp:379/:461/:468/:583/:590 是直接消费者
- 任意签名变更会破坏 C-2 的"行为一致"基线
- 编译期可立即验证（API 冻结 = sm_context.cpp 零 diff）

**替代方案**：
- A. 公开新抽象层 → 引入 v2 接口，过度工程
- B. 允许微小签名变更 → 触发 C-2 串行依赖
- C. **采用**：完全冻结

### 决策 3: SIMT 编排 vs 数据结构边界

**选择**：本 change 只动"编排逻辑"（divergence/reconvergence 处理），不动"数据结构"（已抽离至 simt_stack.cpp）

**理由**：
- simt_stack.cpp 已是独立模块（`src/ptxsim/core/simt_stack.cpp`）
- warp_context.cpp 残留的 :64-143 是 push/pop/check_reconvergence 的**调用方**
- 重抽数据结构会破坏现有测试覆盖

**替代方案**：
- A. 把 simt_stack 合并回 warp_context → 倒退
- B. 抽离整个 push/pop API → 复杂度上升
- C. **采用**：明确边界，仅抽编排调用

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 4 处 sync_to_warp_state() 漏迁移 | 调度器死循环 | MUST 行级 diff 逐项核对，lessons-learned §1 |
| WarpContext public API 签名漂移 | sm_context.cpp:379 等 5 站点编译失败 | MUST 编译期验证（sm_context.cpp 零 diff） |
| set_active_mask overwrite 语义丢失 | ret handler 行为偏移，分歧场景一半 lanes 卡住 | MUST 失败模式速查表 + AGENTS.md ANTI-PATTERNS 引用 |
| SIMT 编排抽离后 push/pop 顺序错 | 汇聚逻辑错误 | MUST 现有 barrier 测试 + 集成测试覆盖 |
| Phase commit 之间相互影响 | 单 Phase 回归难定位 | MUST 独立 commit + revert 验证 |

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `src/ptxsim/core/warp_context.cpp` | 修改 | 主文件，558 → < 300 行 |
| `src/ptxsim/core/warp_context.h` | 修改 | 新增 helper 类前向声明 |
| `src/ptxsim/core/warp_context_dispatch.{h,cpp}` | 新增 | 指令分发策略表 |
| `src/ptxsim/core/warp_context_active_mask.{h,cpp}` | 新增 | active mask helper |
| `src/ptxsim/core/warp_context_simt.{h,cpp}` | 新增 | SIMT 编排 |
| `src/ptxsim/core/CMakeLists.txt` | 修改 | 添加 6 个新源文件 |

**不变范围**：
- `simt_stack.cpp/h`（数据结构已抽离）
- `sm_context.cpp:379/:461/:468/:583/:590` 5 站点（API 冻结证据）
- `BarrierModule` 行为
- `ThreadContext` 行为

## Migration Plan

### 部署步骤（Checklist B 分 Phase commit）

**Phase 1 (1.5h)**: 提取 active mask helper
```bash
# 提取 set_active_mask + 相关 active mask 操作至 warp_context_active_mask.{h,cpp}
# 保留所有 4 处 sync_to_warp_state() 调用（§1 行级核对）
cmake --build build && ctest
git commit -m "refactor(warp): extract active mask helper to warp_context_active_mask"
```

**Phase 2 (2h)**: 提取 SIMT 编排
```bash
# 提取 push/pop/check_reconvergence 编排逻辑（warp_context.cpp:64-143）至 warp_context_simt.{h,cpp}
# 保持 simt_stack.cpp 数据结构 API 兼容
cmake --build build && ctest
git commit -m "refactor(warp): extract SIMT orchestration to warp_context_simt"
```

**Phase 3 (1.5h)**: 提取指令分发
```bash
# 提取 execute_warp_instruction 分发至 warp_context_dispatch.{h,cpp}
# 使用策略表/函数指针替换 switch/if-else
cmake --build build && ctest
git commit -m "refactor(warp): extract instruction dispatch to warp_context_dispatch"
```

**Phase 4 (1h)**: 最终验证
```bash
wc -l src/ptxsim/core/warp_context.cpp  # < 300
ctest --output-on-failure  # 全绿
# 验证 sm_context.cpp 零 diff
git diff --stat src/ptxsim/core/sm_context.cpp  # 应为空
```

### 回滚策略

- 每个 Phase 独立可 revert
- 任何 Phase 失败立即 `git revert HEAD`，定位问题在下一 Phase 重试

## Open Questions

1. **是否将 helper 类作为嵌套类或独立类？**
   - 推荐：独立类（清晰边界）
   - 决定：决策 1 倾向独立类

2. **分发策略表使用函数指针还是 `std::function`？**
   - 推荐：函数指针（零开销）
   - 决定：作为 Phase 3 内部决策

3. **是否同步更新 src/ptxsim/core/AGENTS.md？**
   - 推荐：YES（与现有 AGENTS.md 风格一致）
   - 决定：作为 Phase 4 验收的可选步骤

## 关联文档

- `improvements/refactor-warp-context.md`：完整 5 段提案
- `docs/adr/ADR-0006-simt-stack-management.md`：SIMT stack 设计
- `docs/adr/ADR-0014-independent-thread-scheduling.md`
- `docs/adr/ADR-0019-pc-management-extraction.md`
- `.opencode/skills/ptx-lessons-learned/SKILL.md`：§1, §7, Checklist B
- `improvements/god-class-refactor-sm-context.md`：依赖方（C-2 边界声明）