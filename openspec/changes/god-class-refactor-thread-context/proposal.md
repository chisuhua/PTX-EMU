## Why

`thread_context.cpp` 当前 884 行、22 个 include，跨 SIMT 栈 / 寄存器 / 内存 / 控制流 4 个子系统混在一个类里，是 P1 优先级债务中工时最大项（10h）。每个子系统独立演进时（如新增指令、调整 PC 同步语义），修改点散布在同一文件中，code review 和回归验证成本指数增长。现在执行重构可避免未来每个 PTX handler 改动都牵一发动全身。

## What Changes

- **Phase 1（~3h, Tier 2 合格）**: 提取 SIMT 栈/PC 管理到独立类 `SimtPcManager`，封装 `get_pc()`/`set_pc()`/`commit_pc()`/`sync_from_warp_state()`/`sync_to_warp_state()` 及关联执行状态（`is_active()`/`is_exited()`/`is_at_barrier()`/`set_state()`/`get_state()`）
- **Phase 2（~4h）**: 提取寄存器访问层为 `RegisterAccessLayer`，封装 `acquire_register()`/`register_bank_manager_`/条件码寄存器管理
- **Phase 3（~3h, 跨季度）**: 提取内存访问 + 控制流为独立模块，完成 `ThreadContext` 瘦身
- 每个 Phase 独立 commit、独立 revert、独立验证
- **BREAKING**: 无（所有外部 API 通过保留的 `ThreadContext` 委托方法保持兼容）

## Capabilities

### New Capabilities
- `simt-pc-state`: Phase 1 的 `SimtPcManager` 类，封装 per-thread PC 读取/写入/同步及执行状态管理，与 `warp_state` 的双向同步由该类统一负责
- `register-access-layer`: Phase 2 的 `RegisterAccessLayer` 类，封装寄存器查找/分配及条件码访问
- `memory-control-flow-extract`: Phase 3 的内存访问提取（`get_memory_addr()`/`acquire_operand()` 等）和控制流提取（`collect_operands()`/`commit_operand()` 等）

### Modified Capabilities
<!-- No existing spec-level requirements are changing; this is a pure refactor. -->

## Impact

| 受影响组件 | 影响类型 |
|-----------|---------|
| `src/ptxsim/core/thread_context.cpp` (884 行) | 行数逐步降至 ~200 行（委托层） |
| `include/ptxsim/thread_context.h` (320 行) | 新增 `#include` 到提取的新类；旧成员改为 `std::unique_ptr` 组合 |
| `src/ptxsim/core/CMakeLists.txt` | 新增 `.cpp` 文件 |
| `tests/unit/contexts/` | 新增 `SimtPcManager` 单元测试（类型一） |
| `tests/integration/` | 验证所有 SIMT/barrier/divergence 测试在 refactor 后仍通过 |
| `src/ptxsim/instructions/*.cpp` | 零改动（委托保持 API 兼容） |

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- [ ] Baseline 函数所有 `set_*`/`commit_*`/`force_*` 调用已列出
- [ ] 逐行 diff 计划已写入 design.md
- [ ] 跨模块状态翻译路径（`sync_to_warp_state` → `is_blocked` 等）已文档化

### 多 Phase 推进
- [ ] Phase 拆分方案 + 独立 commit 粒度已说明（Phase 1/2/3，对应 §Design Decisions）
- [ ] 基线 worktree 命令已记录（`git worktree add .worktrees/baseline-pre-c1-phase1 HEAD~1`）
- [ ] 失败处理策略（revert 该 Phase 的单个 commit）已说明

### 文档同步
- [ ] `src/ptxsim/core/AGENTS.md`: 新增 `SimtPcManager` 相关说明
- [ ] ADR 追加：建议新建 ADR-0017 记录 "PC management extraction rationale"
- [ ] `tasks.md` Phase 状态变更已说明