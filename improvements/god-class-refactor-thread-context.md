# god-class-refactor-thread-context

**优先级**: P1 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-1
**阶段**: default | **分类**: arch-design
**类型**: refactor

## 架构依据

- `src/ptxsim/core/thread_context.cpp` 实测 **727 行**，22 个 include
- 跨 SIMT stack / 寄存器 / 内存 / 控制流 4 个子系统职责混合
- **§1 跨模块状态翻译实证站点**：`simt_pc_mgr_->set_state()` (:71, :145, :224) + `commit_pc()` (:149) + `set_state(EXIT)` (:145) —— lessons-learned §1 "看似冗余但不可漏"的典型实例
- 行尾注释 (:726-727) 列出 `sync_from_warp_state, sync_to_warp_state, get_pc, set_pc, get_next_pc, set_next_pc, commit_pc` —— PC 管理 API 密集，是 SIMT 收敛正确性的核心

## 范围

- **In Scope**:
  - 拆分 thread_context.cpp 为 ≤ 4 个职责单一组件（SIMT stack 状态 / 寄存器访问 / 内存访问 / 控制流）
  - Phase 1: 提取 SIMT stack 状态到独立类（~3h，Tier 2 可承载）
  - Phase 2: 提取寄存器访问层（~4h）
  - Phase 3: 提取内存访问 + 控制流（~3h）
- **Out Scope**:
  - 不改变 ThreadContext public API 签名
  - 不修改 WarpContext 对 ThreadContext 的调用路径
  - 不动 SimtPcManager 内部实现

## 关键场景

- GIVEN SIMT stack 状态提取, WHEN 任一 warp 分歧/汇聚, THEN simt stack push/pop 行为与拆分前逐字节一致
- GIVEN PC 管理函数迁移, WHEN 行级 diff, THEN `set_state()` / `commit_pc()` 调用逐行随迁（§1 防漏）
- GIVEN 拆分后, WHEN sm_context / warp_context 编译, THEN 调用点零修改（API 冻结）

## 技术约束

- MUST 遵循 lessons-learned §1 行级 diff（SKILL.md:48-77）：重点站点 :71/:145/:149/:224
- MUST 遵循 Checklist B（SKILL.md:474-483）：baseline worktree + 3 Phase 独立 commit，单 Phase 回归即 revert
- MUST 保持 `commit_pc()` 语义：PC 推进必须在指令执行完成后才提交
- MUST NOT 改变 ThreadContext public API 签名
- MUST NOT 引入递归锁（§2 教训：持锁方法调用同锁其他 public 方法 = deadlock）

## 验收标准

- thread_context.cpp < 250 行
- 新组件 ≤ 4 个
- SIMT 收敛测试（execute_warp_instruction 路径）零回归
- 每个 Phase commit 独立可 revert
- barrier/divergence/reconvergence 测试全绿
