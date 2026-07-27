# god-class-refactor-sm-context — Spec

## ADDED Requirements

### Requirement: 重复 reconvergence 循环去重

The system MUST extract the two duplicated reconvergence orchestration loops (sm_context.cpp:455-490 and :580-623) to a single shared helper.

#### Scenario: 重复循环收敛为 1 个 helper

- **WHEN** 提取后运行 `grep -c 'check_reconvergence' src/ptxsim/core/sm_context.cpp`
- **THEN** 命中 ≤ 2（且均经过 helper）
- **AND** `wc -l src/ptxsim/core/sm_context.cpp` 减少 ≥ 65 行

#### Scenario: 共享 helper 行为一致

- **GIVEN** 两段循环原逐行近似
- **WHEN** 提取为 helper 后
- **THEN** 两处调用点 trace 输出与栈行为逐字节一致

### Requirement: step_b no-op byte-identical fallback 锁定

The system MUST preserve the step_b no-op byte-identical fallback contract per lessons-learned §14.

#### Scenario: 4 分支单元测试锁定

- **GIVEN** `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4 分支测试
- **WHEN** step_b_set_blocked_cycles 移动后
- **THEN** 4 分支测试全绿
- **AND** 测试代码无需修改即可通过

### Requirement: sm_context 拆分

The system MUST reduce sm_context.cpp to < 250 lines by extracting ≤ 4 components.

#### Scenario: 主文件行数 < 250

- **WHEN** 提取完成后运行 `wc -l src/ptxsim/core/sm_context.cpp`
- **THEN** 命中 < 250

#### Scenario: 新增 ≤ 4 组件

- **WHEN** 提取后运行 `ls src/ptxsim/core/sm_context_*.{h,cpp}` (排除原 sm_context.{h,cpp})
- **THEN** 命中 ≤ 8（4 组件 × .h+.cpp）

### Requirement: §1 行级 diff

The system MUST preserve `sm_context.cpp:379` `w->update_active_mask()` 调用及其注释 (:374) 行级随迁。

#### Scenario: update_active_mask 调用保留

- **GIVEN** lessons-learned §1 跨模块状态翻译（sm_context.cpp:379 是活例）
- **WHEN** 拆分后
- **THEN** `update_active_mask()` 调用在主文件或新模块中仍存在
- **AND** 注释 "only updated by update_active_mask(). Without this fix, active_count…" 完整保留

### Requirement: WarpContext API 不变

The system MUST NOT change WarpContext public API signatures (frozen by refactor-warp-context).

#### Scenario: sm_context.cpp 调用点零修改

- **WHEN** 拆分后
- **THEN** sm_context.cpp:379/:461/:463/:464/:468/:583/:585/:586/:590 调用点零 diff
- **AND** 编译期通过（C-18 冻结约束）

## 关联

- `improvements/god-class-refactor-sm-context.md`：完整提案
- `docs/adr/ADR-0020-cpptlm-injection-points.md`：注入点代码
- `openspec/changes/refactor-warp-context/`：依赖 C-18 冻结 API