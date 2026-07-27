# split-ptx-visitor-god-class — Design

## Context

**当前状态**：`src/ptx_parser/ptx_visitor.cpp` 实测 1067 行。原 C-17 提案（post-phase3-debt-roadmap.md §1.2）声称"按 statement 类型拆分 visit* 方法"为主范围，但 Oracle Round-2 实证发现：**该范围已由 10 个类别子文件实施完毕**（`ptx_visitor_{generic,atom,call,branch,barrier,simple,special,warp,memory,abi}.cpp`，由 `src/ptx_parser/AGENTS.md:48` 文档化）。

**剩余 1067 行的真实构成**（grep 实证）：
| 区间 | 行数 | 内容 | 提取目标 |
|------|------|------|---------|
| :138-185 | 47 | qualifier helpers (`tokenToQualifier`, `extractQualifiersFromContext`) | ptx_visitor_operands.cpp |
| :185-317 | 132 | `createOperandFromContext` | ptx_visitor_operands.cpp |
| :318-839 | 521 | 顶层 ANTLR visit* overrides | 保留（主文件） |
| :841-902 | 61 | `visitTcgen05Inst` (ADR-0016) | ptx_visitor_tcgen05.cpp |
| :905-925 | 20 | 10 文件 `#include` 聚合区 | ptx_visitor_dispatch.cpp |
| :927-931 | 4 | X-Macro dispatch | ptx_visitor_dispatch.cpp |
| :937-1067 | 130 | operand visitors (`visitOperand`/...) | ptx_visitor_operands.cpp |

**实证 bug**：`#include "ptx_visitor_warp.cpp"` 在 ptx_visitor.cpp:917 和 :922 **重复两次**（验证：`grep -c '#include "ptx_visitor_warp.cpp"' src/ptx_parser/ptx_visitor.cpp` = 2）。

## Goals / Non-Goals

**Goals:**
- 提取 3 个职责单一子文件（tcgen05 / operands / dispatch），与现有 10 文件类别模式一致
- 删除 :922 重复 include bug
- 减少 ptx_visitor.cpp 至 ≤ 700 行
- 保持 IR 输出字节级一致
- 所有 PTX 语法测试通过

**Non-Goals:**
- 重新拆分按 statement 类型的 visit*（已完成，10 文件存在）
- 修改 `.g4` grammar 文件
- 修改 IR 数据结构 / CFGBuilder 入口（ADR-0007）
- 拆分 `visitFunctionDecl` (ptx_visitor.cpp:486-679) — 另立提案
- 优化 X-Macro dispatch 性能（保持语义）

## Decisions

### 决策 1: 提取顺序（TCGEN05 → OPERANDS → DISPATCH）

**选择**：先 tcgen05（ADR-0016 相关），再 operands（helpers），最后 dispatch（include 聚合）

**理由**：
- tcgen05 是 ADR-0016 关联代码，独立性强，先做风险最低
- operands 包含 `extractQualifiersFromContext` 19 个调用点，签名锁定后操作安全
- dispatch 与 #include 聚合区一起做，删除 :922 重复 include 一并完成

**替代方案**：
- A. 一次性大提取 → 风险高，回归难定位
- B. 倒序（dispatch → operands → tcgen05）→ 一样可达目标，无显著优势
- C. **采用**：分 3 commit 顺序提取，每步独立可 revert（Checklist B）

### 决策 2: 子文件命名约定

**选择**：`ptx_visitor_<category>.cpp` 与现有 10 文件一致

**理由**：
- 现有命名模式已确立（`ptx_visitor_*.cpp`）
- 工程师心智模型一致
- 文件列表可直接通过 `ls src/ptx_parser/ptx_visitor_*.cpp` 枚举

**替代方案**：
- A. 命名空间拆分（`ptx::parser::tcgen05`）→ 引入新概念，过度工程
- B. 拆为多个 header + 单一 cpp → 复杂度上升，编译时间无显著优化
- C. **采用**：保持现有 .cpp 物理分割模式

### 决策 3: X-Macro dispatch 提取为 .cpp 或 .inc

**选择**：.cpp 文件（与现有模式一致）

**理由**：
- .inc 需 CMake 额外配置处理
- 5 行 X-Macro dispatch 配合 :905-925 include 聚合区，组成 25 行小单元，.cpp 自然

**替代方案**：
- A. 保留在 ptx_visitor.cpp → 重构效果有限
- B. 拆为头文件 + cpp → 双文件维护成本
- C. **采用**：单 .cpp 文件

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| `extractQualifiersFromContext` 签名漂移 | 19 个调用点编译失败 | MUST 保持签名不变；行级 diff 锁定 |
| tcgen05 visitor 行为偏移 | IR 字节级不一致 | MUST `cta_group` IMMEDIATE 提取逻辑 (:866-877) 随迁；§13 lessons-learned 锁定 |
| 重复 include 引发 ODR 违规 | 链接错误 | MUST 删除 :922 重复 include；编译验证 |
| ptx_visitor.cpp 仍 > 500 行 | 未达原 < 500 目标 | 接受（顶层 visit* overrides ~520 行合法保留）；验收改为 ≤ 700 |
| CMakeLists.txt 未添加新源文件 | 链接缺失 | MUST 同步更新 `src/ptx_parser/CMakeLists.txt` |

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `src/ptx_parser/ptx_visitor.cpp` | 修改 | 主文件，1067 → ≤ 700 行 |
| `src/ptx_parser/ptx_visitor_tcgen05.cpp` | 新增 | 61 行 |
| `src/ptx_parser/ptx_visitor_operands.cpp` | 新增 | ~310 行 |
| `src/ptx_parser/ptx_visitor_dispatch.cpp` | 新增 | ~25 行 |
| `src/ptx_parser/CMakeLists.txt` | 修改 | 添加 3 个新源文件 |
| `src/ptx_parser/AGENTS.md` | 可选更新 | 若命名约定扩展可记录 |

**不变范围**：
- 10 个现有 `ptx_visitor_<category>.cpp` 文件（仅需在 dispatch.cpp 中聚合 include）
- 19 处 `extractQualifiersFromContext` 调用点（签名不变）
- 所有 ANTLR 生成的 visitor 方法签名
- IR 数据结构（`include/ptx_ir/`）

## Migration Plan

### 部署步骤（Checklist B 分 Phase commit）

**Phase 1 (30 min)**: 删除 :922 重复 include
```bash
# 删除 ptx_visitor.cpp:922 重复 #include "ptx_visitor_warp.cpp"
sed -i '922d' src/ptx_parser/ptx_visitor.cpp
# 验证
grep -c '#include "ptx_visitor_warp.cpp"' src/ptx_parser/ptx_visitor.cpp  # 应为 1
cmake --build build && ctest  # 必须全绿
git commit -m "fix(ptx_visitor): remove duplicate #include ptx_visitor_warp.cpp at :922"
```

**Phase 2 (45 min)**: 提取 tcgen05 visitor
```bash
# 移动 ptx_visitor.cpp:841-902 至新文件 ptx_visitor_tcgen05.cpp
# 在 ptx_visitor.cpp 中替换为 #include "ptx_visitor_tcgen05.cpp"
# 更新 CMakeLists.txt 添加新源
cmake --build build && ctest  # 必须全绿
git commit -m "refactor(ptx_visitor): extract visitTcgen05Inst to ptx_visitor_tcgen05.cpp"
```

**Phase 3 (60 min)**: 提取 operand/qualifier helpers
```bash
# 移动 ptx_visitor.cpp:138-317 + :937-1067 至 ptx_visitor_operands.cpp
# 行级 diff 锁定 extractQualifiersFromContext 签名
cmake --build build && ctest  # 必须全绿
git commit -m "refactor(ptx_visitor): extract operand/qualifier helpers to ptx_visitor_operands.cpp"
```

**Phase 4 (30 min)**: 提取 X-Macro dispatch + include 聚合区
```bash
# 移动 ptx_visitor.cpp:905-931 至 ptx_visitor_dispatch.cpp
# 在 ptx_visitor.cpp 中替换为 #include "ptx_visitor_dispatch.cpp"
cmake --build build && ctest  # 必须全绿
git commit -m "refactor(ptx_visitor): extract X-Macro dispatch + include aggregation"
```

**Phase 5 (15 min)**: 最终验证
```bash
wc -l src/ptx_parser/ptx_visitor.cpp  # 应 ≤ 700
./tests/ptx/test_all_ptx.sh  # 应 47/47
ctest --output-on-failure  # 全绿
```

### 回滚策略

- 每个 Phase 独立可 revert
- 若任一 Phase 引入回归，立即 `git revert HEAD`，定位问题在下一 Phase 重试
- 不合并至 main 直至所有 5 个 Phase 验证通过

## Open Questions

1. **是否同时更新 src/ptx_parser/AGENTS.md 文档化 3 个新子文件？**
   - 推荐：YES（与现有 10 文件 AGENTS.md 风格一致）
   - 决定：作为 Phase 5 验收的可选步骤

2. **是否拆分 visitFunctionDecl (ptx_visitor.cpp:486-679)？**
   - 推荐：NO（另立提案，超出本 change 范围）
   - 决定：明确划入 Out Scope

3. **dispatch.cpp 使用 .cpp 还是 .inc？**
   - 推荐：.cpp（无 CMake 特殊处理）
   - 决定：决策 3 已确定

## 关联文档

- `improvements/split-ptx-visitor-god-class.md`：完整 5 段提案
- `docs/adr/ADR-0007-cfg-post-dominator.md`：post-dominator 算法
- `docs/adr/ADR-0016-blackwell-only-tcgen05.md`：tcgen05 关联
- `.opencode/skills/ptx-lessons-learned/SKILL.md`：§1, §7, Checklist H
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-17`：原债务条目