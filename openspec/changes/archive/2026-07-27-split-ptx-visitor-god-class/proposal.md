# split-ptx-visitor-god-class — Proposal

## Why

`src/ptx_parser/ptx_visitor.cpp` 实测 1067 行（`wc -l` 验证），但债务矩阵 post-phase3-debt-roadmap.md §1.2 C-17 声称的"按 statement 类型拆分"主范围**已由 10 个类别子文件实施完毕**（`ptx_visitor_{generic,atom,call,branch,barrier,simple,special,warp,memory,abi}.cpp`，由 `src/ptx_parser/AGENTS.md:48` 文档化）。

剩余 1067 行的真实结构（grep 实证）：
- qualifier/operand 解析 helpers (ptx_visitor.cpp:138-317, 937-1067，约 310 行)
- tcgen05 visitor (ptx_visitor.cpp:841-902，ADR-0016)
- X-Macro dispatch (ptx_visitor.cpp:927-931)
- 顶层 visit* overrides (ptx_visitor.cpp:318-839)

同时存在**实证 bug**：`#include "ptx_visitor_warp.cpp"` 在 ptx_visitor.cpp:917 与 :922 **重复出现两次**。

现在的问题是：原 C-17 提案重复声明已完成工作，且遗漏了 4 项实证残余（含 1 个 bug）。本 change 重新定基线，专注于这 4 项残余提取。

## What Changes

- **新增** `ptx_visitor_tcgen05.cpp`：提取 `visitTcgen05Inst` (ptx_visitor.cpp:841-902) 至独立文件，与现有 10 文件类别模式一致
- **新增** `ptx_visitor_operands.cpp`：提取 qualifier/operand helpers (ptx_visitor.cpp:138-317 + :937-1067，约 310 行) 至独立文件
- **新增** `ptx_visitor_dispatch.cpp`（或 `.inc`）：提取 X-Macro dispatch 区 (:927-931) 连同类别 `#include` 聚合区 (:905-925)
- **修复 bug**：删除 ptx_visitor.cpp:922 重复的 `#include "ptx_visitor_warp.cpp"`（保留 :917）
- **不可破坏**：IR 输出字节级一致，所有 PTX 语法测试通过

## Capabilities

### New Capabilities
- `ptx-visitor-tcgen05-extraction`: tcgen05 visitor 模块化拆分（ADR-0016 关联）
- `ptx-visitor-operand-extraction`: operand/qualifier helpers 独立化
- `ptx-visitor-dispatch-extraction`: X-Macro dispatch + include 聚合区独立化
- `ptx-visitor-dup-include-cleanup`: 修复 ptx_visitor.cpp:922 重复 include bug

### Modified Capabilities
（无现有 spec-level 行为变更。本 change 为纯重构，不修改 PTX 解析语义。）

## Impact

**受影响代码**：
- `src/ptx_parser/ptx_visitor.cpp`（主文件，1067 → ≤ 700 行）
- `src/ptx_parser/CMakeLists.txt`（需添加 3 个新源文件）

**不受影响**：
- ANTLR grammar `.g4` 文件
- IR 数据结构（`include/ptx_ir/`）
- CFGBuilder 入口（`src/ptx_parser/cfg_builder.cpp`）
- 其他 10 个类别子文件（已存在，仅需 include 区聚合）
- 所有 19 处 `extractQualifiersFromContext` 调用点（保持签名不变）

**依赖**：
- 无 C-2/C-18 依赖，可独立优先执行
- ADR-0007（CFG post-dominator）算法不变
- ADR-0016（tcgen05）相关注释须随迁（ptx_visitor.cpp:863-877）

**Oracle 评审链**：
- Round-1: REJECT 范畴识别错误
- Round-2: NEEDS-MORE-CHANGES（核心 scope 已实施）
- Round-3: APPROVE 优先执行

**Oracle 关键约束**：
- MUST Checklist H 实证验证（`ptx-lessons-learned/SKILL.md:539-555`）
- MUST §1 行级 diff（lessons-learned §1，`SKILL.md:48-77`）：visitTcgen05Inst 内 C3 fix parse-tree walk 注释 (:863-877) 行级随迁
- MUST IR 输出字节级一致
- MUST `./tests/ptx/test_all_ptx.sh` 全绿（47/47 目标）

**工时**: 2-3h（独立 + 范围小）