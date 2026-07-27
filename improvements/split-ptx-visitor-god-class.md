# split-ptx-visitor-god-class

**优先级**: P2 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-17
**阶段**: default | **分类**: arch-design
**类型**: refactor

## 架构依据

- `src/ptx_parser/ptx_visitor.cpp` 实测 **1067 行**（`wc -l` 验证）
- **按 statement 类型的 10 文件拆分已完成，不得重复提案**：`ptx_visitor_{generic,atom,call,branch,barrier,simple,special,warp,memory,abi}.cpp` 均存在，经 `src/ptx_parser/AGENTS.md:48` 文档化，由 ptx_visitor.cpp:905-925 的 `#include` 区聚合
- 剩余 1067 行的真实构成（grep 实证）：
  - qualifier/operand 解析 helpers：`tokenToQualifier` (ptx_visitor.cpp:138)、`extractQualifiersFromContext` (:155)、`createOperandFromContext` (:185-317)
  - 顶层 ANTLR visit* overrides：`visitPtxFile` (:318) … `visitFunctionDecl` (:486) … `visitInstruction` (:703-839)
  - tcgen05 visitor：`visitTcgen05Inst` (:841-902，ADR-0016)
  - X-Macro dispatch：ptx_visitor.cpp:927-931
  - operand visitors：`visitOperand` (:937)、`visitSpecialRegister` (:941)、`visitRegister` (:953)、`visitImmediate` (:992)、`visitAddress` (:1002-1067)
- **实证 bug**：`#include "ptx_visitor_warp.cpp"` 在 ptx_visitor.cpp:917 与 :922 **重复出现两次**
- 债务矩阵归类 🟡 P2（post-phase3-debt-roadmap.md:64）

## 范围

- **In Scope**:
  - (a) 提取 tcgen05 visitor `visitTcgen05Inst` (ptx_visitor.cpp:841-902) → 新文件 `ptx_visitor_tcgen05.cpp`，与现有 10 文件类别模式一致（ADR-0016）
  - (b) 提取 operand/qualifier 解析 helpers（ptx_visitor.cpp:138-317 + :937-1067，共约 310 行）→ 新文件 `ptx_visitor_operands.cpp`
  - (c) 提取 X-Macro dispatch 区（ptx_visitor.cpp:927-931）连同类别 `#include` 聚合区（:905-925）→ `ptx_visitor_dispatch.cpp`（或 `.inc`）
  - (d) **删除 ptx_visitor.cpp:922 重复的 `#include "ptx_visitor_warp.cpp"`**（保留 :917）
- **Out Scope**:
  - 不重新拆分按 statement 类型的 visit*（已完成，见架构依据）
  - 不修改 .g4 grammar 文件
  - 不修改 IR 数据结构 / CFGBuilder 入口（ADR-0007）
  - 不拆分 `visitFunctionDecl` (ptx_visitor.cpp:486-679)——如需进一步减重另立提案

## 关键场景

- GIVEN 重复 include 删除， WHEN 编译， THEN 无重定义/ODR 告警且行为不变
- GIVEN tcgen05 visitor 提取， WHEN 解析含 tcgen05 指令的 PTX， THEN 生成 IR 与提取前字节级一致（含 `cta_group` IMMEDIATE 提取逻辑，ptx_visitor.cpp:866-877，lessons-learned §13 修复点）
- GIVEN operand helpers 提取， WHEN 19 处 `extractQualifiersFromContext` 调用点解析任意指令， THEN qualifier 列表完全一致
- GIVEN 拆分后， WHEN `./tests/ptx/test_all_ptx.sh`, THEN 全绿

## 技术约束

- MUST 执行 Checklist H 实证验证（.opencode/skills/ptx-lessons-learned/SKILL.md:539-555）：本提案所有行号已 `wc -l`/`grep`/`sed` 实证，apply 前复核一次 git HEAD
- MUST 保持 IR 输出字节级一致（解析结果零 diff）
- MUST NOT 修改 `.g4`；MUST NOT 变更 `extractQualifiersFromContext` 签名（19 个调用点，ptx_visitor.cpp:870 注释明示）
- MUST 保留 `visitTcgen05Inst` 内 C3 fix 的 parse-tree walk 注释（ptx_visitor.cpp:863-877），行级随迁（lessons-learned §1, SKILL.md:48-77）
- SHOULD 新文件命名与现有 `ptx_visitor_<category>.cpp` 模式一致

## 验收标准

- ptx_visitor.cpp ≤ 700 行（1067 → 约 650-690：移除 tcgen05 ~62 行 + operand/qualifier ~310 行 + dispatch ~5 行 + 重复 include 1 行）
- `grep -c '#include "ptx_visitor_warp.cpp"' src/ptx_parser/ptx_visitor.cpp` == 1
- `./tests/ptx/test_all_ptx.sh` 全绿（47/47 目标，以 git HEAD 实测为准）
- parser 相关单元测试 + CFG post-dominator 集成测试零回归（ADR-0007）

## Round-3 vs Round-1/2 deltas

- 撤回 Round-1 的"按 statement 类型拆分 visit*"主范围（实证已完成，10 文件存在）
- 新范围锚定 4 项实证残余：tcgen05 visitor、operand/qualifier helpers、X-Macro dispatch、重复 include bug
- 验收行数从 <500 放宽至 ≤700（顶层 visit* overrides ~520 行合法保留）
- 工时 5h → **2-3h**

## Oracle 评审链

- Round-1 (Oracle): REJECT 范畴识别错误（已实施但误判为未实施）
- Round-2 (Oracle): NEEDS-MORE-CHANGES（核心 scope 已实施，需重定基线）
- Round-3 (Oracle): APPROVE（4 项残余均有 file:line 锚点，可独立优先执行）