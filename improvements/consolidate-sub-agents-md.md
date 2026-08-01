# consolidate-sub-agents-md

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-5
**阶段**: default | **分类**: infra-setup
**类型**: docs

## 架构依据

- 8 个子 AGENTS.md 文件与根 AGENTS.md **70%+ 内容重复**：
  - `src/cudart/AGENTS.md`
  - `src/grammar/AGENTS.md`
  - `src/ptx_ir/AGENTS.md`
  - `src/ptx_parser/AGENTS.md`
  - `src/ptxsim/AGENTS.md`
  - `src/ptxsim/barrier/AGENTS.md`
  - `src/ptxsim/core/AGENTS.md`
  - `src/ptxsim/instructions/AGENTS.md`
- 重复内容包括：构建命令、测试命令、编码规范、ANTI-PATTERNS 等
- 维护负担：修改规范需同步 9 个文件，容易遗漏

## 范围

- **In Scope**:
  - 提取公共内容为根 AGENTS.md 的共享 section
  - 子 AGENTS.md 仅保留目录特有的导航信息和 WHERE TO LOOK 表
  - 建立"根文件为 SSOT"的引用约定
- **Out Scope**:
  - 不删除任何子 AGENTS.md（保留目录级导航价值）
  - 不修改 .opencode/skills/ 下的技能文件

## 关键场景

- GIVEN 子 AGENTS.md 精简后, WHEN 开发者查看 src/ptxsim/AGENTS.md, THEN 仅看到该目录特有信息 + 指向根文件的引用
- GIVEN 规范更新, WHEN 修改根 AGENTS.md, THEN 所有子目录自动继承（通过引用而非复制）

## 技术约束

- MUST 保留每个子 AGENTS.md 的目录导航功能
- MUST 保留根 AGENTS.md 的完整规范定义
- SHOULD 子文件首行添加 `> 公共规范: 见根 AGENTS.md`

## 验收标准

- 子 AGENTS.md 平均行数减少 ≥ 50%
- 无规范内容丢失（根文件保留完整定义）
- 所有子文件保留 WHERE TO LOOK 表
