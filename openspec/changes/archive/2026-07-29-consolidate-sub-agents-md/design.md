# consolidate-sub-agents-md - Design

## Overview

PTX-EMU 的 AGENTS.md 体系采用层级结构：根 `AGENTS.md` 定义完整项目规范，8 个子目录 `AGENTS.md` 提供目录级导航。当前子文件大量复制根文件内容（构建命令、测试命令、编码规范、ANTI-PATTERNS），导致 70%+ 重复率。

本 change 将公共规范集中在根文件（SSOT），子文件仅保留目录特有信息 + 引用根文件。

## Design Decisions

### 决策 1: 子文件保留策略 - WHERE TO LOOK 表 + 目录特有信息

**选择**: 子 AGENTS.md 保留以下内容：
1. 首行引用声明：`> 公共规范（构建/测试命令、编码规范、ANTI-PATTERNS）: 见 [根 AGENTS.md](../../AGENTS.md)`
2. 目录特有的 WHERE TO LOOK 表（该目录符号/文件的快速定位）
3. 目录特有的注意事项（如有）
4. 子目录链接（如有下级 AGENTS.md）

**删除内容**：
- STRUCTURE 段（已在根文件）
- COMMANDS 段（已在根文件）
- CONVENTIONS 段（已在根文件）
- ANTI-PATTERNS 段（已在根文件）
- CODE MAP 中已出现在根文件的部分

**理由**:
- WHERE TO LOOK 是子文件的核心价值：开发者已在子目录工作，需要快速定位该目录内文件
- 引用声明明确告知读者去根文件找公共信息，避免"信息去哪了"的困惑

### 决策 2: 引用声明格式

**选择**: Markdown blockquote + 相对路径链接

```markdown
> **公共规范**（构建命令、测试命令、编码规范、ANTI-PATTERNS）: 见 [根 AGENTS.md](../../AGENTS.md)
```

**理由**:
- blockquote 在 Markdown 渲染中视觉醒目
- 相对路径链接确保从任何子目录可正确跳转
- 明确列出"公共规范"包含的内容类别，减少读者回根文件查找的次数

### 决策 3: 根文件角色明确化

**选择**: 在根 `AGENTS.md` 顶部添加简短说明，明确其为 SSOT

```markdown
> **本文件是 PTX-EMU 项目规范的单一真值源（SSOT）。**
> **子目录 AGENTS.md 仅包含目录特有导航信息。**
```

**理由**:
- 明确根文件职责，避免未来开发者再次复制内容到子文件
- 不修改根文件的实质规范内容，仅添加 2 行声明

### 决策 4: 不引入自动化去重工具

**选择**: 手动编辑，不引入脚本自动检测重复

**理由**:
- AGENTS.md 变更频率低（非代码文件），自动化 ROI 不足
- 手动编辑可保留人工判断（某些"重复"是有意的上下文重述）
- 后续可通过 code review 规范约束

## Implementation Plan

### Phase 1: 根文件 SSOT 声明
1. 在根 `AGENTS.md` 顶部添加 SSOT 说明声明
2. 确认根文件包含所有公共规范段落（STRUCTURE、COMMANDS、CONVENTIONS、ANTI-PATTERNS）
3. 验证：根文件内容无丢失

### Phase 2: 精简 8 个子文件（分批）
1. `src/ptxsim/core/AGENTS.md`（最大，142 行 -> ~30 行）
2. `src/ptx_parser/AGENTS.md`（85 行 -> ~25 行）
3. `src/grammar/AGENTS.md`（84 行 -> ~25 行）
4. `src/ptxsim/instructions/AGENTS.md`（88 行 -> ~25 行）
5. `src/ptxsim/AGENTS.md`（81 行 -> ~20 行）
6. `src/ptx_ir/AGENTS.md`（54 行 -> ~20 行）
7. `src/cudart/AGENTS.md`（46 行 -> ~15 行）
8. `src/ptxsim/barrier/AGENTS.md`（47 行 -> ~15 行）

每个文件：
- 添加引用声明首行
- 保留 WHERE TO LOOK 表
- 删除与根文件重复的段落
- 验证：无信息丢失

### Phase 3: 最终验证
1. 逐文件 diff 确认删除的内容均在根文件中存在
2. 确认所有子文件保留 WHERE TO LOOK 表
3. 计算总行数减少百分比

## Testing Strategy

### 验证维度

| 验证项 | 方法 | 预期 |
|--------|------|------|
| 子文件行数减少 | `wc -l` 对比前后 | 平均减少 ≥ 50% |
| WHERE TO LOOK 保留 | grep "WHERE TO LOOK" 各子文件 | 每个文件均存在 |
| 引用声明存在 | grep "公共规范" 各子文件 | 每个文件首行存在 |
| 无信息丢失 | 逐文件 diff，确认删除内容在根文件中 | 无丢失 |
| 根文件完整性 | 确认 STRUCTURE/COMMANDS/CONVENTIONS/ANTI-PATTERNS 存在 | 完整 |

### 行数减少验证

```bash
# 精简前总行数
cat src/*/AGENTS.md src/ptxsim/*/AGENTS.md | wc -l  # 727 行

# 精简后总行数（目标 ≤ 363 行，即减少 ≥ 50%）
cat src/*/AGENTS.md src/ptxsim/*/AGENTS.md | wc -l
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 子文件信息不足，开发者需频繁跳转根文件 | 开发体验下降 | 引用声明明确列出根文件包含的内容类别 |
| 误删目录特有信息 | 信息丢失 | 逐文件 diff 验证，仅删除确认在根文件中存在的内容 |
| 根文件过长 | 阅读负担 | 根文件 118 行，精简后子文件总计 ~180 行，总体减少 |
| 未来开发者再次复制 | 回到重复状态 | SSOT 声明 + code review 约束 |

## Open Questions

1. **是否需要保留子文件中的部分 COMMANDS？**
   - 推荐：NO（统一引用根文件，避免部分保留导致不一致）
   - 决定：全部删除，仅引用

2. **CODE MAP 表如何处理？**
   - 推荐：根文件保留完整 CODE MAP，子文件保留该目录符号子集
   - 决定：子文件保留目录内符号的 CODE MAP 子集（非删除）

## 关联文档

- `improvements/consolidate-sub-agents-md.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-5`：原债务条目
- `AGENTS.md`：根文件（SSOT 目标）
