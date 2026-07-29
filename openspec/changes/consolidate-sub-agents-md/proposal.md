# consolidate-sub-agents-md - Proposal

## Why

PTX-EMU 有 **8 个子目录 AGENTS.md** 文件，与根 `AGENTS.md` 存在 **70%+ 内容重复**。重复内容包括：构建命令（`. env.sh`、`cmake` 命令）、测试命令（`ctest`、`test_all_ptx.sh`）、编码规范（clang-format、命名约定）、ANTI-PATTERNS 列表等。

重复的文件列表（行数统计）：

| 文件 | 行数 |
|------|------|
| `AGENTS.md`（根） | 118 |
| `src/cudart/AGENTS.md` | 46 |
| `src/grammar/AGENTS.md` | 84 |
| `src/ptx_ir/AGENTS.md` | 54 |
| `src/ptx_parser/AGENTS.md` | 85 |
| `src/ptxsim/AGENTS.md` | 81 |
| `src/ptxsim/barrier/AGENTS.md` | 47 |
| `src/ptxsim/core/AGENTS.md` | 142 |
| `src/ptxsim/instructions/AGENTS.md` | 88 |

核心问题：
- 维护负担：修改规范需同步 9 个文件，容易遗漏子文件导致规范不一致
- 信息噪音：开发者查看子目录 AGENTS.md 时被重复内容淹没，难以快速定位目录特有信息
- 违反 SSOT（Single Source of Truth）原则

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-5`

## What Changes

- **提取** 公共内容（构建命令、测试命令、编码规范、ANTI-PATTERNS）保留在根 `AGENTS.md` 作为 SSOT
- **精简** 8 个子 AGENTS.md，仅保留目录特有的导航信息和 WHERE TO LOOK 表
- **建立** "根文件为 SSOT" 的引用约定：子文件首行添加 `> 公共规范: 见根 AGENTS.md`
- **不删除** 任何子 AGENTS.md（保留目录级导航价值）

## Capabilities

### New Capabilities
- `agents-md-ssot-convention`: 子 AGENTS.md 引用约定，公共规范集中在根文件

### Modified Capabilities
- `agents-md-hierarchy`: 8 个子 AGENTS.md 从完整规范变为目录级导航 + 引用

## Impact

**受影响文件**：
- `src/cudart/AGENTS.md`（46 -> ~15 行）
- `src/grammar/AGENTS.md`（84 -> ~25 行）
- `src/ptx_ir/AGENTS.md`（54 -> ~20 行）
- `src/ptx_parser/AGENTS.md`（85 -> ~25 行）
- `src/ptxsim/AGENTS.md`（81 -> ~20 行）
- `src/ptxsim/barrier/AGENTS.md`（47 -> ~15 行）
- `src/ptxsim/core/AGENTS.md`（142 -> ~30 行）
- `src/ptxsim/instructions/AGENTS.md`（88 -> ~25 行）

**不受影响**：
- `AGENTS.md`（根，保留完整规范定义，可能微调措辞明确 SSOT 角色）
- `.opencode/skills/` 下的技能文件
- 任何源代码或测试文件

**依赖**：
- 无前置 change 依赖，可独立执行
- 纯文档变更，无编译/测试影响

**工时**: 1-1.5h（纯文档编辑）

## Design-Time Checklist

- [ ] 确认每个子 AGENTS.md 的 WHERE TO LOOK 表内容完整保留
- [ ] 确认根 AGENTS.md 包含所有公共规范（构建命令、测试命令、编码规范、ANTI-PATTERNS、CONVENTIONS）
- [ ] 确认子文件引用约定格式统一（`> 公共规范: 见根 AGENTS.md`）
- [ ] 确认无规范内容丢失（逐文件 diff 验证）
