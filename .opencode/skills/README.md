# PTX-EMU 技能索引

> 本目录技能由 opencode 自动发现并可在调试/开发时加载。

## 调试类

| 技能 | 触发场景 |
|------|---------|
| `ptx-debug` | PTX-EMU 通用调试 — 配置选择、场景化调试方法 |
| `regression-bisect` | 重构后测试回归 — git bisect + 语义对比 + 最小修复 |
| `state-modification-audit` | 状态值异常 — 全项目读写交叉引用审计 |
| `oracle-prompting` | 咨询 Oracle 时 — 防幻觉提示词模板 |

## PTX 仿真类

| 技能 | 触发场景 |
|------|---------|
| `ptx-instruction-pipeline` | 指令执行流水线 — ExecPipe / Handler / PC 管理 / 危险区 |
| `ptx-barrier-mechanism` | 屏障机制 — S_BAR vs S_BAR_WARP_SYNC / Wbar / PC 覆写链 |

## 架构与合规类

| 技能 | 触发场景 |
|------|---------|
| `adr-compliance-check` | ADR 合规检查 — 开发完成后对照 ADR 检查清单验证 |
| `ptx-lessons-learned` | **项目经验沉淀** — 跨模块状态翻译、递归锁、分 Phase commit、checklist（2026-06 新增）|

## 语法与解析类

| 技能 | 触发场景 |
|------|---------|
| `ptx-grammar-modification` | ANTLR 解析错误 / 修改 .g4 文件 — 强制 TDD 流程 |
| `ptxir-serialization` | PTX 加载慢 — 二进制序列化与反序列化 |

## 测试类

| 技能 | 触发场景 |
|------|---------|
| `three-mode-testing` | 生成 PTX 测试用例 — 从 CUDA 程序自动生成 |
| `test-coverage-enforcer` | 新增 barrier/warp/thread 单元测试 — 确保有对应的集成测试验证 PC |

## OpenSpec 集成

以下 4 个 skill 在 `when_to_use` / `metadata` 字段中引用了 `ptx-lessons-learned`，确保 OpenSpec 流程自动应用项目经验：

| Skill | 集成点 |
|-------|-------|
| `openspec-propose` | 设计阶段 — 强制检查函数迁移完整性、多 Phase 推进、文档同步清单 |
| `openspec-apply-change` | 实施阶段 — 强制基线 worktree、Checklist A/D、失败处理纪律 |
| `openspec-archive-change` | 归档阶段 — 强制 prompt 询问生成 postmortem + lessons-learned 更新 |
| `adr-compliance-check` | 合规检查 — 强制 cross-check 5 个 lessons-learned 失败模式（A-E）|

---

## 技能调用关系

```
ptx-debug (入口)
  ├─ regression-bisect (测试回归 → 找 root cause)
  │   ├─ state-modification-audit (值被覆盖 → 交叉引用)
  │   └─ oracle-prompting (咨询 Oracle → 防幻觉)
  ├─ ptx-instruction-pipeline (PC/ExecPipe 问题)
  │   └─ ptx-barrier-mechanism (屏障问题)
  ├─ ptx-grammar-modification (ANTLR 解析错误)
  └─ cpp-debug (C++ 崩溃/内存)

adr-compliance-check (独立使用)
  └─ 开发完成后 / 代码审查前检查 ADR 合规性
  └─ 强制 cross-check ptx-lessons-learned 失败模式 A-E

ptx-lessons-learned (横向支撑)
  └─ 加载时机: 迁移/重构前、调试失败时、commit 前、归档前
  └─ 包含: 4 个 checklist + 失败模式速查表 + 16 个核心经验

OpenSpec 流程集成
  ├─ openspec-propose → 引用 ptx-lessons-learned（设计阶段 checklist）
  ├─ openspec-apply-change → 引用 ptx-lessons-learned（实施阶段纪律）
  ├─ openspec-archive-change → 引用 ptx-lessons-learned（归档阶段 postmortem 强制 prompt）
  └─ adr-compliance-check → 引用 ptx-lessons-learned（合规检查 cross-check）
```

---

## 经验沉淀的元规则

> **新经验的加入流程**:
> 1. 在实施中发现 bug 模式
> 2. 修复后，写入 `ptx-lessons-learned` + `docs/dev-process/lessons-learned.md` + 相关 ADR postmortem
> 3. 同步更新相关 skill（如 `ptx-barrier-mechanism` 涉及屏障问题时）
> 4. 在 `skills/README.md` 更新调用关系图
> 5. 在 OpenSpec 流程的对应阶段加入引用

> **skill 文档的双层结构**:
> - **skill (本目录)**: agent 主动加载，提供快速决策树 + checklist + 速查表
> - **docs/dev-process/lessons-learned.md**: 完整文档（具体案例、代码片段、长篇解释）
> - **互补关系**: 加载 skill 后能快速判断"我遇到了哪类问题"，再 deep-dive 到 lessons-learned.md

---

**最后更新**: 2026-06-18（新增 ptx-lessons-learned skill + OpenSpec 4 skill 集成）
