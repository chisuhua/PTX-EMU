## Why

[Stale 文档同步] `README.md` "已知限制" 章节在 `2026-07-04` Blackwell tcgen05 完整实施后未同步，仍将 WMMA / Tensor Core 描述为 "是 stub"，与代码现实严重矛盾。

**根因**（基于 commit hash 实证，非审计假设）：

| README.md 现状（line 49-53） | 代码现实（git verify 验证） |
|------------------------------|----------------------------|
| "**WMMA / Tensor Core**：是 stub" | `src/ptxsim/instructions/wmma.cpp` 完整实现 tcgen05.mma/ld/st/commit/wait（19 个 includes、5 个谓词函数），归档于 `archive/2026-07-04-implement-wmma-tensor-core-tcgen05/`（commits `35808d6` / `535dd9d` / `a223d6a` / `0213ff1` / `4151268` Fix #11-#14） |
| "**状态**：SIMT v2.0 (Phase 10 进行中)" | docs/README.md Phase 表格显示 "Phase 10 进行中"，但 `docs/dev-process/post-tcgen05-roadmap.md`（commit `66e3e2e`）已为 H5 规划做参考（即 Phase 10 已近收尾）|
| "**PTX 指令覆盖**：核心 ISA ~67%" | tcgen05 完整实现后实际覆盖更高；audit 数字未自动生成 |
| "**CUDA Toolkit**：11.4.4 测试通过" | `env.sh` 自动检测 `$(which nvcc)`，硬件允许任意版本；README 应说明环境自适应 |

**这是 lessons-learned §6 的第二案例**（继 `fix-cvt-strategy-actual-split` 之后）：
- `cleanup-deprecated-barrier-apis` (2026-06-20) — 实施 commits 未追踪 artifacts → 后续审计误判
- `implement-wmma-tensor-core-tcgen05` (2026-07-04) — 实施 commits ✅ tracked，但**根 README.md 未同步** → 当前 stale
- `2026-07-05-fix-cvt-strategy-actual-split` —— 经验沉淀（"1. 跨模块间接状态翻译" 类的文档同步子集）

**追加教训**："重大功能交付" = 代码 + 单元测试 + e2e + **README 同步**。当前阶段 tcgen05 实施完美，但 README 描述滞后于实现 1 个月。

## What Changes

**核心变更**：

- **修复 `README.md` "已知限制" 章节**：删除 "WMMA / Tensor Core：是 stub"，替换为 "完整实现 tcgen05.mma/ld/st/commit/wait"
- **更新 README "状态" 行**：从 "Phase 10 进行中" → "Phase 10 完成；H5 规划中（参考 post-tcgen05-roadmap.md）"
- **更新 "PTX 指令覆盖"**：从硬编码 "~67%" → 引用 docs/audits 自动统计
- **添加 tcgen05 文档导航**：
  - 指向 `docs/adr/0016-blackwell-only-tcgen05.md`（架构决策）
  - 指向 `docs/dev-process/post-tcgen05-roadmap.md`（未来 H5 规划）
  - 指向 `docs/dev-process/lessons-learned.md`（§19 cross-module state translation post-tcgen05）
- **添加 "已实现功能" 章节**（与"已知限制"并列）：列出 tcgen05、TMA、TMEM、cluster arrive/wait、TcQueue
- **CUDA Toolkit 描述环境自适应**：移除硬编码版本号

**显式标记**：本 change 是 5 个已归档 change 的**文档同步**（不修改功能），引用关系：

- `archive/2026-07-04-implement-wmma-tensor-core-tcgen05/` (Phase 1-3, Fix #1-#14)
- `archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/` (Phase 0 infra, Fix #1-#12)
- `archive/2026-07-04-replace-silent-stub-failures/`
- `archive/2026-07-05-fix-cvt-strategy-actual-split/`（参考资料，本 change 是同模式应用）
- `archive/2026-07-03-docs-readme-rebuild/` (Phase 0 docs 索引已重建，本次仅增量更新 `README.md` 状态描述)

**预估代码改动量**：~15 行修改（`README.md`）+ 0 行代码改动 = **纯文档同步**。

## Capabilities

### New Capabilities

无（不引入新功能）。

### Modified Capabilities

无（README 是描述性文档，behavior-level 零变化）。

## Impact

**受影响的文件**：

| 文件 | 改动类型 | 工作量 |
|------|---------|--------|
| `README.md` | 修改 "状态" / "已知限制" / "PTX 指令覆盖" / "CUDA Toolkit" 4 个章节 | 30min |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 不修改（仅 README 引用） | 0 |
| `docs/dev-process/post-tcgen05-roadmap.md` | 不修改（仅 README 引用） | 0 |

**受影响的受众**：

- **新开发者**（最大影响）— 不会再被 "WMMA 是 stub" 误导
- **架构师** — README 状态描述与实际里程碑对齐
- **贡献者** — 知道 tcgen05 已实施完整，可基于此开发新功能

**风险评估**：

- 🟢 极低风险 — 纯 README.md 文字修改，无代码改动，无构建/测试影响
- 🟢 实施时间 < 1 小时
- 🟢 失败 revert 即 `git revert HEAD`（无副作用）

**Lessons-learned §6 集成**：

- ✅ Checklist E（OpenSpec 实施后）：artifacts git-tracked FIRST（Phase 0）
- ✅ Checklist F（Debt audit）：引用 commit hash（4151268, 35808d6 等）而非文件路径
- ✅ Checklist G（lifecycle）：新建 change，**不 amend** 已归档 change

**Phase 0 强制**：本 change 必须 Phase 0 提交 artifacts，再改 README.md（避免 §6 反模式）。
