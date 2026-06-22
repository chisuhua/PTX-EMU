## Why

PTX-EMU 当前**没有项目级质量门禁**——`build/compile_commands.json` 不存在导致 AGENTS.md 声明的 `lsp_*` 工具链全部失效；`.github/workflows/` 仅有 `generate-ptxir.yml` 且核心步骤是空循环 TODO（`generate-ptxir.yml:33`），PR/Push 不会触发任何 ctest 拦截；2026-06-21 健康审计（commit `baa8c4e`）存在 8 处事实错误但无 Errata 修正，导致未来复审对比的基线数据失真。这三个问题形成**根因级债**：Phase 2/3 的所有正确性修复（membar/fence、god class 拆分）都将无回归拦截，CI 启用后第一波"红"无法判断是真 bug 还是 baseline 偏差。本次 change 是 roadmap Phase 1 的执行载体（T0-1/T0-2/T0-3），建立 CI + baseline + Errata 三件套作为后续 Phase 的质量门禁根因。

## What Changes

- **修复 `compile_commands.json` 生成**（T0-1）：检查 `CMakeLists.txt` 是否设置 `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)`；清理根目录任何错误符号链；运行 `cmake -S . -B build` 让 build 自然生成 `build/compile_commands.json`。修复后 LSP/clang-tidy/IDE 全工作面恢复。
- **创建 `.github/workflows/build-test.yml`**（T0-2）：基于现有 `generate-ptxir.yml` 模板，新建 PR/Push 触发 workflow：装 CUDA Toolkit（锁版本 `11.4.4` 与本地一致）+ ANTLR Java 依赖 + `. env.sh` + `cmake -G Ninja` + `cmake --build build` + `ctest --output-on-failure -E "Disabled"`。首次启用 CI 后采用 xfail 策略（用户决策 2026-06-22）：失败测试标记为 xfail 不阻塞 PR，单独 issue 跟踪修复。
- **存档 baseline + 发布审计 Errata**（T0-3）：跑 `./scripts/sanity.sh` 完整输出存档到 `docs/audits/baseline-2026-06-21.log`；创建 `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` 列出 8 项事实错误 + 1 项严重遗漏（BarWarpSyncHandler 仍用 deprecated wbars[]）。Errata 是审计作为 commit `baa8c4e` 历史快照的官方勘误表，未来季度复审有可信对比基线。

## Capabilities

### New Capabilities

- `quality-gate-infrastructure`: 项目质量门禁根因基础设施（CI pipeline + compile_commands + baseline archive）—— 包含 CI workflow 配置、compile_commands.json 修复、sanity baseline 存档三个运维能力；让 PR 有回归拦截、新人有 LSP/IDE 调试能力、未来 Phase 修复有 baseline 对比依据
- `audit-correction`: 健康审计文档的事实错误官方勘误机制 —— 通过 `HEALTH-AUDIT-2026-06-21-ERRATA.md` 列出所有已验证的事实错误（数值虚增/漏报/严重度过高/顺序错位/destroy 性质/PTX 8.7+ 静默失败）；让审计作为 commit `baa8c4e` 历史快照保持不变的同时，未来复审有可信对比

### Modified Capabilities

无（首次引入 quality-gate 相关 specs；审计文档本身不改，作为历史快照保留，由 Errata 补充）

## Impact

| 类别 | 影响 |
|------|------|
| `CMakeLists.txt` | **可能修改**：添加 `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)`（如缺失） |
| `build/compile_commands.json` | **新增**：由 cmake 自动生成（5 分钟修复） |
| 根目录错误符号链（如有） | **删除**：`compile_commands.json` 错误符号链 |
| `.github/workflows/build-test.yml` | **新增**：CI workflow 定义（PR/Push 触发 ctest） |
| `.github/workflows/generate-ptxir.yml` | **保持不变**：仅作为模板参考，核心步骤仍为 TODO（不在本 change 范围） |
| `docs/audits/baseline-2026-06-21.log` | **新增**：`./scripts/sanity.sh` 完整输出存档（含 build warnings + ctest pass/fail/disabled 列表） |
| `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` | **新增**：8 项事实错误 + 1 项严重遗漏的官方勘误表 |
| `docs/roadmap/` | **新增**：3 个 phase 文件（已在 `docs/roadmap/README.md` + `phase-1-foundation.md` + `phase-2-critical-debt.md` + `phase-3-structural-debt.md` 中建立）—— 本 change 不修改，但作为依赖项引用 |
| CUDA Toolkit 版本约束 | **新增**（CI）：锁 `11.4.4`（与本地 `configs/ampere_a100.json` 默认架构 `compute_100` 不直接相关，但避免 CI 漂移） |
| `Jimver/cuda-toolkit` GitHub Action | **新增依赖**（CI）：用于在 runner 装 CUDA |
| `actions/checkout@v4` + `default-jre` apt 包 | **新增依赖**（CI）：checkout + ANTLR Java 运行时 |
| 现有 131 个 ctest 目标 | **可能新增 xfail 标记**：CI 首次启用后失败的测试（用户决策：不阻塞 PR，单独 issue 跟踪） |

## References

- 审计基线：`docs/audits/HEALTH-AUDIT-2026-06-21.md`（commit `baa8c4e`，2026-06-21）
- Oracle 深度审查：`ses_1155c96adffeBJ5SwSGBXUpgYK`（2m 36s，7 个 Q&A 验证事实错误）
- Roadmap：`docs/roadmap/README.md` + `phase-1-foundation.md`（本 change 的设计输入）
- 项目 AGENTS.md：`/AGENTS.md`（`lsp_*` 工具链声明依赖 `compile_commands.json`）
- 现有 workflow 模板：`.github/workflows/generate-ptxir.yml`（含 TODO `generate-ptxir.yml:33`）
- 用户决策（2026-06-22）：
  - 接受 Tier 0/1 优先级调整（CI 优先于 membar）
  - PTX 8.7+ 占位去留：A + PTX_WARN（在 Phase 3 T2-4 实施）
  - CI 首次失败处置：xfail 不阻塞 PR
  - 修正审计文档 8 处事实错误
- 相关 ADR：无（CI/baseline/Errata 均为运维，不涉及架构决策）
- 相关 Skill：`ptx-debug`（baseline 失败时定位）、`three-mode-testing`（sanity.sh 验证）