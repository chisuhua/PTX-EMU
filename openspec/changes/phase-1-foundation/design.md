## Context

### 现状问题

1. **`build/compile_commands.json` 不存在**：审计 D1 + 多次 grep 实证。根因有二：
   - `CMakeLists.txt:117`（审计声称）设置了 `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` 但 build 输出未生成该文件 —— 可能因 generator 兼容问题（Makefiles vs Ninja）
   - 根目录可能存在错误符号链指向不存在的 build/compile_commands.json
   - **后果**：AGENTS.md 声明的 `lsp_*` 工具链（clangd/clang-tidy/LSP/部分 opencode 工具）全部失效；新人调试成本高；代码补全/跳转/类型推断全失灵

2. **`.github/workflows/` 仅有空 workflow**：grep 实证 `.github/workflows/` 只含 `generate-ptxir.yml`，核心步骤是空循环 TODO（`generate-ptxir.yml:33` `# TODO: implement generate_ptxir as standalone tool`）
   - **后果**：项目实际**无 CI 保障**——所有 PR/Push 不会触发 build/test；Phase 2/3 的正确性修复（membar/fence 实现、god class 拆分）将无回归拦截；债务持续累积

3. **审计 8 项事实错误 + 1 项严重遗漏无 Errata**：审计作为 commit `baa8c4e` 的历史快照保持不变，但以下错误会让未来 commit 的复审对比失真：
   - ThreadContext public 字段：108 → 81（虚增 33%）
   - Symtable 泄漏：5 → 7（漏 `cta_context.cpp:74,104` 2 处）
   - `ptx_visiter` 影响文件：14 → 18（少 4 文件）
   - H2 反向依赖：🔴 H → 🟡 M（4 值枚举 `EXE_STATE` 是合法叶子类型）
   - P0-1 membar 工作量：2 d → 2-3 d（未计 DUAL STATE 修复）
   - Phase 1 顺序：P0-1→P0-2→P0-3 → P0-4→P0-3→P0-2→P0-1（CI 优先）
   - cudaStream_t 性质：漏写 delete → **destroy 是 STUB**（不是漏写）
   - PTX 8.7+ 选项 C 现状被低估（静默失败比编译错误更危险）
   - 严重遗漏：`BarWarpSyncHandler` 仍用 deprecated `warp_state.wbars[]`（per AGENTS.md Phase 5 deferred），是 Phase 2 T1-4 前置依赖

### 目标状态

| # | 现状 | 目标 |
|---|---|---|
| 1 | `build/compile_commands.json` 不存在 | 文件存在且 `wc -l > 0`；`clangd --check` 无解析错误 |
| 2 | `.github/workflows/` 仅有空 TODO workflow | `.github/workflows/build-test.yml` 就位；PR 触发后能在 Actions 页看到 ctest 完整输出 |
| 3 | 审计错误无 Errata | `HEALTH-AUDIT-2026-06-21-ERRATA.md` 发布；baseline `baseline-2026-06-21.log` 存档 |
| 4 | 现有 131 ctest 目标失败无记录 | baseline-2026-06-21.log 含完整 pass/fail/disabled 列表 |

### 关键文件

| 文件 | 当前职责 | 目标职责 |
|------|---------|---------|
| `CMakeLists.txt` | 项目构建配置（顶层） | 确认/添加 `CMAKE_EXPORT_COMPILE_COMMANDS ON` |
| `build/compile_commands.json` | 不存在 | 由 cmake 自动生成，含所有 TU 编译命令 |
| `.github/workflows/build-test.yml` | 不存在 | 新建：PR/Push 触发 ctest |
| `.github/workflows/generate-ptxir.yml` | 核心步骤空循环 TODO | 保持不变（不在本 change 范围） |
| `docs/audits/baseline-2026-06-21.log` | 不存在 | `./scripts/sanity.sh` 完整输出 |
| `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` | 不存在 | 8 项事实错误 + 1 项遗漏的官方勘误 |
| 根目录错误符号链（如有） | 指向不存在的 build/compile_commands.json | 删除 |

### 利益相关方

- **新人开发者**：compile_commands.json 修复后 IDE/调试工具恢复；CI 启用后有自动化验证
- **Phase 2/3 实施者**：所有正确性修复有 CI 拦截；baseline 提供 xfail 比对依据
- **季度复审者**：Errata 让审计作为历史快照可信；baseline 提供客观对比数据

## Goals / Non-Goals

**Goals:**
- 让 `lsp_*` 工具链全工作面恢复（compile_commands.json 修复）
- PR/Push 自动触发 build + test（CI workflow 创建）
- 建立 commit `baa8c4e` 的客观质量基线（baseline 存档）
- 让审计作为历史快照保持不变的同时有可信 Errata（事实错误官方勘误）
- 首次 CI 启用采用 xfail 策略（用户决策 2026-06-22），不阻塞 PR
- 单次 change 完成 T0-1/T0-2/T0-3 三任务合并（基础设施 + CI + baseline 紧密耦合）

**Non-Goals:**
- 不修改审计文档 `HEALTH-AUDIT-2026-06-21.md`（它是历史快照，由 Errata 补充）
- 不实施审计 Phase 2/3 的任何修复（membar/fence、god class 等）—— 本 change 仅建立质量门禁
- 不升级 ANTLR 版本（→ Phase 4 T3-1）
- 不修复 `generate-ptxir.yml` 的 TODO（不在本 change 范围）
- 不迁移 cudart_sim.cpp 双角色（→ Phase 4 T3-2）
- 不实现任何 PTX 指令（→ Phase 2 T1-4 或 Phase 4 T3-3/T3-6）
- 不修改 `tests/CMakeLists.txt:13-31` 的 nvcc 强制策略（→ Phase 4 T3-1 ANTLR 升级时一并处理）

## Decisions

### Decision 1: CI 使用 Ninja generator（而非 Unix Makefiles）

**选择**: `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release`

**理由**:
- 审计 §6.3 M6：当前 `CMakePresets.json` 单一 preset 硬编码 `Unix Makefiles`，增量构建比 Ninja 慢 30-50%
- 663 个 .o 文件在 Ninja 下增量构建时间显著降低（CI runner 时间成本敏感）
- Ninja 严格依赖声明减少 race condition（CI 多次并行 build 稳定）

**替代方案**:
- 保留 Unix Makefiles：增量构建慢；CI 时间成本高
- 引入 `sccache`/`ccache`：额外复杂度，本 change 范围外

**Trade-off**: 需要 CI runner 装 `ninja-build`（apt 包），配置成本 +1 step

### Decision 2: CI CUDA Toolkit 锁版本 `11.4.4`

**选择**: 使用 `Jimver/cuda-toolkit@v0.2.11` action 锁 CUDA `11.4.4`

**理由**:
- 审计 §6.4：当前 CUDA Toolkit 无版本约束；本地路径残留 `/workspace/project/opt/cuda/bin/nvcc`
- 与项目 README 声明的 `cuda (test with 11.4.4)` 对齐
- 避免 CI 升级到 12.x 时 `compute_100` 虚拟架构兼容性破坏

**替代方案**:
- 不锁版本（用 `latest`）：跨版本兼容性风险
- 锁 `12.x`：与本地/项目约定不一致

**Trade-off**: 未来 CUDA 升级需同步修改此 workflow

### Decision 3: xfail 不阻塞 PR（用户决策 2026-06-22）

**选择**: 首次 CI 启用后失败的测试标记为 xfail，不阻塞 PR 合并；单独 issue 跟踪修复

**理由**:
- 避免一次性过载：131 个 ctest 目标可能多个 baseline 失败，全阻塞 PR 会停滞项目
- baseline 已有存档（`baseline-2026-06-21.log`），失败原因可追溯
- xfail 渐进修复：每个失败单独 issue 跟踪，与 Phase 2/3 修复并行推进

**替代方案**:
- 立即修复所有失败再启用 CI：工作量 1-2 周（远超本 change 24h 时间窗）
- 不启用 CI 继续手动 sanity.sh：失去 CI 拦截价值

**Trade-off**: 需要 ctest 支持 `MARK xfail` 或外部 mechanism（详细见 Implementation）

### Decision 4: Errata 单独文档而非修改审计

**选择**: 创建 `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` 列出 8 项事实错误 + 1 项遗漏

**理由**:
- 审计是 commit `baa8c4e` 的历史快照，保持不变保证未来 `git checkout baa8c4e` 可重现
- Errata 是官方补充机制（类似 RFC errata），与 git 历史解耦
- 复审者可读 Errata + 审计双文档对比，避免单一文档被静默修改

**替代方案**:
- 直接修改审计文档添加"已修正"标记：破坏历史快照完整性；git diff 难以追踪修正
- 删除审计文档：失去历史基线

**Trade-off**: 需要双文档同步维护（Errata 列出错误编号 + 实际值，审计保持原始值）

### Decision 5: compile_commands.json 修复优先于 CI 创建

**选择**: T0-1 → T0-3 → T0-2 顺序执行（compile_commands → baseline → CI）

**理由**:
- T0-1 让本地能复现 CI 失败（CI runner 失败时可在本地编译排查）
- T0-3 baseline 在 CI 启用前建立，避免"红"无对比
- T0-2 是最终启用步骤（依赖前两个完成）

**替代方案**:
- T0-2 先于 T0-1/T0-3：CI 失败无法在本地复现；修复周期长
- 三任务并行：无依赖关系的好处；但 CI 启用后第一波"红"无 baseline 比对

**Trade-off**: 串行执行总时间 ≈ 24h（5min + 0.5d + 0.5-1.5d），与并行差异不显著

### Decision 6: baseline 输出含 ctest pass/fail/disabled 三态

**选择**: `docs/audits/baseline-2026-06-21.log` 包含 `./scripts/sanity.sh` 完整输出 + 手动汇总 ctest 状态

**理由**:
- 审计 §4.1：131 ctest 目标（3 Disabled）—— baseline 必须显式标注 disabled 测试
- 未来复审对比时，"新增 failure" vs "baseline 已 disabled" 必须可区分
- sanity.sh 输出含 build warnings（如 cudart_sim.cpp:933 cerr 替代 logger），baseline 存档后 future warning 变化可对比

**替代方案**:
- 仅存档 ctest pass/fail：丢失 build warnings 信息
- 仅存档 sanity.sh 输出（无 ctest 汇总）：需要人工解析才能知道哪些是 disabled

**Trade-off**: baseline 文件可能较大（10-50 MB），但 git LFS 或简单 .gz 可解决

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解 |
|---|:---:|:---:|---|
| **R1**: 首次 CI 启用大量 baseline 失败 | 🟡 高 | 中 | xfail 不阻塞 PR；baseline 对比；分批修复（Phase 2/3 同步） |
| **R2**: NVIDIA CUDA 在 GitHub Actions 拉取慢 | 🟡 中 | 低 | 用 `Jimver/cuda-toolkit@v0.2.11` action 缓存；预留 10 min timeout |
| **R3**: ANTLR Java 在 CI runner 缺失 | 🟢 低 | 中 | apt-get install `default-jre`；已在 env.sh 检查 |
| **R4**: ASan 在 CI 中暴露大量泄漏 | 🟡 中 | 中 | 首次跑不阻塞；按文件分批修复（Phase 2 T1-1/T1-2 同步） |
| **R5**: compile_commands.json 修复后 IDE 配置需更新 | 🟢 低 | 低 | 文档说明 `.clangd` 或 `compile_flags.txt` 配置位置 |
| **R6**: baseline 日志含敏感信息（绝对路径/邮箱） | 🟢 低 | 低 | CI 输出已由 GitHub Actions 自动过滤；baseline 仅含相对路径 |
| **R7**: Errata 文档未来随审计多次复审而膨胀 | 🟢 低 | 低 | Errata 用日期分节（2026-06-21 v1），未来复审追加新 Errata 文档而非修改 |
| **R8**: xfail 机制 ctest 原生不支持 | 🟡 中 | 中 | 用 Catch2 `[!mayfail]` tag 或在 CTestCustom 中标记；详细见 Implementation |
| **R9**: 根目录错误符号链被误删导致 build 失效 | 🟢 低 | 🟡 中 | 先 `ls -la` 确认符号链指向；删除前 cp 备份 |
| **R10**: CI 中 ctest 跑通但本地失败（环境差异） | 🟡 中 | 🟡 中 | CI 输出包含 cmake build 命令；可在本地精确复现；逐步排查 |

### Trade-off 总览

- **运维成本 vs 质量收益**：CI 维护成本（每次 CUDA/ANTLR 升级需同步 workflow）vs 自动化回归拦截价值 —— **收益远大于成本**
- **xfail 短期便利 vs 长期技术债**：xfail 不阻塞 PR 降低短期摩擦 vs 失败测试累积可能让 xfail 列表膨胀 —— **需建立 xfail → issue → 修复闭环**
- **baseline 文件大小**：存档完整 sanity 输出（含 build warnings）vs git 仓库体积 —— **用 .gitattributes + LFS 或 git-lfs track**
- **Errata 双文档 vs 单文档**：审计不变 + Errata 补充 vs 直接修改审计 —— **保持历史快照完整性优先**

## Migration Plan

### 部署步骤

本 change 不涉及生产部署（CI 配置 + 文档），仅是仓库内变更：

```bash
# 1. 在 worktree 中执行
git worktree add ../ptx-emu-foundation -b feat/phase-1-foundation

# 2. T0-1：修复 compile_commands.json
cd ../ptx-emu-foundation
# 检查/修改 CMakeLists.txt（如需要）
grep -n "CMAKE_EXPORT_COMPILE_COMMANDS" CMakeLists.txt
# 删除根目录错误符号链
ls -la compile_commands.json  # 确认是错误符号链
rm compile_commands.json  # 删除（如有）
# 重新生成
cmake -S . -B build
ls -la build/compile_commands.json  # 应存在
git add CMakeLists.txt build/compile_commands.json
git commit -m "fix(build): enable compile_commands.json generation (T0-1)"

# 3. T0-3：存档 baseline + 创建 Errata
./scripts/sanity.sh 2>&1 | tee docs/audits/baseline-2026-06-21.log
# 手动汇总 ctest 三态
ctest -N 2>&1 | grep -E "Test #|Total Tests" >> docs/audits/baseline-2026-06-21.log
# 创建 Errata 文档（8 项事实错误 + 1 项遗漏）
# 详见 tasks.md Task 3
git add docs/audits/baseline-2026-06-21.log docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md
git commit -m "docs(audit): archive baseline + publish errata (T0-3)"

# 4. T0-2：创建 CI workflow
# 创建 .github/workflows/build-test.yml
# 详见 tasks.md Task 4
git add .github/workflows/build-test.yml
git commit -m "ci: add build-test workflow (T0-2)"

# 5. 推 PR 测试 CI
git push origin feat/phase-1-foundation
gh pr create --title "phase-1-foundation: enable quality gates" \
             --body "见 openspec/changes/phase-1-foundation/proposal.md"
```

### 回滚策略

如 CI workflow 有严重问题：

```bash
# 1. 删除 workflow（保留分支历史）
git revert HEAD  # 撤销 CI commit

# 2. 或临时禁用
# 在 .github/workflows/build-test.yml 开头添加:
#   if: false  # 临时禁用
git commit -am "ci: temporarily disable build-test workflow"
```

如 compile_commands.json 修复有副作用：

```bash
# 1. 撤销 CMakeLists.txt 修改（如有）
git checkout CMakeLists.txt
# 2. 让 build/compile_commands.json 由 cmake 重新生成
rm -rf build && cmake -S . -B build
```

如 Errata 文档需修订：

```bash
# 直接修改 docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md
# 不影响审计快照
```

### 渐进式启用

T0-1 → T0-3 → T0-2 顺序执行（compile_commands → baseline → CI 启用）：
- T0-1 完成后即可让 IDE/LSP 恢复（本地收益）
- T0-3 完成后 baseline 存档（CI 启用前的客观对比）
- T0-2 是 CI 实际启用步骤（PR 触发后开始拦截）

## Open Questions

1. **Q1**: xfail 机制在 ctest 中如何标记？
   - 选项 A：Catch2 `[!mayfail]` tag（catch_amalgamated.hpp 支持）
   - 选项 B：CTestCustom.cmake.in 配置 `WILL_FAIL TRUE`
   - 选项 C：外部脚本解析 ctest 输出
   - **倾向**：A（最轻量）；需验证 Catch2 是否支持
   - **决策时机**：T0-2 Task 4.2

2. **Q2**: baseline 日志是否需要 git-lfs？
   - 选项 A：直接提交（文件 10-50 MB 可能膨胀仓库）
   - 选项 B：git-lfs track `*.log`
   - 选项 C：仅存档失败摘要 + 链接外部存储
   - **倾向**：B（保持简洁，未来 baseline 也用同样策略）
   - **决策时机**：T0-3 Task 3.1

3. **Q3**: CI CUDA 版本与本地不一致时如何处理？
   - 现状：本地 CUDA 11.4.4；CI 锁 11.4.4
   - 如未来本地升级到 12.x，CI 需同步
   - **决策时机**：CUDA 升级时（不在本 change 范围）

4. **Q4**: Errata 文档如何处理未来季度复审发现的新错误？
   - 选项 A：每个复审周期新建 Errata v2/v3
   - 选项 B：同一文档追加"v2 (2026-09-21)"节
   - **倾向**：A（保持每次复审独立 Errata，便于 git 历史追踪）
   - **决策时机**：首次季度复审（2026-09-21）时