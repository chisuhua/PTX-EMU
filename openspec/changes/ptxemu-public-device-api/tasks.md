## 1. Pre-Implementation 准备 (per ptx-lessons-learned §4 + §7)

- [ ] 1.1 MUST: 验证基线 worktree `.worktrees/phase2-baseline` 编译通过 (`cmake -S . -B build && cmake --build build`),耗时 15-20min
- [ ] 1.2 MUST: 跑基线 ctest `cd build && ctest` 全部 PASS (per ptx-lessons-learned §4 实测验证)
- [ ] 1.3 MUST: 跑 Metis pre-impl review 审计 4 OpenSpec artifacts (proposal.md/design.md/specs/{public-device-api,ptxemu-core-library,statement-ir-public,ci-drift-check}/spec.md),输出 GO / ⚠️ CONDITIONAL / ❌ NO-GO 决策
- [ ] 1.4 应用 Metis MUST-RESOLVE 列表 (若有),修订 artifacts 重审直至 ⚠️→GO 或 ✅ 确认
- [ ] 1.5 NOTE: 4 artifacts 范围数字一致性按 Checklist J 校验 (proposal.md 涉及的 openspec 文件清单 + design.md Migration Plan Phase 列表 + 4 specs 涉及的 capability 列表 + 本 tasks.md Phase 列表四者交叉一致)
- [ ] 1.6 MUST: `git add openspec/changes/ptxemu-public-device-api/` + commit "docs(openspec): ptxemu-public-device-api initial 4 artifacts" (per Checklist E artifacts-first 纪律)
- [ ] 1.7 创建 `feat/ptxemu-public-device-api` 分支从 `origin/main` (738b412c) HEAD
- [ ] 1.8 NOTE: 严禁基于 commit `c2038a93` 或更早 (HSK-8 spec §3 Risk 4 Mitigation - 保留 `g_cpptlm_bridge` 引用)

## 2. Phase 0 闭包净化 (per design.md Decision 1 + spec/statement-ir-public §0) ✅ **COMPLETE**

- [x] 2.1 创建 `include/ptxemu/ir/` 目标目录 (空, 等待 Phase 1 填充) — *deferred to Phase 1*
- [x] 2.2 审计污染点 A: `grep -rn "operand_phy_addr\|setPhyAddr\|invalidatePhyAddr" src/ include/` 列出 8 active sites (per Metis audit) ✅
- [x] 2.3 MUST: 净化污染点 A — `include/ptx_ir/operand_context.h:59` 移除 `mutable void *operand_phy_addr`,改用 ThreadContext-local index-keyed cache ✅ **(Phase 0.3a-d 4 commits)**
- [x] 2.4 验证污染点 A 净化 — `git grep "operand_phy_addr" include/ptxemu/` 为 0; `thread_context.cpp:407` 直接读 cache 不再 mutate OperandContext ✅
- [x] 2.5 审计污染点 B: `grep -rn 'InstructionState state' src/ include/` 验证 0 active readers/writers (per Metis audit) ✅
- [x] 2.6 MUST: 净化污染点 B — `include/ptx_ir/statement_context.h:306` 移除 `InstructionState state` 字段 ✅ **(commit 586ea14f)**
- [x] 2.7 验证污染点 B 净化 — `git grep "InstructionState state =" include/ptx_ir/` 为 0 ✅
- [x] 2.8 MUST: 删除 dead code `BarWarpSyncInstr::reconvergenceLabel` + writer `ptx_visitor_barrier.cpp:119` ✅ **(commit 602bfc30 + 359579ec for test fixture fix)**
- [x] 2.8a 验证 reconvergenceLabel 0 reader ✅
- [x] 2.9 全量 ctest 验证 Phase 0 净化零回归 ✅ **(246/246 PASS 多次: 33.37s / 37.57s / 35.82s / 35.02s / 29.95s)**
- [x] 2.10 commits 完成: d8b6ca56 (0.3a) / a6c9bdaf (0.3b) / 66ca4875 (0.3c) / 1fb15d89 (0.3d) / 586ea14f (B) / 602bfc30+359579ec (0.8+0.8b)

## 3. Phase 1 5 文件晋升 + namespace (per design.md Decision 1)

- [ ] 3.1 复制 `include/ptx_ir/statement_context.h` → `include/ptxemu/ir/statement.h` (正文改名 `StatementContext` → `Statement`,加 `namespace ptxemu { namespace ir { ... } }` 包裹)
- [ ] 3.2 复制 `include/ptx_ir/operand_context.h` → `include/ptxemu/ir/operand_context.h` (同名 namespace 包裹)
- [ ] 3.3 复制 `include/ptx_ir/ptx_types.h` → `include/ptxemu/ir/ptx_types.h` (namespace 包裹)
- [ ] 3.4 复制 `include/ptxsim/execution_types.h` → `include/ptxemu/ir/execution_types.h` (仅暴露 `InstructionState` enum, `EXE_STATE`/`BAR_TYPE`/`CTAId` 通过前向声明隔离或保留 internal)
- [ ]  3.5 复制 `include/ptx_ir/ptx_qualifier.def` → `include/ptxemu/ir/ptx_qualifier.def`
- [ ] 3.6 复制 `include/ptx_ir/ptx_op.def` → `include/ptxemu/ir/ptx_op.def`
- [ ] 3.7 旧 `include/ptx_ir/` 路径改为 forwarding header: 内容 ` #pragma once + #include <ptxemu/ir/statement.h> + namespace ptx_ir = ptxemu::ir; `
- [ ] 3.8 验证 5 文件自洽 — `g++ -fsyntax-only -I include -I include/ptxemu/ir include/ptxemu/ir/statement.h` 0 错误
- [ ] 3.9 grep 全部 `include/ptx_ir/` 调用方 (`grep -rn 'include.*ptx_ir/' src/ tests/ docs/`),记录 32 callsites 清单(暂不修,等 release 周期)
- [ ] 3.10 跑全量 ctest 验证 Phase 1 零回归 (若失败 → 立即 `git revert HEAD`, 不混入后续)
- [ ] 3.11 commit "refactor(ptxemu): Phase 1 5 文件晋升 + namespace ptxemu::ir (forwarding header 兼容)"

## 4. Phase 2 device_api.h + impl + 库目标 (per spec/public-device-api + design.md Decision 2)

- [ ] 4.1 创建 `include/ptxemu/device_api.h` (~200 行) 含 5 项契约内容 (namespace ptxemu + IPtxEmuDevice + 4 DTO + 2 factory + VERSION)
- [ ] 4.2 抽取 S1 facade.cc (CppTLM 仓 archive `b68abe6f`) 12 callsites 1:1 映射为虚方法
- [ ] 4.3 MUST: `PTXEMU_API_VERSION 1` + static_assert 自检
- [ ] 4.4 创建 `src/ptxemu/device_api_impl.cc` (~400 行) 实现 IPtxEmuDevice 适配层 (调 PTX-EMU 内部 SMContext/WarpContext)
- [ ] 4.5 创建 `src/ptxemu/cmake/ptxemu_core.cmake` (或直接 add_library in root) — 显式源清单 + `target_include_directories(ptxemu_core PUBLIC include/ptxemu PRIVATE include/ptx_ir include/ptxir include/ptxsim src/ptxsim src/cudart)`
- [ ] 4.6 更新 root `CMakeLists.txt` + `src/CMakeLists.txt` 包含 ptxemu_core 库目标
- [ ] 4.7 跑 `cmake -S . -B build` 配置 + `cmake --build build --target ptxemu_core` 编译
- [ ] 4.8 跑全量 ctest 验证 Phase 2 零回归 (新增 1 个 device_api 自身的简单 smoke 测试)
- [ ] 4.9 commit "feat(ptxemu): Phase 2 public device_api.h + ptxemu_core library target"

## 5. Phase 3 隔离 + install (per spec/ptxemu-core-library §PROJECT_IS_TOP_LEVEL + design.md Decision 2)

- [ ] 5.1 NOTE: 不要修改现有 `CMakeLists.txt` 顶层结构, 仅追加 option + if 块
- [ ] 5.2 在 root `CMakeLists.txt` 顶部添加 `option(PTXEMU_BUILD_TESTING "Build PTX-EMU tests" OFF)` + `if(PROJECT_IS_TOP_LEVEL OR PTXEMU_BUILD_TESTING)` 包裹 `enable_testing() + add_subdirectory(tests)`
- [ ] 5.3 添加 `install(TARGETS ptxemu_core EXPORT ptxemu_core_targets ARCHIVE DESTINATION lib INCLUDES DESTINATION include)` 规则
- [ ] 5.4 配置 `-DPTXEMU_BUILD_TESTING=OFF` 验证 CppTLM 风格消费 (`cmake -S . -B build-off` 测试 `install` 而不 `enable_testing`)
- [ ] 5.5 配置默认 (无 flag) 验证 PTX-EMU 顶层构建仍跑测试 (`ctest -L unit` 抽样 5 个 case)
- [ ] 5.6 跑全量 ctest 验证 Phase 3 零回归
- [ ] 5.7 commit "build(cmake): Phase 3 PROJECT_IS_TOP_LEVEL 隔离 + PTXEMU_BUILD_TESTING option + install 规则"

## 6. Phase 4 CI drift_check workflow (per spec/ci-drift-check)

- [ ] 6.1 创建 `.github/workflows/drift_check.yml` (per spec §Requirement 1)
- [ ] 6.2 验证 trigger: 修改 `include/ptxemu/**` 或 `include/ptx_ir/**` 自动 run
- [ ] 6.3 验证 drift_check 不依赖 CppTLM submodule (HSK-6 单向消费关系)
- [ ] 6.4 NOTE: Phase 2 PR 不含 `consumer_smoke` (HSK-9 准入, per HSK-8 ack 决策点 2)
- [ ] 6.5 跑全量 ctest 验证 Phase 4 (改动最小, 仅新增 yml)
- [ ] 6.6 commit "ci: Phase 4 add drift_check workflow (header version + virtual method count guard)"

## 7. Phase 5 文档同步 (per ptx-lessons-learned §21 + AGENTS.md)

- [ ] 7.1 创建 `include/ptxemu/AGENTS.md` (新目录)
- [ ] 7.2 更新 `include/ptx_ir/AGENTS.md` 标注 deprecated (forwarding header only)
- [ ] 7.3 同步 root `AGENTS.md` 顶层 HSK 链路段追加 HSK-8 + 跨仓协调顺序
- [ ] 7.4 更新 `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` (新建) 记录实施进度
- [ ] 7.5 NOTE: 不修改根 README.md (本 PR 是 protocol change, 根 README 同步是 archive 阶段)
- [ ] 7.6 commit "docs: Phase 5 sync AGENTS.md + audit append (HSK-8 progress)"

## 8. PR 开 + 验证 + 合入 (跨仓协调顺序 Step 2-4)

- [ ] 8.1 创建 `feat/ptxemu-public-device-api` 分支并 push 到 origin
- [ ] 8.2 NOTE: 不 commit 任何 openspec/changes/ 内容到 PR (artifacts 已在 init commit 1.6 完成, PR 仅包含实施 commits)
- [ ] 8.3 跑 `./scripts/sanity.sh` 验证 PTX-EMU 自身健康检查全绿
- [ ] 8.4 跑 `ctest` 全量测试 PASS (含 Phase 8 ctest 在 worktree 内 baseline 对比)
- [ ] 8.5 跑 drift_check workflow (本地: `act -j drift-check` 或 GitHub PR UI)
- [ ] 8.6 开 PR to `main`, tag @ptx_emu_owner @ptx_emu_architecture_team review
- [ ] 8.7 PR 描述含 7 条验收清单 (per HSK-8 spec §"CppTLM 端接受条件" 5 条 + `OPENSPEC deliverable` 2 条)
- [ ] 8.8 PR review 通过后 merge to main (squash or rebase, per project 约定)
- [ ] 8.9 推 ack 更新到 issue #22: "PTX-EMU Phase 2 PR 合入 at <hash>" + DRIFT_CHECK PASS evidence
- [ ] 8.10 update HSK-PROTOCOL-NOTES.md 追加 HSK-8 行 (split: HSK-6 → HSK-8)

## 9. 跨仓协调 + 清理 (HSK-8 spec §"跨仓协调顺序" Step 5 + 6)

- [ ] 9.1 通知 CppTLM owner Phase 2 PR 合入完成 (issue #22 评论 hash 链接)
- [ ] 9.2 等 CppTLM owner 开 Phase 3 bump PR (submodule pin + add_subdirectory + 桥接残留簇删除)
- [ ] 9.3 CppTLM bump PR 合入后验证 PTX-EMU 仓 side unchanged (跨仓协调反向验证)
- [ ] 9.4 release 周期后 (建议 1 release) 删除 `include/ptx_ir/` forwarding header
- [ ] 9.5 archive change: `openspec archive ptxemu-public-device-api` 创建 archive commit

## MUST/NOTE 总结

**MUST (硬约束)**:
- 1.1 + 1.2 (基线 worktree 验证)
- 1.3 + 1.4 (Metis pre-impl review)
- 1.6 (artifacts-first commit 纪律)
- 1.8 (PR base 约束)
- 2.3 + 2.6 (Phase 0 净化 2 污染点)
- 2.8 (dead code 删)
- 2.9 + 3.10 + 4.8 + 5.6 (各 Phase 后 ctest 零回归验证)
- 4.3 (PTXEMU_API_VERSION 静态断言)

**NOTE (软提示)**:
- 1.5 (4 artifacts 一致性)
- 1.7 (分支创建时机)
- 5.1 (CMakeLists 改动范围限制)
- 6.4 (consumer_smoke 延后)
- 7.5 (根 README 不同步)
- 8.2 (PR 不含 artifacts)
- 9.4 + 9.5 (release 后清理)
