## 1. Baseline & Preparation

- [ ] 1.1 Create baseline worktree: `git worktree add .worktrees/ptxir-baseline main` from commit `5592886d` (current main HEAD)
- [ ] 1.2 Verify baseline ptxir sub-build succeeds: `cmake --build .worktrees/ptxir-baseline/build --target ptxir` (skip if OOM on 2-core system; document as expected limitation)
- [ ] 1.3 Create implementation worktree: `git worktree add .worktrees/ptxir-compliance -b ptxir-format-compliance main`
- [ ] 1.4 Review gap analysis: read `docs/architecture/ptxir-serialization-gaps-gap-analysis.md` and verify all 9 G-items + 5 D-items understood
- [ ] 1.5 Review ADR-0023 7 decisions: read `docs/adr/ADR-0023-ptxir-binary-format.md` §决策内容 and verify writer/reader code map

## 2. Phase 1: Reader Instruction Coverage (G9 修复)

**目标**: 补全 Reader 12 种缺失指令类型 + 移除 default 静默跳过
**Commit**: `feat(ptxir): complete reader instruction coverage (24/24)`
**依赖**: 无
**回退策略**: `git revert <commit>` → reader 回到 12/24 状态

- [ ] 2.1 在 `src/ptx_ir/ptxir_reader.cpp` 中添加 `case S_MEMBAR`: 读取 qualifiers (u8 count + u16[]) → 构造 `MembarInstr` → `stmt.data = instr`
- [ ] 2.2 添加 `case S_FENCE`: 读取 qualifiers → 构造 `FenceInstr`
- [ ] 2.3 添加 `case S_REDUX_SYNC`: 读取 qualifiers + operands (u8 count + u32[]) → 构造 `ReduxSyncInstr`（参考 writer `write_redux_sync`）
- [ ] 2.4 添加 `case S_MBARRIER`: 读取 qualifiers + operands → 构造 `MbarrierInstr`
- [ ] 2.5 添加 `case S_CALL`: 读取 qualifiers + operands → 构造 `CallInstr`
- [ ] 2.6 添加 `case S_VOTE`: 读取 qualifiers + operands → 构造 `VoteInstr`
- [ ] 2.7 添加 `case S_SHFL`: 读取 qualifiers + operands → 构造 `ShflInstr`
- [ ] 2.8 添加 `case S_ATOM`: 读取 qualifiers + operands → 构造 `AtomInstr`
- [ ] 2.9 添加 `case S_TEXTURE`: 读取 qualifiers + operands → 构造 `TextureInstr`
- [ ] 2.10 添加 `case S_SURFACE`: 读取 qualifiers + operands → 构造 `SurfaceInstr`
- [ ] 2.11 添加 `case S_REDUCTION`: 读取 qualifiers + operands → 构造 `ReductionInstr`
- [ ] 2.12 添加 `case S_PREFETCH`: 读取 qualifiers + operands → 构造 `PrefetchInstr`
- [ ] 2.13 添加 `case S_CP_ASYNC`: 读取 qualifiers + operands → 构造 `CpAsyncInstr`
- [ ] 2.14 添加 `case S_ABI_DIRECTIVE`: 无字段（参考 writer `write_abi_directive`）→ 构造空 `AbiDirective`
- [ ] 2.15 添加 `case S_PREDICATE_PREFIX`: 读取 qualifiers → 构造 `PredicatePrefix`
- [ ] 2.16 修改 `default` 分支：抛 `std::runtime_error("Unknown StatementType: " + std::to_string(type))` 替代静默跳过
- [ ] 2.17 编译验证: `cmake --build .worktrees/ptxir-compliance/build --target ptxir_reader` (应成功)
- [ ] 2.18 静态检查: 用 `grep` 确认 `default` 分支不再有 `// skip` 注释或 `stmt.data = instr;` 静默赋值
- [ ] 2.19 Commit Phase 1 (单独 commit): `git add src/ptx_ir/ptxir_reader.cpp && git commit -m "feat(ptxir): complete reader instruction coverage (24/24)"`
- [ ] 2.20 立即验证: `cmake --build build && ctest -R "ptxir|ptxir_serialization" -V` (如 2 核 OOM，至少验证 ptxir 库构建)

## 3. Phase 2: Format Contract Alignment (D1-D5 修复)

**目标**: Writer/Reader 实现完全符合 ADR-0023 Decision 1
**Commit 1**: `refactor(ptxir): align writer with ADR-0023 Section TOC layout`
**Commit 2**: `refactor(ptxir): align reader with ADR-0023 Section TOC layout`
**依赖**: Phase 1 完成
**回退策略**: 单 commit revert；writer/reader 各一次 revert

- [ ] 3.1 (Commit 1) 修改 `PtxirWriter::write_header()`: 在写入 24 字节 header 后，预留 `section_count * 6` 字节 TOC 空间（用 seek + pad），更新 `section_count` 字段
- [ ] 3.2 (Commit 1) 实现 `PtxirWriter::write_toc_entries()`: 按 section 写入顺序（REGDECL → KERNEL → STRING_TABLE）写入 TOC 条目
- [ ] 3.3 (Commit 1) 修改 `PtxirWriter::write()` 调用顺序: `pre_pass → write_header → write_toc_entries → write_regdecl_section → write_kernel_section → write_string_table → backfill_header_offsets`
- [ ] 3.4 (Commit 1) 实现 `PtxirWriter::backfill_header_offsets()`: 回填 `string_table_offset` (offset 12-15) 和 `string_table_size` (offset 16-19)
- [ ] 3.5 (Commit 1) 实现 `PtxirWriter::write_regdecl_section()`: 写入操作数表（从 `reg2id_` 推导）；如 `reg2id_` 为空则不写入 section，但 TOC 仍占位
- [ ] 3.6 (Commit 1) 编译验证: `cmake --build .worktrees/ptxir-compliance/build --target ptxir_writer` (应成功)
- [ ] 3.7 (Commit 1) Commit: `git add src/ptx_ir/ptxir_writer.cpp && git commit -m "refactor(ptxir): align writer with ADR-0023 Section TOC layout"`
- [ ] 3.8 (Commit 2) 修改 `PtxirReader::read_header()`: 读取 TOC 条目到 `std::vector<PtxirSectionTOC>` 成员
- [ ] 3.9 (Commit 2) 实现 `PtxirReader::read_toc_entries()`: 从 header 后读取 `section_count` 个 TOC 条目
- [ ] 3.10 (Commit 2) 修改 `PtxirReader::read()`: 按 TOC 类型分派到 `read_regdecl_section` / `read_kernel_section` / `read_string_table`
- [ ] 3.11 (Commit 2) 修改 `PtxirReader::read_string_table()`: 改为通过 TOC 定位（不再 seek `sizeof(PtxirHeader)`）
- [ ] 3.12 (Commit 2) 实现 `PtxirReader::read_regdecl_section()`: 从 REGDECL section 重建操作数表
- [ ] 3.13 (Commit 2) 在 `read()` 中添加: 检测重复 TOC type → 抛异常；TOC offset 越界 → 抛异常；未知 section type → 抛异常
- [ ] 3.14 (Commit 2) 编译验证: `cmake --build .worktrees/ptxir-compliance/build --target ptxir_reader` (应成功)
- [ ] 3.15 (Commit 2) Commit: `git add src/ptx_ir/ptxir_reader.cpp && git commit -m "refactor(ptxir): align reader with ADR-0023 Section TOC layout"`
- [ ] 3.16 Phase 2 整体验证: 完整 build ptxir 库 + 现有测试（如果存在）通过

## 4. Phase 3: Test Suite & Tooling Completion (G1, G3, G4 修复)

**目标**: roundtrip 测试 + generate_ptxir() + load_ptxir(apply_cfg) 完整路径
**Commit 1**: `test(ptxir): add roundtrip unit tests for all 24 instruction types`
**Commit 2**: `feat(ptxir): add generate_ptxir() offline tool`
**Commit 3**: `feat(ptxir): complete load_ptxir(apply_cfg=true) path with CFGBuilder integration`
**依赖**: Phase 2 完成
**回退策略**: 单 commit revert

- [ ] 4.1 (Commit 1) 创建 `tests/unit/test_ptxir_serialization.cpp`: 引入 Catch2 header + `ptxir_serialization.h` + `statement_factory.h`
- [ ] 4.2 (Commit 1) 添加测试用例: `TEST_CASE("Roundtrip: BranchInstr")` — 构造 S_BRA with target/predicate/reconvergence_pc → serialize → deserialize → 验证字段
- [ ] 4.3 (Commit 1) 添加测试用例: 24 种指令类型各一个 TEST_CASE（GenericInstr 覆盖 S_MOV/S_ADD/S_SUB/S_MUL/S_LD/S_ST/S_SETP/S_CVT 8 种）
- [ ] 4.4 (Commit 1) 添加测试用例: `TEST_CASE("Roundtrip: mixed 100+ statements")` — 混合类型 + 验证所有字段
- [ ] 4.5 (Commit 1) 添加测试用例: `TEST_CASE("Error: invalid magic")` — 写入错误 magic → deserialize 抛异常
- [ ] 4.6 (Commit 1) 添加测试用例: `TEST_CASE("Error: unsupported version")` — version=99 → 抛异常
- [ ] 4.7 (Commit 1) 添加测试用例: `TEST_CASE("Error: unknown section type")` — 手动写入 .ptxir with bad section type → 抛异常
- [ ] 4.8 (Commit 1) 添加测试用例: `TEST_CASE("Error: unknown opcode")` — 手动写入 .ptxir with invalid opcode → 抛异常
- [ ] 4.9 (Commit 1) 在 `tests/CMakeLists.txt` 中注册新测试 target: `add_executable(unit_ptxir_serialization tests/unit/test_ptxir_serialization.cpp)` + `target_link_libraries(unit_ptxir_serialization PRIVATE ptxir ptxir_writer ptxir_reader Catch2::Catch2)` + `add_test(NAME unit_ptxir_serialization COMMAND unit_ptxir_serialization)` + 标签 `[unit;ptxir]`
- [ ] 4.10 (Commit 1) 编译验证: `cmake --build .worktrees/ptxir-compliance/build --target unit_ptxir_serialization` (应成功，不依赖 ANTLR)
- [ ] 4.11 (Commit 1) 运行测试: `ctest -R unit_ptxir_serialization -V` (24/24 roundtrip + 4 error 测试通过)
- [ ] 4.12 (Commit 1) Commit: `git add tests/unit/test_ptxir_serialization.cpp tests/CMakeLists.txt && git commit -m "test(ptxir): add roundtrip unit tests for all 24 instruction types"`
- [ ] 4.13 (Commit 2) 在 `include/ptxir/ptxir_serialization.h` 添加 `bool generate_ptxir(const std::string& ptx_path, const std::string& ptxir_path, const std::string& kernel_name = "");`
- [ ] 4.14 (Commit 2) 在 `src/ptxir/ptxir_serialization.cpp` 实现 `generate_ptxir()`: 调用 `load_ptx_statements(ptx_path, kernel_name, false)` → 失败返回 false → 否则 `serialize_statements(stmts, ptxir_path)` → 返回 `out.good()`
- [ ] 4.15 (Commit 2) 添加 `generate_ptxir` 单测（`tests/unit/test_ptxir_serialization.cpp`）: 模拟 PTX 文本 → generate → deserialize → 验证语句数量
- [ ] 4.16 (Commit 2) 验证 generate_ptxir: `ctest -R "unit_ptxir_serialization" -V` (如 ANTLR OOM 跳过，标记 expected fail)
- [ ] 4.17 (Commit 2) Commit: `git add include/ptxir/ptxir_serialization.h src/ptxir/ptxir_serialization.cpp tests/unit/test_ptxir_serialization.cpp && git commit -m "feat(ptxir): add generate_ptxir() offline tool"`
- [ ] 4.18 (Commit 3) 在 `include/ptxir/ptxir_serialization.h` 添加 `std::vector<StatementContext> load_ptxir(const std::string& ptxir_path, bool apply_cfg = false);`
- [ ] 4.19 (Commit 3) 在 `src/ptxir/ptxir_serialization.cpp` 实现 `load_ptxir()`: 调用 `deserialize_statements(path)` → 如 `apply_cfg=true` 则调用 `CFGBuilder::build(stmts)` → 返回 stmts
- [ ] 4.20 (Commit 3) 添加 `load_ptxir(apply_cfg)` 单测: 构造带 S_BRA + S_LABEL 的 statements → serialize → load_ptxir(apply_cfg=true) → 验证 reconvergence_pc 已填充
- [ ] 4.21 (Commit 3) 验证 load_ptxir: `ctest -R "unit_ptxir_serialization" -V` (全部通过)
- [ ] 4.22 (Commit 3) Commit: `git add include/ptxir/ptxir_serialization.h src/ptxir/ptxir_serialization.cpp tests/unit/test_ptxir_serialization.cpp && git commit -m "feat(ptxir): complete load_ptxir(apply_cfg=true) path with CFGBuilder integration"`
- [ ] 4.23 Phase 3 整体验证: 完整运行 `ctest -R "ptxir|ptxir_serialization" -V` (应 100% 通过,除 ANTLR 依赖测试 expected fail)

## 5. Phase 4: Documentation & Protocol (G2 修复)

**目标**: AGENTS.md 协议 + 测试文档升级
**Commit 1**: `docs(ptxir): add StatementContext modification protocol to AGENTS.md`
**Commit 2**: `docs(ptxir): update THREE-MODE-TESTING-GUIDE.md to four-mode framework`
**依赖**: Phase 3 完成（工具有了才能更新文档）
**回退策略**: 单 commit revert

- [ ] 5.1 (Commit 1) 在 `src/ptx_ir/AGENTS.md` 添加 "## StatementContext 修改协议" 章节（参考 [.opencode/skills/ptx-lessons-learned/SKILL.md](https://github.com/chisuhua/PTX-EMU/blob/main/.opencode/skills/ptx-lessons-learned/SKILL.md) 跨模块状态翻译经验）
- [ ] 5.2 (Commit 1) 在协议章节列出 4 项 checklist: (1) 同步 ptxir_writer.cpp (2) 同步 ptxir_reader.cpp (3) 添加 roundtrip test (4) 更新 X-Macro dispatch
- [ ] 5.3 (Commit 1) 如 `include/ptxir/AGENTS.md` 不存在则创建；存在则添加 "## 公共头文件修改协议" 章节
- [ ] 5.4 (Commit 1) 在 `include/ptxir/AGENTS.md` 与 `src/ptx_ir/AGENTS.md` 之间建立交叉引用
- [ ] 5.5 (Commit 1) Commit: `git add src/ptx_ir/AGENTS.md include/ptxir/AGENTS.md && git commit -m "docs(ptxir): add StatementContext modification protocol to AGENTS.md"`
- [ ] 5.6 (Commit 2) 阅读 `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md` 了解当前结构
- [ ] 5.7 (Commit 2) 添加 "## Mode 4: PTXIR 快速加载" 章节：API 示例（`load_ptxir(apply_cfg=true)`）、工作流、限制（参考 [.opencode/skills/ptxir-serialization/SKILL.md](https://github.com/chisuhua/PTX-EMU/blob/main/.opencode/skills/ptxir-serialization/SKILL.md) §Workflow）
- [ ] 5.8 (Commit 2) 更新文档标题: "THREE-MODE-TESTING-GUIDE" → "FOUR-MODE-TESTING-GUIDE"（如适用，保留别名）
- [ ] 5.9 (Commit 2) 更新文件名前缀: 如考虑重命名为 `FOUR-MODE-TESTING-GUIDE.md`，添加旧文件名 `THREE-MODE-TESTING-GUIDE.md` 作为 redirect / deprecated alias
- [ ] 5.10 (Commit 2) 在 `docs/developer-guide/README.md` 索引中更新 Mode 4 引用
- [ ] 5.11 (Commit 2) 验证文档链接: 搜索 `tests/three_mode_testing` 引用 → 更新为 `tests/ptxir` 或四模式对应路径
- [ ] 5.12 (Commit 2) Commit: `git add docs/developer-guide/ && git commit -m "docs(ptxir): update THREE-MODE-TESTING-GUIDE.md to four-mode framework"`
- [ ] 5.13 Phase 4 整体验证: 文档交叉引用无死链（`grep -r "three_mode_testing\|three-mode" docs/` 应无 dead reference）

## 6. Final Verification & Archive

- [ ] 6.1 完整 build: `cmake --build .worktrees/ptxir-compliance/build` (允许 ANTLR 依赖 OOM 失败)
- [ ] 6.2 完整 ctest: `ctest --test-dir .worktrees/ptxir-compliance/build -V` (记录 expected fail)
- [ ] 6.3 运行 sanity.sh: `./scripts/sanity.sh` (健康检查)
- [ ] 6.4 验证 PTX 语法测试: `./tests/ptx/test_all_ptx.sh` (应 45/45 通过)
- [ ] 6.5 在 `openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/tasks.md` §10 添加完成记录:
  ```
  - [x] 10.1-10.4 COMPLETED via openspec/changes/ptxir-format-compliance/ (commit <hash>)
  - [x] 10.5 clang-format all new source files (run clang-format if available)
  ```
- [ ] 6.6 合并到 main: `cd /workspace/project/PTX-EMU && git checkout main && git merge ptxir-format-compliance --no-ff -m "Merge ptxir-format-compliance: align with ADR-0023"`
- [ ] 6.7 清理 worktree: `git worktree remove .worktrees/ptxir-compliance && git worktree remove .worktrees/ptxir-baseline`
- [ ] 6.8 Archive change: `openspec archive ptxir-format-compliance --yes` (合并后)
- [ ] 6.9 验证 archive 成功: `ls openspec/changes/archive/2026-07-30-ptxir-format-compliance/` 应存在
- [ ] 6.10 更新 `proposal-approved.md`: `ptxir-format-compliance` 从"已批准提案"移到"已实施"表格
- [ ] 6.11 更新 `proposal-suggestions.md`: 移除 `ptxir-format-compliance` 索引条目（如有）
- [ ] 6.12 提交归档变更: `git add openspec/ proposal-approved.md proposal-suggestions.md && git commit -m "chore: archive ptxir-format-compliance change"`
