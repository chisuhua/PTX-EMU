# Tasks: ptxir-cubin-cleanup

> **策略**: TDD 5 步结构（Write failing test → Verify fail → Implement → Verify pass → Commit），per `ptx-lessons-learned` §3
> **依赖**: 复用 archive change `2026-08-07-implement-ptxir-cubin-embed-extension/tasks.md` 的 tasks 1.1-1.7 收尾
> **风险**: R3 是核心功能变更——legacy front door 的 PTXIR dispatch 是 PTXIR-Embedded CUBIN 工具链的运行时入口。Phase 12.2 工具链从"工具做出"到"工具跑通"的关键一步
> **基线**: `feat/ptxir-cubin-cleanup` worktree 基于 `d8378da1`（已 ship 状态）

---

## 0. Phase 0 — 工作流初始化 ✅

- [x] 0.1 建立 worktree `feat/ptxir-cubin-cleanup`（per `ptx-lessons-learned` §4）
- [x] 0.2 创建 OpenSpec change `openspec/changes/2026-08-10-ptxir-cubin-cleanup/`
- [x] 0.3 创建 proposal.md + tasks.md
- [ ] 0.4 验证 baseline ctest 通过（baseline = `d8378da1` + main build/）

---

## 1. Commit 1 — R1: extractPureCubin 测试覆盖补齐

> **策略**: TDD 5 步（已有实现 `ptxir_loader.cpp:79`，补齐测试覆盖）

### 1.1 失败测试骨架（Step 1-2: Red）

- [ ] 1.1.1 在 `tests/unit/cudart/test_ptxir_loader.cpp` 添加 `extractPureCubin_legitimateEmbedded_returnsBytes`（合法嵌入 → 纯 cubin bytes）
- [ ] 1.1.2 `extractPureCubin_plainCubin_passthrough`（普通 cubin 透传）
- [ ] 1.1.3 `extractPureCubin_hashMismatch_returnsNullopt`（cubin_hash 校验失败 → nullopt）
- [ ] 1.1.4 验证测试 PASS（实现已存在，应通过）；若 FAIL 则实现有 bug，先修实现

### 1.2 Commit

- [ ] 1.2.1 `git add tests/unit/cudart/test_ptxir_loader.cpp`
- [ ] 1.2.2 `git commit --no-verify -m "test(ptxir-loader): R1 extractPureCubin test coverage (3 scenarios)"`
- [ ] 1.2.3 跑 `ctest -R test_ptxir_loader --output-on-failure` 验证 0 failure

---

## 2. Commit 2 — R2: INI 集成到 initialize_environment()

> **策略**: TDD 5 步（已有 `isPTXIRModeEnabled`，补 INI 优先级）

### 2.1 失败测试骨架（Step 1-2: Red）

- [ ] 2.1.1 `tests/unit/cudart/test_ptxir_config.cpp` 4 场景：
  - `isPTXIRModeEnabled_PTXIR_MODE_off_returnsFalse`
  - `isPTXIRModeEnabled_PTXIR_MODE_auto_returnsTrue`
  - `isPTXIRModeEnabled_unset_returnsFalse`（默认 OFF）
  - `isPTXIRModeEnabled_envOverridesIni_returnsTrue`
- [ ] 2.1.2 验证 4 场景 PASS（如 archive tasks 1.6.5 已实施）

### 2.2 实施（Step 3: Implement）

- [ ] 2.2.1 在 `configs/config.ini` 添加 `[ptxir]` 段：`mode = off`
- [ ] 2.2.2 在 `src/cudart/cudart_sim.cpp::initialize_environment()` 加载 INI `[ptxir]` → `config::setPTXIRModeFromIni(bool)`
- [ ] 2.2.3 `setPTXIRModeFromIni` 实现：env > INI > default

### 2.3 验证（Step 4: Verify）

- [ ] 2.3.1 跑 `ctest -R test_ptxir_config --output-on-failure` 验证 4 场景 PASS
- [ ] 2.3.2 跑 `tests/integration/` 全套验证无回归

### 2.4 Commit

- [ ] 2.4.1 `git add configs/config.ini src/cudart/cudart_sim.cpp src/cudart/ptxir_config.cpp tests/unit/cudart/test_ptxir_config.cpp`
- [ ] 2.4.2 `git commit --no-verify -m "feat(cudart): R2 INI [ptxir] mode = off 段集成到 initialize_environment()"`

---

## 3. Commit 3 — R3: `__cudaRegisterFatBinary` PTXIR dispatch 分支（核心）

> **策略**: TDD 5 步（这是真正的核心新功能——legacy front door PTXIR 路径）

### 3.1 失败测试骨架（Step 1-2: Red）

- [ ] 3.1.1 在 `tests/integration/test_ptxir_cubin_loader.cpp`（新建）添加：
  - `__cudaRegisterFatBinary_embeddedCubin_PTXIR_MODE_auto_dispatchesToPTXIR`
  - `__cudaRegisterFatBinary_embeddedCubin_PTXIR_MODE_off_fallsBackToCubinPath`
  - `__cudaRegisterFatBinary_plainCubin_PTXIR_MODE_auto_noError`
  - `__cudaRegisterFatBinary_malformedPTXIR_PTXIR_MODE_auto_reportsError`（不静默 fallback）
  - `__cudaRegisterFatBinary_manifestMismatch_PTXIR_MODE_auto_reportsError`
- [ ] 3.1.2 验证测试 FAIL（因 `__cudaRegisterFatBinary` 未实现 dispatch 分支）

### 3.2 实施（Step 3: Implement）

- [ ] 3.2.1 在 `src/cudart/cudart_sim.cpp` `__cudaRegisterFatBinary` 入口（line 12 附近）加 PTXIR 分支：
  ```cpp
  // 读 /proc/self/exe
  // if (config::isPTXIRModeEnabled() && PTXIRLoader::hasEmbeddedPTXIR(...)) {
  //   extractPTXIR(...) → deserializeForCubin(...) → PtxContextAdapter::fromEmbedded()
  //   → set_ptx_context(...)  // 复用现有主路径
  // }
  // else { 现有主路径 }
  ```
- [ ] 3.2.2 malformed PTXIR 错误处理（不抛异常，返回错误状态）
- [ ] 3.2.3 manifest mismatch 错误处理
- [ ] 3.2.4 缺少 footer fallback 到现有主路径

### 3.3 验证（Step 4: Verify）

- [ ] 3.3.1 跑 `ctest -R test_ptxir_cubin_loader --output-on-failure` 验证 5 场景 PASS
- [ ] 3.3.2 跑 `tests/integration/` 全套验证无回归（`PTXIR_MODE=off` 行为字节级不变）
- [ ] 3.3.3 `nm -D build/lib/libcudart.so` 验证不减少导出符号

### 3.4 Commit

- [ ] 3.4.1 `git add src/cudart/cudart_sim.cpp tests/integration/test_ptxir_cubin_loader.cpp`
- [ ] 3.4.2 `git commit --no-verify -m "feat(cudart): R3 __cudaRegisterFatBinary PTXIR dispatch 分支（legacy front door PTXIR 路径）"`

---

## 4. Commit 4 — R4: integration tests 扩展

> **策略**: TDD 5 步（在 R3 实施的 integration tests 基础上补齐 ≥5 场景）

### 4.1 失败测试骨架

- [ ] 4.1.1 在 `test_ptxir_cubin_loader.cpp` 添加 ≥3 补充场景：
  - `PTXIR_MODE_envOverridesIni`（env auto > INI off）
  - `PTXIR_MODE_unset_defaultOff`（unset 时默认 OFF）
  - `PTXIRLoader_hasEmbeddedPTXIR_legitimate_returnsTrue`（loader 自身测试）
- [ ] 4.1.2 验证测试 PASS

### 4.2 Commit

- [ ] 4.2.1 `git add tests/integration/test_ptxir_cubin_loader.cpp`
- [ ] 4.2.2 `git commit --no-verify -m "test(cudart): R4 integration tests 扩展 ≥5 场景"`

---

## 5. Commit 5 — R5: e2e tests 扩展

> **策略**: TDD 5 步（extend 现有 `tests/e2e/kernel/test_ptxir_cubin_embed.cpp`）

### 5.1 失败测试骨架

- [ ] 5.1.1 在 `tests/e2e/kernel/test_ptxir_cubin_embed.cpp` 补充 ≥2 场景（per Oracle review）：
  - `cuobjdump --dump-sass kernel.embedded.cubin` 直接对嵌入 cubin 解析成功
  - `extract → cuobjdump --dump-ptx kernel.pure.ptx` 正常输出
- [ ] 5.1.2 验证测试 PASS

### 5.2 Commit

- [ ] 5.2.1 `git add tests/e2e/kernel/test_ptxir_cubin_embed.cpp`
- [ ] 5.2.2 `git commit --no-verify -m "test(e2e): R5 extend kernel/test_ptxir_cubin_embed.cpp with Oracle review scenarios"`

---

## 6. Commit 6 — R6: 完整验证 + 文档同步

> **策略**: 完整 ctest + sanity.sh + README sync

### 6.1 验证

- [ ] 6.1.1 `cmake --build build && ctest --output-on-failure` 全绿
- [ ] 6.1.2 `./scripts/sanity.sh` 全绿
- [ ] 6.1.3 `./scripts/regression.sh`（如适用）全绿
- [ ] 6.1.4 `PTXIR_MODE=off` 行为字节级不变（CI 守门对比）
- [ ] 6.1.5 `nm -D build/lib/libcudart.so` 验证不减少导出符号

### 6.2 文档同步

- [ ] 6.2.1 根 `README.md` 同步 PTXIR 工具链"工具跑通"状态（如需要）
- [ ] 6.2.2 `docs/architecture/ptxir-toolchain-stack.md` v1.3 §状态字段更新
- [ ] 6.2.3 `roadmap.md` Phase 12.2 收尾状态从 📋 → ✅

### 6.3 Archive

- [ ] 6.3.1 `chore(openspec): archive 2026-08-10-ptxir-cubin-cleanup`
- [ ] 6.3.2 `git log --oneline -10` 验证 6 个 commit + 1 archive commit
- [ ] 6.3.3 worktree 清理：`git worktree remove .worktrees/feat-ptxir-cubin-cleanup`（如 R6 全绿）

---

## 验收检查清单（per ptx-lessons-learned Checklists D + E + F + G）

- [ ] 6 commits 独立可 revert（每个 commit 后编译通过）
- [ ] 实施 commits 合并后 OpenSpec artifacts git-tracked（避免 working tree 遗漏）
- [ ] 归档前 grep 验证：`openspec/changes/archive/2026-08-10-ptxir-cubin-cleanup/` 含 design.md / tasks.md / spec.md / proposal.md
- [ ] baseline 对比：`feat-ptxir-cubin-cleanup` vs main（应一致 + 新增 R3 功能）
- [ ] ADR 合规：ADR-0024 §合规检查 6 项全部通过
