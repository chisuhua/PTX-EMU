# Tasks: Implement WMMA / Tensor Core (Phase 1: m8n8k4 f16)

> **前置依赖**：`replace-silent-stub-failures` 已合并（archived
> `2026-07-04`），提供了 explicit-failure 合约。
> **基线 worktree**：复用 `.worktrees/fix-pre-p0-baseline`。
> **Lessons-learned 集成**：Checklist B（重构前基线）+ D（commit 前同步）
> + E（artifacts 必 tracked）。
> **Phase 拆分**：3 commits，每 commit 独立可 revert
> （ptx-lessons-learned §3）。

---

## 0. Artifacts Tracking（必做！）

- [ ] 0.1 在 main 上创建分支：`git checkout -b feat/implement-wmma-tensor-core`
- [ ] 0.2 `git add openspec/changes/implement-wmma-tensor-core/`
- [ ] 0.3 `git status` 验证 4 个文件 tracked（proposal/design/specs/tasks）
- [ ] 0.4 commit: `git commit -m "docs(openspec): add implement-wmma-tensor-core artifacts"`
- [ ] 0.5 `git ls-files openspec/changes/implement-wmma-tensor-core/` 验证非空

---

## 1. 文件改名 + 基线（Fix #5: rename tensor.cpp → wmma.cpp）

- [ ] 1.1 建立基线 worktree：`git worktree add ../wmma-impl feat/implement-wmma-tensor-core`
- [ ] 1.2 在 worktree 中跑基线：`cmake --build build && ctest -L "unit;integration;e2e"`
- [ ] 1.3 `git mv src/ptxsim/instructions/tensor.cpp src/ptxsim/instructions/wmma.cpp`
- [ ] 1.4 修改 `src/CMakeLists.txt`: `tensor.cpp` → `wmma.cpp`
- [ ] 1.5 修改 `src/ptxsim/instructions/AGENTS.md` STRUCTURE 章节反映改名
- [ ] 1.6 自检：`cmake --build build --target ptxsim && ctest --output-on-failure`
- [ ] 1.7 commit: `git commit -m "chore: rename tensor.cpp to wmma.cpp (Fix #5)"`
- [ ] 1.8 验证独立可 revert（git revert HEAD 应编译通过）

---

## 2. 实现 m8n8k4 f16 fragment arithmetic（Fix #6）

- [ ] 2.1 阅读 PTX ISA §9.7.13 fragment layout，确定 m8n8k4 f16
      的 row/col layout（8 行 × 4 列 f32 输出，4-element K reduction）
- [ ] 2.2 阅读 `include/ptxsim/utils/half_utils.h` 确认 f16 ↔ f32 接口
- [ ] 2.3 实现 `WmmaHandler::processWmmaOperation` m8n8k4 f16 分支：
      - 解析 qualifiers 判定变体（Q_F16 / Q_F32）
      - 检查 `active_mask == 0xFFFFFFFF`，否则 throw
        `ExecutionStateException`
      - 复用 `half_utils.h::f16_to_f32` 转换
      - 8×4 输出片段写入 dst（保留 fragment layout）
- [ ] 2.4 重写 `tests/unit/ptx/test_wmma_not_implemented.cpp` → 改名为
      `test_wmma_m8n8k4.cpp`，断言 32 个 fragment 元素正确（类型一）
- [ ] 2.5 在 `tests/unit/CMakeLists.txt` 注册 `unit_wmma_m8n8k4`：
      ```cmake
      add_catch_test(unit_wmma_m8n8k4
          ptx/test_wmma_m8n8k4.cpp
      )
      set_tests_properties(unit_wmma_m8n8k4 PROPERTIES LABELS "unit;ptx;wmma")
      ```
- [ ] 2.6 自检：
      ```bash
      grep -n "throw\|m8n8k4\|fragment" src/ptxsim/instructions/wmma.cpp
      cmake --build build --target ptxsim
      ctest -R "unit_wmma_m8n8k4" --output-on-failure
      ```
- [ ] 2.7 验证无回归：`ctest -L "unit;integration;e2e"`
- [ ] 2.8 commit: `git commit -m "feat(wmma): implement m8n8k4 f16 fragment arithmetic (Fix #6)"`
- [ ] 2.9 验证独立可 revert（git revert HEAD 应回到 throw-only 行为）

---

## 3. 集成 + E2E 测试 + 文档同步（Fix #7, #8）

- [ ] 3.1 创建 `tests/integration/wmma/test_wmma_mma_sync.cpp`（类型二）：
      - 使用 `execute_warp_instruction` 驱动
      - 验证 uniform warp + m8n8k4 路径写入正确结果
      - 验证 divergent warp 抛 `ExecutionStateException`
- [ ] 3.2 在 `tests/integration/CMakeLists.txt` 注册新测试
- [ ] 3.3 创建 `tests/e2e/kernel/test_wmma_gemm.cu`（类型三）：
      - 16×16 GEMM kernel
      - nvcc 编译 → simulator 执行 → host 验证结果
- [ ] 3.4 在 `tests/e2e/kernel/CMakeLists.txt` 注册 E2E 测试
- [ ] 3.5 修改 `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS：
      移除 `tensor.cpp (WmmaHandler)` 异常说明，标注已实现
- [ ] 3.6 修改根 `AGENTS.md` 已知限制表：WMMA 条目从"抛异常" → "已实现"
- [ ] 3.7 修改 `openspec/specs/stub-explicit-failure/spec.md`：WMMA-MUST
      改为"已实现变体执行 + 未实现变体抛异常"
- [ ] 3.8 自检：
      ```bash
      ./scripts/sanity.sh --quick
      grep -n "WMMA\|Tensor\|stub\|已实现" AGENTS.md
      ```
- [ ] 3.9 commit: `git commit -m "feat(wmma): integration/e2e tests + spec relaxation (Fix #7, #8)"`

---

## 4. 最终验证 + 合并 + 归档

- [ ] 4.1 完整 sanity check：`./scripts/sanity.sh`
- [ ] 4.2 PTX 语法测试：`./tests/ptx/test_all_ptx.sh`
- [ ] 4.3 与 baseline (`.worktrees/fix-pre-p0-baseline`) 对比无新增 FAIL
- [ ] 4.4 合并到 main：`git merge --no-ff feat/implement-wmma-tensor-core`
- [ ] 4.5 验证 artifacts 在 main 已 tracked：
      ```bash
      git ls-files openspec/changes/implement-wmma-tensor-core/
      ```
- [ ] 4.6 清理 worktree：`git worktree remove ../wmma-impl`
- [ ] 4.7 归档：`openspec archive "implement-wmma-tensor-core" --yes`

---

## 失败回滚速查

| 失败 Phase | 立即动作 |
|-----------|---------|
| Phase 1 (rename) | `git revert HEAD` → CMakeLists + 文件名恢复 |
| Phase 2 (m8n8k4 impl) | `git revert HEAD` → 回到 throw-only，rename 保留 |
| Phase 3 (tests + docs) | `git revert HEAD` → 仅回滚测试和文档 |

---

## 关键约束（必读）

⚠️ **MUST**：
- 复用 `include/ptxsim/utils/half_utils.h`，不重新实现 f16 ↔ f32
- 类型一测试覆盖 32 个 fragment 元素（不只 spot-check 几个）
- 实施 commits 合并前先 `git add openspec/changes/<name>/`（避免
  lessons-learned §6 模式）

⚠️ **MUST NOT**：
- 不要在 WMMA 实现里用 `qualifiers.back()` 判断类型（lessons-learned §5）
- 不要修改 `UnsupportedInstructionException` / `ExecutionStateException`
  类定义本身
- 不要破坏 cute_rmsnorm / cute_hello_* 等已通过的 E2E 测试

---

## 未来 Phases（不在本次 change 范围）

- **Phase 2**: m16n16k16 f16 — 更大片段，建议单独 change
- **Phase 3**: mma.sync (sm_70+) 通用路径
- **Phase 4**: tcgen05.mma (sm_100) — Hopper/Blackwell
- **Phase 5**: mma.sp sparse variants

每个 Phase 独立 propose → apply → archive。