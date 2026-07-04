# Tasks: Replace Silent Stub Failures

> **总 Phase 数**：5（0 + 1 + 2 + 3 + 4）
> **前置依赖**：C1 + C2 + C3 已合并到 main（HEAD `3f46a3e`）
> **基线 worktree**：复用 `.worktrees/fix-pre-p0-baseline`
> **Lessons-learned 集成**：Checklist B（重构前基线）+ D（Commit 前同步）+ E（artifacts 必 tracked）+ G（lifecycle）

---

## 0. Artifacts Tracking（必做！避免 lessons-learned §6 教训）

> **强制第一 Phase**：`git add` OpenSpec artifacts 在实施 commits 之前。
> 防止 working tree 遗漏 → 后续债务审计误判为 active debt。

- [ ] 0.1 在 main 上创建工作分支：`git checkout -b fix/replace-silent-stub-failures`
- [ ] 0.2 `git add openspec/changes/replace-silent-stub-failures/`
- [ ] 0.3 `git status` 验证无遗漏（应显示 4 个文件：proposal.md, design.md, specs/stub-explicit-failure/spec.md, tasks.md）
- [ ] 0.4 commit: `git commit -m "docs(openspec): add replace-silent-stub-failures artifacts"`
- [ ] 0.5 `git ls-files openspec/changes/replace-silent-stub-failures/` 验证非空

---

## 1. WMMA Stub 改造（Fix #1）—— 核心改动

- [ ] 1.1 创建实施 worktree：`git worktree add ../c5-impl fix/replace-silent-stub-failures`
- [ ] 1.2 在 worktree 中建立基线：
  ```bash
  cd ../c5-impl && . env.sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  cd build && ctest --output-on-failure 2>&1 | tee /tmp/c5-baseline.log
  ```
- [ ] 1.3 修改 `src/ptxsim/instructions/tensor.cpp` 第 8-15 行：
  - [ ] 1.3.1 添加 `#include "ptxsim/ptx_exceptions.h"`
  - [ ] 1.3.2 添加 `#include "utils/logger.h"`
  - [ ] 1.3.3 `WmmaHandler::processWmmaOperation` 实现改为：
    ```cpp
    void WmmaHandler::processWmmaOperation(ThreadContext *context, void **operands,
                                            const std::vector<Qualifier> &qualifiers) {
        PTX_ERROR_EMU("WMMA instruction not implemented (qualifiers=%zu)",
                      qualifiers.size());
        throw UnsupportedInstructionException(
            "wmma.*",
            "Tensor Core not yet implemented in ptx-emu (ref: c5.1)");
        // MUST 显式传 PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION —— 但当前
        // 基类构造函数默认 INTERNAL_ERROR，需在头文件中确认或扩展。
        // 见 tasks.md 1.3.4
    }
    ```
  - [ ] 1.3.4 **MUST** 验证异常构造函数：阅读 `include/ptxsim/ptx_exceptions.h:97`
        确认 `UnsupportedInstructionException` 是否自动设置
        `UNSUPPORTED_INSTRUCTION` 错误码。若**仅接受** `INTERNAL_ERROR`，
        则需要在 `ptx_exceptions.h` 添加默认错误码参数或在调用处显式 cast
- [ ] 1.4 创建 `tests/unit/ptx/test_wmma_not_implemented.cpp`:
  - [ ] 1.4.1 添加 `#include "ptxsim/ptx_exceptions.h"`
  - [ ] 1.4.2 测试用例：
    ```cpp
    TEST_CASE("WmmaHandler throws UnsupportedInstructionException", "[unit][ptx][wmma][stub]") {
        ThreadContext ctx;
        void* ops[4] = {nullptr, nullptr, nullptr, nullptr};
        std::vector<Qualifier> quals;
        REQUIRE_THROWS_AS(
            WmmaHandler::processWmmaOperation(&ctx, ops, quals),
            UnsupportedInstructionException
        );
    }
    ```
  - [ ] 1.4.3 验证 `what()` 返回 message 以 `"wmma."` 开头
- [ ] 1.5 在 `tests/unit/CMakeLists.txt` 注册新测试：
  ```cmake
  add_catch_test(unit_wmma_not_implemented
      ptx/test_wmma_not_implemented.cpp
  )
  set_tests_properties(unit_wmma_not_implemented PROPERTIES LABELS "unit;ptx;wmma;stub")
  ```
- [ ] 1.6 自检命令：
  ```bash
  grep -n "throw\|PTX_ERROR_EMU" src/ptxsim/instructions/tensor.cpp  # 确认 throw 已添加
  cmake --build build --target ptxsim  # 编译验证
  ctest -R "unit_wmma_not_implemented" --output-on-failure  # 测试验证
  ```
- [ ] 1.7 验证无回归：
  ```bash
  ctest -L "unit;integration;e2e" --output-on-failure  # 全测试
  ./scripts/sanity.sh --quick  # 快速 sanity
  ```
- [ ] 1.8 commit: `git commit -m "fix(ptxsim): throw on WMMA stub execution (Fix #1)"`
  - [ ] 1.8.1 **MUST** 包含 `(Fix #1)` 标记（ptx-lessons-learned Checklist D）
- [ ] 1.9 验证 commit 独立可 revert：
  ```bash
  git revert HEAD  # 应编译仍通过（修改的是抛异常而非行为语义）
  cmake --build build
  git revert HEAD  # 撤销 revert
  ```

---

## 2. 删除 wmma.cpp 死代码（Fix #2）

- [ ] 2.1 验证死代码状态：
  ```bash
  grep -rn "wmma\.cpp" src/CMakeLists.txt  # 预期空（未被编译）
  ls -la src/ptxsim/instructions/wmma.cpp  # 应存在
  ```
- [ ] 2.2 验证无残留引用：
  ```bash
  grep -rn "WMMA_Handler" src/ include/  # 仅 wmma.cpp 自身
  ```
- [ ] 2.3 物理删除：`rm src/ptxsim/instructions/wmma.cpp`
- [ ] 2.4 验证 LSP 错误已消失（若 LSP 服务器运行）：
  - 在 `.h` 中不应再有 `WMMA_Handler` undeclared identifier
- [ ] 2.5 验证编译：
  ```bash
  cmake --build build --target ptxsim
  ```
- [ ] 2.6 验证无回归：
  ```bash
  ctest -L "unit;integration;e2e" --output-on-failure
  ```
- [ ] 2.7 commit: `git commit -m "chore: remove dead wmma.cpp (Fix #2)"`
- [ ] 2.8 验证独立可 revert：
  ```bash
  git revert HEAD  # 撤销删除
  ls src/ptxsim/instructions/wmma.cpp  # 应重新存在
  git revert HEAD  # 撤销 revert
  ```

---

## 3. Multi-PTX Warning（Fix #3）

- [ ] 3.1 阅读 `src/utils/cubin_utils.cpp:118-148` 完整逻辑
- [ ] 3.2 修改 `src/utils/cubin_utils.cpp`:
  - [ ] 3.2.1 添加 `#include "utils/logger.h"`（如未引入）
  - [ ] 3.2.2 在 while 循环外添加 `int ptx_section_count = 0;`
  - [ ] 3.2.3 while 循环内每次成功读取 section 后 `ptx_section_count++;`
  - [ ] 3.2.4 while 循环后添加：
    ```cpp
    if (ptx_section_count > 1) {
        PTX_WARN_EMU("Multiple PTX sections found in cubin (count=%d) - "
                     "all sections extracted (c5.3)", ptx_section_count);
    }
    ```
- [ ] 3.3 创建 `tests/unit/parser/test_multi_ptx_warning.cpp`:
  - [ ] 3.3.1 创建 `tests/unit/parser/` 子目录（如不存在）
  - [ ] 3.3.2 测试用例：构造多 section cubin mock → 验证 PTX_WARN_EMU 被调用
  - [ ] 3.3.3 标签：`[unit][parser][cubin][warning]`
- [ ] 3.4 在 `tests/unit/CMakeLists.txt` 注册新测试（如新建子目录需先加目录模板）
- [ ] 3.5 自检：
  ```bash
  cmake --build build --target cudart
  ctest -R "unit_multi_ptx_warning" --output-on-failure
  ```
- [ ] 3.6 验证无回归：
  ```bash
  ctest -L "unit;parser;cudart" --output-on-failure
  ```
- [ ] 3.7 commit: `git commit -m "feat(cudart): warn on Multi-PTX cubin extraction (Fix #3)"`
- [ ] 3.8 验证独立可 revert（warning 不影响功能，应无破坏）

---

## 4. AGENTS.md 文档同步（Fix #4）

- [ ] 4.1 修改 `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS 章节:
  - [ ] 4.1.1 将 `wmma.cpp` 条目改为 "Removed in c5 - see replace-silent-stub-failures"
  - [ ] 4.1.2 新增条目说明：`tensor.cpp::WmmaHandler` 现在抛
        `UnsupportedInstructionException`
- [ ] 4.2 修改根 `AGENTS.md` "已知限制" 章节:
  - [ ] 4.2.1 "WMMA / Tensor Core"：从"是 stub" → "抛 `UnsupportedInstructionException` 异常（commit `(Fix #1)`）"
  - [ ] 4.2.2 "Multi-PTX cubins"：从"仅提取第一个 PTX" → "输出 `PTX_WARN_EMU` 警告并保留所有 sections（commit `(Fix #3)`）"
- [ ] 4.3 在 `tests/ptx/test_wmma.cpp`（已知破损）顶部加注释：
  ```cpp
  // NOTE (2026-07): 此测试引用不存在的 tests/test_wmma.ptx，
  // 不在 ctest 列表。已知破损，应在 `implement-wmma-tensor-core`
  // change 中修复。
  ```
- [ ] 4.4 自检命令（验证文档同步）：
  ```bash
  grep -n "wmma\|stub\|UnsupportedInstruction" src/ptxsim/instructions/AGENTS.md
  grep -n "WMMA\|Tensor\|stub\|警告" AGENTS.md  # 根 AGENTS.md
  ```
- [ ] 4.5 commit: `git commit -m "docs: sync AGENTS for stub failure handling (Fix #4)"`

---

## 5. 最终验证与归档

- [ ] 5.1 完整 sanity check：
  ```bash
  ./scripts/sanity.sh  # 完整（非 --quick）
  ```
- [ ] 5.2 PTX 语法测试套件：
  ```bash
  ./tests/ptx/test_all_ptx.sh  # 必做（非 ctest 替代）
  ```
- [ ] 5.3 与 baseline 对比无新增 FAIL：
  ```bash
  diff <(grep -E "FAIL|PASS" /tmp/c5-baseline.log | sort) \
       <(grep -E "FAIL|PASS" <(ctest --output-on-failure 2>&1) | sort)
  # 预期：差异仅为"无新增 FAIL"（可能有新增 PASS）
  ```
- [ ] 5.4 LSP 检查：所有 `.cpp` `.h` 无 `WMMA_Handler` 编译错误（仅 pre-existing
      类型错误保留，与本 change 无关）
- [ ] 5.5 合并到 main：
  ```bash
  git checkout main
  git merge --no-ff fix/replace-silent-stub-failures
  # 预期 merge commit 列出 5 个 Phase commits
  ```
- [ ] 5.6 **MUST** 验证 artifacts 在 main 已 tracked：
  ```bash
  git ls-files openspec/changes/replace-silent-stub-failures/
  # 应非空
  ```
- [ ] 5.7 清理 worktree：
  ```bash
  git worktree remove ../c5-impl
  git branch -d fix/replace-silent-stub-failures
  ```
- [ ] 5.8 归档 change：
  ```bash
  openspec archive "replace-silent-stub-failures" --yes
  ```
- [ ] 5.9 验证归档成功：
  ```bash
  ls openspec/changes/archive/ | grep replace-silent-stub  # 应有
  openspec status  # 无 active changes
  ```
- [ ] 5.10 **建议立即 propose follow-up** `implement-wmma-tensor-core` change
      （在 C5 基础上实施真实 WMMA / Tensor Core 实现）

---

## 失败回滚速查

| 失败 Phase | 立即动作 |
|-----------|---------|
| Phase 1（WMMA throw） | `git revert HEAD` → 检查 cute_rmsnorm_debug 是否触发异常 |
| Phase 2（删除 wmma.cpp） | `git revert HEAD` → 编译应仍通过（死代码删除不影响） |
| Phase 3（Multi-PTX warning） | `git revert HEAD` → 重新跑 PTX 解析测试 |
| Phase 4（文档同步） | `git revert HEAD` → 仅文档回滚 |
| Phase 5（合并后整体） | `git revert -m 1 <merge-commit>` → main 回到 merge 前 |

---

## 关键约束（必读）

⚠️ **MUST NOT** 触碰以下（即使"看起来顺手"）：
- `UnsupportedInstructionException` 类定义（除非 1.3.4 验证需要扩展）
- `PTX_ERROR_EMU` / `PTX_WARN_EMU` 宏定义
- `instruction_handlers.cpp:186-189` X-Macro weak dispatch
- `tensor.cpp` 中的强覆盖实现（必须保留）
- `cubin_utils.cpp` 的 append 逻辑（已正确）

⚠️ **MUST** 应用 ptx-lessons-learned：
- 每个 commit 独立可 revert（Checklist B）
- commit message 包含 `(Fix #N)` 标记（Checklist D）
- artifacts git-tracked before 实施（Checklist E）
- 不 amend 已归档的 18 个 OpenSpec changes（Checklist G）

⚠️ **MUST** 验证：
- 每个 Phase 后跑 `./scripts/sanity.sh --quick`
- Phase 5 跑完整 `./scripts/sanity.sh`
- `./tests/ptx/test_all_ptx.sh`（非 ctest 替代）
- baseline 对比无新增 FAIL