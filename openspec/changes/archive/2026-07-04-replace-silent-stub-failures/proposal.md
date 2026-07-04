# Replace Silent Stub Failures — WMMA / Tensor Core / Multi-PTX

## Why

PTX-EMU 在 3 处 stub 处发生**静默失败**（silent failure）：遇未实现指令
时既不报错也无 warning，导致目标寄存器得到未初始化值。最坏情况下
e2e 测试通过但数值错误，难调试、难定位。

**核心触发点**：`docs/audits/debt-audit-2026-07-02.md` §3.1 P0-A3/A4
（WMMA/Tensor Core stub） + P2-A5（Multi-PTX 静默截断）。
本 change 的真实目标：

1. **消灭** silent failure：把 3 处"假装能跑"的 stub 改造为显式失败；
2. **建立**未实现 stub 的统一处理范式（首例接通异常基础设施）；
3. **同步**`UnsupportedInstructionException` 异常基础设施的实际使用方式
   + 文档已知限制章节。

**当前状态**（已验证，HEAD `3f46a3e`）：

| Stub | 位置 | 实测影响 |
|------|------|---------|
| `tensor.cpp::WmmaHandler::processWmmaOperation` | `src/ptxsim/instructions/tensor.cpp:8-15` | 遇 wmma/mma 指令无任何操作，dst 寄存器得未初始化值 |
| `wmma.cpp::WMMA_Handler::processWmmaOperation` | `src/ptxsim/instructions/wmma.cpp:6-13` | **死代码**（类名错全大写，未被 CMake 编译，无运行路径） |
| `cubin_utils.cpp` Multi-PTX | `src/utils/cubin_utils.cpp:118-148` | 实测**正确**追加所有 section，但无 warning 提示用户 |

**前置条件**（依赖已合并）：
- C1（cleanup-deprecated-barrier-apis）+ C2（migrate-bar-warp-sync）+ C3（dead-code-cleanup）
  已在 main（commits `8a5573d`/`7914764`/`6ec8efd`/`4b9d6e1`/`f564004`）
- `UnsupportedInstructionException` 已定义（`include/ptxsim/ptx_exceptions.h:97`）
  但**从未被任何 handler 调用**——本 change 是首例接通
- `PTX_ERROR_EMU` / `PTX_WARN_EMU` 宏已稳定使用（5+ 处现有用法）

## What Changes

将 3 处 stub 改造为显式失败，物理删除 1 处死代码，并同步文档：

- **代码改造**：
  - `tensor.cpp:8-15` `WmmaHandler::processWmmaOperation`：调用
    `PTX_ERROR_EMU` + `throw UnsupportedInstructionException`
  - `cubin_utils.cpp:118-148` Multi-PTX 提取：添加 section 计数器 + 警告
- **死代码删除**：
  - `src/ptxsim/instructions/wmma.cpp` 物理删除（类名错全大写 + 未编译）
- **测试新增**（`tests/unit/ptx/` 而非 `tests/unit/instructions/`）：
  - `test_wmma_not_implemented.cpp`：验证 `WmmaHandler` 抛
    `UnsupportedInstructionException`（`REQUIRE_THROWS_AS`）
  - `test_multi_ptx_warning.cpp`：验证 multi-section cubin 触发
    `PTX_WARN_EMU`
- **文档同步**：
  - `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS 部分
  - 根 `AGENTS.md` "已知限制"章节
  - `tests/ptx/test_wmma.cpp`（已损坏的独立 main() 测试）标注为"已知破损"

## Non-Goals

- **不实施** WMMA / Tensor Core 真实实现：超 C5 范围，应单独建
  `implement-wmma-tensor-core` change 跟踪。`tensor.cpp` 未来改名
  `wmma.cpp`（文件名与内容不符）也属该 change 范围。
- **不修改** `cubin_utils.cpp` 的 append 逻辑（已正确）。
- **不修改** `UnsupportedInstructionException` 类定义本身。
- **不修改** `PTX_ERROR_EMU` / `PTX_WARN_EMU` 宏定义或调用模式。
- **不触碰** `instruction_handlers.cpp:186-189` 的 `__attribute__((weak))`
  分发机制（tensor.cpp 是强覆盖源）。
- **不修复** `tests/ptx/test_wmma.cpp`（引用不存在的 `tests/test_wmma.ptx`，
  不在 ctest 列表，仅记录为已知破损）。

## Goals

1. WMMA / Tensor Core 遇指令时**显式抛 `UnsupportedInstructionException`**
   + 调用 `PTX_ERROR_EMU`，目标寄存器**不再**得未初始化值
2. Multi-PTX cubin 提取时**输出 `PTX_WARN_EMU`**，提示用户二进制含多个
   .cu 来源
3. 删除 `src/ptxsim/instructions/wmma.cpp` 死代码
4. `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS 反映新行为
5. 现有 ctest 全 PASS，无新增 FAIL（cute_rmsnorm_debug / barrier_warp_sync
   等不依赖 wmma/mma 指令）
6. 4 个 Phase commit（参见 tasks.md）+ 每个 commit 独立可 revert

## Risks

- **实测风险低但理论存在**：`tests/integration/divergence/` 下 227 行集成测试
  + cute/cutlass 框架可能间接触发 wmma 路径（**实测** `grep -r "wmma\.\|
  mma\.sync" tests/ bench/` 在整个项目零匹配，但 cutlass 头文件存在
  `mma_sm_*.hpp` 等）。必须 `./scripts/sanity.sh --quick` 完整通过。
- **`UnsupportedInstructionException` 构造函数签名陷阱**：
  `PtxEmuException(message, error_code = INTERNAL_ERROR)`。若漏掉第二参数，
  所有异常被记为 `INTERNAL_ERROR` 而非 `UNSUPPORTED_INSTRUCTION`。
- **`tensor.cpp` 是 X-Macro 强覆盖**：删除它不会"自动"回到 X-Macro 默认
  实现（`__attribute__((weak))` 在链接器层行为有差异）。**必须保留
  tensor.cpp，仅修改其实现**。
- **死代码删除 wmma.cpp 不影响构建**：因该文件**本就未被 CMake 编译**，
  删除后 CMakeLists.txt 不需要同步。
- **AGENTS.md "已知限制"章节**：必须精确表述"抛异常"而非"是 stub"，避免
 误导新人以为可以运行 wmma 测试。

## Design-Time Checklist (Lessons-Learned)

参考 `.opencode/skills/ptx-lessons-learned/SKILL.md` Checklist B/D/E/G。

### B. 重构前准备
- [x] 基线 worktree 计划：复用 `.worktrees/fix-pre-p0-baseline`
      （已存在，含 Phase 0-7 完整基线）
- [x] Phase 拆分：4 个独立 commit（参见 tasks.md）
- [x] 失败处理策略：任何已有测试回归 → 立即 revert 该 Phase，
      **不混入**后续 commit

### D. Commit 前同步清单
- [x] `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS 章节同步
- [x] 根 `AGENTS.md` 已知限制章节同步
- [x] commit message 格式：`fix(ptxsim): throw on WMMA stub (Fix #1)` /
      `chore: remove dead wmma.cpp (Fix #2)` / `feat(cudart): warn on
      Multi-PTX cubin (Fix #3)` / `docs: sync AGENTS for stub failure
      handling (Fix #4)`

### E. OpenSpec 实施后（强制）
- [x] 所有 artifacts (`proposal.md` / `design.md` / `specs/` / `tasks.md`)
      已 git-tracked（提交时 `git add openspec/changes/<name>/`）
- [x] 实施 commits 合并后立即 git-tracked artifacts（避免 working tree
      遗漏——参考 lessons-learned §6 `cleanup-deprecated-barrier-apis`
      `8a5573d` 教训）

### G. OpenSpec lifecycle 约束
- [x] 当前 HEAD `3f46a3e` 0 个活动 OpenSpec 变更（本 change 是首个）
- [x] 不 amend 已归档的 18 个 OpenSpec 变更

## 审计依据

- `.opencode/notes/debt-audit-2026-07-02.md` §3.1 (P0-A3/A4 stub) + §3.3 (P2-A5 Multi-PTX)
- `.opencode/skills/ptx-lessons-learned/SKILL.md` §6 (OpenSpec artifacts 提交)
- `src/ptxsim/barrier/barrier_module.cpp:26,51,88,94,145,164,186,192,220`
  （`PTX_ERROR_EMU` 模式参考）
- `include/ptxsim/ptx_exceptions.h:97`（`UnsupportedInstructionException` 定义）
- `include/utils/logger.h:625-626/666-667`（`PTX_WARN_EMU` 定义）

## Capabilities

### New Capabilities
- `stub-explicit-failure`: PTX-EMU 中所有未实现 stub 必须显式失败（throw 或
  warn），禁止静默无操作

### Modified Capabilities
- 无（不修改现有 spec 的 REQUIREMENTS，仅新建）

## Impact

**修改文件**：
- `src/ptxsim/instructions/tensor.cpp`（WMMA handler 实现）
- `src/utils/cubin_utils.cpp`（Multi-PTX 警告）
- `src/ptxsim/instructions/AGENTS.md`（KNOWN STUBS 同步）
- `AGENTS.md`（已知限制章节同步）
- `openspec/changes/replace-silent-stub-failures/{proposal,design,tasks}.md`
  + `specs/stub-explicit-failure/spec.md`（本 change artifacts）

**删除文件**：
- `src/ptxsim/instructions/wmma.cpp`（死代码）

**新建测试**：
- `tests/unit/ptx/test_wmma_not_implemented.cpp`
- `tests/unit/parser/test_multi_ptx_warning.cpp`（新建
  `tests/unit/parser/` 子目录）

**影响范围**：
- 仅 runtime 行为（throw / warn），不修改语义
- 不影响 cute_rmsnorm_debug / barrier_warp_sync / cute_rmsnorm_bar_sync_pattern
  等已验证 e2e 测试
- 不影响 PTXIR 序列化层（`src/ptxir/`）或 cudart runtime 拦截
- 不影响指令分发（X-Macro `instruction_handlers.cpp:186-189`）