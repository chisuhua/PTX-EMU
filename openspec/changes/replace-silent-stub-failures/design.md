# Design: Replace Silent Stub Failures

## Context

PTX-EMU 当前在 3 处 PTX 指令/解析路径存在 **silent failure**：

1. **WMMA / Tensor Core stub**（`src/ptxsim/instructions/tensor.cpp:8-15`）：
   `WmmaHandler::processWmmaOperation` 调用 4 个 operands 后无任何操作，
   dst 寄存器保留未初始化值。**这是真实运行路径**（X-Macro 生成的强覆盖）。

2. **死代码 stub**（`src/ptxsim/instructions/wmma.cpp:6-13`）：
   `WMMA_Handler::processWmmaOperation`（类名错全大写），**未被 CMake 编译**
   （`src/CMakeLists.txt` 零引用），无运行路径。保留造成编译错误
   （LSP 已报告 `WMMA_Handler` undeclared identifier）。

3. **Multi-PTX 警告缺失**（`src/utils/cubin_utils.cpp:118-148`）：实测已正确
   追加所有 PTX section，但**无 warning 提示**用户二进制含多个 .cu 来源。

**项目异常基础设施**（已建立但从未接通）：
- `UnsupportedInstructionException` 定义于 `include/ptxsim/ptx_exceptions.h:97`
- `PTX_ERROR_EMU` / `PTX_WARN_EMU` 宏定义于 `include/utils/logger.h:625-626/666-667`
- 已稳定使用于 `src/ptxsim/barrier/barrier_module.cpp` 共 9 处（line 26, 51, 88, 94, 145, 164, 186, 192, 220）

**约束**：
- C5 不实施 WMMA/Tensor Core 真实版本（超范围，应单独建
  `implement-wmma-tensor-core` change）
- 不修改 `UnsupportedInstructionException` 类定义或宏本身
- 现有 ctest 全 PASS，无新增 FAIL

## Goals / Non-Goals

**Goals**（设计目标）：
1. WMMA/Tensor Core 遇指令时**显式抛 `UnsupportedInstructionException`** + 调用
   `PTX_ERROR_EMU`，dst 寄存器**不再**得未初始化值
2. Multi-PTX cubin 提取时**输出 `PTX_WARN_EMU`**，提示用户二进制含多个 .cu 来源
3. 物理删除 `src/ptxsim/instructions/wmma.cpp` 死代码（编译错误源）
4. 4 个 Phase commit + 每个 commit 独立可 revert（ptx-lessons-learned §3）
5. 所有 artifacts git-tracked（避免 lessons-learned §6 模式：实施 commit 遗漏
   artifacts → 12 天后债务审计误判为 active）

**Non-Goals**（明确排除）：
- WMMA/Tensor Core 真实实现（`implement-wmma-tensor-core` change 跟踪）
- `tensor.cpp` 文件名改正（应随真实实现一起改名）
- `UnsupportedInstructionException` 类 API 扩展（如添加新错误码）
- `PTX_ERROR_EMU` / `PTX_WARN_EMU` 宏重构
- X-Macro `__attribute__((weak))` 分发机制修改
- `tests/ptx/test_wmma.cpp` 修复（已知破损，仅记录）

## Decisions

### Decision 1: 修改 `tensor.cpp` 而非 `wmma.cpp`

**Context**: 两个文件都包含 `processWmmaOperation`，但：
- `tensor.cpp` 定义 `WmmaHandler`（驼峰式，与 X-Macro `ptx_op.def` 一致）→ **真实运行路径**
- `wmma.cpp` 定义 `WMMA_Handler`（全大写）→ **死代码**，未被编译

**Choice**: 修改 `tensor.cpp:8-15`，删除 `wmma.cpp`。

**Rationale**:
- `tensor.cpp` 是 X-Macro 强覆盖（`instruction_handlers.cpp:186-189` 的
  `__attribute__((weak))` 链接行为）
- 删除 `tensor.cpp` 不会"自动"回到 weak 默认实现，**必须保留**
- `wmma.cpp` 已是编译错误源（LSP 报告 `WMMA_Handler` undeclared）

**Alternatives considered**:
- ❌ 改名 `tensor.cpp` → `wmma.cpp`：超 C5 范围，应随真实实现一起
- ❌ 同时保留两个文件 + 注释其中一个为 deprecated：增加维护负担，无收益
- ❌ 统一两个文件名为 `wmma.cpp` + 合并：破坏 X-Macro 现有引用

### Decision 2: 使用 `UnsupportedInstructionException` 而非新定义异常类

**Context**: 项目已有 `UnsupportedInstructionException`（`ptx_exceptions.h:97`），
但**从未被任何 handler 调用**。

**Choice**: 使用现有 `UnsupportedInstructionException` + 标准 message 格式
`"wmma.*"`（含指令名前缀便于日志过滤）。

**Rationale**:
- 项目约定："不引入新错误码，使用现有异常体系"（参考 `tests/unit/memory/`
  4 处现有用法）
- 接通已有基础设施 = 建立"首例"使用模式（proposal.md A.1）
- 保持错误码枚举稳定，便于上游 catch 处理

**Constructor 签名陷阱**（必须在 implementation 中注意）：
```cpp
// PtxEmuException 基类构造函数
explicit PtxEmuException(
    const std::string& message,
    PtxEmuErrorCode error_code = PtxEmuErrorCode::INTERNAL_ERROR) noexcept;
```
**必须显式传第二参数**为 `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION`，
否则所有异常被记为 `INTERNAL_ERROR`（AI 实施易错点）。

**Alternatives considered**:
- ❌ 定义新的 `WMMAStubException` 类：重复定义 + 破坏既有约定
- ❌ 仅 `PTX_ERROR_EMU` 不抛异常：违反 proposal.md Goal 1（异常必须可捕获）

### Decision 3: Multi-PTX 警告保持 warning 而非 error

**Context**: `cubin_utils.cpp` 当前 append 已正确，无功能 bug。

**Choice**: 输出 `PTX_WARN_EMU`（不抛异常，不中断执行）。

**Rationale**:
- 与 `qualifier_utils.cpp:313` 的
  `PTX_WARN_EMU("Unsupported immediate qualifier: %s, zeroing value")` 一致
  （"降级继续"语义）
- Multi-PTX 警告仅起"诊断/日志"作用
- 不应中断正常 kernel 执行（用户实际只需要知道这个信息）

**Alternatives considered**:
- ❌ 升级为 error 停止模拟：过度反应，会破坏现有 cute_rmsnorm 测试
- ❌ 静默无操作（现状）：违背本 change 目的

### Decision 4: 测试路径用 `tests/unit/ptx/` 而非 `tests/unit/instructions/`

**Context**: `tests/unit/instructions/` **不存在**（实测 `ls tests/unit/instructions/`）
而 `tests/unit/ptx/` 已有 9 个 PTX 指令测试。

**Choice**: 新建测试文件放 `tests/unit/ptx/`（与现有约定一致）。

**Rationale**:
- 沿用 `tests/unit/ptx/test_fma_rn_f32.cpp` 等已有约定
- `tests/unit/instructions/` 若新建子目录需更新 `tests/unit/CMakeLists.txt` 模板
- 对应测试目录 ctest label 为 `unit;ptx;...`（与现有命名空间一致）

**Alternatives considered**:
- ❌ 新建 `tests/unit/instructions/`：偏离项目已有约定
- ❌ 放在 `tests/unit/wmma/`：过细分类，未来 tensor 重命名时需迁移

### Decision 5: 4 个 Phase commit 而非聚合 commit

**Context**: 本 change 涉及 3 个文件 + 1 个文档同步，3+ commits 是 ptx-lessons-learned §3 的强制建议。

**Choice**: 4 个独立 commit，每个独立可 revert。

**Rationale**:
- ptx-lessons-learned §3："复杂迁移必须分 Phase commit，每个 Phase 独立可回退"
- 已归档 `2026-07-03-dead-code-cleanup` 4b9d6e1 即采用此模式
- 失败时仅 revert 受影响 Phase，不污染后续 commit

**Phase 拆分**（详细 tasks 见 tasks.md）：
- Phase 1: `fix(ptxsim): throw on WMMA stub (Fix #1)` — 修改 `tensor.cpp`
- Phase 2: `chore: remove dead wmma.cpp (Fix #2)` — 删除 `wmma.cpp`
- Phase 3: `feat(cudart): warn on Multi-PTX cubin (Fix #3)` — 修改 `cubin_utils.cpp`
- Phase 4: `docs: sync AGENTS for stub failure handling (Fix #4)` — 文档同步

**Alternatives considered**:
- ❌ 1 个聚合 commit：失败回退成本高，违反项目既定模式
- ❌ 6+ 个微 commit：过度切分，每个 commit 信息价值低

## Risks / Trade-offs

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| cute/cutlass 框架某测试间接触发 wmma 路径 | 中 | `./scripts/sanity.sh --quick` 完整通过 + 4 个 Phase 每个都跑 ctest |
| `UnsupportedInstructionException` 构造函数漏第二参数 | 中 | implementation review checklist 显式列出 + tasks.md 1.5 加自检命令 |
| `tensor.cpp` 修改破坏 X-Macro 强覆盖 | 中 | Phase 1 commit 后立即跑 `ctest -L "unit;wmma"` 验证 |
| `wmma.cpp` 删除后有 CMake 引用残留 | 低 | 删除前 grep 验证：预期空 |
| `cubin_utils.cpp` 修改破坏现有 PTX 提取 | 低 | Phase 3 commit 后跑 `ctest -L "unit;parser;cudart"` |
| AGENTS.md 文档与实现脱节 | 低 | Phase 4 commit 包含 `grep` 验证命令 |
| 用户测试用例实际依赖 silent WMMA（虽未发现） | 低 | grep 验证：项目内 `wmma\.mma\|mma\.sync` 零匹配 |

## Migration Plan

**实施流程**（4 个 Phase，1 个 worktree）：

```bash
# 0. 准备 baseline worktree（复用已有）
cd /workspace/project/PTX-EMU
# .worktrees/fix-pre-p0-baseline 已存在

# 1. 创建实施 worktree
git worktree add ../c5-impl -b fix/replace-silent-stub-failures main
cd ../c5-impl
. env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. 基线验证（必须全 PASS）
cd build && ctest --output-on-failure 2>&1 | tee /tmp/c5-baseline.log

# 3. 实施 4 个 Phase
# Phase 1: 修改 tensor.cpp + 新增 test_wmma_not_implemented.cpp
# Phase 2: 删除 wmma.cpp
# Phase 3: 修改 cubin_utils.cpp + 新增 test_multi_ptx_warning.cpp
# Phase 4: 同步 AGENTS.md

# 每个 Phase 后跑 sanity.sh --quick
./scripts/sanity.sh --quick

# 4. 最终验证
./scripts/sanity.sh  # 完整

# 5. 提交（4 个 commit + 1 个 merge）
git add openspec/changes/replace-silent-stub-failures/
git commit -m "docs(openspec): replace-silent-stub-failures design adjustments"
git checkout main
git merge --no-ff fix/replace-silent-stub-failures

# 6. 归档
openspec archive "replace-silent-stub-failures" --yes
```

**回滚策略**：

| Phase 失败 | 回滚动作 |
|-----------|---------|
| Phase 1 (WMMA throw 失败) | `git revert <commit>` + 检查 cute_rmsnorm |
| Phase 2 (wmma.cpp 删除失败) | `git revert <commit>` + 检查编译错误 |
| Phase 3 (Multi-PTX warning 失败) | `git revert <commit>` + 检查 PTX 解析测试 |
| Phase 4 (文档同步失败) | `git revert <commit>` 即可（仅文档） |

**合并后清理**：
```bash
git worktree remove ../c5-impl
git branch -d fix/replace-silent-stub-failures
```

## Open Questions

1. **Future change "implement-wmma-tensor-core"**：本 change 完成后，应立即
   propose `implement-wmma-tensor-core` 跟踪：
   - 删除 `tensor.cpp` 中的 throw，替换为真实实现
   - 重命名 `tensor.cpp` → `wmma.cpp`
   - 添加 unit + integration + e2e 测试覆盖 wmma.mma 指令

2. **C5 是否需要 OpenSpec proposal 引用 ADR**？当前未引用（短期工程行为）。
   建议**不**建 ADR，但 proposal.md 引用 ADR-0003（PC API 废弃）作为"删除
   死代码"的先例，证明项目已有同类先例。

3. **`tests/ptx/test_wmma.cpp` 独立破损测试**：引用不存在的
   `tests/test_wmma.ptx`，不在 ctest 列表。是否在 C5 中修复？
   - 建议**不**修复（超范围），仅在 tasks.md Phase 4 文档同步中标注为
     "已知破损"

## 影响范围

| 组件 | 影响类型 | 详情 |
|------|---------|------|
| `src/ptxsim/instructions/tensor.cpp` | 修改 | `WmmaHandler::processWmmaOperation` 抛异常 |
| `src/ptxsim/instructions/wmma.cpp` | **删除** | 死代码 |
| `src/utils/cubin_utils.cpp` | 修改 | Multi-PTX 计数器 + PTX_WARN_EMU |
| `tests/unit/ptx/test_wmma_not_implemented.cpp` | 新建 | Catch2 unit test |
| `tests/unit/parser/test_multi_ptx_warning.cpp` | 新建 | Catch2 unit test |
| `tests/unit/CMakeLists.txt` | 修改 | 注册 2 个新测试 |
| `src/ptxsim/instructions/AGENTS.md` | 修改 | KNOWN STUBS 章节 |
| `AGENTS.md` | 修改 | 已知限制章节 |
| `openspec/changes/replace-silent-stub-failures/` | 新建 | artifacts |
| `src/CMakeLists.txt` | 无变化 | wmma.cpp 本就未被编译 |

## 相关 ADR 引用

- **ADR-0003**（commit-pc-pattern）：作为"删除死代码"先例参考
- **ADR-0008**（barrier-semantics）：PTX_ERROR_EMU 使用模式参考
- **未来 ADR 候选**：若 WMMA 实现成熟，可建 `0016-wmma-tensor-core.md`