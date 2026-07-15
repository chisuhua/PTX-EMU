# Internal Implementation Plan: cpptlm-d1-full

> **Status**: Draft (companion to OpenSpec change `openspec/changes/cpptlm-d1-full/`)
> **Audience**: PTX-EMU 团队工程师（含未来 6 个月后的自己）
> **From**: PTX-EMU Architecture Team
> **关联**: [proposal.md](./proposal.md) + [design.md](./design.md) + [specs/cpptlm-d1-full/spec.md](./specs/cpptlm-d1-full/spec.md) + [tasks.md](./tasks.md) + [ADR-0021](../../../docs/adr/0021-cpptlm-d1-full-integration.md)

---

## 0. 这份文档是什么

OpenSpec artifacts（proposal/design/spec/tasks）是**对外契约**——给 CppTLM 团队审阅，给未来审计追溯。
本 internal-plan 是 PTX-EMU 团队**自用**的完整实施手册：

- 实施时**实际**打开的文件路径、命令、行号
- 经验沉淀（来自 `docs/dev-process/lessons-learned.md` 的 16 章）
- 失败模式速查（发现 bug 时先翻这里）
- 5 Phase commit 节奏 + 每步验证命令
- 与姊妹 change `cpptlm-phase8b-injection-points` 的协调

---

## 1. 完整实施路径（5 Phase commit 节奏）

> ⚠️ **Lessons Learned #3 强制**: 每个 Phase 独立 commit + 独立可回退。已有测试回归 → 立即 revert 该 Phase。

### Internal Phase ↔ tasks.md Phase 映射

| Internal Phase | tasks.md Phase | 说明 |
|:---:|:---:|---|
| A | Phase 1 | ABI 真值源（HSK-1 准备）|
| B | Phase 2 | cudaLaunchKernel 异步路径 |
| C | Phase 3 + 4 | cudaStreamSynchronize + LD/ST bridge（合并实施）|
| D | Phase 5 | CMake 集成 + libcpptlm_cudart.so |
| E | Phase 6 | 全量回归 + 性能基线 |
| — | Phase 0 | 基线 worktree（internal-plan 前置步骤，不独立编号）|
| — | Phase 7 | 测试完善（融入各 Phase 验证步骤）|
| — | Phase 8 | Handshake 发出 + 文档同步（internal-plan §7 覆盖）|

### Phase A: ABI 真值源（HSK-1 准备）

**目标**：`include/cudart/cpptlm_bridge.h` 可独立编译 + `CPPTLMBRIDGE_VERSION=1` + `static_assert`

**关键文件**：
- `include/cudart/cpptlm_bridge.h`（新建，~70 LOC）
- `include/cudart/cpptlm_bridge_impl.h`（新建 stub，optional）

**关键命令**：
```bash
cd /workspace/project/PTX-EMU
# 编辑文件（参照综合任务书 §2.1 Task #1 完整代码）
cmake --build build --target cudart  # 必须 PASS
grep '#include' include/cudart/cpptlm_bridge.h  # MUST ONLY <cstdint> + <cuda_runtime.h>
```

**commit hash 记录**：实施完成后立即记录 commit hash 输出给 CppTLM（HSK-1）

### Phase B: SingletonGuard + 三段式桥接（~1.5d）

**目标**：`g_gpu_context`/`g_ptx_interpreter`/`CudaDriver::instance()`/`HardwareMemoryManager::instance()` 重复初始化 → FATAL

**关键文件**：
- `src/cudart/cudart_sim.cpp`（修改 `__cudaRegisterFatBinary` 附近代码）

**实施关键点**：
```cpp
class SingletonGuard {
public:
    SingletonGuard() {
        if (initialized_) {
            std::cerr << "FATAL: PTX-EMU global singleton already initialized";
            std::abort();
        }
        initialized_ = true;
    }
    static bool initialized_;
};

// 在 4 个 init 入口前加: SingletonGuard guard;
```

**验证**：`integration_singleton_guard` PASS

### Phase C: cudaLaunchKernel 异步化 + Stream sync（~2d）

**目标**：`cudaLaunchKernel` 异步 + `cudaStreamSynchronize` 真实轮询 + `cudaDeviceSynchronize` + `cudaStreamCreate`

**关键文件**：
- `src/cudart/cudart_sim.cpp`（修改 `cudaLaunchKernel` 函数体 line 332-386 + 新增 helper 函数）

**实施关键点**（Lessons Learned #1 + #2 + #5 强制）：
- `g_pending_kernels` 用 `unordered_map`，删除前先 `completed_ids.push_back(id)` 收集（Lessons #2）
- `g_active_streams` 跟踪 stream 句柄
- `kernel_id` 用 `std::atomic<uint64_t>` 保证唯一（线程安全）
- bridge 路径 vs nullptr 路径**完全分支**，主路径不能在 bridge 分支意外修改

**验证**：
```bash
ctest -L "unit;integration;e2e" --output-on-failure
# 重点验证：
# 1. bridge == nullptr 时现有测试 100% pass（字节级回退）
# 2. bridge != nullptr 时 mock 测试 + 真实 kernel 测试 PASS
```

### Phase D: GLOBAL LD/ST Bridge（~0.3d）

**目标**：`LdHandler/StHandler` 在 GLOBAL 空间走 `bridge->global_access()`

**关键文件**：
- `src/ptxsim/instructions/memory.cpp`（修改 LdHandler/StHandler::processOperation）

**实施关键点**（Lessons Learned #5 强制）：
- `is_global_space()` 必须遍历整个 qualifier 列表而非仅 `back()`
- bridge != nullptr + is_global_space(addr) → `bridge->global_access()` → 立即读/写 `SimpleMemory` + 返回 latency
- UINT64_MAX 或 bridge == nullptr → 原有 PTX-EMU 内部路径

**验证**：`integration_ld_st_bridge` + `[unit;memory]` 0 回归

### Phase E: CMake 集成 + ANTLR4 修正（HSK-2/3 准备，~0.2d）

**目标**：`libcpptlm_cudart.so` CMake 集成 + `.github/copilot-instructions.md` 版本修正

**关键文件**：
- `CMakeLists.txt`（末尾追加 `option(BUILD_LIB_CPPTLM_CUDART)` + `find_package(cpptlm)` + `add_subdirectory`）
- `.github/copilot-instructions.md`（4.13.1 → 4.13.2）
- `src/cudart/cpptlm_bridge/CMakeLists.txt`（新建 stub）

**HSK-3 草案准备**：
```cmake
# 选项 1 (默认) : ExternalProject_Add
ExternalProject_Add(cpptlm
    GIT_REPOSITORY https://github.com/chisuhua/CppTLM.git
    GIT_TAG <cppTLM_COMMIT_HASH>
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/cpptlm-install
)
# 选项 2 : find_library + 环境变量
find_library(CPPTLM_CUDART_LIB cpptlm_cudart HINTS $ENV{CPPTLM_PTXEMU_LIBDIR})
# 选项 3 : pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(CPPTLM REQUIRED IMPORTED_TARGET cpptlm)
```

---

## 2. 经验沉淀检查清单（实施时边做边勾）

### Checklist A: 函数迁移完整性

- [ ] 列出 baseline `cudart_sim.cpp:332-386` 中所有 `set_*` / `g_*` / `bridge->*` 调用点
- [ ] 列出所有 `mutex_` / `lock_guard` / `atomic` 使用（F12b-LD 单线程，Phase 9+ 加 mutex）
- [ ] 对每个 `g_pending_kernels.erase()`，确认采用迭代器安全模式（先 `completed_ids.push_back(id)` → 循环外统一 erase，避免 range-for 中 `unordered_map::erase` 触发 UB）

### Checklist B: 重构前

- [ ] 建立基线 worktree（参见 tasks.md Phase 0.5）
- [ ] 列出本 change 的所有 5 个 Phase
- [ ] 决定哪些 Phase 需要基线对比
- [ ] 准备 revert 策略：每个 Phase 独立 commit，失败立即 `git revert HEAD`

### Checklist D: Commit 前

- [ ] 跑过 baseline worktree 对比（Phase 完成后）
- [ ] `AGENTS.md` 是否需要同步（Phase 8 统一处理）
- [ ] ADR-0021 是否需要追加状态（Proposed → Active）
- [ ] `openspec/changes/cpptlm-d1-full/tasks.md` Phase 状态变更（每 Phase 完成后勾选）
- [ ] commit message 列出独立的 fix 编号

### Checklist E: OpenSpec change 实施后

- [ ] `git ls-files openspec/changes/cpptlm-d1-full/` 不为空（Lessons Learned #6 强制）
- [ ] commit message 引用 openspec change ID
- [ ] 每个 commit 独立可 revert（`git revert HEAD` 后编译通过）

### Checklist H: Pre-implementation Review

- [ ] 实施前调用 Metis 审计 5 个 OpenSpec artifacts（proposal/design/spec/tasks + 本 internal-plan）
- [ ] 验证关键假设：
  - `wc -l cudart_sim.cpp` 验证 933 行（实际）
  - `grep -n "cppTLMBridge\|cpptlm_bridge"` 验证 cpptlm_bridge.h 是否真不存在
  - `ls antlr4/antlr4-cpp-runtime-*` 验证真实版本（已确认 4.13.2）
  - `ctest -N -L cudart` 验证 oracle 测试数量

---

## 3. 实施时常见陷阱速查

### 陷阱 #1: `cudaLaunchKernel` 异步路径破坏现有同步测试

**症状**：开启 `g_cpptlm_bridge != nullptr` 测试路径后，大量现有测试失败

**解决路径**：
1. 验证 `g_cpptlm_bridge` 默认 `nullptr`（必须）
2. 验证测试中是否有人不小心设置了 `g_cpptlm_bridge`（grep test files）
3. 确认异步/同步分支完全独立，互不干扰

### 陷阱 #2: `g_pending_kernels.erase()` 在 range-for 中触发迭代器失效

**症状**：`cudaStreamSynchronize` 中段错误或部分 kernel 未被 erase

**解决路径**：
1. 严格遵守迭代器安全模式：先 `completed_ids.push_back(id)` → 循环外统一 erase（避免 range-for 中 `unordered_map::erase` 触发 UB）
2. 验证：`grep -n "g_pending_kernels.erase" src/cudart/cudart_sim.cpp` 检查所有 erase 位置

### 陷阱 #3: `is_global_space()` 仅看 `qualifiers.back()` 导致 float 误判

**症状**：cute_rmsnorm simpleGEMM 等 kernel 通过 bridge 后输出错误

**解决路径**：
1. 实施时必须审查 `is_global_space()` 实现
2. 必须遍历整个 qualifier 列表（reference: cute_rmsnorm float 类型判断 bug）
3. 验证：handler 入口加临时 print 确认 `is_global_space` 标志

### 陷阱 #4: ANTLR4 版本声明双源冲突

**症状**：CppTLM 团队困惑"PTX-EMU CI 是否会安装 ANTLR4"

**解决路径**：
1. **HSK-2 必须** 在 Phase 1 前发出
2. 修复 `.github/copilot-instructions.md`：4.13.1 → 4.13.2
3. 验证 vendored 目录：`ls antlr4/antlr4-cpp-runtime-*` → 真实版本
4. 验证 CI yml：`.github/workflows/*.yml` 不安装 ANTLR4

### 陷阱 #5: `g_cpptlm_bridge` 初始化时机错误

**症状**：`bridge == nullptr` 时 cudaLaunchKernel 仍然异步（fallback 路径误触发）

**解决路径**：
1. 严格：`g_cpptlm_bridge == nullptr` → 同步路径；非 null → 异步
2. 验证：D-PTX-1 决策中明确初始化时机（libcpptlm_cudart.so 加载时赋值）
3. 必须设置 `static CppTLMBridge* g_cpptlm_bridge = nullptr;` 在 cudart_sim.cpp 单一定义

### 陷阱 #6: 错误码映射不一致

**症状**：`bridge->submit_kernel()` 返回 0xFFFF 但 `cudaLaunchKernel` 返回 0

**解决路径**：
1. D-PTX-5 表格强制 5 类条件 + 返回值
2. 桥接错误码：`int ret = bridge->submit_kernel(...); if (ret != 0) return (cudaError_t)ret;`
3. 测试覆盖：mock bridge 返回 5 种典型 cudaError_t 值

---

## 4. 验证策略（实施时按此顺序）

### 4.1 每个 Phase 完成后

```bash
cd /workspace/project/PTX-EMU

# 编译验证
cmake --build build --target cudart

# 单元测试
cd build && ctest -L "unit" --output-on-failure

# 集成测试
ctest -L "integration" --output-on-failure

# e2e 测试
ctest -L "e2e" --output-on-failure

# 完整 sanity
cd .. && ./scripts/sanity.sh
```

### 4.2 Phase 6 集成验证

```bash
# 验证 CMake OFF 路径（默认）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 验证 CMake ON 路径（mock find_package）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIB_CPPTLM_CUDART=ON \
      -Dcpptlm_DIR=/path/to/mock/cpptlm/lib/cmake
cmake --build build
```

### 4.3 Handshake 验证（Phase 8 准备）

```bash
# HSK-1: cpptlm_bridge.h commit hash
git log --oneline -- include/cudart/cpptlm_bridge.h | head -1

# HSK-2: ANTLR4 版本一致性
grep -nE "antlr4|ANTLR" AGENTS.md README.md .github/copilot-instructions.md
ls antlr4/antlr4-cpp-runtime-*

# HSK-3: CMake 草案存在
[ -f src/cudart/cpptlm_bridge/CMakeLists.txt ] && echo "stub CMakeLists exists"
```

---

## 5. 失败时回退策略

### 5.1 Phase N 失败 → 立即 revert

```bash
# Phase N commit 后跑测试发现回归
cd /workspace/project/PTX-EMU
git log --oneline -3  # 找到 Phase N commit hash
git revert HEAD  # 自动 revert 改动
cmake --build build  # 验证 revert 后编译通过
ctest -L "unit;integration;e2e" --output-on-failure  # 验证回归消失
```

### 5.2 多个 Phase 失败 → 回退到 worktree baseline

```bash
cd ../ptxemu-baseline-f12b  # 切到 Phase 0.5 建立的 worktree
git log --oneline -5  # 找到最后一个 known-good commit
git checkout <known-good-commit>
cmake --build build
ctest -L "unit;integration;e2e" --output-on-failure
```

---

## 6. 与姊妹 change `cpptlm-phase8b-injection-points` 的协调

### 6.1 不冲突区域（可并行）

| 文件 | 本 change | 姊妹 change |
|------|-----------|-------------|
| `cudart_sim.cpp` | ✅ 修改 | ❌ 不触碰 |
| `memory.cpp` | ✅ 修改 | ❌ 不触碰 |
| `CMakeLists.txt` | ✅ 修改 | ❌ 不触碰 |
| `sm_context.cpp` | ❌ 不触碰 | ✅ 修改 |
| `warp_context.cpp/h` | ❌ 不触碰 | ✅ 修改 |
| `scoreboard/pipeline/tensor_core_interface.h` | ❌ 不触碰 | ✅ 新增 |
| `cpptlm_bridge.h` | ✅ 新增 | ❌ 不触碰 |

### 6.2 潜在关联区域（需协调）

- **blocked_cycles 双重 timing 注入**：姊妹 change 的 `pipeline_provider_` 与本 change 的 `global_access()` 都设置 `blocked_cycles_remaining`，需 `max-of-two` 语义（与 design.md §11.3 一致）
  - 协调点：本 change `global_access()` 返回的 latency 不直接调用 `set_blocked_cycles_for_active()`，而是**由 CppTLM 端的 KernelLaunchTLM** 在 `tick()` 中统一管理；KernelLaunchTLM 在 tick() 中实现 `max(pipeline_provider_cycles, global_access_cycles)` 二选一更高值
  - 注：综合任务书 §5.1 G-D3 验证这一点（≤1 cycle 误差）

- **3 个纯虚接口与 bridge 的关系**：本 change 不修改 `IScoreboard` 等；但 `cppTLMBridge` 的 `global_access()` 是 D2 路径（D1 之外的 memory 路径）
  - 协调点：本 change 的 `cpptlm_bridge.h` 与姊妹 change 的 3 个接口头文件**完全独立**，无 mutual include

---

## 7. HSK-1/2/3 回传模板（Phase 8 准备）

### HSK-1 回传

```
Subject: [HSK-1] cpptlm_bridge.h ABI source-of-truth ready for CppTLM integration

Cc: CppTLM Team (#cpptlm-integration Slack)

CppTLM Team,

PTX-EMU 仓库已完成 CppTLMBridge ABI 真值源首发。

- commit hash: <COMMIT_HASH>
- CPPTLMBRIDGE_VERSION: 1
- ABI path: include/cudart/cpptlm_bridge.h
- Test path: tests/unit/cpptlm/test_cpptlm_bridge.cpp
- ABI fields: version() + submit_kernel() + poll_kernel() + synchronize_stream() + global_access()
- ABI verifications:
    - 5 虚方法签名一致
    - cudaStream_t static_assert 通过
    - kernel_args deep-copy 契约明确
- 后续 bump 流程: 修改 cpptlm_bridge.h → CPPTLMBRIDGE_VERSION+1 → 通知 rebase

请同步 rebase CppTLM MemoryBridge::version()。
```

### HSK-2 回传

```
Subject: [HSK-2] PTX-EMU ANTLR4 4.13.2 + CI 不会牵连 CppTLM

Cc: CppTLM Team

CppTLM Team,

确认 PTX-EMU 仓库 ANTLR4 版本一致性：

- vendored: antlr4/antlr4-cpp-runtime-4.13.2-source/  ← 实际
- AGENTS.md: 4.13.2  ← 一致
- README.md: 4.13.2   ← 一致
- copilot-instructions.md: 原 4.13.1（错误）→ 修正为 4.13.2

CI 不会牵连 CppTLM：
- .github/workflows/*.yml 不安装 ANTLR4  ← 截图证据：<PATH>
- vendored 目录独立编译
- CppTLM CI 集成 libcpptlm_cudart.so 时不会触发 ANTLR4 重新生成
```

### HSK-3 回传

```
Subject: [HSK-3] libcpptlm_cudart.so CMake 暴露方式草案

Cc: CppTLM Team

CppTLM Team,

PTX-EMU 端 libcpptlm_cudart.so CMake 暴露方式草案：

[选项 1 默认] ExternalProject_Add
- 优点：版本 pin + build 隔离
- 缺点：首次 build 需访问 CppTLM 仓库

[选项 2] find_library + CPPTLM_PTXEMU_LIBDIR 环境变量
- 优点：build 时解耦
- 缺点：需预先安装 libcpptlm_cudart.so

[选项 3] pkg-config
- 优点：标准 Linux 工具集成
- 缺点：Windows/macOS 需替代

PTX-EMU 倾向选项 1（ExternalProject_Add），CPPTLM_COMMIT_HASH 由 D5 EOD 时确认。

请 CppTLM 团队 review 后反馈。
```

---

## 8. Postmortem 模板（归档后使用）

实施完成后（apply phase 完成 + 全部测试通过 + archive commit 完成），**强制**生成 postmortem：

```markdown
## cpptlm-d1-full Postmortem (Date)

### 实施时长
- 计划：~5d
- 实际：<X>d
- 偏差原因：<Reason>

### 遇到的关键问题
1. <Problem>: <Resolution>
2. <Problem>: <Resolution>

### Lessons 沉淀（追加到 docs/dev-process/lessons-learned.md）
- §N: <新经验>（教训 + 真实案例 + 检查命令）

### 元规则执行情况
- ✅ 基线 worktree: <建立时机>
- ✅ 分 Phase commit: <commit hash 列表>
- ✅ OpenSpec artifacts tracked: <git ls-files>
- ✅ 教训沉淀: <changelog commits>
```

---

**最后更新**: 2026-07-15（v1.0 草案）
**下次 review**: Phase 0 完成 + 决策全部签署后
