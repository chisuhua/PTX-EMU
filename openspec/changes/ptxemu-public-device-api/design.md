## Context

### 现状问题 (per Oracle 闭包审计 `ses_fd5ef471cffeWvINOBm5E1GMYd`)

PTX-EMU 仓自 2026-08-21 完成 cpptlm bridge Phase 1-4 cleanup (commits `a9a14e1d`/`292022a3`/`e4d7e369`/`09786635`) 后:
- 公共 include 目录布局: `include/cudart/` (8 文件) + `include/ptx_ir/` (4 文件) + `include/ptxir/` (4 文件) + `include/ptxsim/` (4 文件) + `include/memory/` (1 文件) + `include/register/` (1 文件) + `include/utils/`
- **目标公共设备 API 不存在**: `include/ptxemu/` 目录、`IPtxEmuDevice` 抽象接口、`ptxemu_core` 库目标均缺失
- `StatementContext` (`include/ptx_ir/statement_context.h` 338 行) 是 CppTLM S1 facade 的公共值类型 (`sizeof` 可见性强制),但当前零公共头入口
- CMake 无 `if(PROJECT_IS_TOP_LEVEL)` 隔离模式,被 `add_subdirectory` 消费时测试会污染调用方
- CI 3 workflow (`build-and-test`/`docs-validate`/`Generate PTXIR Files`) 无 `drift_check`,无 header 一致性 check gate

### 目标状态

实施 HSK-8 spec `3b8f7a5` 锁定的 PTX-EMU 端公共设备 API 契约:

```
PTX-EMU include layout (Phase 2 PR 合入后):
include/
├── cudart/                    # [unchanged] fake libcudart.so 公共头
├── ptxemu/                    # [NEW] HSK-8 公共契约入口
│   ├── device_api.h          # IPtxEmuDevice + DTO + 工厂 + VERSION
│   └── ir/                    # [NEW] 晋升自 ptx_ir/
│       ├── statement.h        # (从 statement_context.h)
│       ├── operand_context.h
│       ├── ptx_types.h
│       ├── execution_types.h
│       ├── ptx_qualifier.def  # (X-Macro 表)
│       └── ptx_op.def         # (X-Macro 表)
├── ptx_ir/                    # [fallback] 旧路径 forwarding header, 一个 release 周期
│   ├── statement_context.h    # → #include "../ptxemu/ir/statement.h"
│   └── ...                    # 其他 4 文件同样 forwarding
├── ptxir/                     # [unchanged] PTXIR 序列化
├── ptxsim/                    # [unchanged] 内部 SIMT 实现
├── memory/                    # [unchanged]
├── register/                  # [unchanged]
└── utils/                     # [unchanged]

src/
└── ptxemu/                    # [NEW]
    ├── device_api_impl.cc     # 薄适配层
    └── cmake/ptxemu_core.cmake # 库目标配置 (可选, 也可直接 add_library)
```

CMake library:
- `add_library(ptxemu_core STATIC ...)` 显式源清单
- PUBLIC `include/ptxemu/`
- PRIVATE `${CMAKE_CURRENT_SOURCE_DIR}/src/ptxsim/` + `src/cudart/` (内部依赖)
- `option(PTXEMU_BUILD_TESTING "Build PTX-EMU tests" OFF)` 默认值

CI:
- `.github/workflows/drift_check.yml` (新增) — header 一致性 check
- `.github/workflows/build-and-test.yml` (更新) — `PTXEMU_BUILD_TESTING=ON` 时跑测试,默认 OFF

### 涉及 stakeholders

| 角色 | 关注点 |
|------|--------|
| PTX-EMU owner (@ptx_emu_owner) | ack 决策, 实施 Phase 2 PR, CI gate 维护 |
| PTX-EMU architecture team | 闭包审计, ABI 锁决策, `reconvergenceLabel` 删除 |
| CppTLM maintainer (@cpp_tlm_owner) | 等 Phase 2 PR 合入后开 Phase 3 bump PR |
| UsrLinuxEmu architecture team (@usr_linux_emu_architecture_team) | 仅通知, 无需 ack |

## Goals / Non-Goals

### Goals

- **G1**: 公共设备 API `IPtxEmuDevice` + `ptxemu_core` 库目标就绪, CppTLM 端 `add_subdirectory(external/PTX-EMU)` 即可消费
- **G2**: StatementContext 晋升公共 IR 头, 通过 CppTLM Decision 5 "pure data, no implementation" 前置硬校验 (Phase 0 净化完成后)
- **G3**: 5 文件闭包自洽 — `g++ -fsyntax-only` 单 TU 编译 `include/ptxemu/ir/statement.h` 无未声明 symbol
- **G4**: `drift_check` CI workflow 验证 local-only invariants: `PTXEMU_API_VERSION=1` 守卫宏保留 + `IPtxEmuDevice` 虚方法数量 >= 12 (覆盖 S1 facade 12 callsites 1:1)。**不**做 cross-repo CppTLM submodule hash 比对 — 避免 PTX-EMU CI 依赖 CppTLM 构建链 (违背 HSK-6 单向消费关系)
- **G5**: `if(PROJECT_IS_TOP_LEVEL)` 隔离 PTX-EMU 自身测试, CppTLM 消费时不触发 PTX-EMU 测试构建 (避免 ctest gate 冲突)
- **G6**: `reconvergenceLabel` dead code 在晋升前清理, 公共化后语义冻结

### Non-Goals

- **NG1**: 不重写 PTX-EMU 内部 SIMT 实现 (`GPUContext`/`SMContext`/`WarpContext`/`ThreadContext` 不变)
- **NG2**: 不修改 PTXIR 二进制格式 (`ptxir_writer.h`/`ptxir_format.h` 不变)
- **NG3**: 不引入新 ABI 表面 — HSK-4 vendored 3 接口 (`IScoreboard*`/`IPipelineLatencyProvider*`/`ITensorCoreTiming*`) 已存在,本 change 仅复用其位置
- **NG4**: 不重做 S1 commit `b68abe6f` 的 PTX-EMU integration (`cleanup-cudart-cpptlm-bridge-coupling` 已 archive)
- **NG5**: 不在本次 PR 中合入 `consumer_smoke` — HSK-9 准入(独立 CI workflow,避免 PTX-EMU CI 依赖 CppTLM 构建链)

## Decisions

### Decision 1: StatementContext 路径 — 选 (a) 晋升 ✅ **[Phase 0 COMPLETE]**

**Why** (3 关键论据,per Oracle Hypothesis 1 验证):
1. CppTLM Decision 5 显式锁 "sizeof visibility is mandatory" — 与路径 (b) opaque handle 语义直接冲突
2. 路径 (b) 题面自相矛盾:"CppTLM 通过现有 PtxirReader" 即返回 `vector<StatementContext>`, 即暴露
3. 1 年维护税率: 路径 (a) `ptx_op.def` 表驱动 X-Macro 增列不破 ABI;路径 (b) 每次版本 bump 双仓回归

**实施结果** (Phase 0 全部完成):
- 污染点 A (`operand_phy_addr`): 4 commits (d8b6ca56/a6c9bdaf/66ca4875/1fb15d89), ThreadContext-local index-keyed cache 替代
- 污染点 B (`InstructionState state`): 1 commit (586ea14f), 字段已移除
- reconvergenceLabel dead code: 2 commits (602bfc30 + 359579ec)
- ctest 246/246 PASS 全程验证
- **无需降级路径 (b)** — 路径 (a) 成功实施

**Alternatives considered** (已否决,无需降级):
- **(b) StatementHandle 不透明 + `decode_ptxir` 字节流**:
  - 优点: 内部 IR 完全私有
  - 缺点: 与 CppTLM Decision 5 冲突, 长期维护税高
  - **未启用**: Phase 0 实施顺利, 路径 (a) 验证通过

**污染点 A**: `include/ptx_ir/operand_context.h:59` — `mutable void *operand_phy_addr = nullptr`
- **实际调用点 (8 sites, 4 文件, per Metis audit)**:
  - WRITE: `src/ptxsim/core/thread_context.cpp:362` (热路径, vecOp 元素) | `src/ptxsim/instruction_base.cpp:145,152` (setPhyAddr 热路径)
  - READ: `src/ptxsim/core/thread_context.cpp:404` (collect_operands, const ref) | `src/ptxsim/instruction_handlers.cpp:116-118` (CP_ASYNC debug 宏)
  - DEBUG: `src/ptx_ir/operand_context.cpp:79-98` (toString, const) | `src/cudart/ptx_interpreter.cpp:141,144` (invalidatePhyAddr, CUDART 遗留)
- **方案**: 不使用 `unordered_map<OperandContext*, void*>` (vector<OperandContext> 元素地址稳定性问题 + `mutable` const 路径问题)
- **采用** (per Metis MUST-RESOLVE #4): **index-keyed cache on ThreadContext**
  - 扩展现有 `operand_collected[i]` 数组 (per `src/ptxsim/core/thread_context.cpp:404` 已存在) 为 authoritative runtime cache
  - WRITE 站点 (`instruction_base.cpp:145,152`) 改为 `thread_ctx->operand_phy_cache_[stmt_idx][i] = result` — 不再 mutate OperandContext
  - READ 站点 (`thread_context.cpp:404`) 改为直接读 `thread_ctx->operand_phy_cache_[stmt_idx][i]`,跳过 OperandContext 访问
  - `toString()` (const 路径) 通过 `friend` 声明或 accessor method 暴露 cache, 不再依赖 `mutable`
  - CP_ASYNC macro (`instruction_handlers.cpp:116-118`) 通过 `thread_ctx->lookup_phy_addr(stmt, i)` 函数化访问

**污染点 B**: `include/ptx_ir/statement_context.h:310` — `InstructionState state = InstructionState::READY`
- **关键修正 (per Metis audit)**: 该字段声明后**从未被读写** (`grep -rn '\.state' src/ptxsim/` 仅命中 1 个 commented-out 代码 `thread_context.cpp:132`)。`src/ptxsim/instruction_base.cpp:100-102` 注释明确 "do not write to stmt.state here to avoid a data race... The state begins as READY and need not be reset" — 确认是死字段
- 之前 proposal.md 声称的 "8+ 处 set_state() 调用点" 实际是 `ThreadContext::set_state(EXE_STATE)` / `SimtPcManager::set_state(EXE_STATE)`,使用 **`EXE_STATE` 枚举** (定义于 `simt_pc_manager.h`),**非 `InstructionState`** (定义于 `execution_types.h:22`)
- 净化方案: 直接删除字段 + 文档化 `InstructionState` enum 仅保留为 schema placeholder, 后续无 caller 时可删 enum
- **VERIFY before delete**: `git grep -E '\.state\b' include/ptx_ir/statement_context.h` + grep 整个 src/ptx_ir/ 确保无 reader/writer
  - 净化方案: 移出值类型, 改用 `unordered_map<StatementContext*, InstructionState> runtime_state_table`
  - 风险: 与 ptx-lessons-learned §1 案例同构 — 调度器 `sync_to_warp_state()` 翻译路径需逐一审计, 防止遗漏 set_state

### Decision 2: 库目标 PUBLIC/PRIVATE 拆分 + `if(PROJECT_IS_TOP_LEVEL)` 隔离

**Why**: 实现 CppTLM Decision 4 "PTX-EMU 内部头封装性" 承诺 — 编译时 PTX-EMU 内部头不可见

```cmake
# include/CMakeLists.txt 或 root CMakeLists.txt 新增
option(PTXEMU_BUILD_TESTING "Build PTX-EMU tests when not top-level" OFF)

# ptxemu_core 库目标
add_library(ptxemu_core STATIC
    src/ptxemu/device_api_impl.cc
)
target_include_directories(ptxemu_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include/ptxemu
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/ptx_ir       # Phase 1 净化后变 PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ptxir
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ptxsim
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ptxsim
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cudart
)

# 隔离: PTX-EMU 顶层构建时启用测试
if(PROJECT_IS_TOP_LEVEL OR PTXEMU_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# install 规则: 仅导出库, 不导出测试
install(TARGETS ptxemu_core
    EXPORT ptxemu_core_targets
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
```

**Alternatives considered**:
- **A1 (find_package 模式)**: 适合二进制分发, 增 CI 安装步骤 + 前缀管理; 当前双方源码紧耦合不适用
- **A2 (ExternalProject_Add)**: HSK-3 旧方向已废止, 与新 add_subdirectory 模型冲突
- **A3 (约定级 PUBLIC/PRIVATE, 不强制)**: S1 模式, 头文件漂移照样炸, 不可接受

### Decision 3: `PTXEMU_API_VERSION` 守卫宏 + frozen 规则

**Why**: HSK-8 spec §7 锁定 "公共签名变更 → 触发 HSK-9 (新版本号), 不允许就地 bump"

```cpp
// include/ptxemu/device_api.h
#define PTXEMU_API_VERSION 1

// 静态自检: impl 端
static_assert(PTXEMU_API_VERSION == 1,
              "PTXEMU_API_VERSION 已冻结于 1; 公共签名变更必须签发 HSK-9 增加版本号");
```

**变更规则**:
- ✅ 加方法到 `IPtxEmuDevice` 末尾 → 不 bump VERSION (下游可选实现)
- ❌ 修改已有方法签名 → 必须 bump VERSION (= HSK-9 触发)
- ❌ 移除已有方法 → 必须 bump VERSION

### Decision 4: 决策点 4 的 Phase 2 PR 排期 — 12-15d

(详见 proposal.md "What Changes" §"受影响 API" 引用的 HSK-8 ack body §3 决策点 4)

**PR base**: `origin/main` (post `09786635`) — 严禁基于 `c2038a93` 或更早

**Phase 拆分** (per ptx-lessons-learned §3 phase commit 纪律, 每 Phase 独立 commit + 可 revert):
- **Phase 0**: 闭包净化 2 污染点 (~3-4d) — **必须先跑 Metis pre-impl review**
- **Phase 1**: 5 文件晋升 + `ptxemu::ir` namespace + 旧路径 forwarding header (~1.5d)
- **Phase 2**: `device_api.h` + `device_api_impl.cc` + `ptxemu_core` 库目标 (~2-3d)
- **Phase 3**: `if(PROJECT_IS_TOP_LEVEL)` 隔离 + `option(PTXEMU_BUILD_TESTING OFF)` + install 规则 (~0.5d)
- **Phase 4**: `drift_check` workflow + `consumer_smoke` 基础 (~2d; Phase 2 PR 仅含 `drift_check`)
- **Phase 5**: 文档同步 (`AGENTS.md`/`H2 跨仓 audit`/3 README Fix #1-#3 per ptx-lessons-learned §21) (~1-2d)

### Decision 5: `reconvergenceLabel` dead code 清理窗口

**Why**: HSK-8 公共化前是清理它的最后低成本窗口 — 公共化后改 `BarWarpSyncInstr` 结构需 HSK-9

**Source**: `include/ptx_ir/statement_context.h:229` — `std::string reconvergenceLabel;` (per Oracle session `ses_fd5ef471cffeWvINOBm5E1GMYd`)

**清理方案**: Phase 1 与晋升同 commit 删除 (单字段 + 1 行注释 + 单元测试 grep 验证 0 caller)

## Risks / Trade-offs

| Risk | 严重度 | 缓解策略 |
|------|:-----:|---------|
| **R1**: Phase 0 净化发现 `state` 字段被 5+ 处执行引擎深度依赖无法剥离 → 必须降级路径 (b) | High | Metis pre-impl review 强制, 提前 5-7d 评估可剥离性; 路径 (b) fallback 已写在 HSK-8 spec §7 |
| **R2**: 5 文件晋升后 namespace 冲突 (`ptxemu::ir::Qualifier` vs 内部 `Qualifier`) | Medium | namespace 重命名 — `ptxemu::ir::StatementQualifier` 等; 旧 `Qualifier` 保留 type alias 一个 release |
| **R3**: forwarding header 旧路径在 release 周期内被删 → 旧调用方编译失败 | Medium | 一个 release 周期后必删, 但删除前 grep 全部 `include/ptx_ir/` 路径验证 0 调用方 |
| **R4**: `drift_check` 与 CppTLM submodule 强耦合 → submodule pin 变更触发误报 | **已避免** | drift_check 是 local-only invariants check (per spec/ci-drift-check §Requirement 1), 不读 CppTLM submodule, 不存在该耦合风险 |
| **R5**: 4 artifacts 内部范围数字不一致 (per ptx-lessons-learned §23 Checklist J) | Medium | Metis 强制校验; 4 artifacts 同源 (Oracle 闭包审计结果), 同步引用 |
| **R6**: 跨仓协调时序 — PTX-EMU PR 延迟 → CppTLM bump PR 阻塞 | Low | 14d ack 窗口充分, ETD 2026-09-19 前合入; CppTLM 端可先本地分支验证 bump |
| **R7**: `PTXEMU_API_VERSION` 误 bump → CppTLM 端 binary 不兼容 | High | Decision 3 静态断言 + CI grep 验证 VERSION 改动只在 release notes |
| **R8**: `add_subdirectory` 把 PTX-EMU 的 options/tests 带入 CppTLM 构建 | Medium | Decision 2 `if(PROJECT_IS_TOP_LEVEL)` 隔离 + `option(PTXEMU_BUILD_TESTING OFF)` 默认 |
| **R9**: 基线 worktree build 失败 → 工作时间浪费 | Low | ptx-lessons-learned §4 实测验证 baseline worktree 全量 build 15-20min PASS |

## Migration Plan

### Phase 0: Metis pre-impl review + 闭包净化 (3-4d)

1. 完成 4 artifacts (proposal/design/specs/tasks)
2. 启动 Metis 子代理审计 4 artifacts + 2 污染点下游影响 (per ptx-lessons-learned §7 + Checklist H)
3. 应用 Metis MUST-RESOLVE (若有)
4. `git commit` artifacts FIRST (per Checklist E — 避免 working tree 遗漏)
5. Phase 0.1: 净化 `operand_context.h:59` — `unordered_map<OperandContext*, void*>` 替代 (commit 1)
6. Phase 0.2: 净化 `statement_context.h:310` — `unordered_map<StatementContext*, InstructionState>` 替代 (commit 2)
7. Phase 0.3: dead code 删 (`reconvergenceLabel`) — (commit 3)
8. 全部 Phase 0 commits 完成后跑 PTX-EMU 全量 ctest 验证零回归 (per ptx-lessons-learned §3 "Phase N 通过但 Phase N+1 失败" 防御)
9. **失败处理**: 任一 ctest 回归 → 立即 `git revert` 该 Phase commit, 不混入后续

### Phase 1: 5 文件晋升 + namespace 包装 (1.5d)

1. 新建 `include/ptxemu/ir/` 目录
2. 复制 5 文件 (statement_context → statement / 其他 4 文件保留名) 至新目录
3. 新文件加 `namespace ptxemu::ir { ... }` 包裹 + `namespace ptxemu { namespace ir { ... } }` 双 namespace
4. 旧 `include/ptx_ir/` 路径改为 forwarding header:
   ```cpp
   #pragma once
   #include <ptxemu/ir/statement.h>
   namespace ptx_ir = ptxemu::ir;  // 兼容 alias
   ```
5. grep 全部 `include/ptx_ir/` 调用方 (预期: 主要在 src/ + 测试), 不动 src/ 等一个 release
6. `git commit` Phase 1 整批 (1 commit)

### Phase 2: device_api.h + 实现 + 库目标 (2-3d)

1. 新建 `include/ptxemu/device_api.h` (~200 行)
2. 新建 `src/ptxemu/device_api_impl.cc` (~400 行薄适配层)
3. 新建 `src/ptxemu/cmake/ptxemu_core.cmake` (或直接 add_library in root)
4. 更新 `CMakeLists.txt` (root + src/) 包含库定义
5. `git commit` Phase 2 (1-2 commits, 按依赖拆)

### Phase 3: 隔离 + install (0.5d)

1. CMake 顶部加 `option(PTXEMU_BUILD_TESTING ...)`
2. `if(PROJECT_IS_TOP_LEVEL OR PTXEMU_BUILD_TESTING)` 隔离 tests/
3. install 规则
4. `git commit` Phase 3 (1 commit)

### Phase 4: CI workflow (2d)

1. 新建 `.github/workflows/drift_check.yml`
2. 验证 PTX-EMU PR build 时 workflow 触发
3. 本地 skip `consumer_smoke` (HSK-9 准入)
4. `git commit` Phase 4 (1-2 commits)

### Phase 5: 文档同步 (1-2d)

1. 同步 `include/ptxemu/AGENTS.md` (新目录)
2. 更新 `include/ptx_ir/AGENTS.md` 标注 deprecated
3. 同步 `src/AGENTS.md` HSK 链路段追加 HSK-8
4. 跨仓 audit append: `docs/audits/` (per checklist §F)
5. 3 README Fix (per ptx-lessons-learned §21) — 但本 change 不动根 README (Phase 2 PR 仅 protocol change, root README 是 a9a14e1d+ 级别)

### Rollback 策略

- **Phase 5 / Phase 4 / Phase 3**: 标准 `git revert <commit>` (每 Phase 1-2 commit, 互不依赖)
- **Phase 2**: revert Phase 2 + Phase 1 — 必须保持 include/ptxemu/ 路径已生成但不含实现; 不影响其他组件
- **Phase 1**: revert Phase 1 + Phase 0 — 恢复 include/ptx_ir/ 为权威路径
- **Phase 0**: revert Phase 0 commits (3 commits) — 恢复 2 污染点 + reconvergenceLabel

每个 Phase 独立可回退, 不混入后续 commit (per ptx-lessons-learned §3)。

### 跨仓协调 (HSK-8 spec §"跨仓协调顺序")

```
[1] PTX-EMU ack     ✅ 已完成 (commit 738b412c + comment 5381166580 @ 2026-08-22)
[2] PTX-EMU Phase 2 PR   🔜 feat/ptxemu-public-device-api → origin/main
                            Phase 0 → Phase 5 顺序 commit (ETD 2026-09-19 前合入)
[3] PTX-EMU CI 全绿       ⏳ drift_check + ctest 全部 PASS
[4] PTX-EMU PR 合入 main  🎯 2026-09-19 前
[5] CppTLM bump PR        ⏳ 等 Step 4 后由 CppTLM owner 触发 (独立 change)
```

**禁止跨级**:
- ❌ CppTLM bump PR 在 PTX-EMU Phase 2 PR 合入前提交 (submodule pin 解析失败)
- ❌ PTX-EMU Phase 2 PR 基于 `c2038a93` 或更早 (保留 `g_cpptlm_bridge` 引用, 库目标无法独立链接)
- ❌ 修改 PTXIR 二进制格式 (HSK-8 锁定不变)
- ❌ 修改 HSK-4 vendored 3 接口 (`IScoreboard*` 等)

## Open Questions

- **Q1 (Phase 0 执行前必答)**: `InstructionState state` 字段在所有 8+ 处 `set_state()` 调用点是否能干净剥离为 runtime side-table, 而不破坏调度器 invariant? — 答: 必须 Metis pre-impl review 验证
- **Q2 (Phase 1 执行前必答)**: namespace 包装是否会破坏 `ptx_ir` 已有 consumer (32 callsites grep 待确认)? — 答: 计划用 `namespace ptx_ir = ptxemu::ir` alias 一个 release 周期
- **Q3 (Phase 2 执行前必答)**: `IPtxEmuDevice` 抽象方法集是否覆盖 CppTLM S1 facade 所有使用点 (12 callsites)? — 答: HSK-8 spec §"CppTLM 端接受条件"已锁定 5 条, 实施时机械抽取 S1 facade.cc 调用点
- **Q4 (Phase 4 CI 必须项)**: `drift_check` workflow 是否需 PTX-EMU CI 配置 CppTLM submodule (本来没有)? — 答: **不** — drift_check 是 **local-only** invariants check (`PTXEMU_API_VERSION=1` + 虚方法数量 >= 12), 不读 CppTLM submodule, 不破坏 HSK-6 单向消费关系
