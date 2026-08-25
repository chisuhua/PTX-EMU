# HSK-8: PTX-EMU 端公共设备 API 契约 — PTX-EMU ACK (消费方确认)

> **日期**: 2026-08-22
> **起草方**: PTX-EMU Architecture Team (@ptx_emu_owner)
> **回传目标**: CppTLM 仓 issue #22 评论 + CppTLM `docs/superpowers/specs/2026-08-21-hsk-8-cpptlm-response.md` (mirror)
> **关联**:
> - CppTLM HSK-8 spec: [`3b8f7a5`](https://github.com/chisuhua/CppTLM/commit/3b8f7a5) (`docs/superpowers/specs/2026-08-21-hsk-8-ptxemu-public-api.md`)
> - CppTLM HSK-8 ack 模板: `docs/superpowers/specs/2026-08-21-hsk-8-cpptlm-response.md`
> - CppTLM openspec change: `openspec/changes/cpptlm-ptxemu-public-device-api/`
> - Oracle session: `ses_fd5ef471cffeWvINOBm5E1GMYd` (跨项目闭包审计, 共 4 hypothesis 验证)
> - 关联 commit base: `09786635` (cleanup Phase 3 of 4) 之后的 `origin/main`

---

## 元数据

| 字段 | 值 |
|------|---|
| **HSK 编号** | **HSK-8**（序列：HSK-1 ABI header @ `8dc000ec` / HSK-2 ANTLR4 4.13.2 / HSK-3 ExternalProject_Add (废止方向) / HSK-4 `ptx_arg_sizes[]` (vendored 3 interfaces) / HSK-5 `advance()` deferred → CANCELLED by HSK-6 / HSK-6 bridge deprecation @ PTX-EMU `25e36f60` + CppTLM ack `369cf71` / HSK-7 (🔵 未签发, ABI 解冻 CPPTLMBRIDGE_VERSION 触发) / **HSK-8 (本 ack)**）|
| **发起方** | **CppTLM maintainer** (chisuhua) — 真因 PTX-EMU 仓 `09786635` Phase 1-4 cleanup 后, CppTLM 侧需新握手取代 HSK-3 旧方向 |
| **消费方** | **PTX-EMU Architecture Team** (本仓 owner ack) |
| **状态** | ✅ **PTX-EMU ACCEPTED** (含 4 决策点明确答复 + 闭包审计) → 触发 CppTLM 端 Phase 3 bump PR 跨仓协调 |
| **公告日** | 2026-08-22 |
| **Ack 截止** | **2026-09-05**（14 天 + 超时无异议兜底, per HSK-6 precedent）|

## 仓 HEAD 锚点

```
PTX-EMU    @09786635 (Phase 3 of 4 cleanup 完成, PTX-EMU 真相源侧锚点)
CppTLM     @3b8f7a5 (HSK-8 spec commit, 含 ack response 模板 + .gitignore 修复)
```

---

## 1. 协议范围 (What)

HSK-8 锁定 PTX-EMU 端公共设备 API 契约 (`ptxemu/device_api.h`), CppTLM 端通过
`add_subdirectory(external/PTX-EMU)` 消费 `ptxemu_core` 静态库, 取代 S1
`cmake/PTXEmuCore.cmake` 编译防火墙 (约定级封装, 头文件漂移照样炸 CppTLM)。

**核心契约 7 条**: 详见 CppTLM `3b8f7a5` spec §"HSK-8 核心契约"。本 ack 端确认
PTX-EMU 接受全部 7 条契约作为 Phase 2 PR 实施依据。

## 2. Oracle 闭包审计结果 (决策点 1 前置硬校验)

> **Oracle session**: `ses_fd5ef471cffeWvINOBm5E1GMYd`
> **验证状态**: 全部引用 `file:line` 均经 `sed -n` / `grep` 实测, 标记 ⚠️ 1 处未验证

### 2.1 StatementContext 实际闭包（5 文件, 非题面 3 文件）

| 文件 | 行数 | 被谁拉入 | 类型 |
|---|---|---|---|
| `include/ptx_ir/statement_context.h` | 338 | 目标 | 值类型 + 20 struct |
| `include/ptx_ir/operand_context.h` | 95 | 直接 include | 值类型 |
| `include/ptx_ir/ptx_types.h` | 32 | 直接 include | Qualifier enum + X-Macro 表头 |
| `include/ptxsim/execution_types.h` | 53 | 直接 include | EXE_STATE + BAR_TYPE + InstructionState |
| `include/ptx_ir/ptx_qualifier.def` | **332** | `ptx_types.h:11` `#include "ptx_qualifier.def"` | X-Macro (Qualifiers 表) |
| `include/ptx_ir/ptx_op.def` | **203** | `ptx_types.h:21` `#include "ptx_op.def"` | X-Macro (6-tuple op 表: `X(enum_val, struct_name, str, opcount, _, instr_kind)`) |
| **总计** | **~1053 LOC** | 全部纯数据 + enum + X-Macro | 零实现代码 |

### 2.2 20 struct 字段类型核查 (Oracle Hypothesis 1 证伪方法实证)

| struct | 字段引用类型 | 闭包污染? |
|---|---|---|
| `DeclarationInstr` (line 17) | `Qualifier`, `std::optional<int>`, `std::vector<int>` | ❌ 无 |
| `DollarNameInstr`, `PragmaInstr`, `LabelInstr`, `VoidInstr`, `BranchInstr`, `BarrierInstr`, `MembarInstr`, `FenceInstr`, `ReduxSyncInstr`, `MbarrierInstr`, `CallInstr`, `PredicatePrefix`, `AtomInstr`, `VoteInstr`, `ShflInstr`, `ActivemaskInstr`, `BarWarpSyncInstr` | `std::string`, `std::vector<Qualifier>`, `std::vector<OperandContext>`, 本地 enum | ❌ 无 |
| `GenericInstr` (line 140-143) | `std::vector<Qualifier>`, `std::vector<OperandContext>` | ❌ 无 |
| `Tcgen05Instr` (line 180-191) | 本地 enum `Tcgen05OpKind` + `Tcgen05Dtype` (line 152/167), 内嵌定义 | ❌ 无 |

**结论**: 闭包审计通过 CppTLM Decision 5 前置硬校验 ("pure data, no implementation" 门槛的**结构条件**)。

### 2.3 2 处非纯数据污染点 (Phase 0 MUST-RESOLVE)

Oracle hypothesis 1 发现的 2 个污染点, 影响路径 (a) 是否仍为最佳选择:

#### 污染点 A: `operand_context.h:59` — 运行时物理地址指针混入值类型

```cpp
// include/ptx_ir/operand_context.h:59 (CONFIRMED via sed -n '59p')
mutable void *operand_phy_addr = nullptr;
```

伴随方法: `setPhyAddr(void *addr)` (line 68), `invalidatePhyAddr()` (line 69)。

**风险**: CppTLM 拿到 `sizeof(OperandContext)` 可见性后, 这个 `void*` 字段会成为公共 ABI 一部分。但它是运行时 cache, 不属于值语义。

**Phase 0 净化方案**: 
- 选项 1 (推荐): 移出 `OperandContext`, 改用 `unordered_map<OperandContext*, void*>` runtime cache
- 选项 2: 文档化为 non-ABI 保证 + `[[deprecated]]`

#### 污染点 B: `statement_context.h:310` — 执行态状态机嵌入 IR 值类型

```cpp
// include/ptx_ir/statement_context.h:305-311 (CONFIRMED via sed -n '305,320p')
class StatementContext {
public:
    StatementType type;
    std::vector<Qualifier> qualifier;
    InstrVariant data;
    InstructionState state = InstructionState::READY;  // <-- 污染点
    std::string instructionText;
};
```

**风险**: 这是 ptx-lessons-learned §1 "跨模块间接状态翻译" 教科书案例 —— IR 值类型携带执行状态, 调度器 `sync_to_warp_state()` 翻译时易遗漏, 重现 `barrier 迁移 set_state() 翻译事故` 结构。

**Phase 0 净化方案**:
- 选项 1 (推荐): 移出 `StatementContext`, 改 `unordered_map<StatementContext*, InstructionState>` runtime side-table
- 选项 2: 文档化为 non-ABI + 加 `[[nodiscard]]` 强制读取

⚠️ **Oracle 引用 `include/ptxir/ptxir_format.h:16` PTXIR_VERSION=4 claim 未验证** (该文件不存在), 故 PTXIR 版本频率不做 rate-claim。

---

## 3. 4 决策点答复 (PTX-EMU owner 明确答复)

### 决策点 1: StatementContext 公共化路径 → **(a) 晋升 `ptxemu/ir/statement.h`** [CONDITIONAL]

**答复**: 选 (a) 晋升路径, 但**前置 Phase 0 净化** §2.3 的 2 个污染点。

**理由**:
- 闭包审计 (§2.1) 通过 CppTLM Decision 5 的 "pure data, no implementation" 结构门槛
- CppTLM Decision 5 显式锁 "sizeof visibility is mandatory", 与路径 (b) opaque handle 直接冲突
- 路径 (b) 题面描述含内在矛盾 (CppTLM 用现有 PtxirReader 仍会暴露 StatementContext)
- Phase 0 净化 2 污染点后, 路径 (a) 无 CppTLM 侧阻塞项
- 1 年维护税率低于路径 (b) (不需每次 ptx_op.def 增列触发 PTXIR 跨仓回归)

**降级 fallback**: 若 Phase 0 净化发现 `state` 字段被 5+ 处执行引擎深度依赖无法剥离, 降级路径 (b) opaque StatementHandle + decode_ptxir 直提交字节流。

### 决策点 2: PTX-EMU 端 CI 集成 → **本期 drift_check, 下期 consumer_smoke**

**答复**: 
- **本期** (`Phase 2 PR` 内): 新建 `.github/workflows/drift_check.yml` workflow, 仅做 `device_api.h` PUBLIC 接口 vs CppTLM submodule `external/PTX-EMU/include/ptxemu/device_api.h` 一致性对比 (~5min)
- **下期** (HSK-9 准入): `consumer_smoke` 独立 workflow, 不进 `build-and-test` 主 gate (避免 PTX-EMU CI 依赖 CppTLM 构建链, 违背 HSK-6 单向消费关系)

### 决策点 3: `PROJECT_IS_TOP_LEVEL` 隔离模式 → **接受 + 强化约束**

**答复**: 接受 spec 默认配置, 增加约束:

```cmake
# PTX-EMU CMakeLists.txt 顶部 (新增)
option(PTXEMU_BUILD_TESTING "Build PTX-EMU tests" OFF)

if(PROJECT_IS_TOP_LEVEL OR PTXEMU_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# include/CMakeLists.txt 仅在目标 PUBLIC 时安装 (新增)
install(TARGETS ptxemu_core
    EXPORT ptxemu_core_targets
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include)
```

**约束强化**:
- 默认 `PTXEMU_BUILD_TESTING=OFF` — CppTLM submodule 消费时不触发 PTX-EMU 测试构建, 避免与 CppTLM 自身 ctest gate 冲突
- `consumer_smoke` 必须在 `PTXEMU_BUILD_TESTING=ON` 才编译, 默认 OFF
- `install()` 规则仅导出 `ptxemu_core` 库, 不导出 tests

### 决策点 4: Phase 2 PR 排期 → **12-15d, 目标 2026-09-19 前合入**

**答复**: PR base = `origin/main` (post `09786635`), 工作量因 Phase 0 净化上调:

| 内容 | Oracle 调整后估算 | MUST-RESOLVE 因素 |
|---|---|---|
| Phase 0 闭包净化 (§2.3 2 污染点) | **3-4d** (必须前置) | `operand_phy_addr` 38 处调用点审计 + `state` 字段 8 处 set_state 翻译路径审计 |
| `include/ptxemu/device_api.h` (~200 行) | 1d | 无变化 |
| 5 文件晋升至 `ptxemu/ir/` + `ptxemu` namespace | 1.5d | 含 2 `.def` X-Macro 文件路径调整 + 旧路径 forwarding header 一个 release 周期 |
| `src/ptxemu/device_api_impl.cc` (~400 行薄适配层) | 1-2d | EXE_STATE ↔ ThreadState static_assert 锁 |
| `add_library(ptxemu_core STATIC ...)` + PUBLIC/PRIVATE 拆分 | 0.5d | 无变化 |
| `PROJECT_IS_TOP_LEVEL` 隔离 + `option(PTXEMU_BUILD_TESTING OFF)` | 0.5d | install 规则新增 |
| `tests/build_cpptlm_consume/consumer_smoke.cc` + `drift_check.cmake` | 2d | drift_check workflow YAML 配套 |
| 内部 `EXE_STATE` ↔ 公共 `ThreadState` static_assert 锁 | 0.5d | 无变化 |
| Metis pre-impl review (per ptx-lessons-learned §7) | 1-2d | 开 Phase 0 前必跑, 验证闭包闭包自洽 |
| **总计** | **12-15d** (= 2.4-3 man-week) | Oracle 上调 50% over spec 1-2d 估算 |

**目标合入窗口**: **2026-09-19 前** (= spec ack 截止 2026-09-05 后 + 14d 实施 + 1d 缓冲)

---

## 4. PTX-EMU 端 5 条验收条件

| # | 验收条件 | 当前状态 | Phase 2 PR 后状态 |
|---|---------|---------|-----------------|
| 1 | `include/ptxemu/device_api.h` 已新增 | ⏳ 待 Phase 2 | ✅ Phase 2 PR |
| 2 | `add_library(ptxemu_core STATIC ...)` 可被 `add_subdirectory` 消费 | ⏳ 待 Phase 2 | ✅ Phase 2 PR |
| 3 | `consumer_smoke` 测试 PASS | ⏳ 待 Phase 2 (本期仅 drift_check, consumer_smoke 下期) | ⏳ HSK-9 准入 |
| 4 | `drift_check` 通过 | ⏳ 待 Phase 2 | ✅ Phase 2 PR |
| 5 | PTX-EMU maintainer 在 #22 评论 +1 ack | ✅ **本文件 + issue #22 评论** | ✅ |

## 5. 跨仓协调顺序 (per HSK-8 spec §"跨仓协调顺序")

| 步骤 | 责任方 | 内容 | 当前状态 |
|---|---|---|---|
| 1 | PTX-EMU | HSK-8 ack commit (即本文件) | ✅ 本 commit |
| 2 | PTX-EMU | Phase 2 PR (`feat/ptxemu-public-device-api` → main) | 🔜 2026-08-23 开工 |
| 3 | PTX-EMU | CI 全绿 (drift_check ✅ + 自身测试 ✅) | ⏳ |
| 4 | PTX-EMU | PR 合入 main | 🎯 2026-09-19 前 |
| 5 | CppTLM | bump PR (submodule pin + add_subdirectory + 桥接残留簇删除) | ⏳ 等 Phase 2 PR 合入后 |

**禁止跨级**:
- ❌ PTX-EMU PR 基于 `c2038a93` 或更早 commit (引用 `g_cpptlm_bridge`, 库目标无法独立链接) — PTX-EMU 已知 `c2038a93` 不在 main commit graph, 此约束自动满足
- ❌ CppTLM bump PR 在 PTX-EMU PR 合入前提交 (submodule pin 解析失败)

---

## 6. 已知债务与监控点

### 6.1 死代码清理窗口 (关闭点)

`include/ptx_ir/statement_context.h:229` — `BarWarpSyncInstr::reconvergenceLabel` 是已标注 dead code (per ptx-lessons-learned §Oracle 引用), 公共化前是清理它的最后低成本窗口。

**建议**: Phase 2 PR 中加入 `reconvergenceLabel` 删除子 commit (1 行 diff + 1 include 减少)。

### 6.2 ptx_op.def 6-tuple 成为跨仓契约

晋升后 `ptx_op.def` 的格式 `X(enum_val, struct_name, str, opcount, _, instr_kind)` (line 20) 成为冻结列序, 改列序是 breaking change。

**缓解**: ptx-op.def 新增 X 行仅追加 (现有 read_legacy_v1/v2 reader 不受影响); 列序改动需 HSK-9 触发。

---

## 7. PTX-EMU owner 决策签字

| 项目 | 答复 |
|---|---|
| HSK-8 主协议 (公共设备 API 契约) | ✅ ACCEPTED |
| StatementContext 公共化路径 | ✅ (a) 晋升 [CONDITIONAL Phase 0] |
| CI 集成策略 | ✅ 本期 drift_check / 下期 consumer_smoke |
| PROJECT_IS_TOP_LEVEL 隔离 | ✅ 接受 + option PTXEMU_BUILD_TESTING OFF |
| Phase 2 PR 排期 | ✅ 12-15d, 目标 2026-09-19 前 |
| 跨仓协调 5 步顺序 | ✅ 接受 |
| Ack 截止 2026-09-05 + 14d 缓冲 | ✅ 接受 |

**Cc**: @ptx_emu_owner · @ptx_emu_architecture_team · @usr_linux_emu_architecture_team · @cpp_tlm_owner

**Refs**:
- CppTLM HSK-8 spec: `3b8f7a5` (https://github.com/chisuhua/CppTLM/blob/openspec/2026-08-21-cpptlm-v05-mvp-s1-ptxemu-integration/docs/superpowers/specs/2026-08-21-hsk-8-ptxemu-public-api.md)
- CppTLM openspec change: `openspec/changes/cpptlm-ptxemu-public-device-api/`
- CppTLM HSK-6 ack 前例: `369cf71`
- PTX-EMU HSK-6 公告: `25e36f60` (commit 模式参考)
- Oracle session: `ses_fd5ef471cffeWvINOBm5E1GMYd` (4 hypothesis 跨项目分析)
- CppTLM 5 条验收条件 (spec §CppTLM 端接受条件): #1 #2 #3 #4 #5
- ptx-lessons-learned §1 (跨模块状态翻译) · §7 (Metis pre-impl review) · §3 (Phase commit 纪律)

---

**起草**: PTX-EMU Architecture Team · 2026-08-22
**接收方**: CppTLM Team (chisuhua)
**状态**: ✅ PTX-EMU Owner ACCEPTED → 触发 Phase 2 PR 实施 (ETD 2026-09-19)

---

## §HSK-8 实践示例：Phase 2.2.1/2.3.1 follow-up 案例 (2026-08-25)

> **目的**: 展示 HSK-8 ack 之后的追加实施如何遵守"public ABI 不变"约束 — 这是 HSK-9 准入准备完成案例,所有 12 IPtxEmuDevice 方法均已 wired,无需新 HSK 即可提交 CppTLM submodule bump。

### 实施约束验证

| 约束 | Phase 2.2.1/2.3.1 验证 |
|------|------------------------|
| HSK-8 §Decision 5 sizeof visibility | ✅ `WarpStatus` 5-field struct 保持原样,无新字段 |
| HSK-8 §Decision 6 ThreadState enum 1:1 | ✅ `map_state` + `map_thread_status` 映射保持 4 值,`Yielded → kIdle` 保守默认 |
| `PTXEMU_API_VERSION=1` 冻结 | ✅ 未 bump,`include/ptxemu/device_api.h` 静态断言锁定 |
| drift_check Invariant 6 (空 body stubs) | ✅ Exemption list 3 → 0,所有 12 方法都有真实 delegation |
| CppTLM 端 0 内部 header includes | ✅ `cpp 不暴露` 约束保持 |
| ctest 零回归 | ✅ 249 → 251 (新增 2 integration tests) |

### 跨仓协调步骤 (per HSK-8 ack §"跨仓协调顺序" + follow-up plan)

| 步骤 | 操作 | 状态 |
|------|------|------|
| 1 | PTX-EMU 完成 Phase 2.2.1/2.3.1 实施 (commit `eb207378` + `4c2fb143`) | ✅ |
| 2 | PTX-EMU CI 全绿 — ctest 251/251 + drift_check 7 invariants PASS | ✅ |
| 3 | PTX-EMU push branch + open PR (#19) | ✅ |
| 4 | CppTLM owner 接收通知 + 验证 PR | 🔄 |
| 5 | PTX-EMU merge PR (squash) | 🔄 |
| 6 | CppTLM submodule bump PR | ⏳ 等 Phase 2.2.1/2.3.1 合入后 |
| 7 | 跨仓 双向 smoke test | ⏳ |

### HSK-9 准入准备完成状态

| HSK-9 准入条件 | 当前状态 |
|----------------|----------|
| 12/12 IPtxEmuDevice 方法 wired | ✅ Phase 2.2.1/2.3.1 完成 |
| drift_check Invariant 6 全 wired | ✅ Exemption list EMPTY |
| 单元 + 集成 + e2e 测试覆盖 | ✅ integration_warp_status_snapshot + integration_device_api_delegation_e2e |
| `cpp 不暴露` 约束保持 | ✅ CppTLM 侧 0 PTX-EMU 内部 header includes |
| 公共 ABI 冻结 | ✅ PTXEMU_API_VERSION=1, WarpStatus 5-field 不变 |

**结论**: HSK-9 准入条件全部满足。任何后续公共签名变更(新方法、字段增删)必须先签发 HSK-9。

### 参考链接

- [OpenSpec change `openspec/changes/phase-2-2-1-3-1-followup/`](../changes/phase-2-2-1-3-1-followup/) — 本次实施 artifacts
- [Postmortem `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` §2026-08-25 Follow-up](../../audits/2026-08-13-hsk8-ptxemu-public-api.md)
- [PR #19](https://github.com/chisuhua/PTX-EMU/pull/19) — Phase 2.2.1/2.3.1 PR
- [Commits `eb207378` + `4c2fb143`](https://github.com/chisuhua/PTX-EMU/commits/feat/phase-2-2-1-3-1-followup) — 实施 commits
