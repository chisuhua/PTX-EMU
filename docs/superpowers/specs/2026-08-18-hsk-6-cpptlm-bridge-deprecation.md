# HSK-6: CppTLM 桥接消费关系废止预告 (Deprecation Announcement)

## 元数据

| 字段 | 值 |
|---|---|
| **HSK 编号** | **HSK-6**（序列：HSK-1 ABI header @ `8dc000ec` / HSK-2 ANTLR4 4.13.2 / HSK-3 ExternalProject_Add / HSK-4 `ptx_arg_sizes[]` / HSK-5 `advance()` deferred）|
| **发起方** | **PTX-EMU Architecture Team**（真相源持有方 `PTX-EMU@ccd34155:include/cudart/cpptlm_bridge.h:14-16` 自述 "PTX-EMU 是 ABI 提供方, CppTLM 是消费方"）|
| **状态** | 🔄 **PROPOSED** → ACCEPTED（CppTLM maintainer ack + UsrLinuxEmu 利益相关方 ack 后）|
| **公告日** | 2026-08-18 |
| **Ack 截止** | **2026-09-01**（14 天，含周末；超时 = 视为无异议 per HSK-1~5 历史惯例）|
| **关联 ADR** | UsrLinuxEmu [ADR-090 v2 commit `37a91b6`](https://github.com/chisuhua/UsrLinuxEmu/blob/main/docs/00_adr/adr-090-ptxir-via-h2d-dma-v2.md) §D5/§D6 |
| **关联 commit** | UsrLinuxEmu `e03b5a1` (ADR-090 v2 Accepted) + `37a91b6` (§D5/§D6 扩充) |

## 仓 HEAD 锚点（v2 §C0.4 新规强制）

```
PTX-EMU    @ccd34155 (真相源持有方)
CppTLM     @585e4ff (消费方)
TaskRunner @cdb3633 (submodule, parallel work item)
UsrLinuxEmu@e03b5a1 (ADR-090 v2 Accepted) + 37a91b6 (Oracle F-NEW-2 修订)
```

## 1. 协议范围（What）

HSK-6 宣告 **CppTLM 仓对 PTX-EMU `cpptlm_bridge.h` 的消费关系废止**。

- **真相源**（**保留不删**）: `PTX-EMU@ccd34155:include/cudart/cpptlm_bridge.h`（294 行）。`CPPTLMBRIDGE_VERSION` 冻结于 2，进入 maintenance-only。任何解冻企图必须发出 HSK-7 公告。
- **消费方 vendored 副本**（**待物理删除**）: `CppTLM@585e4ff:include/cudart/cpptlm_bridge.h`（14837 字节 / 308 行）。与 PTX-EMU 真相源 294 行存在 diff，引用时不得混用行号。
- **替代路径**: `CppTLM v3.0.0`（DGpuBar + Doorbell + SQ/CQ 三件套，详见 UsrLinuxEmu ADR-090 v2 §D3.3）+ PTX-EMU git submodule（Option B per ADR-090 v2 §D3.2）

### 1.1 HSK-5 关闭（`advance()` deferred → CANCELLED）

HSK-5 中保留为 deferred 状态的 `PtxEmuDriverApi::advance()` 函数随 `IPtxEmuDriver` 接口整体删除而永久废止。**HSK-5 状态由 deferred → CANCELLED by HSK-6**。协议序列不再保留悬空项。

## 2. 影响符号清单（11 项，跨两个阶段）

### 2.1 Phase 1 冻结（公告日起 — Mode B E2E 通过前）

冻结期间不删不扩展，仅 Phase 2 才物理删除：

| # | 项 | 位置 | 备注 |
|---|---|---|---|
| 1 | `MemoryBridge` 类 | `CppTLM@585e4ff:include/tlm/gpu/memory_bridge.hh` | 已 ship, 不删不扩展 |
| 2 | `IPtxEmuDriver` 接口 | `CppTLM@585e4ff:include/tlm/gpu/ptx_emu_driver.hh:19` | HSK-5 `advance()` 同步关闭 |
| 3 | `DriverWrapper` 类 | `CppTLM@585e4ff:include/tlm/gpu/ptx_emu_driver.hh:51` |  |
| 4 | `g_ptx_emu_driver` 全局符号 | `CppTLM` 仓（PTX-EMU ↔ CppTLM 入口点） |  |
| 5 | `cpptlm_set_driver` ABI 入口 | `CppTLM` 仓（PTX-EMU 方向） |  |
| 6 | `ptx_emu_driver_shim.cc` | `CppTLM@585e4ff:src/tlm/gpu/ptx_emu_driver_shim.cc` |  |
| 7 | vendored `cpptlm_bridge.h` | `CppTLM@585e4ff:include/cudart/cpptlm_bridge.h` | 14837 字节, 与真相源 294 行有 diff |
| 8 | vendored `pipeline_interface.h` | `CppTLM@585e4ff:include/cudart/pipeline_interface.h` | 1659 字节, 被 `ptx_emu_driver.hh:18` include |
| 9 | vendored `scoreboard_interface.h` | `CppTLM@585e4ff:include/cudart/scoreboard_interface.h` | 1278 字节, 被 `ptx_emu_driver.hh:19` include |
| 10 | vendored `tensor_core_interface.h` | `CppTLM@585e4ff:include/cudart/tensor_core_interface.h` | 1709 字节, 被 `ptx_emu_driver.hh:20` include |
| 11 | `PtxEmuDriverApi` 布局锁 | `CppTLM@585e4ff:include/tlm/gpu/ptx_emu_driver.hh:27` | `static_assert(sizeof(PtxEmuDriverApi) == 64)` |

### 2.2 P0-1 硬门禁（G-D4 静态断言守卫迁移）

**Phase 2 物理删除前置条件**：以下 17 条 static_assert 必须先迁移到新建的 `CppTLM@585e4ff:include/cudart/abi_guards.h`：

| 来源文件 | 行号 | 守卫数量 | 内容 |
|---|---|---|---|
| `cpptlm_bridge.h` | `:243-306` | 16 条 | 6 PipelineId + 6 TcPrecision + 4 `is_same_v` |
| `ptx_emu_driver.hh` | `:27` | 1 条 | `sizeof(PtxEmuDriverApi) == 64` 布局锁 |

**验证**：迁移完成后, 用 PTX-EMU 真相源 `ccd34155:include/cudart/cpptlm_bridge.h:223-290`（14 条 PTX-EMU 侧断言）对比 vendored 副本，确保所有 static_assert 组完整迁移且无遗漏。

### 2.3 Phase 2 物理删除（Mode B E2E 通过后）

满足 P0-1 + Mode B E2E 测试通过 + HSK-6 ACCEPTED 后：

1. 删除 §2.1 表 11 项
2. 同步更新 `CppTLM/CMakeLists.txt:8`（v2.1.0 → v3.0.0 BREAKING bump）
3. 新建 `CppTLM/include/cpptlm_version.h`
4. 移除 `cpptlm_bridge.h` vendored 规则
5. PTX-EMU submodule 在 UsrLinuxEmu 仓 submodule 中已注册
6. **门禁未过, 不进入 Phase 2**（per ADR-090 v2 §D6.3）

## 3. 双向确认 Checklist

### 3.1 PTX-EMU 端（发起方）

- [ ] **已发出**: 本公告 commit `??` in chisuhua/PTX-EMU
- [ ] **同步**: 通知 CppTLM maintainer + UsrLinuxEmu Architecture Team
- [ ] **CPPTLMBRIDGE_VERSION 锁定 2**: 任何解冻触发 HSK-7 公告
- [ ] **真相源保留**: PTX-EMU 仓 `include/cudart/cpptlm_bridge.h` 不删

### 3.2 CppTLM 端（消费方）

- [ ] **P0-1 门禁**: G-D4 静态断言（17 条）全部迁移至 `abi_guards.h`
- [ ] **删除清单标注**: Phase 1 冻结（11 项）；Phase 2 待 P0-1 + Mode B E2E 通过
- [ ] **替代路径**: 实施 CppTLM v3.0.0 dGPU board（DGpuBar + Doorbell + SQ/CQ per ADR-090 v2 §D3.3）
- [ ] **Module B**: `git submodule` 集成 PTX-EMU per ADR-090 v2 §D3.2

### 3.3 UsrLinuxEmu 端（利益相关方）

- [ ] **已 commit**: ADR-090 v2 `e03b5a1` (✅ Accepted) + `37a91b6` (§D5/§D6 扩充)
- [ ] **Module B 准备**: Mode A 双轨测试 + E2E 验证（per ADR-090 v2 §E）
- [ ] **Fence 信号**: CompletionRing push → host_notify → HAL fence signal 链路（per §D3.4）
- [ ] **ANTLR4 spike**: 已确认 OWNER = UsrLinuxEmu + PTX-EMU（**不属 CppTLM scope**，per CppTLM #19 ack）

## 4. 监控点（per HSK-2 历史惯例）

### 4.1 CPPTLMBRIDGE_VERSION 冻结

`CPPTLMBRIDGE_VERSION = 2` 永久冻结。任何解冻企图（包括但不限于 PTX-EMU 升 VERSION 3、CppTLM 兼容新 VERSION）必须发出 **HSK-7** 公告，禁止静默 bump（per `PTX-EMU@ccd34155:include/cudart/AGENTS.md` "CPPTLMBRIDGE_VERSION bump 治理"）。

### 4.2 替代路径触发监控

- CppTLM v3.0.0 版本号 bump（`CMakeLists.txt:8`）作为 Phase 2 物理删除的就绪信号
- UsrLinuxEmu submodule bump（`external/PTX-EMU`）作为 Mode B E2E 集成测试的触发条件

### 4.3 双向 ack 超时兜底

按 HSK-1~5 历史惯例（响应周期均 1-2 天）,14 天已含 10 倍余量。**超时未 ack = 视为无异议**（deferred to next HSK），防止 P0 窗口期死锁。

## 5. 跨仓 commit 顺序（per ADR-035 §R5.1）

```
[1] PTX-EMU commit `??`: 本 HSK-6 公告发出 ✅ (此处)
   ↓ (CppTLM ack + UsrLinuxEmu ack)
[2] CppTLM commit `??`: P0-1 门禁 + G-D4 迁移 + openspec change `cpptlm-v3-dgpu-extract/`
   ↓
[3] UsrLinuxEmu commit `??`: submodule bump external/PTX-EMU + E2E 集成测试
   ↓
[4] CppTLM commit `??`: Phase 2 物理删除 + v2.1.0 → v3.0.0 bump
   ↓
[5] TaskRunner (并行): tadr-308 创建 + `IGpuDriver::load_kernel_module` 1 method 新增
```

## 6. 参考

- **真相源**（保留）: `PTX-EMU@ccd34155:include/cudart/cpptlm_bridge.h:14-16`
- **HSK-1 历史**: `PTX-EMU@8dc000ec:include/cudart/cpptlm_bridge.h`（commit `8dc000ec` 是 HSK-1 真相源锁点）
- **HSK-2~5**: `PTX-EMU@main:docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md:446-452` + `2026-07-15-cpptlm-hsk-response.md:132-134`
- **HSK 协议定义**: `PTX-EMU@main:docs/superpowers/specs/2026-07-15-cpptlm-hsk-response.md`
- **AGENTS.md 治理**: `PTX-EMU@ccd34155:include/cudart/AGENTS.md` "CPPTLMBRIDGE_VERSION bump 治理"
- **PTX-EMU 真值源**: `PTX-EMU@ccd34155:include/cudart/cpptlm_module.h:12-52`（8 ABI HSK-6 不影响）
- **CppTLM 消费方**: `CppTLM@585e4ff:include/cudart/{cpptlm_bridge,pipeline_interface,scoreboard_interface,tensor_core_interface}.h`
- **CppTLM 桥接层**: `CppTLM@585e4ff:include/tlm/gpu/{ptx_emu_driver,memory_bridge}.hh` + `src/tlm/gpu/ptx_emu_driver_shim.cc`
- **CppTLM 消费方 ACM**: `CppTLM@585e4ff:include/tlm/gpu/ptx_emu_driver.hh:27`（布局锁）, `:51`（DriverWrapper）
- **ADR-090 v2（上游 canonical）**: UsrLinuxEmu `e03b5a1` + `37a91b6` [docs/00_adr/adr-090-ptxir-via-h2d-dma-v2.md](https://github.com/chisuhua/UsrLinuxEmu/blob/main/docs/00_adr/adr-090-ptxir-via-h2d-dma-v2.md) §D5/§D6
- **CppTLM RFC v3.0** (Gate #2 ack): [chisuhua/CppTLM#19](https://github.com/chisuhua/CppTLM/issues/19)
- **HSK-6 公告草稿（Oracle session 评估）**: UsrLinuxEmu `37a91b6` commit message + 本文档

---

**起草**: UsrLinuxEmu Architecture Team (Sisyphus, 2026-08-18)
**复核**: Oracle session `ses_fef78854dffeLfDJh7p8ELuMLy`（起草前 4 轮评估 + 5-step self-check 通过）
**目标审阅方**: PTX-EMU Architecture Team owner（发起方）+ CppTLM maintainer（消费方 ack）+ UsrLinuxEmu Architecture Team（利益相关方 ack）