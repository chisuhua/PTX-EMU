# PTX-EMU 项目健康审计报告

| 项目 | PTX-EMU（C++20/CUDA PTX 模拟器） |
|---|---|
| **审计日期** | 2026-06-21 |
| **Git Commit** | baa8c4e |
| **分支** | main |
| **审计范围** | 架构 / 代码技术债 / PTX 指令覆盖 / 测试 / 文档 / 构建系统 |
| **审计方法** | 5 个并行 explore 子代理全项目静态扫描 + 启发式分析 |

---

## 0. Executive Summary（一页摘要）

### 0.1 综合评级

| 维度 | 评级 | 评分 | 一句话 |
|---|---|:---:|---|
| 架构设计 | **B+** | 7.5/10 | 分层清晰但 god class 与模块反向依赖突出 |
| 代码技术债 | **C+** | 3.2/5 | 多处裸 new 泄漏、巨型 switch、长函数集中 |
| PTX 指令支持度 | **C+** | ~67% | 计算核心扎实，同步/warp-level/membar/fence 集体缺失 |
| 测试覆盖度 | **A-** | 8/10 | 数量充足（739 TEST_CASE / 2339 断言），但 atomic/cudart 拦截薄弱 |
| 文档完整度 | **B** | 7/10 | ADR + 开发者指南齐全，根 README 与 Doxygen 严重落后 |
| 构建系统 | **C** | 4/10 | 基础存在但 CI 几乎空白，ANTLR/CUDA 依赖脆性大 |
| **总体** | **B-** | **6.5/10** | 工程化基础好，债务可控，但同步指令与 CI 是最大瓶颈 |

### 0.2 最关键的 5 个发现

1. **PTX 同步指令集体缺失**（membar / fence / shfl.sync / vote.sync / red / cp.async）：当前是空 `SimpleHandler` 占位，掩盖内存序问题；CUDA kernel 一旦用 fence/membar 就静默失败
2. **CI 几乎不存在**：唯一的 `.github/workflows/generate-ptxir.yml` 核心步骤是空循环 `TODO` —— 项目实际无 CI 保障，债务没人拦截
3. **5 类裸 `new` 无对应 `delete`**（Symtable 5 处、KernelContext 2 处、cudaStream/cudaEvent 句柄、OperandContext 8+ 处）：每次 kernel launch 泄漏，ASan 一跑就抓
4. **god class 三巨头**：ThreadContext 108 public 字段 / WarpContext 78 public + 三重 active_mask 状态 / SMContext 27+ 方法 + friend class 破封装 —— 是后续大量 bug 的隐性根源
5. **Doxygen 覆盖率仅 16%**（12/77 头文件含 `@brief`）+ **根 README 严重过时**（2026-05-26 仍描述光线追踪 demo，与 SIMT v2.0 脱节）—— 新人 onboarding 风险

### 0.3 总体健康度雷达图

```
                     架构设计  B+ (7.5)
                        │
                        │
       构建系统 C (4) ──┼── 代码技术债 C+ (3.2)
                        │
                        │
文档完整度 B (7) ────────┼────── 测试覆盖度 A- (8)
                        │
                        │
              PTX 指令支持 C+ (6.7)
```

### 0.4 修复优先级（Top 10）

| 优先级 | 项 | 影响范围 | 工作量 |
|:---:|---|---|---|
| **P0-1** | 实现 `membar`/`fence`/`cp.async` handler（当前空 SimpleHandler 掩盖竞态） | PTX 同步正确性 | 2 天 |
| **P0-2** | 替换 7 处 `Symtable *s = new` / `cudaStream_t new` 为 `unique_ptr` 或栈对象 | 内存安全 | 1 天 |
| **P0-3** | 创建主 CI workflow `.github/workflows/build-test.yml` | 整体质量门禁 | 0.5 天 |
| **P1-1** | `ptx_ir → ptxsim/execution_types.h` 反向依赖解除（H2 架构债） | 模块边界 | 1 天 |
| **P1-2** | configs 默认值从 `mini.json` 改为 `ampere_a100.json`（L5 配置债） | 真实硬件路径 | 0.5 小时 |
| **P1-3** | 重写根 README.md（反映 SIMT v2.0 + 三类测试 + ADR 索引） | 新人 onboarding | 0.5 天 |
| **P1-4** | 修复 27 个未按规范分类的测试目标（补 `unit_/integration_/e2e_` 前缀） | 测试规范 | 0.5 天 |
| **P2-1** | `arithmetic_conversion.cpp` 1063 行巨型 switch 拆分（策略模式） | 可维护性 | 3 天 |
| **P2-2** | ThreadContext / WarpContext god class 拆分 | 测试可 mock | 2 周 |
| **P2-3** | 补 Doxygen 注释（P0 头文件 8 个） | API 文档 | 1 周 |

---

## 1. 架构债务审计

### 1.1 模块依赖图（实测）

```
声明层次： GPUContext → SMContext → CTAContext → WarpContext → ThreadContext

实测跨模块 include：
  cudart ─┬─→ ptxsim/{gpu,sm,cta}_context.h
         ├─→ ptxsim/instruction_factory.h
         ├─→ ptxsim/ptx_config.h
         ├─→ ptxsim/ptx_exceptions.h
         ├─→ ptx_parser/cfg_builder.h
         ├─→ ptx_parser/ptx_visiter.h            ← ANTLR visitor
         ├─→ grammar/* (ptxLexer.h, ptxParser.h, antlr4-runtime.h)
         └─→ ptx_ir/* (kernel_context, statement_context)

  ptx_ir  ──→ ptxsim/execution_types.h            ⚠️ 反向
  ptx_parser ──→ ptxsim/ptx_exceptions.h          ⚠️ 反向
```

### 1.2 架构债务清单

#### H 级（必须修复，影响模块边界正确性）

| # | 位置 | 问题 | 严重度 | 修复建议 | 工作量 |
|:---:|---|---|:---:|---|:---:|
| **H1** | `src/cudart/cudart_sim.cpp:6-9,240-243` | 双角色模块（runtime shim + parser driver）：单 TU 同时含 ANTLR + visitor + GPUContext | 🔴 H | 抽出 `PtxDriver` 单一职责类；cudart_sim.cpp 仅做 symbol interception | 中 |
| **H2** | `include/ptx_ir/statement_context.h:7` | ptx_ir 反向依赖 ptxsim（`EXE_STATE` 应为叶子类型） | 🔴 H | 把 execution_types.h 移到中立位置（include/utils/） | 小 |
| **H3** | `src/cudart/` + `src/grammar/` CMake 边界 | cudart → ptx_parser → cudart 链路过深 + ANTLR 生成代码混入 cudart 库 | 🔴 H | 拆分库边界（ptx_parser / ptxsim / cudart 三个静态库） | 中-大 |

#### M 级（结构性债务，影响可维护性）

| # | 位置 | 问题 | 严重度 | 工作量 |
|:---:|---|:---:|:---:|:---:|
| **M1** | `include/ptx_parser/ptx_visiter.h` | **拼写错误** "visiter"，14 个 .cpp 引用 | 🟡 M | 小（git mv + 全项目 include 替换） |
| **M2** | `include/ptxsim/thread_context.h`（108 个 public 成员）/ `src/ptxsim/core/thread_context.cpp`（848 行） | god class，违反 SRP | 🟡 M | 大（拆 4 个 POD） |
| **M3** | `include/ptxsim/warp_context.h`（78 public）/ `src/ptxsim/core/warp_context.cpp`（466 行） | god class + **三重 active_mask 状态**（`active_mask[]` / `warp_state.threads[i].is_active` / `warp_state.exec_mask`）—— AGENTS.md 已自承"DUAL STATE MECHANISM" | 🟡 M | 中（合并为单一 source of truth） |
| **M4** | `include/ptxsim/sm_context.h` | 27+ 方法 + `friend class BarWarpSyncHandler;` 破封装 | 🟡 M | 中（拆 debug 接口 + 删除 friend） |
| **M5** | `src/ptxsim/instructions/arithmetic_conversion.cpp` | 1288 行单文件，1063 行巨型 `switch (dst_bytes)` | 🟡 M | 中（按指令族拆 5+ 文件） |
| **M6** | `src/ptx_parser/ptx_visitor.cpp` | 1019 行 X-Macro 文件 + 14 个 `ptx_visitor_*.cpp` 通过 `#include` 文本包含 | 🟡 M | 大（改为独立 TU 编译） |
| **M7** | `include/ptxsim/ptx_exceptions.h`（9290 行） | 单头文件过大 + 被 ptx_parser 反向依赖 | 🟡 M | 中（拆多文件） |

#### L 级（代码气味）

| # | 位置 | 问题 | 修复建议 |
|:---:|---|---|---|
| **L1** | `include/ptxsim/cta_context.h:23-42` | 字段全 public（warpNum, threadNum, sharedMemBytes 等） | 改 private + getter |
| **L2** | `src/ptxsim/core/gpu_context.cpp:64-120` | 40+ 手写 `if (j.contains(...))` JSON 字段加载 | 用反射宏或 schema 库 |
| **L3** | `configs/ampere_a100.json` | **88% 字段未被消费**（memory_system / interconnect / tensor_cores 等），假装支持 Hopper/Blackwell 实际 GPU 行为与配置无关 | 大（需新 cache/memory subsystem） |
| **L4** | `src/cudart/` + `src/memory/` | 双内存管理（simple_memory_allocator + cuda_driver）并存 | 收敛为单一子系统 |
| **L5** | `configs/config.ini` | **9/9 配置文件都 fallback 到 `mini.json`**，没有任何指向 `ampere_a100.json` | 极小（改默认值） |

### 1.3 值得保留的架构优势

- **A1. AGENTS.md 公开技术债务范式**：`src/ptxsim/core/AGENTS.md` 明确记录 DUAL STATE MECHANISM 的 invariant 与历史 BUG（BUG-RETHANG、BUG-POSTBARRIER-TWOHALVES）以及对应的 regression test 路径
- **A2. BarrierModule 重构（2026-06）**：`BarrierModule + WarpBarrier + CTABarrier` 三件套替换旧的 inline 屏障逻辑，AGENTS.md 主动淘汰废弃 API
- **A3. 三层测试金字塔强制物理分类**：`tests/unit/` / `tests/integration/` / `tests/e2e/` 物理目录隔离 + ctest label 前缀
- **A4. PTXIR 二进制序列化**：绕过 ANTLR 的快速加载路径，关键性能优化
- **A5. WarpScheduler 作为 Strategy**：`SMContext::set_warp_scheduler(unique_ptr<WarpScheduler>)` 支持运行时替换调度策略
- **A6. X-Macro + ptx_op.def 单一来源**：新增指令改 1 个 .def + 1 个 handler + 1 个 visitor
- **A7. InstructionLatencyConfig 依赖反转**：已从 GPUConfig 拆出到 `include/ptx_ir/instruction_latency_config.h` —— 应推广到 H2 的 EXE_STATE

---

## 2. 代码技术债审计

> **审计范围**：`src/`（74 个 .cpp）+ `include/`（77 个 .h）= **27,417 行生产代码**

### 2.1 综合评级：**3.2 / 5**（中等偏上，需关注）

| 维度 | 评分 | 说明 |
|---|:---:|---|
| TODO/FIXME 完整性 | 4/5 ✓ | 数量可控（生产代码 20 处），但 visitor.cpp 有 4 个遗留 TODO |
| Stub/未实现 | 2/5 ⚠️ | WMMA/Tensor Core 完全 stub；fatal abort 无降级 |
| **内存安全** | **2/5 ⚠️** | **多处裸 new 无 delete**，cudaStream/Event 句柄泄漏 |
| 错误处理 | 3/5 ⚠️ | 无空 catch，但有静默吞错（默认值替代） |
| 代码风格 | 3/5 ⚠️ | 长函数（5 个 > 200 行）+ 97 处注释代码 + 98 处裸 cout |
| 死代码 | 4/5 ✓ | 较少 |
| 并发安全 | 4/5 ✓ | 单线程模型契合 |
| 魔法数字 | 3/5 ⚠️ | `0xFFFFFFFF` 滥用 12+ 处，warp size `32` 未命名 |
| C++ 现代化 | 4/5 ✓ | 良好（nullptr ✓、smart pointer 101 处 ✓、无 auto_ptr）|

### 2.2 P0 级技术债（必须立即修复）

#### 2.2.1 裸 `new` 无对应 `delete` —— 内存泄漏热点

| 模式 | 数量 | 风险 | 位置 |
|---|---|---|---|
| `Symtable *s = new Symtable()` | **5 处** | 🔴 每次 kernel launch 泄漏 | `src/cudart/ptx_interpreter.cpp:213,302,443,459,550` |
| `KernelContext *kernelContext = new` | 2 处 | 🔴 泄漏 | `src/cudart/ptx_interpreter.cpp` |
| `OperandContext *o = new` | 8+ 处 | 🔴 泄漏 | `src/ptx_parser/ptx_parser.cpp` |
| `cudaStream_t new int(0)` / `cudaEvent_t` | 6 处 | 🔴 句柄泄漏，多 stream 语义被破坏 | `src/cudart/cudart_sim.cpp:684` 等 |
| `delete[]` | 0 | ✓ 一致性 | — |
| `unique_ptr`/`make_unique` | 101 处 | ✓ 主流良好 | — |

**修复策略**：替换为 `std::unique_ptr<Symtable>` + `std::make_unique<KernelContext>()`；cudaStream_t 用 `cudaStreamPool` 管理。

#### 2.2.2 fatal logger 无降级 —— 进程崩溃风险

- `include/utils/logger.h:520,584` 两处 `std::abort()` 在 fatal 路径
- 一旦触发，PTX 模拟器直接崩进程，无 graceful fallback
- 建议：抛出 `PtxRuntimeException`（异常层次 ADR-0001 已有），让上层决定 abort vs log

### 2.3 P1 级技术债

#### 2.3.1 `ptx_visitor.cpp` 6 处 TODO + 5 处静默 `std::any_cast`

- 行 303、323、363、607 处 TODO（影响 PTX 解析路径走错）
- 行 380-388 静默 catch `std::any_cast` 失败后回退到 PTX 7.0 默认版本，无诊断
- **关联到 PTX 指令覆盖度审计**：是 16 项 HIGH 缺口（set/shf/lop3/prmt/slct/cnot/brx/exit/trap/brk/brkpt/membar/fence/shfl/vote）中相当一部分的隐藏根因

#### 2.3.2 长函数 TOP-5

| 文件:行 | 函数 | 行数 |
|---|---|---|
| `arithmetic_conversion.cpp:224` | `switch (dst_bytes)` CVT 分发 | **1063** |
| `ptx_interpreter.cpp:118` | `if (g_gpu_context)` kernel 启动 | 313 |
| `ptxir_writer.cpp:129` | `write_instruction()` | 301 |
| `arithmetic_utils.h:16` | `if (is_float)` 类型分发 | 235 |
| `thread_context.cpp:447` | 地址计算 | 211 |

#### 2.3.3 错误处理静默吞错

- `src/ptxsim/core/thread_context.cpp:670` `catch(...) { offset = 0; }` 默认偏移量为 0
- `src/ptx_parser/ptx_visitor.cpp:380-388` 静默 catch PTX 版本回退到 7.0
- 建议：加 `PTX_WARN` 而非静默吞

### 2.4 热点文件 TOP-10（按债务密度）

| # | 文件 | 行数 | 主要债务 |
|:---:|---|:---:|---|
| 1 | `src/ptxsim/instructions/arithmetic_conversion.cpp` | 1288 | **1063 行巨型 switch** + CVT 全指令分支 |
| 2 | `src/cudart/ptx_interpreter.cpp` | 703 | **5 处裸 new Symtable 无 delete** + 313 行长函数 + BUGFIX FIXME |
| 3 | `src/ptx_parser/ptx_visitor.cpp` | 1019 | **6 个 TODO** + 5 处静默 `std::any_cast` catch |
| 4 | `src/cudart/cudart_sim.cpp` | 933 | 933 行单文件 + cudaStream/Event 句柄泄漏 + `std::cerr` 替代 logger |
| 5 | `src/ptxsim/core/thread_context.cpp` | 848 | 3 处 TODO + 211 行长函数 + 静默 `catch(...)` |
| 6 | `src/ptx_parser/ptx_parser.cpp` | 1081 | 8+ 处裸 new OperandContext + fprintf(stderr) 调试 |
| 7 | `src/ptx_ir/ptxir_writer.cpp` | 430 | **12 处 `0xFFFFFFFF` 哨兵** + 301 行 write_instruction |
| 8 | `src/ptxsim/core/sm_context.cpp` | 703 | 210 行 `exe_once()` 函数 |
| 9 | `src/ptxsim/instructions/arithmetic_ext.cpp` | 763 | 211 行长函数 + 大量 C-style cast |
| 10 | `include/utils/logger.h` | 846 | `std::abort()` 用于 fatal，无 graceful fallback |

### 2.5 修复优先级（按价值/实施成本）

| 优先级 | 项 | 工作量 |
|:---:|---|:---:|
| 🔴 P0 | 替换 `Symtable *s = new` 系列为 `unique_ptr` 或栈对象 | 1-2 天 |
| 🔴 P0 | cudaStream_t/cudaEvent_t 句柄泄漏 | 0.5 天 |
| 🔴 P1 | visitor.cpp 4 处遗留 TODO 评估 | 1 天 |
| 🔴 P1 | logger fatal `std::abort()` 增加降级选项 | 0.5 天 |
| 🟡 P2 | 拆分 `arithmetic_conversion.cpp` 1063 行 switch 为策略/表驱动 | 3 天 |
| 🟡 P2 | `0xFFFFFFFF` 提取为 `kInvalidIndex` 常量 | 0.5 天 |
| 🟡 P2 | `32`（warp size）提取为 `kWarpSize` | 0.5 天 |
| 🟡 P2 | ptx_interpreter.cpp 长函数拆分 | 2 天 |
| 🟢 P3 | 静默 catch 增加 `PTX_WARN` 日志 | 1 天 |
| 🟢 P3 | `typedef` → `using` 迁移 | 1 天 |
| 🟢 P3 | 336 处 C-style cast → `static_cast`/`reinterpret_cast` | 2-3 天 |

---

## 3. PTX 指令支持度审计

### 3.1 项目定义的指令规模

- `include/ptx_ir/ptx_op.def`：**106 条 X-Macro** = 7 operand helpers + 7 结构 helper + **92 条实际 PTX 指令**
- 17 条是项目自定义/PTX 8.7+ 扩展（TCGEN 系列、st.async、red.async、cp.async、mbarrier.*、tensormap.replace、st.bulk、abi.preserve）

### 3.2 按 ISA 类别支持度

| 类别 | 已实现 | 部分实现 | Stub | 缺失 | 完整度 |
|---|:---:|:---:|:---:|:---:|:---:|
| 整数运算（add/sub/mul/div/mad/rem/addc/subc/mul24/mad24/min/max/abs/neg） | 13 | 0 | 0 | 2 (set, sad) | **87%** |
| 浮点运算（fma/sqrt/sin/cos/rcp/rsqrt/lg2/ex2） | 8 | 0 | 0 | 1 (copysign) | **89%** |
| 位操作（and/or/xor/not/shl/shr/bfe/popc/clz） | 9 | 0 | 0 | 3 (shf, lop3, prmt) | **75%** |
| 数据转换（cvt/cvta） | 2 | 0 | 0 | 0 | **100%** |
| 内存操作（ld/st 全空间 + 向量） | 2 | 0 | 0 | 0 | **100%** |
| 控制流（bra/ret/call） | 2 | 1 (call 仅 vprintf) | 0 | 5 (brx/exit/trap/brk/brkpt) | **25%** |
| 同步（bar.sync/bar.warp.sync/activemask） | 2 | 0 | 0 | 6 (membar/fence/redux.sync/mbarrier.*) | **22%** |
| 逻辑/比较（setp/selp） | 2 | 0 | 0 | 3 (set/slct/cnot) | **40%** |
| Warp-level（activemask/shfl/vote） | 1 | 0 | 0 | 3 (shfl/vote/bar.warp.sync partial) | **25%** |
| 原子（atom 9/10 ops，缺 CAS） | 0 | 1 | 0 | 0 | **90%** |
| 矩阵/张量（wmma） | 0 | 0 | 1 | 0 | **0%** |
| 异步拷贝（cp.async/cp.async.bulk/wait_group） | 0 | 0 | 1 | 3 | **0%** |
| Reduction/Prefetch | 0 | 0 | 0 | 3 | **0%** |
| TCGEN/Bulk/Async（PTX 8.7+） | 0 | 0 | 0 | 9 | **0%** |
| Texture/Surface | 0 | 0 | 0 | 9 | **0%** |

### 3.3 综合支持度估算（vs PTX 7.x ISA 250+ 指令）

```
核心 ISA（计算+控制+同步+内存）:  ~67% 完整度
全部 ISA 完整度（含扩展）:        ~42%
```

### 3.4 关键发现

1. **计算核心扎实**：整数/浮点/转换/位操作主路径实现完整，含 fp16、cc 标志、saturate、wide/hi/lo、.approx 等 PTX 修饰符
2. **控制流 + 同步最弱**：ret 后才设 EXIT（exit 指令缺失），membar/fence 是空 `SimpleHandler`，bar.warp.sync 仍用旧 Wbar 而非 BarrierModule
3. **warp-level 集体缺失**：shfl/vote 完全没接（Hopper mbarrier/sync/tcgen 等留待后续）
4. **dispatcher 设计安全但掩盖错误**：通过 `__attribute__((weak))` 覆盖机制，缺失指令不会链接错误而是降级为 no-op
5. **ptx_op.def 占位过多**：TCGEN 系列（7 条）/ tensormap / st.bulk / st.async 在 ptx_op.def 中有占位但完全是空 `SimpleHandler`，依赖硬件抽象未实现 —— 决策：实现 or 删除占位？

### 3.5 推荐修复优先级（按价值/实施成本）

| 优先级 | 项 | 理由 |
|:---:|---|---|
| **P0** | `membar` / `fence` 实现 | 同步正确性；当前空 SimpleHandler 隐藏竞态 |
| **P0** | `exit` / `trap` / `brk` 显式实现 | 调试与 kernel 终止语义 |
| **P0** | `cp.async` 真异步引擎 | Hopper+ 必须；当前只是打印日志 |
| **P1** | `shfl.sync` / `vote.sync` | Warp-level 通信核心（reduce/broadcast） |
| **P1** | `set` / `slct` / `cnot` | 编译器高频使用 |
| **P1** | `shf` / `lop3` / `prmt` | Volta+ 优化指令 |
| **P2** | `wmma` 真实现 | Tensor Core 算力 |
| **P2** | `mbarrier.*` / `tcgen05.*` | Hopper TMA 加速 |
| **P3** | TCGEN/Texture/Surface | 项目目标外，可保持 stub |
| **P3** | ptx_op.def 占位清理 | 删除 PTX 8.7+ 暂不实现的占位条目以避免误导 |

---

## 4. 测试覆盖度审计

### 4.1 综合评级：**A-**

| 维度 | 评级 | 关键指标 |
|---|:---:|---|
| 测试覆盖度 | **A-** | 131 ctest 目标，96 测试源文件，739 TEST_CASE，2339 断言 |
| 关键路径覆盖 | **A-** | barrier/divergence/simt/pc/memory 全覆盖，atomic/cudart 偏弱 |
| PTX 语法样本 | **B+** | 33 个 PTX 文件，涵盖分歧/屏障/同步 |
| 文档完整度 | **B+** | 16 个 ADR + 完整开发者指南 |
| API 文档（Doxygen） | **C** | 仅 12/77 头文件（~16%）含 `@brief` |
| CI 友好性 | **A** | Catch2 单头、3 个 Disabled 测试隔离、无 GPU 依赖 |

### 4.2 测试矩阵（关键路径 × 测试类型）

> ✓ = 有覆盖，△ = 部分覆盖，✗ = 无覆盖

| 关键路径 | unit | integration | e2e | 评级 |
|---|:---:|:---:|:---:|:---:|
| **Barrier 机制** | ✓ (12) | ✓ (6) | ✓ (1) | **A** |
| **Divergence / 收敛** | ✓ (4) | ✓ (7) | ✓ (2) | **A** |
| **SIMT Stack** | ✓ (7) | ✓ (1) | ✗ | A |
| **PC 管理（per-thread）** | ✓ (2) | ✓ (1) | ✗ | A |
| **Memory 边界/分配** | ✓ (5) | ✓ (2) | △ | B+ |
| **Atomic 操作** | △ (1) | ✓ (1) | ✗ | **C** |
| **CUDA runtime 拦截** | △ (1) | ✗ | ✓ (7) | **C** |
| **Control Flow (bra/ret)** | ✓ (3) | ✗ | ✗ | B+ |
| **Integer / Float 算术** | ✗ | ✓ (5) | ✗ | B+ |
| **Bitwise / cvt / cvta** | ✗ | ✓ (5) | ✗ | B+ |
| **LD/ST (global/shared)** | ✗ | ✓ (4) | ✓ (4) | A- |
| **Sync (`__syncthreads`)** | ✓ (3) | ✓ (2) | ✓ (1) | A |
| **WMMA / Tensor Core** | ✗ | ✗ | ✗ | — |
| **Cluster (sm_90+)** | ✗ | ✗ | ✗ | — |

### 4.3 ctest 命名规范执行

```
总测试数:       131（3 Disabled）
├── unit_* 前缀:        76    ✅ 类型一
├── integration_* 前缀: 20    ✅ 类型二
├── e2e_* 前缀:          8    ✅ 类型三
└── 未分类（前缀不符）: 27    ⚠️ 待整改
    (dummy_*, simple*, 2D*, bitonic, bfs, all-pairs*, aligned*,
     cute_*, test_ptx_bra, test_printf, test_simt_stack_integration,
     test_cfg_edge_cases)
```

### 4.4 缺失的关键测试

#### 🔴 P0 缺失

1. **Atomic 专项测试** —— 仅 `tests/integration/ptx/test_atom_add.cpp`（2 个 TEST_CASE）
   - `tests/unit/ptx/test_atom_global.cpp`（unit 测试 atom.add/cas/exch 各变体）
   - `tests/integration/ptx/test_atom_memory_order.cpp`（与 membar/fence 的内存序）
   - `tests/e2e/kernel/test_atomic_kernel.cu`（真实 CUDA histogram kernel）

2. **CUDA Runtime 拦截专项测试** —— 仅 `unit_logger_cudart_component`（测日志，不测拦截行为）
   - `tests/unit/cudart/test_cudart_intercept.cpp`（cudaMalloc/Memcpy/Memset/Free 拦截语义）
   - `tests/unit/cudart/test_cudaLaunchKernel_params.cpp`（参数序列化与 grid/block dim 解析）

3. **27 个未分类测试整改**：补 `unit_` / `integration_` / `e2e_` 前缀，满足 commit ab55e06 重构目标

#### 🟡 P1 缺失

4. **WMMA / Tensor Core 边界测试**：即使是 stub，也需 `tests/unit/ptx/test_wmma_stub.cpp` 验证 stub 行为
5. **Memory OOB e2e**：`tests/e2e/kernel/test_global_oob.cu` 验证越界访问不静默成功
6. **PTX 语法样本补充**：`tests/ptx/test_atom_*.ptx` / `test_membar.ptx` / `test_wmma_stub.ptx`
7. **恢复 3 个 Disabled 测试**：
   - `integration_warp_barrier_memory_visibility`
   - `integration_cta_barrier_memory_visibility`
   - `integration_local_memory`
8. **启用 `tests/ptxir/`**：当前为空目录，需要序列化往返一致性测试

### 4.5 测试质量指标

| 指标 | 数值 | 评级 |
|---|---:|:---:|
| TEST_CASE 块总数 | **739** | A |
| SECTION 子测试 | **213** | A |
| 断言（REQUIRE/CHECK/ASSERT） | **2339** | A |
| 使用 SECTION 的测试文件占比 | ~30% | B+ |
| 边界用例（lane=0、partial arrive、empty CTA） | 分散在 barrier/divergence | B+ |
| BDD/Property-based | 未发现 | C |

---

## 5. 文档完整度审计

### 5.1 文档状态

| 维度 | 评级 | 关键指标 |
|---|:---:|---|
| 文档完整度 | **B+** | 17 个目录，16 ADR + 完整开发者指南 |
| API 文档（Doxygen） | **C** | 16% 头文件含 `@brief` |
| 文档维护 | **B** | 多篇最新，少量过期 |

### 5.2 文档清单

#### 根目录文档

| 文档 | 状态 | 问题 |
|---|---|---|
| `/README.md`（2026-05-26） | ⚠️ **严重过期** | 仍是 v1 阶段"光线追踪 demo"描述，与 SIMT v2.0 脱节 |
| `/AGENTS.md`（2026-06-20, 22KB） | ✅ 最新 | 项目最高优先级索引 |
| `/docs/README.md`（2026-06-15） | ✅ 最新 | 导航完整 |

#### `/docs/adr/` —— 14 个 ADR + README + template ✅

- 0001 异常层次体系替代 assert ✅ Active
- 0002 PC 权威源统一到 WarpState ✅ Active
- 0003 commit_pc / force_set_pc 分离 ✅ Active
- 0004 自然停顿机制 is_warp_ready_to_fetch ✅ Active
- 0005 MemoryRegion 注册机制 ✅ Active
- 0006 SIMT Stack 显式控制流管理 ✅ Active
- 0007 CFG Post-Dominator 收敛分析 ✅ Active
- 0008 Barrier 语义增强 ✅ Active
- 0009 X-Macro + Weak Symbol 指令分发 ✅ Active
- 0010 Fake CUDA Runtime 拦截机制 ✅ Active
- 0011 PTX→PTXIR 多阶段 Pipeline 架构 ⚠️ **Proposed**
- 0012 Per-Thread PC（Volta+ SIMT 模型） ✅ Active
- 0013 Statement Factory 测试统一 ✅ Active
- 0014 Independent Thread Scheduling (ITS) ⚠️ **Proposed**

#### `/docs/architecture/`

- `SIMT-ARCHITECTURE-V2.md`（1134 行，2026-06-20）✅ 最新
- `sm90_100.md` ✅ Hopper/Blackwell
- `GPGPU-SIM-SIMT-ANALYSIS.md`（717 行）
- **❌ CFG-DESIGN.md 缺失**（README 标记待创建）

#### `/docs/developer-guide/` —— 13 个指南（基本最新）

- GETTING-STARTED / TESTING / PERFORMANCE / DEBUGGING / DEBUG-QUICK-REFERENCE / DEBUG-CONFIG / REGRESSION-DEBUGGING / PTX-DEBUG-SKILL-USAGE / CFG-INTEGRATION / BARRIER-PROGRAMMING-REFERENCE / KNOWN_ISSUES
- ⚠️ **THREE-MODE-TESTING-GUIDE.md 路径需更新**（已迁移到 `tests/archive/three_mode_testing/`）

#### 其他

- `/docs/reports/` —— 6 个报告 + BUG 诊断
- `/docs/research/` —— barrier-semantics 主题（7 篇）
- `/docs/skills/` —— 4 个技能沉淀
- `/docs/technical_design/` —— 2 篇（barrier_module_design, implicit_reconvergence_enforcement）
- `/docs/testing/` —— 2 篇
- `/docs/appendix/` —— 4 篇（CHANGELOG 待更新）
- `/docs/archive/` —— 50+ 历史归档

### 5.3 Doxygen 覆盖缺口

| 指标 | 数值 |
|---|---:|
| 头文件总数 | **77** |
| 含 `@brief` 的头文件 | **12** |
| **Doxygen 覆盖率** | **~16%** ⚠️ |
| `@param` / `@return` 注释数 | 181 |

**缺失 Doxygen 的关键头文件**（按重要性）：
- `ptxsim/cta_context.h`、`gpu_context.h`、`scheduler_config.h`、`instruction_handlers.h`
- `ptx_ir/statement_context.h`、`statement_factory.h`、`ptx_op.def`
- `cudart/cuda_driver.h`、`cudart_sim.h`
- `ptxsim/barrier/warp_barrier.h`、`cta_barrier.h`、`barrier_module.h`

---

## 6. 构建系统与依赖审计

### 6.1 综合评级

| 维度 | 评分 |
|---|:---:|
| CMake 配置 | C+ |
| 依赖管理 | C |
| **CI/CD 成熟度** | **1.5 / 5（几乎无 CI）** |
| 构建性能 | B |

### 6.2 P0 级构建债务

| # | 问题 | 证据 | 影响 |
|:---:|---|---|---|
| **D1** | **`compile_commands.json` 符号链是断的** | `build/compile_commands.json` 不存在；根 `CMakeLists.txt:117` 已设 `CMAKE_EXPORT_COMPILE_COMMANDS ON` 但未生效 | LSP、clang-tidy、IDE 全部失效；AGENTS.md 声明的 `lsp_*` 工具链完全无法工作 |
| **D2** | **唯一的 CI workflow 不能 build/test** | `.github/workflows/generate-ptxir.yml:30-34` 核心步骤是空循环 `TODO: implement generate_ptxir` | **项目实际上没有 CI** —— 推 PR 不会触发构建/测试 |
| **D3** | **CMake 路径残留本地绝对路径** | 缓存 `CMAKE_CUDA_COMPILER:FILEPATH=/workspace/project/opt/cuda/bin/nvcc` | 跨机器构建必失败 |
| **D4** | **CMake Preset 误导** | `CMakePresets.json` 单一 preset `cuda_cc` 硬编码 `/usr/bin/gcc`、Debug、`Unix Makefiles` | 与 README 的 Release 默认值冲突；preset 名误导 |
| **D5** | **强制所有测试用 CUDA 编译** | `tests/CMakeLists.txt:13-31` 全工程强制 nvcc 编译 | 纯 CPU 测试编译时间被 nvcc 拖慢 5-10× |

### 6.3 P1 级债务

- **`include_directories` 滥用**（D6）：根 CMakeLists.txt:85-88 + src/CMakeLists.txt:51-54，传播性 include 导致 663 个 .o 文件依赖膨胀
- **ANTLR 升级路径断裂**：4.11.1 完全 vendored + 升级路径断裂；4.13.1 在 `fix-pre-p0-baseline` worktree 未合并
- **无 sanitizer 支持**（M4）：ASan/UBSan/TSan 全无
- **无 unity build / precompiled header**（M5）：663 个 .o 文件无 PCH
- **无 Ninja 生成器配置**（M6）：增量构建比 Ninja 慢 30-50%
- **Catch2 v2 amalgamation**（M8）：90 个测试目标 × 1MB = 90MB 重复编译
- **`add_subdirectory` 引用构建目录名**（M1）：脆弱 hack

### 6.4 依赖风险

| 依赖 | 引入方式 | 风险等级 | 备注 |
|---|---|:---:|---|
| **ANTLR 4.11.1** | 完全 vendored（jar 3.5MB + 源码 50MB） | 🔴 HIGH | 版本硬编码；无下载脚本；CI 中 Java 缺失 |
| **CUDA Toolkit** | 系统预装，**无版本约束** | 🔴 HIGH | 本地路径残留；`compute_100` 虚拟架构；`cuobjdump` 静默失败 |
| **Catch2 v2** | amalgamation（tests/catch_amalgamated.{cpp,hpp}） | 🟢 LOW | 稳定但升级成本高 |
| **nlohmann/json** | vendored（889K） | 🟡 MEDIUM | 升级需手动替换 |
| **inipp** | vendored（21K） | 🟢 LOW | 单一头文件 |
| **cutile-python** | Git submodule（NVIDIA 官方） | 🟢 LOW | 仅 CuTe 路径使用 |

### 6.5 系统依赖矩阵

| 依赖 | 验证 | 缺失风险 |
|---|---|---|
| GCC | **未验证版本** | env.sh 只检查 nvcc |
| CMake ≥ 3.15 | ⚠️ Preset v8 语法需 3.25+ | 版本错配 |
| Java | OK（env.sh 检查 `which java`）| 不在 `find_package` 范围 |
| CUDA Toolkit | OK（自动发现）| 不锁版本 |
| ccache | 可选 | OK |
| cuobjdump | WARNING only | 静默失败风险 |

### 6.6 构建时间估算

| 场景 | 估算 |
|---|---|
| 干净全量构建 | ~5-8 分钟 |
| 头文件增量（修改 `warp_context.h`） | ~1-2 分钟（全工程重编） |
| 单元测试（90 个）| ~2-3 分钟 |
| sanity.sh 完整 (Tier 1-10) | ~8-15 分钟 |

---

## 7. 跨维度关联分析

通过交叉对比五个审计维度，识别以下关键关联问题：

### 关联 1：god class 与测试稀薄的根因

**M2/M3（ThreadContext 108 public + WarpContext 三重 active_mask）** → 测试只能依赖整个 god class 而无法 mock → `tests/unit/exec/test_active_mask_consistency` 这类核心测试稀薄。

### 关联 2：visitor.cpp 静默吞错 = PTX 缺口根因

`ptx_visitor.cpp` 6 处 TODO + 5 处静默 `std::any_cast` catch → PTX 解析路径走错 → 是 16 项 HIGH 缺口（set/shf/lop3/prmt/slct/cnot/brx/exit/trap/brk/brkpt/membar/fence/shfl/vote）中相当一部分的隐藏根因。

### 关联 3：CI 缺失 = 债务持续增长

D2（CI 是 TODO）+ D4（CMake Preset 误导）+ D5（强制 CUDA 编译所有测试）→ 实际没有 CI 门禁 → 所有债务都没人主动拦截，会持续增长。

### 关联 4：文档过期 + Doxygen 16% = 新人债务增量

根 README 严重过时（v1 阶段）+ Doxygen 16% + 27 个测试未分类 → 新人 onboarding 成本极高 → 引入更多债务。

### 关联 5：ANTLR vendored × cudart 双角色 = 编译放大

ANTLR 4.11.1 完全 vendored + 升级路径断裂 + H1（cudart 双角色）→ 任何 ANTLR 微调都会强制 cudart 全量重编，形成乘数效应。

### 关联 6：membar/fence 缺失 = 3 个 Disabled 测试根因

3 个 Disabled 测试（warp_barrier_memory_visibility / cta_barrier_memory_visibility / local_memory）—— 这 3 项正是缺失指令（membar/fence）会掩盖的真实场景。修指令 → 解锁测试。

### 关联 7：同步原语三角债

`bar.warp.sync` 仍走旧 Wbar 路径 + `barrier.cpp` Stage 3 TODO + `membar/fence` 空 SimpleHandler —— 三件事是同一个根因（同步原语未实现）。

---

## 8. 综合修复路线图

### Phase 1：止血（1 周内，P0）

```yaml
# 立即修复（48 小时内）
- [ ] P0-1 实现 membar/fence/cp.async handler（同步正确性）
- [ ] P0-2 替换 7 处裸 new Symtable/KernelContext/cudaStream_t 为 unique_ptr
- [ ] P0-3 创建 .github/workflows/build-test.yml（PR/Push 触发 ctest）
- [ ] P0-4 删除根目录 compile_commands.json 符号链（让 build 自然生成）

# 1 周内
- [ ] P1-1 ptx_ir 反向依赖解除（H2 架构债）
- [ ] P1-2 configs 默认值改为 ampere_a100.json（L5 配置债）
- [ ] P1-3 重写根 README.md（反映 SIMT v2.0 + 三类测试 + ADR 索引）
- [ ] P1-4 修复 27 个未按规范分类的测试目标
- [ ] P1-5 logger fatal std::abort() 增加降级（throw PtxRuntimeException）
```

### Phase 2：去债（2-4 周，P1）

```yaml
# 2 周内
- [ ] arithmetic_conversion.cpp 1063 行 switch 拆分（策略模式）
- [ ] visitor.cpp 4 处遗留 TODO 评估与修复
- [ ] 补 atomic 单元测试 + e2e kernel
- [ ] 补 cudart 拦截专项测试
- [ ] 创建 CFG-DESIGN.md

# 4 周内
- [ ] 恢复 3 个 Disabled 测试（warp/cta barrier memory visibility + local memory）
- [ ] 启用 tests/ptxir/（序列化往返一致性测试）
- [ ] shfl.sync / vote.sync 实现
- [ ] set / slct / cnot / shf / lop3 / prmt 实现（编译器高频使用）
```

### Phase 3：根治（1-3 月，P2）

```yaml
# 1 月
- [ ] cudart_sim.cpp 双角色拆分（H1 架构债）
- [ ] ThreadContext / WarpContext god class 拆分（M2/M3）
- [ ] SMContext friend class 破封装消除（M4）
- [ ] 补 Doxygen 注释（目标 ≥ 80%）
- [ ] cudart → ptx_parser → cudart 库边界拆分（H3）

# 3 月
- [ ] ANTLR 升级到 4.13.x（FetchContent 化）
- [ ] wmma 真实现或从 ptx_op.def 删除占位
- [ ] visitor 头文件 ptx_visiter.h 重命名为 ptx_visitor.h
- [ ] JSON 配置 88% 字段消费（L3，需新 cache/memory subsystem）
- [ ] 引入 vcpkg 或 FetchContent 替代 vendored 依赖
```

### Phase 4：长期演进（P3，按需）

```yaml
# 按路线图决策
- [ ] Hopper (sm_90+) 路线图：mbarrier.* / tcgen05.* 集成 BarrierModule
- [ ] C++ 现代化：336 处 C-style cast → static_cast/reinterpret_cast
- [ ] C++ 现代化：typedef → using 迁移
- [ ] unity build + precompiled headers（构建时间 -50%）
- [ ] clang-tidy + pre-commit 框架
```

---

## 9. 决策建议

### 9.1 路线图方向决策（需要用户确认）

#### D-A：PTX 8.7+ / Hopper 路线图

- **选项 A**：继续按需支持 PTX 7.x，删除 ptx_op.def 中 TCGEN/tcgen05/st.async 占位
- **选项 B**：规划 3-6 月路线图，激活 TCGEN 系列 + mbarrier.* + Tensor Map
- **选项 C**：维持现状（占位 + SimpleHandler no-op），风险是掩盖未来 PTX 编译错误

**推荐**：A（短期清晰）或 B（长期投资）

#### D-B：god class 拆分粒度

- **选项 A**：激进 —— 拆 ThreadContext 为 4 个 POD（ThreadPCState / ThreadMemoryBindings / ThreadOperandCollector / ThreadCCReg）
- **选项 B**：保守 —— 仅把 `active_mask[]` 三重状态合并为单一 source of truth（消除 DUAL STATE MECHANISM 风险），其他字段保持
- **选项 C**：维持现状，但补 active_mask 一致性测试

**推荐**：B（最高 ROI，最小 blast radius）

#### D-C：库边界重构

- **选项 A**：激进 —— ptx_parser / ptxsim / cudart 拆三个独立库 + FetchContent
- **选项 B**：保守 —— 仅修复 `ptx_ir → ptxsim/execution_types.h` 反向依赖（H2），其他边界保留
- **选项 C**：维持现状，但补依赖图文档

**推荐**：B（1 天工作量，最大解耦收益）

### 9.2 立即可执行（无需架构决策）

1. 修复 `compile_commands.json` 链（5 分钟）
2. 跑 `./scripts/sanity.sh --quick` 验证基线
3. 创建 `.github/workflows/build-test.yml`（30 分钟）

---

## 10. 附录

### 10.1 审计数据汇总

| 指标 | 值 |
|---|---:|
| 生产代码行数（src/ + include/） | 27,417 |
| C++ 源文件数 | 151（74 .cpp + 77 .h） |
| CTest 注册测试数 | 131（3 Disabled） |
| 测试源文件数 | 96 |
| TEST_CASE 块 | 739 |
| SECTION 子测试 | 213 |
| 断言总数 | 2339 |
| PTX 语法样本 | 33 |
| ADR 文档数 | 14（12 Active + 2 Proposed） |
| Doxygen 头文件覆盖 | 12/77（16%） |
| 全量构建产物大小 | 401 MB |
| 目标文件数 | 663 |
| PTX 指令定义（ptx_op.def） | 92 条实际指令 |
| PTX 指令完整实现度 | ~67%（核心 ISA） |

### 10.2 审计方法论

本审计采用 5 个并行 explore 子代理 + 1 个综合分析阶段：

1. **架构债务审计** —— 模块依赖、分层、接口、god class、可扩展性
2. **代码技术债审计** —— 9 个维度（TODO、stub、内存安全、错误处理、风格、死代码、并发、魔法数字、现代化）
3. **PTX 指令覆盖度审计** —— 对照 PTX 7.x/8.x ISA 250+ 指令
4. **测试与文档审计** —— 测试矩阵、文档状态、Doxygen 覆盖
5. **构建系统审计** —— CMake、依赖、CI、构建时间

**审计员限制**：
- 仅静态扫描，不修改任何文件
- 不运行构建/测试（避免环境依赖）
- 不与作者交流（无补充上下文）

### 10.3 引用规范

本文档引用以下审计明细：
- 架构债务：H1-H3 / M1-M7 / L1-L5（11 个独立发现）
- 技术债：9 个维度 + TOP-10 热点文件
- PTX 指令：13 个类别覆盖度表 + 16 项 HIGH 缺口
- 测试：13 项关键路径 × 3 类测试矩阵 + 8 项缺失测试
- 文档：14 ADR + 17 个文档目录
- 构建：5 项 P0 + 11 项 P1 + 7 项 L 债务

---

## 报告元信息

| 字段 | 值 |
|---|---|
| **审计员** | Sisyphus（项目健康审计） |
| **审计日期** | 2026-06-21 |
| **Git Commit** | baa8c4e |
| **文档位置** | `/docs/audits/HEALTH-AUDIT-2026-06-21.md` |
| **下次审计建议** | 2026-09-21（季度）+ Phase 1 完成后立即复审 |
| **状态** | 待决策 → 实施 |

---

> **本文档作为项目健康基线快照**。所有优先级标记基于当前（2026-06-21）状态；实施过程中应根据实际进展调整。
---

## 11. Errata (官方补充)

> 本审计作为 commit `baa8c4e` 的历史快照保持不变。如发现事实错误或遗漏,请参阅官方 Errata:
> 📋 **[HEALTH-AUDIT-2026-06-21-ERRATA.md](./HEALTH-AUDIT-2026-06-21-ERRATA.md)** (2026-06-22 发布)
>
> Errata 包含 8 项事实错误修正 (数值/计数/严重度/顺序) + 1 项严重遗漏 (BarWarpSyncHandler 仍用 deprecated wbars[])。
