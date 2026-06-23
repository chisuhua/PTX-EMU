# ADR-0015: CVT 指令策略模式重构 (Composition over Inheritance)

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-06-23 |
| **关联任务** | T2-6 (Phase 3) |
| **关联 PR** | 5 commits: `86e0786`, `d3c77b5`, `2f3c150`, `620d066`+`edbce54`, `fc3c352`+`9837d44`+`d6123e0`, `204b5cd`, `40b331b` |
| **作者** | Sisyphus orchestrator |
| **审核人** | TBD |

## 上下文

### 问题：`CvtHandler::processOperation` 单方法 1145 行

在 T2-6 之前，`src/ptxsim/instructions/arithmetic_conversion.cpp` 中的 `CvtHandler::processOperation`
是单方法 1145 行（line 141-1286），包含：

- **4 层嵌套 switch**：`dst_bytes` (1/2/4/8) → `is_float` → `is_signed`/`signed` ↔ `src_bytes` (1/2/4/8)
- **5×4×4×4 = 320+ case 分支**：覆盖 f16/f32/f64 × int8/16/32/64 × signed/unsigned × with/without .sat/.rni/.rzi 等
- **4 个 case 间 int→int 4×4×4 模板化代码**几乎重复
- **2 个 helper 函数** (`half_to_float` / `float_to_half`) 与 `half_utils.h` 重复 ~70 行

### 维护灾难（量化）

| 指标 | T2-6 前 | T2-6 后 |
|------|--------:|--------:|
| `CvtHandler::processOperation` 行数 | 1145 | 31 (Sub-task 3) → 0 (Sub-task 6) |
| `arithmetic_conversion.cpp` 总行数 | 1288 | 0 (Sub-task 6 删除) |
| `cvt_strategy.cpp` 行数 | 0 | 1048 (5 strategies + select + dispatch) |
| CVT 单元测试数 | 1 (`test_cvt`) | 6 unit + 8 integration + 1 cvta = **15** |
| CVT 集成测试断言数 | ~30 | 94 + P1-4.1 fix 启用 |

### 关键约束（来自 Phase 3 master plan §关键约束）

> **DO NOT 拆 CvtHandler 类**：X-Macro 在 `instruction_factory.cpp:14-17` 实例化 `new CvtHandler()`，多 Handler 会改变注册机制。T2-6 必须用 **Composition over Inheritance** — 策略在 CvtHandler 内部。

> **DO NOT 改 `CvtHandler::processOperation` 签名**：所有指令 handler 通过 X-Macro 统一注册，签名变会引发链接错误。

这两条约束把"拆 Handler"的大门关上了：不能改为 `CvtFloatToIntHandler` / `CvtIntToFloatHandler` 等多 Handler 类。

## 决策驱动因素

1. **零外部破坏面**：`CvtHandler` 仍由 X-Macro 单一注册，外部调用方（`InstructionFactory::get_handler(S_CVT)`）完全不变。
2. **零行为变更**：5 sub-task 严格 TDD，每步都有 14 个 CVT 测试做 regression guard。
3. **可读性 + 可扩展性**：5 个具体策略 + 1 个 select_strategy() 工厂，新增 modifier（`.rna`/`.rs`）只需扩 strategy，**不需改主流程**。
4. **测试粒度细**：5 个 strategy 各自单元测试（5×~10 assertions）+ 8 个 integration test 覆盖 cross-strategy 场景。
5. **删除冗余**：`half_to_float` / `float_to_half` 改用 `half_utils.h`（精度 0 差异，Sub-task 1 验证）。

## 考虑的替代方案

### 方案 A: 拆 Handler 类（继承方案）

**描述**：把 `CvtHandler` 拆为 `CvtFloatToFloatHandler` / `CvtFloatToIntHandler` / `CvtIntToFloatHandler` / `CvtIntToIntHandler` 4 个类，分别在 `instruction_factory.cpp` 注册。

**优点**:
- 符合"Open/Closed"开闭原则
- 每个 Handler 独立可测

**缺点**:
- ❌ 违反 Phase 3 master plan 关键约束 (DO NOT 拆 CvtHandler 类)
- ❌ 改变 X-Macro 注册机制，所有 `S_CVT` 调用点需更新
- ❌ 改 `processOperation` 签名 = 改 4 个 handler + 链接错误风险

**结论**: 拒绝

### 方案 B: 纯表驱动分发（Table-Driven）

**描述**：用 5×4×4 静态表 (5 types × 4 src_bytes × 4 dst_bytes) 存放函数指针或函数对象。

**优点**:
- 极致紧凑（~50 行核心 + 静态表）
- 编译器可优化 switch

**缺点**:
- ❌ 不易处理 `.sat` / `.rni` / `.rzi` / `.rmi` / `.rpi` / `.rna` / `.rs` 7 个 modifier
- ❌ 表条目爆炸（5×4×4×7 = 560 entries）vs 现状 5 strategies
- ❌ 修改 modifier 需重排整张表
- ❌ 不可 OOP 单元测试（无法单独测 `.sat` 路径）

**结论**: 拒绝

### 方案 C: 策略模式 — Composition (✅ 选中)

**描述**：保留单一 `CvtHandler` 类（X-Macro 约束），其 `processOperation` 委托给 `ConversionStrategy` 抽象基类 + 5 个具体策略（`FloatToFloatStrategy` / `IntToFloatStrategy` / `FloatToIntStrategy` / `IntToIntStrategy`）+ `select_strategy()` 工厂。

**优点**:
- ✅ 符合 X-Macro 约束（`CvtHandler` 仍是单一类）
- ✅ 零外部破坏面（所有调用方不变）
- ✅ 5 strategies 各自单测，调试容易
- ✅ 新 modifier / 新 type 只需扩 strategy 或新增 strategy
- ✅ 31 行 dispatch 干净（`build_context` → `select_strategy` → `convert`）

**缺点**:
- 运行时多态开销（一次 `static_cast<const ConversionStrategy&>` + 一次虚函数调用）— 可忽略（~2ns）
- 5 strategies 占 5 个文件（vs 表驱动 1 个文件）— 文件数增加 ~80 行 boilerplate

**选择理由**: 在 5 维度嵌套（4 bytes × 2 signed × 2 is_float × 7 modifier）下，策略模式的 OOP 表达力远超表驱动；X-Macro 约束强制走 Composition；运行时多态开销可忽略。

## 决策内容

### 架构：Composition Strategy Pattern

```
CvtHandler (X-Macro 单例)
    │
    │ processOperation(operands, qualifiers, ...)
    ▼
build_context(qualifiers)              // 抽取强类型 CvtContext (int dst_bytes, bool dst_is_float, ...)
    │
    ▼
select_strategy(ctx)                   // 工厂：返回 static 单例
    │
    ├──► FloatToFloatStrategy    (f16/f32/f64 → f16/f32/f64)
    ├──► IntToFloatStrategy      (int8/16/32/64 → f16/f32/f64)
    ├──► FloatToIntStrategy      (f16/f32/f64 → int8/16/32/64, 含 .sat + 5 个 .rn*)
    └──► IntToIntStrategy        (int8/16/32/64 → int8/16/32/64, 含 .sat + 4 个 .rn*)
    │
    ▼
strategy.convert(dst, src, ctx)        // 实际写寄存器
```

### 设计原则

1. **零行为变更**：5 sub-task 严格 TDD，每步 14 CVT 测试作 regression guard
2. **CvtContext 强类型化**：替换原 30+ `Qualifier::Q_xxx` 检查为 `ctx.has_sat` / `ctx.dst_bytes` 等
3. **5 strategies 互斥**：`select_strategy` 根据 `dst_is_float × src_is_float` 唯一选择
4. **Modifier 收敛在 strategy 内部**：`.sat`/`.rni` 等由 strategy 处理，select_strategy 不感知
5. **测试金字塔**：5 unit (per strategy) + 8 integration (cross-strategy) + 1 cvta

### 实现要点

- `cvt_strategy.h`: `CvtContext` (强类型) + `ConversionStrategy` (抽象) + `select_strategy()` (工厂)
- `cvt_*.cpp`: 5 个具体 strategy，~50-200 行 each
- `cvt_helpers.cpp`: 抽离的 4 个 helper（`round_half_to_even` / `half_to_float` / `float_to_half` / `should_saturate_uint32`），其中 `half_to_float` / `float_to_half` 委托给 `half_utils.h`（精度 0 差异）
- `arithmetic_conversion.cpp` (Sub-task 6 之前 31 行) → Sub-task 6 **删除**；`CvtHandler::processOperation` 内联到 `cvt_strategy.cpp`

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `arithmetic_conversion.cpp` | **删除** | 31 行单纯 dispatch，cvt_strategy.cpp 接管 |
| `cvt_strategy.cpp` | 新增 | 1048 行（5 strategies + select + helpers + dispatch） |
| `include/ptxsim/instructions/cvt/*.h` | 新增 | 6 个 header（CvtContext, ConversionStrategy, 5 strategies） |
| `src/CMakeLists.txt` | 修改 | 移除 arithmetic_conversion.cpp |
| `instruction_factory.cpp` | **不变** | X-Macro 仍 `new CvtHandler()` |
| `instruction_handlers.h` | **不变** | `DECLARE_GENERIC_INSTR_HANDLER(Cvt)` 仍声明 CvtHandler 类 |

## 后果

### 正面影响

- **可读性 ↑↑**：从 1145 行单方法 → 5 个 ~100 行 strategy + 31 行 dispatch
- **可测试性 ↑↑**：5 strategy 各自单测，定位 bug 不需 trace 整个 switch
- **可扩展性 ↑**：新增 modifier (`.rna` 已支持) / 新 type 只需扩对应 strategy
- **冗余代码 ↓**：`half_to_float` / `float_to_half` 委托 `half_utils.h`，减 ~70 行重复
- **测试覆盖 ↑↑**：14 → 15 测试目标，断言数 ~30 → ~250，0 行为变更

### 负面影响

- **文件数 ↑**：cvt/ 子目录 6 个新文件（含 header），5 strategy 各自独立
- **运行时多态开销**：1 次虚函数调用（~2ns/次）— 可忽略
- **`GeneralCvtStrategy` 死代码**：Sub-task 3 过渡策略（保留 1063 行 switch），Sub-task 4 拆 5 strategies 后 select_strategy 永远不返回它。**待后续清理**（标注 TODO，不在 T2-6 范围）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 拆分后行为与原 switch 字节级不一致 | 🟢 低 | 🔴 高 | Sub-task 3-4 严格 TDD：每个 sub-step 写测试 + 跑全量 14 CVT 测试 |
| `.rna` / `.rs` modifier 边界条件 bug | 🟡 中 | 🟡 中 | 5 strategy 各自单测 + 8 integration 覆盖 cross-strategy |
| 删 `arithmetic_conversion.cpp` 链接错误 | 🟢 低 | 🔴 高 | Sub-task 6 先把 `CvtHandler::processOperation` 移到 `cvt_strategy.cpp`（global scope），再删原文件 |
| 5 strategies 文件数过多 | 🟢 低 | 🟢 低 | 各自 < 200 行，可读性 > 文件数 |
| `GeneralCvtStrategy` 死代码浪费空间 | 🟢 低 | 🟢 低 | 标注 TODO，待 Phase 4 清理（不在 T2-6 范围） |

## 合规检查

后续相关开发应检查：

- [ ] 新增 CVT modifier 时，在对应 strategy 内添加（不在 select_strategy）
- [ ] 新增 CVT type 时（罕见），决定新增 strategy vs 扩现有 strategy
- [ ] 修改 `CvtContext` 字段时，所有 5 strategies 必须同步（编译期强制）
- [ ] `GeneralCvtStrategy` (cvt_strategy.cpp:108-1031) 是过渡死代码，**勿使用** select_strategy 已不返回它
- [ ] 不要在 strategy 外直接调用 `half_to_float` / `float_to_half`，统一用 `half_utils.h`

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-06-23 | 初始版本 | Sisyphus orchestrator |

## 参考

- Phase 3 master plan: `docs/superpowers/plans/2026-06-23-phase3-critical-debt.md`
- T2-6 task list: `openspec/changes/phase3-t2-6-cvt-strategy-pattern/tasks.md`
- X-Macro dispatch ADR: `docs/adr/0009-xmacro-instruction-dispatch.md`
- half_utils ADR (planned): `openspec/changes/half_utils-bugfix/`
- Sub-task 1 commit: `86e0786` (Red test) + `d3c77b5` (Green: extract helpers)
- Sub-task 2 commit: `2f3c150` (delegate to half_utils.h)
- Sub-task 3 commit: `620d066` (Red test) + `edbce54` (Green: CvtContext + select skeleton)
- Sub-task 4 commits: `fc3c352` (FloatToFloat), `9837d44` (IntToInt + 5 wired), `d6123e0` (unit tests)
- Sub-task 5 commit: `204b5cd` (P1-4.1 fix + 94 integration tests)
- Sub-task 6 commit: `40b331b` (delete arithmetic_conversion.cpp)
