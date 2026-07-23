# ADR-0019: ThreadContext 持续瘦身（MemoryAccessor + InstructionPipeline accessor 方案）

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-07-14 |
| **关联任务** | `openspec/changes/god-class-refactor-thread-context-phase3/` |
| **关联 PR** | 计划中（Phase 3.0–3.3 提交） |
| **作者** | PTX-EMU Architecture Team |
| **审核人** | 待 Metis 复核 + Oracle PC 生命周期 review |

## 上下文

`ThreadContext` 是 PTX-EMU 中负责每条 thread 的寄存器查找、指令执行、内存地址解析、PC 生命周期管理的核心类。在 Phase 1+2 之前，它是一个"god class"（884 行），混合了至少 6 个独立职责。

**Phase 1+2**（归档于 `archive/2026-07-06-god-class-refactor-thread-context/`）已成功提取两个子模块：

1. `SimtPcManager`（`include/ptxsim/simt_pc_manager.h`）— PC + execution state（state machine, set_pc/commit_pc/set_next_pc, warp_state access）
2. `RegisterAccessLayer`（`include/ptxsim/register_access_layer.h`）— 寄存器查找 + bank manager 委托

Phase 1+2 后 `ThreadContext` 仍有 **727 行**（实测 2026-07-14），承担三类未迁移职责：

- **内存访问**：`get_memory_addr()` (260+ 行)、`mov_data()`、`mov()`、`initialize_shared_memory()`、外部直接赋值的 5 个 public 内存字段
- **控制流/执行编排**：`_execute_once()` (50 行)、`execute_thread_instruction()`、`collect_operands()`、`commit_operand()` 等
- **遗留 POD 回填**：`exec_state_`、`reg_pred_`、`memory_`、`program_ref_` 4 个在 `init()` (lines 79-91) 与 `reset()` (lines 225-226) 仍被回填

**Phase 3.1 + 3.2 第一次尝试**（已取消）：

- 提取 `MemoryAccessor`，用 `std::function` 回调传入 `get_memory_addr` — 9 个测试因状态发散而回归
- 计划给 `IInstructionHandler::ExecPipe` 添加 `InstructionPipeline*` 第二参数 — 未实施
- **根因**：`shared_mem_space` / `local_mem_space` / `name2Sym` / `name2Share` / `cta_context_` 5 个字段是 `public`，外部代码（`cta_context.cpp:320` 等）可直接赋值，导致 `MemoryAccessor` 拷贝的副本与 `ThreadContext` 字段发散

Metis 审查 `ses_0a11eea61ffe0HTZX5uQEUvP7L`（2026-07-14）识别 10 项阻塞问题；本 ADR 记录了修正后的 Phase 3 决策。

## 决策驱动因素

1. **跨模块状态翻译**（lessons-learned §1）：任何状态字段在多模块之间共享时，必须有单一权威源 + 强制 setter 通道，不能让外部代码绕过
2. **递归锁死锁**（lessons-learned §2）：`MemoryAccessor` 通过 `thread_->acquire_register()` 调用必须保证 `acquire_register` 不持有与 `MemoryAccessor` 同一把锁 — 当前 `acquire_register` 无锁，安全
3. **多 Phase 独立 commit**（lessons-learned §3）：每个 Phase/Sub-step 必须可独立 revert；本设计把 Phase 3 拆为 8 个 commit（3.0、3.1、3.2.0–3.2.4、3.3.a、3.3.b）
4. **基线 worktree 绑定到 prerequisite commit**（lessons-learned §4）：`git worktree add .worktrees/baseline-pre-c3-phase1 <phase-3.0-commit-sha>`，**不**是 `HEAD~1`，否则基线不包含 Phase 3.0 的 setter 转换
5. **PC 生命周期不变式保留**（AGENTS.md §CONVENTIONS, ADR-0003）：`set_next_pc(current_pc + 1)` → `handler->ExecPipe(this, statement)` → `commit_pc()` 顺序与行号在 Phase 3.2.4 迁移后**必须字节对齐**
6. **TDD 纪律**（AGENTS.md §TDD）：新类（`MemoryAccessor`、`InstructionPipeline`）必须先写类型一单元测试（TDD Red），再实现
7. **新结构数据必须配单元测试**（AGENTS.md §测试覆盖率）：`MemoryAccessor` 3 个测试 + `InstructionPipeline` 3 个测试 + 1 个 PC 生命周期集成测试
8. **PRD-stable API**：handler `ExecPipe(ThreadContext*, StatementContext&)` 签名**不**变更，避免触及 40+ 具体 handler

## 考虑的替代方案

### 方案 A：直接给 `ExecPipe` 加 `InstructionPipeline*` 第二参数（已取消的设计）

**描述**: 修改基类 `IInstructionHandler::ExecPipe` 签名为 `void ExecPipe(ThreadContext*, InstructionPipeline*, StatementContext&)`，所有 40+ 具体 handler 更新签名。

**优点**:
- "概念上"显式传递 pipeline 给 handler
- 一次到位

**缺点**:
- 触碰 40+ handler 签名（`instruction_handlers.cpp` X-Macro 调度点 + 每个具体 handler 的 `ExecPipe` override）
- 实际只有 **4 个站点**真正读取 `operand_collected` / `operand_is_immediate_`：`instruction_base.cpp:172-173, 200, 231` + `barrier.cpp:92-93`
- 改签名后增加 4 步迁移（基类 + X-Macro 调度 + 40+ handler）
- 改变虚函数表布局，binary compatibility 风险
- 风险高、收益低

### 方案 B：Accessor 方案（✅ 选中）

**描述**: 保持 `ExecPipe` 签名不变。`ThreadContext` 添加 `get_operand_collected()` / `get_operand_is_immediate()` 两个 accessor；初始返回 `ThreadContext` 自身字段，Phase 3.2.3 之后转发到 `InstructionPipeline`。**只**改 4 个实际读取点。

**优点**:
- handler 签名零变化，零风险
- 只改 4 个实际读取点（`instruction_base.cpp:172-173, 200, 231` + `barrier.cpp:92-93`）
- 每个 sub-step（3.2.0/3.2.1/3.2.2/3.2.3/3.2.4）独立可回退
- 与 Phase 1+2 的"小步快跑、独立 commit"模式一致

**缺点**:
- `ThreadContext` 多 2 个 accessor 方法
- 实际读取点后续若新增需注意通过 accessor（合规检查项）

**选择理由**: 最小变更面 + 最大可回退性 + 与 lessons-learned §3 一致。

### 方案 C：把 `InstructionPipeline` 整体替换 `ThreadContext`（`IExecutionContext` 接口）

**描述**: 引入 `IExecutionContext` 抽象基类，让 `InstructionPipeline` 和 `ThreadContext` 都实现它；handler 接收 `IExecutionContext*`。

**优点**:
- 长期可扩展

**缺点**:
- 触及所有 40+ handler 签名
- 当前无第二个 context 实现，YAGNI
- 远超出 Phase 3 范围

**选择理由**: 不在 Phase 3 范围；如有需要可作未来 ADR-0020+ 提案。

## 决策内容

### 设计原则

1. **单一权威源 + 强制 setter 通道**：所有跨模块共享状态必须通过 `ThreadContext` 的 setter；setter 转发到 `MemoryAccessor` / `InstructionPipeline`
2. **handler 签名零变更**：`ExecPipe(ThreadContext*, StatementContext&)` 永久稳定
3. **TDD 先行**：新类先有 Red 测试，再实现
4. **每个 sub-step 独立 commit + 独立 revert**
5. **PC 生命周期字节对齐**：Phase 3.2.4 迁移的代码与原 `thread_context.cpp:101-150` 在行号、调用顺序、参数上**完全一致**

### 实现要点

#### Phase 3.0 — public → private + setter

```cpp
// thread_context.h
private:
    void *shared_mem_space_ = nullptr;
    void *local_mem_space_ = nullptr;
    std::map<std::string, std::unique_ptr<Symtable>> *name2Sym_ = nullptr;
    std::map<std::string, std::unique_ptr<Symtable>> *name2Share_ = nullptr;
    CTAContext *cta_context_ = nullptr;
public:
    void set_shared_memory_space(void *);
    void *get_shared_memory_space() const;
    // ... 同样为 local_mem_space / name2Sym / name2Share / cta_context_ 提供 setter/getter
```

外部直接赋值迁移：`src/ptxsim/core/cta_context.cpp:320` `thread->shared_mem_space = shared_mem_space` → `thread->set_shared_memory_space(shared_mem_space)`

#### Phase 3.1 — MemoryAccessor 提取

```cpp
// include/ptxsim/core/memory_accessor.h
class MemoryAccessor {
public:
    MemoryAccessor(ThreadContext *thread);
    void *get_memory_addr(const AddrOperand&, const std::vector<Qualifier>&);
    void mov_data(void *src, void *dst, std::vector<Qualifier> &q);
    void mov(void *from, void *to, const std::vector<Qualifier> &q);
    void initialize_shared_memory(const std::string &name, uint64_t address);
    // ... setters / getters ...
private:
    void *shared_mem_space_ = nullptr;
    void *local_mem_space_ = nullptr;
    std::map<std::string, std::unique_ptr<Symtable>> *name2Sym_ = nullptr;
    std::map<std::string, std::unique_ptr<Symtable>> *name2Share_ = nullptr;
    CTAContext *cta_context_ = nullptr;
    ThreadContext *thread_ = nullptr;  // for acquire_register
    static uint64_t SHMEMADDR_;  // 类静态，保留单实例语义
};
```

**关键决策**:
- `SHMEMADDR_` 是**类静态成员**（不是文件静态），保留"全程序一份"的当前行为
- `get_memory_addr` **不**使用 `std::function` 回调（已取消的方案），改为 `thread_->acquire_register()` 直接调用
- 3 个类型一单元测试在 TDD Red 阶段先写后实现

#### Phase 3.2 — InstructionPipeline 提取（accessor 方案）

```cpp
// include/ptxsim/thread_context.h
public:
    std::vector<void*> &get_operand_collected();
    const std::vector<void*> &get_operand_collected() const;
    std::vector<char> &get_operand_is_immediate();
    const std::vector<char> &get_operand_is_immediate() const;
private:
    std::unique_ptr<InstructionPipeline> instruction_pipeline_;

// include/ptxsim/core/instruction_pipeline.h
class InstructionPipeline {
public:
    InstructionPipeline(ThreadContext *thread);
    void _execute_once();  // 字节对齐 thread_context.cpp:101-150
    void execute_thread_instruction();
    void collect_operands(StatementContext&, const std::vector<OperandContext>&, const std::vector<Qualifier>*);
    void commit_operand(StatementContext&, const OperandContext&, const std::vector<Qualifier>&);
    void clear_temporaries();
    bool isIMMorVEC(OperandContext&);
    void dump_state(std::ostream&) const;
    void prepare_breakpoint_context(std::unordered_map<std::string, std::any>&);
    void print_instruction_status(StatementContext&);
    std::vector<void*>& get_operand_collected();
    std::vector<char>& get_operand_is_immediate();
private:
    std::vector<void*> operand_collected_;  // MAX_OPERANDS_PER_INSTR=4
    std::vector<char> operand_is_immediate_;
    std::vector<std::vector<void*>> vecOp_phy_addrs_;
    ThreadContext *thread_ = nullptr;
};
```

**PC 生命周期保留（不变量）**:

```cpp
void InstructionPipeline::_execute_once() {
    assert(thread_->is_valid_pc());
    int current_pc = thread_->get_pc();
    StatementContext &statement = (*thread_->statements)[current_pc];
    thread_->set_next_pc(current_pc + 1);          // line 1: 预推进
    InstructionHandler *handler = InstructionFactory::get_handler(statement.type);
    if (handler) {
        handler->ExecPipe(thread_, statement);     // line 2: handler 可能改 next_pc
    } else {
        thread_->set_state(EXIT);
    }
    thread_->commit_pc();                          // line 3: 提交 PC
}
```

**关键决策**:
- `call_stack` 保留在 `ThreadContext`（跨 kernel 生命周期状态）
- `dst_operand_reg_name_` 不在本期（grep 验证当前代码中不存在）
- 4 个 sub-step（3.2.0 accessor / 3.2.1 基类 / 3.2.2 BarWarpSync / 3.2.3 pipeline 类 / 3.2.4 控制流迁移）每个独立 commit
- 3.2.4 迁移前调 Oracle 审查 PC 生命周期 byte-level diff

#### Phase 3.3 — POD 删除 + 文档

两阶段删除 `exec_state_` / `reg_pred_` / `memory_` / `program_ref_`：
1. **3.3.a**：删除 `init()` (lines 79-91) 与 `reset()` (lines 225-226) 的回填代码 — 单独 commit
2. **3.3.b**：删除 4 个 POD 字段本身 — 单独 commit

每个 sub-step 通过 `grep -rn` 验证零读者；如发现读者则停止删除，先迁移读者。

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `src/ptxsim/core/thread_context.cpp` | 大幅缩减 | 727 → ~300 行 |
| `include/ptxsim/thread_context.h` | 字段 private + 5 setter + 2 accessor | 324 → ~280 行 |
| `src/ptxsim/core/memory_accessor.{h,cpp}` | 新增 | ~250 行 |
| `src/ptxsim/core/instruction_pipeline.{h,cpp}` | 新增 | ~350 行 |
| `src/ptxsim/instruction_base.cpp` | 3 处（172-173, 200, 231） | 通过 accessor 读取 |
| `src/ptxsim/instructions/barrier.cpp` | 1 处（92-93） | 通过 accessor 读取 |
| `src/ptxsim/core/cta_context.cpp` | 1 处（line 320） | 改用 setter |
| `tests/unit/core/test_memory_accessor.cpp` | 新增 | 3 个类型一单元测试 |
| `tests/unit/core/test_instruction_pipeline.cpp` | 新增 | 3 个类型一单元测试 |
| `tests/integration/pc/test_pc_lifecycle_invariant.cpp` | 新增 | 1 个集成测试 |
| `src/ptxsim/core/AGENTS.md` | 更新 WHERE TO LOOK + KEY FILES | 文档同步 |
| `docs/adr/README.md` | 添加本 ADR 链接 | 已于 2026-07-14 完成 |

## 后果

### 正面影响

- `ThreadContext` 缩减到 ~300 行，成为纯委托 hub（与 Phase 1+2 目标一致）
- 内存相关逻辑集中到 `MemoryAccessor`，未来易测试与替换
- 控制流 / 操作数管理集中到 `InstructionPipeline`，PC 生命周期更显式
- handler 签名零变化，二进制兼容性好
- 5 个 public 字段封闭为 private + setter，杜绝外部直接赋值导致的状态发散（lessons-learned §1 防御）
- 新增 7 个测试覆盖新类，提升覆盖率
- 每个 sub-step 独立可回滚

### 负面影响

- 多 2 个新文件需要维护
- `ThreadContext` 多 5 个 setter + 2 个 accessor 方法，公共 API 表面增加
- `InstructionPipeline::_execute_once` 与 `ThreadContext` 之间通过 `thread_->` 转发，长期可能有"是不是应该完全独立"的疑问
- Phase 3.0 的 public → private 转换会破坏 1 个外部直接赋值（`cta_context.cpp:320`），需要先迁移

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| `_execute_once` 迁移后 PC 生命周期漂移 | 中 | 严重（影响所有 warp 调度） | TDD Red 测试 `test_instruction_pipeline.cpp` case 3 + 集成测试 `test_pc_lifecycle_invariant.cpp` + Oracle 审查 byte-level diff |
| `set_shared_memory_space` 调用遗漏 | 中 | 测试回归 | Phase 3.0 commit 前用 `grep -rn 'shared_mem_space\s*='` 审计所有调用点 |
| `SHMEMADDR_` 跨测试文件残留 | 低 | 单测间状态污染 | `test_memory_accessor.cpp` case 2 验证 duplicate-detection 抛异常；如需要可加 `static void reset_SHMEMADDR_for_test()` |
| `operand_is_immediate_` 私有化后破坏外部访问 | 低 | 编译错误 | 验证 `instruction_base.cpp:172-173` + `barrier.cpp:92-93` 是仅有的外部直接读取点（已 grep 验证） |
| 工作区未提交变更混入 Phase 3 commit | 中 | 历史不清 | Phase 0 步骤 0.3 强制 stash / 单独 commit 不相关变更 |

## 合规检查

后续相关开发应检查：

- [ ] 新增字段若是跨模块共享状态，必须提供 `set_xxx` / `get_xxx` 配对
- [ ] 新增字段若是 per-`ThreadContext` 状态，必须评估是否应属于 `MemoryAccessor` / `InstructionPipeline`
- [ ] 修改 `_execute_once` 之前必须调 Oracle 审查 PC 生命周期 byte-level diff
- [ ] 修改 handler 基类（`PipelineHandler` 等）之前必须先验证 `accessor` 是 entry point
- [ ] 删除字段前必须 `grep -rn` 验证零读者；先停后删（两个独立 commit）
- [ ] 任何 `WarpContext` / `CTAContext` 等外部上下文不能直接赋值 `ThreadContext` 字段

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-07-14 | 初始版本（基于 Metis 审查 `ses_0a11eea61ffe0HTZX5uQEUvP7L` 修正后的 Phase 3 设计） | PTX-EMU Architecture Team |

## 参考

- **Metis 审查**：`ses_0a11eea61ffe0HTZX5uQEUvP7L` (2026-07-14) — 10 项阻塞问题全部修正
- **Phase 3 artifacts**：`openspec/changes/god-class-refactor-thread-context-phase3/`
  - `proposal.md` — 提案（含 capability / impact / prerequisites / design-time checklist）
  - `design.md` — 设计决策（8 个 Decision + 行级 diff 计划 + 单元测试规划）
  - `specs/control-flow/spec.md` — `InstructionPipeline` 行为合约
  - `specs/memory-access/spec.md` — `MemoryAccessor` 行为合约
  - `tasks.md` — 8 个独立 commit 任务清单
- **Phase 1+2 归档**：`openspec/changes/archive/2026-07-06-god-class-refactor-thread-context/`
- **Lessons-Learned**：`docs/dev-process/lessons-learned.md` §1（跨模块状态翻译）、§2（递归锁）、§3（多 Phase commit）、§4（基线 worktree）、§6（artifacts-first）、§7（pre-impl Metis review）
- **AGENTS.md 约定**：`src/ptxsim/core/AGENTS.md` §CONVENTIONS（PC 生命周期）、§SINGLE SOURCE OF TRUTH（T2-1 active_mask 不变量）
- **关联 ADR**：
  - ADR-0002（PC 权威源统一到 WarpState）
  - ADR-0003（`commit_pc` / `force_set_pc` 分离）
  - ADR-0012（Per-Thread PC 设计，Volta+ SIMT 模型）
- **债务审计**：`docs/audits/debt-audit-2026-07-02.md` §2.2 C-1 (P1)
