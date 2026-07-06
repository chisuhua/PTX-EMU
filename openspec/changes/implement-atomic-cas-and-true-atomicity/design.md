# Design: implement-atomic-cas-and-true-atomicity

> **Phase 1 (本会话实施)**: CAS handler 实施 (3h)
> **Phase 2/3 (后续会话)**: mutex + oracle test (5h)
> **Metis pre-impl audit**: ⚠️ CONDITIONAL → 6 MUST-RESOLVE 全部锁定

---

## 1. 实施基线验证 (Metis MR-1 ~ MR-6 完成度)

| MR | 项 | 验证结果 | 来源 |
|----|----|---------|------|
| MR-1 | `Q_CAS_ATOM` 解析 | ✅ PASS | `ptx_qualifier.def:251` 唯一匹配,`ptx_visitor_atom.cpp:88-94` remap 逻辑不需 CAS 项 (无 Q_DOTCAS 冲突) |
| MR-2 | 4-operand 传递 | ✅ PASS | `ptx_visitor_atom.cpp:75-77` 循环 `for (i=2; i<min(size, opcount); ++i)` 在 opcount=3 + operandCtxs.size()=3 (CAS) 时正确将 cmp push 到 operands[3] |
| MR-3 | Baseline worktree | ⏸ Phase 2 前置 | Phase 1 scope 明确,不需;Phase 2 mutex 设计需对照 baseline |
| MR-4 | Phase 1 测试验收 | ✅ 设计见 `specs/atomic-cas-handler/spec.md` | 5 个 SHALL Requirements 覆盖单 warp + cmp-match/cmp-mismatch/边界 |
| MR-5 | Phase 2 mutex 审计 | ⏸ Phase 2 前置 | `grep -rn "mutex_\|lock_guard" src/ptxsim/` + 锁序证明 (lessons-learned §2) |
| MR-6 | Scope 边界 | ✅ 已锁定 (proposal.md §Scope 边界) | 不实现 memory ordering / non-global / grammar / opcount |

---

## 2. Phase 1 设计: CAS Handler

### 2.1 函数签名选择

**选择**: 新增 `AtomHandler::processAtomicCAS` 专用函数 (vs. 扩展 `processAtomicOperation` 签名)

**理由**:
- **可读性**: CAS 语义独立 (compare 步骤独有),函数名直接表达语义
- **测试隔离**: 类型一单元测试可直接传入 4 operands,无需 mock 其他 3-operand 路径
- **Future-proof**: Phase 2 引入 mutex 时,`processAtomicCAS` 可加锁而不污染 `processAtomicOperation` (per lessons-learned §2 持锁方法不调用其他持锁方法的边界)

```cpp
// src/ptxsim/instructions/atomic.h 新增
class AtomHandler {
public:
    // ... 现有
    static void processAtomicCAS(
        ThreadContext* context,
        void* dst,        // dst register address (write old value)
        void* addr,       // memory address
        void* cmp_buffer, // compare value source (register or immediate)
        void* val_buffer, // new value source (register or immediate)
        size_t data_size, // from qualifiers (e.g., .b32 → 4)
        MemorySpace space // from qualifiers (e.g., .global → GLOBAL)
    );
};
```

### 2.2 算法 (单 warp 串行模型)

```cpp
void AtomHandler::processAtomicCAS(
    ThreadContext* context, void* dst, void* addr,
    void* cmp_buffer, void* val_buffer,
    size_t data_size, MemorySpace space
) {
    if (!dst || !addr || !cmp_buffer || !val_buffer) return;

    // 1. 读取目标地址旧值 (与现有 processAtomicOperation 一致)
    uint64_t old_val = 0;
    HardwareMemoryManager::instance().access(addr, &old_val, data_size,
                                             /*is_write=*/false, space);

    // 2. 读取 cmp + val
    uint64_t cmp_val = 0, new_val = 0;
    std::memcpy(&cmp_val, cmp_buffer, data_size);
    std::memcpy(&new_val, val_buffer, data_size);

    // 3. CAS 核心: old == cmp → 写入 new,否则不写入
    if (old_val == cmp_val) {
        HardwareMemoryManager::instance().access(addr, &new_val, data_size,
                                                 /*is_write=*/true, space);
    }

    // 4. dst 写回 old (PTX ISA 语义)
    std::memcpy(dst, &old_val, data_size);
}
```

**关键不变量**:
- 与现有 9 个非 CAS op 相同的 load → compute → store 模式
- "No concurrency guarantee" 在 Phase 1 阶段与现有 atom ops 对等 (现有 atom 也无 mutex);Phase 2 mutex 引入后行为升级

### 2.3 Operand 收集路径验证

```
PTX source:       atom.global.cas.b32 %r0, [%r1], %r2, %r3;
                  ↓ ANTLR parse
ctx->operandCtxs[0] = dst (%r0)
ctx->operandCtxs[1] = src (%r2)  ← 注意: PTX grammar 把 cmp 和 val 都算作 src operand
ctx->operandCtxs[2] = src (%r3)  ← 第二 src = val
ctx->addressExpr()  = [%r1]
                  ↓ ptx_visitor_atom.cpp line 56-77 (existing fix)
operands[0] = dst
operands[1] = addr (inserted)
operands[2] = cmp (loop i=2, source = operandCtxs[1])
operands[3] = val (loop i=3 ... wait, but opcount=3, std::min(size, opcount) = 3)
                  ↑ 问题: loop 终止条件 `i < std::min(operandCtxs.size(), opcount)`
                          opcount=3 → loop 跑 i=2,3 → push operands[2]=cmp + operands[3]=val
                          operandCtxs.size()=3 → std::min(3,3)=3 → i=2..3 → push operandCtxs[1]+[2]? 
                  
```

**等等 — 让我重新核对**: 在 ptx_visitor_atom.cpp 第 75-77 行:
```cpp
for (size_t i = 2; i < std::min(operandCtxs.size(), (size_t)opcount); ++i) {
    operands.push_back(createOperandFromContext(operandCtxs[i]));
}
```

**operandCtxs 来自** `ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>()`。对于 `atom.global.cas.b32 %r0, [%r1], %r2, %r3;`,**注意 grammar 设计**: grammar 允许 `operand COMMA addressExpr COMMA operand (COMMA operand)?`:

```
atom.global.cas.b32 %r0, [%r1], %r2, %r3;
             ──┬── ──┬── ──┬── ──┬──
              %r0   [%r1]  %r2   %r3
              │     addr   │     │
              dst           src1  src2  (= cmp + val)
```

ANTLR 的 `getRuleContexts<OperandContext>()` 应返回 [dst(%r0), src1(%r2), src2(%r3)] = 3 个 operand 上下文 (地址 [%r1] 通过 `ctx->addressExpr()` 单独获取,**不在** operandCtxs 中 — 见 visitor 第 23 行注释)。

所以 operandCtxs 大小 = 3。

**Op_count = 3** (per `ptx_op.def:126`)。Loop `i = 2; i < min(3,3) = 3`:
- i=2 → push createOperandFromContext(operandCtxs[2]) = src2 = val
- i=3 时循环退出

最终 operands = [dst, addr, cmp(来自 operandCtxs[1]=src1), val(来自 operandCtxs[2]=src2)] = **4 个元素** ✅

(注:cmp = src1 = operandCtxs[1] 在第一次 push 时被作为 src 加入,但实际是 cmp 角色,这正是 `processAtomicCAS` 需要理解的语义)

**结论**: visitor 现有的循环逻辑已正确收集 4-operand。Handler 需访问 operands[2]=cmp + operands[3]=val。**无需修改 ptx_op.def 或 ptx_visitor_atom.cpp**。

### 2.4 Qualifier 处理

`atomic.cpp` line 36-53 的 `atom_op` 检测循环:
```cpp
for (auto q : qualifiers) {
    switch (q) {
    case Qualifier::Q_ADD_ATOM: ...
    case Qualifier::Q_EXCH_ATOM: ...
    // ... 其他 7 个
    }
}
```

**新增**:
```cpp
case Qualifier::Q_CAS_ATOM:
    atom_op = q;
    break;
```

由于 `Q_CAS_ATOM` 在 enum 单独存在,无 DOT 冲突需要 remap。

### 2.5 调度路径

`instruction_base.cpp:200` (推测,Metis 未完整审计此点) 调用 `AtomHandler::processAtomicOperation(context, &(context->operand_collected[0]), instr.qualifiers)` — Phase 1 实施时需:

1. 检查 `processAtomicOperation` 当前是否支持 4-operand 调度
2. 若否,新增 `processAtomicCAS` 调用路径 (dispatcher 中识别 `Q_CAS_ATOM` 后路由)

**待实施时验证**:读取 `src/ptxsim/instructions/instruction_base.cpp` 第 195-220 行,确认 dispatcher 模式。

### 2.6 测试覆盖

| 类型 | 文件 | 内容 |
|------|------|------|
| 类型一 (unit) | `tests/unit/atomic/test_cas_handler_basic.cpp` | 直接调用 `processAtomicCAS`,验证 dst/memory 状态,不涉及 warp 调度 |
| 类型二 (integration) | `tests/integration/atomic/test_atom_global_cas.cpp` | 使用 `ptxsim::testing` 工具,驱动 warp 执行 PTX 序列 |
| PTX 语法 | `tests/ptx/atom_cas_basic.ptx` | 真实 PTX 文本,`test_all_ptx.sh` 验证解析 |

**Cmp 语义测试矩阵** (per MR-4):
1. 单 warp 32 lanes 对同一地址 + cmp == old → 验证 dst=old + memory=new (1 lane wins)
2. 单 warp 32 lanes 对同一地址 + cmp != old → 验证 dst=old + memory 不变
3. Warp 内混合 cmp (前 16 lane cmp==old, 后 16 lane cmp!=old) → "winner-takes-all" 语义
4. 边界:不同 data_size (.b8/.b16/.b32/.b64)
5. 边界:不同地址 (32 lanes 操作 32 个不同地址)

**Multi-warp 场景测试推迟到 Phase 3** (single-warp 测试足够验证 Phase 1 的算法正确性,multi-warp race 需要 mutex 才能正确测试)。

---

## 3. Phase 2 设计预览 (本会话不实施)

> **目标**: 引入 per-warp 串行化 + cross-warp mutex,让 multi-warp 并发 CAS 产生确定结果

### 3.1 锁设计要点

**两阶段加锁**:
1. **Per-warp serialize**: warp 内的 lanes 串行执行 CAS (避免同一个 warp 内 32 lane 同时对同一地址 CAS)
2. **Cross-warp mutex**: 跨 warp 时,所有 atomic op 共享一个 mutex (per-address 锁复杂度高,本次先用全局锁,性能后续优化)

**关键风险** (lessons-learned §2):
- `processAtomicCAS` 持锁期间**不调用**其他可能持锁的方法
- 不要将 `mutex_` 加在 `barrier_module_->mutex_` 之上 (锁序死锁: barrier 持锁 + atomic 持锁 = 死锁)

### 3.2 实施顺序

1. `grep -rn "mutex_\|lock_guard\|unique_lock" src/ptxsim/` 列出所有现有锁点
2. 选择所有锁的全局顺序 (例如 always barrier_mutex < atomic_mutex)
3. 引入 `AtomicMutex` 单例 (`src/ptxsim/atomic/atomic_mutex.{h,cpp}`)
4. 在 `processAtomicCAS` (以及 `processAtomicOperation` 用于其他 9 个 op) 进入时 lock,离开时 unlock (RAII)

### 3.3 测试 (Phase 3)

**Oracle test**: `tests/integration/atomic/test_atom_global_cas_multiwarp.cpp`
- 2 个 warp 同时对同一地址执行 `atom.add` + `atom.cas` 混合操作
- 验证最终结果确定化 (single mutex 串行化的输出)

---

## 4. Documentation Sync 计划

| 代码改动 | 同步文档 |
|---------|---------|
| 新增 `processAtomicCAS` | `src/ptxsim/instructions/AGENTS.md` 新增 "CAS handler" 章节 |
| Phase 1 完成 | `openspec/CHANGELOG.md` (若存在) 或 `openspec/specs/atomic-cas-handler/spec.md` §Changelog |
| Phase 1 通过 | Update `docs/roadmap/post-phase3-debt-roadmap.md` §1.1 A-9 行状态 |

---

## 5. 验证清单

### Phase 1 完成时必跑

```bash
# 1. 类型一单元测试
cd build && ctest -L "unit;atomic" --output-on-failure

# 2. 类型二集成测试
ctest -L "integration;atomic" --output-on-failure

# 3. PTX 语法测试
./tests/ptx/test_all_ptx.sh

# 4. 全量 sanity
./scripts/sanity.sh --quick
```

### 全部已有测试无回归

- 0 失败 ctest (139 existing PASS, 新增 ~5 PASS for CAS)
- 0 `qualifiers.back()` 引入 (per lessons-learned §5)
- 0 递归锁 (Phase 1 不引入锁;Phase 2 必查)

---

## 6. 失败处理策略

按 `lessons-learned.md` §3 复杂迁移分 Phase commit 模式:

- Phase 1 commit 独立 (单一 atomic commit),失败立即 `git revert HEAD`
- Phase 2/3 暂未启动 — 启动前必跑 baseline worktree
- 任何已有测试回归 → revert 该 Phase, 不混入后续 commit

---

## 7. Refs

- Metis pre-impl audit: `bg_566b7fc3` 详细 6 项 MUST-RESOLVE 决策
- Lessons-learned §2 (recursive lock): `docs/dev-process/lessons-learned.md`
- Lessons-learned §5 (qualifier.back): `docs/dev-process/lessons-learned.md`
- Roadmap §3.2: `docs/roadmap/post-phase3-debt-roadmap.md`
- PTX ISA atomic.cas: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#atomic-instructions-cas>
