## Context

Oracle 2026-07-10 (`ses_0b3791d78ffewb52428kJJ2Irz`) 审计报告 `tcgen05_fragment_mma_f16` helper 存在 2 个 HIGH confidence FlashAttention-readiness 阻塞:

- **H1**: Helper 无 `+=` 累加器（QK^T/PV 矩阵乘需要 `C += A*B` 沿 K 维循环累加）
- **H2**: Helper 输出存为 f16（与 PTX ISA §9.7.16 `f16×f16→f32` 矛盾）

Metis pre-implementation review (`ses_0b1a0cdb1ffenbhbciQ1n0x236`) 给出 CONDITIONAL GO，3 个 MUST-RESOLVE 全部采纳:
1. Persistence T1 断言反转 + 新增 T1_overwrite ✅
2. idesc semantic gap 进入 ADR-0016 debt ✅
3. 2-phase commit 而非 1 combined commit ✅

## Goals / Non-Goals

**Goals**:
- H1: helper 新增 `bool accumulate` 参数（默认 `false` 保留向后兼容）
- H2: helper 改写 f32 storage（符合 PTX ISA §9.7.16 + 硬件行为）
- 现有测试迁移（persistence T1 反转 + readback 机械修改）
- ADR-0016 追加 postmortem 段（idesc semantic gap 进入 debt）
- 2 atomic commits（每个 Phase 独立可 revert）

**Non-Goals**:
- 不修改 grammar（PTX ISA `idesc.accumulate` bit 解析需要 grammar/parser/visitor/handler 全栈修改，超出 scope）
- 不实现 64×64 warp-cooperative fragment layout（P2 follow-up — Oracle 报告 P2 nice-to-have）
- 不修复 H3 (ld/st/cp slot 0) 或 H4 (cluster wait 阻塞) — 独立 `fix-*` change
- 不更新 E2E tests（Priority 3 fallback 与 helper 改动无关）

## Decisions

### D1: Accumulator API 形状 — `bool accumulate=false` function arg

**采纳**: 修改 helper signature 加默认参数

```cpp
// include/ptxsim/instructions/tcgen05_helpers.h:51
void tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate = false);
```

**拒绝的备选**:
- (b) 新增并行函数 `tcgen05_fragment_mma_f16_accumulate(Tmem&)` — 单调用者无必要
- (c) 新增 `Q_TCGEN_ACCUM` Qualifier — 需 grammar + 5 文件改动，投入产出不成比例

**理由**:
1. grep 验证 `tcgen05_fragment_mma_f16` 只在 `tcgen05.cpp:383` 调用（per Metis A1）
2. 默认值 `false` 保留所有现有测试的 overwrite 行为（per Metis C1 mitigation — persistence T1 默认走 overwrite 路径）
3. function arg 是编译期强制 — 漏更新调用点会编译失败（per Metis C6）

**Tradeoff**: 不符合真实 PTX `idesc.accumulate` 语义 — 是 simulator 内部 workaround（ADR-0016 debt）

### D1.1: idesc Reading Path (per Oracle Q1 analysis, session `ses_0b026333bffePgrqVq7PDJNeR1`)

**关键事实**: PTX ISA §9.7.16 syntax `tcgen05.mma [taddr], adesc, bdesc, idesc, pred;` 的 idesc 是**寄存器引用**（`RegOperand`），不是立即数。

**Evidence** (verified file:line):
- `include/ptx_ir/ptx_op.def:133` — `X(S_TCGEN05_MMA, tcgen05.mma, Tcgen05, 4, TCGEN05_INSTR, tensor)` — op_count=4
- `include/ptx_ir/operand_context.h:12-21` — `RegOperand` struct: `std::string name` + `int index` (e.g., `%r5`)
- `include/ptx_ir/operand_context.h:27-29` — `ImmOperand` 是 textual `std::string value` (e.g., "0x123")
- `include/ptx_ir/statement_context.h:180-190` — `Tcgen05Instr.operands` is `std::vector<OperandContext>` (4 entries: `[taddr]`=AddrOperand, `[adesc]`=RegOperand, `[bdesc]`=RegOperand, `[idesc]`=RegOperand)

**含义**: idesc 的 accumulate bit **不是 IR 层直接可读的立即数值** — 它是寄存器名（如 `%r5`），运行时值需要从 `ThreadContext::register_bank_` 读取该寄存器的 `uint32_t` 值，然后解析特定 bit。idesc 的 bit 布局（accumulate 在哪一位）是 **NVIDIA 内部微架构细节，未公开**。

**拒绝的备选**（per Oracle Q1 三选项分析）:

| 选项 | 描述 | 拒绝理由 |
|------|------|---------|
| (a) 运行时从寄存器读 idesc | 需 `ThreadContext::register_bank_` API 访问 | idesc bit 布局未公开；需 1 个新 accessor |
| (b) 新增 `Q_TCGEN_ACCUMULATE` Qualifier | grammar + parser + visitor + handler 全栈 | 真实 PTX 不发射 `.accumulate` qualifier；5 文件改动 |
| **(c) 跳过 idesc 解析，bool 参数**（采纳） | `processTcgen05Mma` 显式传 `accumulate=false` | 与当前 proposal D1 一致；零 grammar 改动 |

**Follow-up**: `fix-tcgen05-idesc-parsing` change（占位符待 propose）— 届时可在 visitor 层添加 `instr.accumulate` 字段（参考已有 `cta_group`/`dtype`/`num_regs`/`has_block_scale` 的 visitor 提取模式）替代 helper 参数。

### D2.1: accumulate Parameter Default Value (per Oracle Q2 analysis)

**已知事实**: 真实硬件 `tcgen05.mma` **总是累加**（idesc.accumulate bit 控制）。"Overwrite" = idesc 设 `accumulate=0` 时的退化情况。

**Oracle 硬件参考**: PTX ISA §9.7.16 硬件语义：默认 accumulate=true。

**但 Simulator 当前不是硬件正确的** — helper 设计就是单次乘 + 覆写。proposal D1 决定将 helper 行为扩展到可累加，默认 `false` 是 pragmatic 过渡：

| Default | Pros | Cons |
|---------|------|------|
| `true` (硬件正确) | 匹配硬件语义 | 需反转 persistence T2/T3；handler 必须显式 `accumulate=false` 才能保留现有测试；违反 "helper 默认 = 硬件默认" 论点（handler 自身应正确） |
| **`false` (proposal 当前)** | 无静默行为变化；所有现有测试零修改通过（除 T1 计划反转） | 语义不硬件正确；记录为 ADR-0016 debt |

**采纳 `false` 的理由**:
1. 所有现有测试（ws 测试 golden 验证 + persistence T2/T3）依赖 `processTcgen05Mma` 的 overwrite 行为
2. `processTcgen05Mma` 作为 PTX handler 在 idesc 解析实施前不应改变语义
3. **零静默行为变化** — 不需要 review 每个调用点
4. 未来 `fix-tcgen05-idesc-parsing` change 实施时，`processTcgen05Mma` 将从 idesc 寄存器动态决定 `accumulate` 参数，那时 helper 默认就不重要了

**未来行为**（per Oracle Q2）: idesc 解析实施后，`processTcgen05Mma` 读 `instr.operands[3]`（idesc RegOperand）→ 从寄存器文件读其 uint32_t 值 → 提取 accumulate bit → 动态决定 `accumulate` 参数。在那之前，`false` 默认是保险选择。

### D2: Output Type 策略 — 无条件 f32 storage

**采纳**: helper body 改 `c_frag` 类型从 `uint16_t` → `float`，删除 `f32_to_f16` 转换

```cpp
// src/ptxsim/instructions/tcgen05_helpers.cpp:42
// BEFORE: std::array<uint16_t, ROWS * COLS_B> c_frag{};
// AFTER:  std::array<float, ROWS * COLS_B> c_frag{};
// 
// src/ptxsim/instructions/tcgen05_helpers.cpp:50
// BEFORE: c_frag[i * COLS_B + j] = f32_to_f16(sum);
// AFTER:  c_frag[i * COLS_B + j] = sum;  // 直接写 f32
```

**拒绝的备选**:
- (b) 拆函数 `tcgen05_fragment_mma_f16_f32` 与 `tcgen05_fragment_mma_f16_f16` — 创建无意义路径
- (c) Qualifier 路由（`Q_F16`/`Q_F32`）— 输出 dtype 是 hardware-fixed

**理由**:
1. PTX ISA §9.7.16 明确 f16×f16→f32（Oracle H2 + 真实硬件 TC f32 storage）
2. slot 利用率从 50%（32 f16 = 64B / 128B slot）提升到 100%（32 f32 = 128B / 128B slot）
3. A slot/B slot 都是 128B 满载，C slot 之前只用 64B 浪费，H2 后统一满载（per Metis A5 验证无冲突）

**Tradeoff**: readback 测试需机械修改（per Metis C2 mitigation — grep `c_buf\[idx \* 2\]` 全项目搜）

### D3: 测试迁移策略 — Persistence T1 反转 + readback 机械修改

**采纳**:
1. `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:184-203` T1:
   - TC 名: `"processTcgen05Mma called twice with identical A,B accumulates into C (2nd mma yields 2× golden)"`
   - 断言: `2 * GOLDEN_MMA_F16_F16_F32`
2. 新增 `T1_overwrite` TC（同文件）:
   - 显式构造 `tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false)`
   - 断言: `GOLDEN_MMA_F16_F16_F32`（验证 overwrite 仍可用）
3. `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp:154-163, 188-194` readback:
   ```cpp
   // BEFORE: c_buf[idx*2] | (c_buf[idx*2+1] << 8) + f16_to_f32
   // AFTER:  float val; memcpy(&val, &c_buf[idx*4], 4);
   ```
4. `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:155-176, 280-292` readback: 同样模式

**拒绝的备选**:
- (a) 直接修改现有测试 — 丢失 overwrite 历史
- (b) 新增并行测试 + 保留旧测试 — 旧测试改名为 `_legacy`，制造 churn

**理由**:
- T1 反转明确测试 accumulate 语义（2nd mma 后 C 累加到 2× golden）
- T1_overwrite 保留 overwrite 行为的回归保险
- mma_ws readback 是纯机械性（数值不变，存储格式变）

**Tradeoff**: T1 名字保留但语义反转（注释必须明确）

### D4: Golden 文件名 — 保留 `GOLDEN_MMA_F16_F16_F32`

**采纳**: 保留原名 + 更新注释段

```cpp
// tests/reference/ptx_tcgen05/tcgen05_mma_golden.h:6-7
// Layout: 8 rows × 4 cols = 32 f32 elements (per-lane fragment output).
// Storage format: f32 (per PTX ISA §9.7.16, mma output dtype is f32;
// previously stored as f16 with f16→f32 readback, fixed in
// fix-tcgen05-mma-accumulator-and-f32-storage Phase 2 commit).
```

**拒绝的备选**:
- (b) 重命名 `GOLDEN_MMA_F16_F16_F32_F32` — 制造 churn
- (c) 新增 deprecated 旧名 — 维护负担

**理由**: 名字中的 `F32` 一直指 output dtype（per `tcgen05_mma_golden.h:33` 注释 "32 f32 elements"），H2 只是把 storage 与类型名对齐

**Tradeoff**: 注释必须明确历史变更（per Metis B4）

### D5: Commit 粒度 — 2 commits (H1 first, H2 second)

**采纳**: Phase 1 = H1 commit，Phase 2 = H2 commit，Phase 3 = Archive commit（3 commits 总数）

**拒绝的备选**: 1 combined commit — 双倍 breakage 调试噩梦（per Metis C5）

**理由**:
1. lessons-learned §3 强制："复杂迁移必须分 Phase commit"
2. H1 单独 commit 后 mma_ws 测试 + persistence T2/T3 仍然过（只 T1 overwrite 断言需要改）— Phase 边界清晰
3. H2 单独 commit 后 readback 改动是 mechanical — diff 集中在 2-3 个文件
4. 任何 Phase 失败可独立 revert（不污染对方）

**Tradeoff**: 略增 commit 数（2 vs 1）— 严格符合 lessons-learned §3

### D6: OpenSpec 结构 — 新建 fix-* change + Ref 链接

**采纳**:
- 新建 `openspec/changes/fix-tcgen05-mma-accumulator-and-f32-storage/`
- 4 个 artifacts (proposal.md + design.md + tasks.md + spec.md)
- Ref 链接到 `archive/2026-07-10-implement-tcgen05-handlers-extended/`

**拒绝的备选**: amend 已归档 change — 违反 lessons-learned §6/G Checklist G

**理由**: 已归档 change 不可 amend；任何修补需求应新建 `fix-*` change + Ref 链接

**Tradeoff**: 比 inline 修改多 4 个 md 文件 — 严格符合 lessons-learned 强制 OpenSpec 流程

### D7: ADR 处理 — 追加 postmortem 段

**采纳**: 在 `docs/adr/ADR-0016-blackwell-only-tcgen05.md` 追加 "2026-07-11 Postmortem: H1+H2 fix" 段

```markdown
## 2026-07-11 Postmortem: H1+H2 fix

### H1 Root Cause
`tcgen05_fragment_mma_f16` (per `tcgen05_helpers.cpp:42,45,57`) 零初始化 c_frag
并覆写写入，从未读取 c_slot 已有值。FlashAttention QK^T/PV 矩阵乘需要 `+=`
累加器（沿 K 维循环），helper 缺乏此能力。

### H1 Fix
新增 `bool accumulate` 参数（默认 `false`）。`accumulate=true` 时先
`tmem.read(c_slot)` 预加载 C，f16→f32 转换，与新 sum 累加，写回。
`processTcgen05Mma` 显式传 `accumulate=false` 保持现有行为。

### H2 Root Cause
Helper 输出存为 `uint16_t` (f16)，与 PTX ISA §9.7.16 规定 `f16×f16→f32` 矛盾。
`tcgen05_mma_golden.h:6` 声称 "32 f32 elements" 但实际是 f16 storage + f16→f32
readback 掩盖不一致。

### H2 Fix
helper body 改 `c_frag` 类型从 `uint16_t` → `float`，删除 `f32_to_f16` 转换。
slot 利用率从 50% 提升到 100%。

### Known Semantic Gap (debt for future)
Helper `accumulate` 参数是 simulator 内部决策，**不解析真实 PTX `idesc.accumulate` bit**。
完整修复需要 grammar + parser + visitor + handler 全栈修改。
PTX ISA §9.7.16 标准语法 `tcgen05.mma [taddr], adesc, bdesc, idesc, pred`
中 accumulate 由 `idesc` 第 N 位的 bit 控制。
Follow-up change: `fix-tcgen05-idesc-parsing` (未 propose).
```

**拒绝的备选**: 创建独立 ADR `0017-helper-accumulator-f32.md` — 制造 churn

**理由**: H1+H2 是 helper 行为修正，归 ADR-0016 Blackwell tcgen05 范围

**Tradeoff**: ADR-0016 越来越长（符合 lessons-learned §G — 同一主题增量追加）

## Migration Plan（per lessons-learned Checklist A）

### Phase 1: H1 — Accumulator 支持（commit 1）

#### Baseline 函数清单

| 函数 | 文件:行 | 调用者 |
|------|---------|--------|
| `tcgen05_fragment_mma_f16(Tmem&)` | `include/ptxsim/instructions/tcgen05_helpers.h:51` | `src/ptxsim/instructions/tcgen05.cpp:383` |

#### 逐行 diff 计划

```cpp
// include/ptxsim/instructions/tcgen05_helpers.h:51
// BEFORE: void tcgen05_fragment_mma_f16(Tmem& tmem);
// AFTER:  void tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate = false);
//         // accumulate=true: read existing C slot, f16→f32, accumulate with new sum.
//         // Caller must ensure single-warp execution (currently safe per SM scheduler).
```

```cpp
// src/ptxsim/instructions/tcgen05_helpers.cpp (行号基于当前代码)
// AFTER Phase 1 (保持 f16 storage，仅加 accumulate):
void tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate) {
    // ... (lines 1-40 不变)
    
    // NEW: accumulate path 预加载 C (替换 line 42)
    std::array<uint16_t, ROWS * COLS_B> c_frag{};
    if (accumulate) {
        std::array<uint8_t, Tmem::kSlotSize> existing_buf{};
        tmem.read(c_slot, existing_buf.data(), Tmem::kSlotSize);
        const uint16_t* existing_raw = 
            reinterpret_cast<const uint16_t*>(existing_buf.data());
        for (int i = 0; i < ROWS * COLS_B; ++i) {
            c_frag[i] = existing_raw[i];  // 保留 f16 bits，累加时再转 f32
        }
    }
    
    // Lines 43-52 (累加循环): 改为 c_frag[i*COLS_B+j] = f32_to_f16(sum) 当 accumulate 时 sum += f16_to_f32(existing)
    // 实现: 先 f16_to_f32(c_frag[i*COLS_B+j]) 再加 sum
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            float sum = 0.0f;
            if (accumulate) {
                sum += f16_to_f32(c_frag[i * COLS_B + j]);  // 现有值
            }
            for (int k = 0; k < COLS_A; ++k) {
                sum += a_flat[i * COLS_A + k] * b_flat[k * COLS_B + j];
            }
            c_frag[i * COLS_B + j] = f32_to_f16(sum);
        }
    }
    
    // Lines 54-57 (写回): 不变
}
```

```cpp
// src/ptxsim/instructions/tcgen05.cpp:383
// BEFORE: tcgen05_fragment_mma_f16(tmem);
// AFTER:  tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false);  // 显式 overwrite
```

#### 跨模块状态翻译表

- `accumulate=true` 时: `tmem.read(c_slot)` → `f16_to_f32(uint16_t bits)` → 与新 `sum` 累加 → `f32_to_f16(sum)` → 后续 `tmem.write(c_slot)`
- `accumulate=false` 时: 与现有行为相同（零初始化 + 覆写）

#### 回退策略

- Phase 1 commit 独立可 revert (`git revert <commit>` 后 helper 回到零参数版)
- Phase 1 测试变更（T1 反转 + T1_overwrite）独立 commit 可 revert

### Phase 2: H2 — f32 Output Storage（commit 2）

#### 逐行 diff 计划

```cpp
// src/ptxsim/instructions/tcgen05_helpers.cpp
// AFTER Phase 2:
void tcgen05_fragment_mma_f16(Tmem& tmem, bool accumulate) {
    // ... (lines 1-40 不变)
    
    // Line 42 改类型:
    std::array<float, ROWS * COLS_B> c_frag{};  // uint16_t → float
    
    if (accumulate) {
        // Q4 通用 load_c_slot helper (在 tcgen05_helpers.cpp 顶部定义):
        alignas(float) std::array<uint8_t, Tmem::kSlotSize> existing_buf{};
        tmem.read(c_slot, existing_buf.data(), Tmem::kSlotSize);
        std::memcpy(c_frag.data(), existing_buf.data(),
                    ROWS * COLS_B * sizeof(float));
    }
    
    // Line 50 删除 f32_to_f16 转换:
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            float sum = 0.0f;
            if (accumulate) {
                sum += c_frag[i * COLS_B + j];  // 直接 float 累加
            }
            for (int k = 0; k < COLS_A; ++k) {
                sum += a_flat[i * COLS_A + k] * b_flat[k * COLS_B + j];
            }
            c_frag[i * COLS_B + j] = sum;  // 直接写 float
        }
    }
    
    // Line 55 改 memcpy size:
    std::memcpy(c_buf.data(), c_frag.data(), c_frag.size() * sizeof(float));  // 32*4 = 128 bytes
    tmem.write(c_slot, c_buf.data(), Tmem::kSlotSize);
}
```

#### readback 模式（per Oracle Q3 analysis）

**关键事实**（verified）:
- `c_buf` 定义为 `std::array<uint8_t, Tmem::kSlotSize>` (`kSlotSize=128`，`tmem.h:29`)
- `std::array<uint8_t, 128>` 的对齐是 `alignof(uint8_t) = 1` — **没有 4 字节对齐保证**
- `reinterpret_cast<const float*>` 在 1 字节对齐地址上是 **UB**（C++20 `[basic.align]/4`）
- 即使对齐了，还有严格别名违规（`uint8_t* → float*` 读取是 `[basic.lval]/11` 违规）

**推荐模式**（per Oracle Q3 选项 ii）: 全数组 `memcpy` 到 `alignas(16) float[32]`

```cpp
// 在 test helper 或 TEST_CASE 开头（替换原 readback 循环内）:
alignas(16) float c_arr[32];
std::memcpy(c_arr, c_buf.data(), sizeof(c_arr));
// 然后用 c_arr[idx] 替代 c_buf[idx*2] 模式
const float actual = c_arr[idx];
REQUIRE(actual == Catch::Approx(expected));
```

**优势**:
- 严格安全（`memcpy` 通过 `char*` 访问，对任何对齐都有效，C++20 `[basic.types]/3`）
- 性能最优（1 次 128 字节拷贝 + 32 次 aligned float 加载）
- 可读性最好（`c_arr[idx]` vs `c_buf[idx*4]`）

#### readback 机械修改清单

| 文件 | 行 | 修改 |
|------|-----|------|
| `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp` | 156, 192 | `c_buf[idx*2] \| (c_buf[idx*2+1] << 8)` + `f16_to_f32` → `alignas(16) float c_arr[32]; memcpy(c_arr, c_buf.data(), sizeof(c_arr)); float actual = c_arr[idx];` |
| `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` | 167, 288 | 同样 |

**验证**: `grep -rn "c_buf\[idx \* 2\]" tests/` 输出应为空（确认无遗漏）

#### load_c_slot 通用 helper（per Oracle Q4 analysis）

为 Phase 1 + Phase 2 统一 accumulate 预加载逻辑，在 `tcgen05_helpers.cpp` 顶部添加：

```cpp
// internal helper for accumulate pre-load
template <typename T>
static void load_c_slot(Tmem& tmem, size_t c_slot, T* c_frag, size_t count) {
    alignas(T) std::array<uint8_t, Tmem::kSlotSize> buf{};
    tmem.read(c_slot, buf.data(), Tmem::kSlotSize);
    std::memcpy(c_frag, buf.data(), count * sizeof(T));
}
```

**Phase 1 调用**: `load_c_slot<uint16_t>(tmem, c_slot, c_frag.data(), ROWS * COLS_B);`
**Phase 2 调用**: `load_c_slot<float>(tmem, c_slot, c_frag.data(), ROWS * COLS_B);`

**优势**:
- 避免 `reinterpret_cast` 的对齐 UB（`alignas(T)` 保证 `buf.data()` 对齐到 `alignof(T)`）
- `std::memcpy` 解决严格别名违规
- Phase 1 和 Phase 2 统一接口

#### readback 机械修改清单（更新版）

| 文件 | 行 | 修改 |
|------|-----|------|
| `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp` | 156, 192 | `c_buf[idx*2] \| (c_buf[idx*2+1] << 8)` + `f16_to_f32` → `alignas(16) float c_arr[32]; memcpy(c_arr, c_buf.data(), sizeof(c_arr)); float actual = c_arr[idx];` |
| `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` | 167, 288 | 同样 |

### Phase 3: Archive（commit 3）

- 4 个 md artifacts git-tracked
- ADR-0016 postmortem 追加
- archive change → commit

## Acceptance Criteria

### Phase 1 (H1) Acceptance

1. `tcgen05_fragment_mma_f16` signature 包含 `bool accumulate = false` 参数
2. `processTcgen05Mma` 显式调 `accumulate=false`（保留现有行为）
3. persistence T1 反转断言为 `2 × GOLDEN`
4. 新增 T1_overwrite TC 用 `accumulate=false` 显式调 helper，断言 `1 × GOLDEN`
5. `cd build && ctest -R "tcgen05" --output-on-failure` 全部 PASS
6. baseline worktree 对比：仅 T1 行为变化，其他测试不变

### Phase 2 (H2) Acceptance

1. `c_frag` 类型 `float`，写回 128 bytes 填满 slot
2. mma_ws readback 机械改为 `memcpy<float>`
3. persistence readback 机械改为 `memcpy<float>`
4. golden header 注释更新（f32 storage 说明）
5. `cd build && ctest -R "tcgen05" --output-on-failure` 全部 PASS
6. baseline worktree 对比：所有 tcgen05-tagged 测试不变（仅 readback 实现改）

### Phase 3 (Archive) Acceptance

1. 4 个 md artifacts git-tracked（`git ls-files openspec/changes/fix-tcgen05-mma-accumulator-and-f32-storage/` 不应为空）
2. ADR-0016 postmortem 追加
3. `cd build && ctest --output-on-failure` 全量 PASS
4. `./tests/ptx/test_all_ptx.sh` 全量 PASS
5. archive commit 含 postmortem 引用

## Known Risks (per Metis C 节)

| ID | 风险 | Severity | Mitigation |
|----|------|----------|------------|
| C1 | T1 断言反转被误判为 regression | critical | OpenSpec proposal 显式列出"persistence T1 预期失败，需要反转"；Apply 阶段第一步同时更新 T1 + helper（不分两步） |
| C2 | f16 readback 漏改 → 静默返回错误值 | critical | H2 commit 前 grep `c_buf\[idx \* 2\]\|f16_to_f32` 列出所有 readback 点；helper 注释加 `static_assert` 强制警觉 |
| C3 | Helper 锁语义未审计 | major | H1+H2 不改变锁语义（仍不持锁，依赖 Tmem）；新 accumulate 路径在 helper header 添加 explicit comment |
| C5 | H1+H2 一起实施双倍 breakage | critical | 严格按 Phase 拆分 — 2 commits + lessons-learned §3 |
| C7 | 试图 amend 已归档 change | major | 新建 `fix-*` change + Ref 链接（per lessons-learned §6/G Checklist G） |
| C8 | ADR 不更新 → debt 丢失 | minor | ADR-0016 追加段（per D7） |

## References

- Oracle 2026-07-10 report: session `ses_0b3791d78ffewb52428kJJ2Irz` (5 blockers)
- Metis pre-implementation review: session `ses_0b1a0cdb1ffenbhbciQ1n0x236`
- ADR-0016: [docs/adr/ADR-0016-blackwell-only-tcgen05.md](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md)
- Ref (archived): [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/)
- ptx-lessons-learned: [.opencode/skills/ptx-lessons-learned/SKILL.md](../../../.opencode/skills/ptx-lessons-learned/SKILL.md)
- step 1 commit (persistence test): `d3be589 test(tcgen05): add multi-op TMEM persistence integration test`
- PTX ISA §9.7.16 (tcgen05.mma semantics)