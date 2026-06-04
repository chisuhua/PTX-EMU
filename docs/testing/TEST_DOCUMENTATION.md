# PTX-EMU 测试文档

> 生成日期: 2026-05-05
> 维护者: PTX-EMU Team
> 状态: 持续更新

---

## 1. 测试概览

### 1.1 测试框架
- **单元测试框架**: Catch2 (`catch_amalgamated.cpp`)
- **CUDA 测试**: 使用 `-keep` 保留中间 PTX，编译架构 `sm_100`（虚拟架构）
- **PTX 语法测试**: `tests/ptx/test_all_ptx.sh` (使用 cuobjdump 提取真实 PTX)

### 1.2 测试分类总览

| 类别 | 标签 | 测试数量 | 说明 |
|------|------|---------|------|
| 内存管理 | `memory` | 3 | 内存分配、边界检查 |
| 立即数解析 | `parse` | 1 | PTX 立即数解析 |
| PTX 指令 | `ptx;integer/float/bitwise/ld_st/cvta/cvt` | 6 | 按指令类型分类 |
| SIMT 执行 | `simt_entry/exec_mask/active_mask` | 25+ | SIMT 堆栈、分支 divergency |
| Warp 调度 | `warp/scheduler` | 3 | Warp 上下文和调度器 |
| 屏障同步 | `barrier/sync` | 15+ | 屏障 reconvergence、deadlock |
| 集成测试 | `integration` | 10+ | 端到端测试 |
| Benchmark | `mini/basic/extra` | 21+ | 真实 CUDA 程序 |

---

## 2. 单元测试详情 (Catch2)

### 2.1 内存管理测试

| 测试文件 | 标签 | 测试目的 | 关键覆盖 |
|---------|------|---------|---------|
| `test_memory_manager.cpp` | `memory` | 测试 SimpleMemoryAllocator 和 MemoryManager | 基本分配/释放、全局/参数/managed内存、内存重叠保护 |
| `test_memory_bounds.cpp` | `memory` | 测试内存边界检查 | 有效访问边界、越界访问抛出异常、只读区域保护 |

**测试用例详情:**

```
test_memory_manager.cpp:
├── SimpleMemoryAllocator basic operations
│   ├── Basic allocation (1024/2048字节分配)
│   └── Allocation and deallocation (分配释放复用)
├── MemoryManager global memory operations
│   ├── Global memory allocation
│   └── Memory access operations (read/write)
├── MemoryManager param memory operations
│   └── Parameter memory allocation and access
├── MemoryManager memory overlap protection
│   └── 大量分配/释放碎片整理
└── MemoryManager managed memory operations
    └── Managed memory allocation and access

test_memory_bounds.cpp:
├── SimpleMemory bounds checking
│   ├── Valid access at offset 0
│   ├── Valid access at last valid offset (1008)
│   ├── Out of bounds: offset beyond total size (2000)
│   ├── Out of bounds: offset + size exceeds total
│   ├── validate_offset: edge cases
│   ├── Valid write access
│   └── Out of bounds write throws
└── HardwareMemoryManager region bounds checking
    ├── Access within registered region succeeds
    ├── Out of bounds access throws
    ├── Write to read-only region throws
    └── Read from read-only region succeeds
```

### 2.2 PTX 解析测试

| 测试文件 | 标签 | 测试目的 | 关键覆盖 |
|---------|------|---------|---------|
| `test_parse_immediate.cpp` | `parse` | 测试立即数解析 | 各种格式（hex/decimal/scientific）的 f32/f64/s32/s64/s16/s8 解析，predicate解析，IEEE 754特殊值 |

**测试用例详情:**

```
test_parse_immediate.cpp (约40个SECTION):
├── Float32 parsing
│   ├── standard hex format (0f3F800000)
│   ├── direct hex format (0x3F800000)
│   ├── decimal format (1.0, -1.5, 0.0)
│   ├── hex-float format (0x1.0p0, 0x1.0p2)
│   └── IEEE 754 special values (inf, -inf, nan)
├── Float64 parsing
│   └── standard hex, direct hex, decimal, negative, zero
├── Int64/32/16/8 parsing
│   ├── positive, negative, hex values
│   └── overflow handling
├── Predicate parsing
│   ├── zero → false (0)
│   ├── non-zero → true (2, 100, -5)
└── Error handling
    ├── Invalid qualifier
    ├── Empty string
    ├── Whitespace
    └── Large numbers (max int64), scientific notation
```

### 2.3 PTX 指令测试 (CUDA 编译)

| 测试文件 | 标签 | 源文件 | 测试目的 |
|---------|------|--------|---------|
| `test_ptx_integer` | `ptx;integer` | `test_ptx_integer.cu` + `ptx_integer_arith.cu` | 整数算术指令 (add, sub, mul, div, mod, abs, neg, min, max, mad, mul24, mulhi 等) |
| `test_ptx_float` | `ptx;float` | `test_ptx_float.cu` + `ptx_float_arith.cu` | 浮点算术指令 (add, sub, mul, div, sqrt, rsqrt, sin, cos, lg2, ex2, RCP, FMA 等) |
| `test_ptx_extended` | `ptx;integer` | `test_ptx_extended.cu` + `ptx_extended_prec.cu` | 扩展精度整数 (addc, subc, mad24, mulhi, sad, div, rem) |
| `test_ptx_bitwise` | `ptx;bitwise` | `test_ptx_bitwise.cu` + `ptx_bitwise_shift.cu` | 逻辑移位 (and, or, xor, not, shl, shr, bfe, bfi, popc, clz, vabsdiff) |
| `test_ptx_cvt` | `ptx;cvt` | `test_ptx_cvt.cu` + `ptx_cvt_arith.cu` | 类型转换 (cvt, cvta, mov, movmatrix) |
| `test_ptx_ld_st` | `ptx;ld_st` | `test_ptx_ld_st.cu` + `ptx_ld_st.cu` | 内存访问 (ld, st, ldu, stu, atom, red) |
| `test_ptx_cvta` | `ptx;cvta` | `test_ptx_cvta.cu` + `ptx_cvta.cu` | 地址转换 |

### 2.4 Warp 层级测试

| 测试文件 | 类型 | 测试目的 | 关键覆盖 |
|---------|------|---------|---------|
| `warp/test_warp_context.cpp` | 独立 | WarpContext 创建和状态管理 | 创建、活跃掩码管理、线程添加、完成状态 |
| `warp/test_sm_context.cpp` | 独立 | SMContext 创建、资源管理、执行 | SM创建、CTA添加、执行、资源限制 |
| `warp/test_warp_scheduler.cpp` | 独立 | Warp 调度器 | RoundRobin、Greedy调度器、inactive warp处理 |

### 2.5 SIMT 执行模型测试

#### 2.5.1 SIMT Stack 测试

| 测试文件 | 标签 | 测试目的 |
|---------|------|---------|
| `test_simt_stack_entry.cpp` | `simt_entry;bug;critical` | **BUG-002**: 已退出线程阻塞 SIMT stack reconvergence |
| `test_simt_stack_catch2.cpp` | `simt_stack` | SIMT Stack 基本操作 |

**关键测试用例 (test_simt_stack_entry.cpp):**
```
B1: all threads at reconvergence PC
B2: partial convergence (部分线程未到达)
B3: return_mask excludes unaffected threads
B4: exited threads don't block convergence [BUG-002][critical]
B5: empty return_mask converges immediately
B6: toString produces expected format
```

#### 2.5.2 Exec Mask 测试

| 测试文件 | 标签 | 测试目的 |
|---------|------|---------|
| `test_exec_mask.cpp` | `exec_mask;bug;critical` | **BUG-001**: reconvergence 后 exec_mask 未恢复 |

**关键测试用例 (test_exec_mask.cpp):**
```
F1: default exec_mask is full active
F2: exec_mask after divergent branch
F3: exec_mask restored after reconvergence [BUG-001][critical]
F4: set_exec_mask and get_exec_mask roundtrip
F5: nested divergence exec_mask recovery
F6: exec_mask and active_mask independence
```

#### 2.5.3 Active Mask 测试

| 测试文件 | 标签 | 测试目的 |
|---------|------|---------|
| `test_active_mask_consistency.cpp` | `active_mask` | **ISSUE-004**: active_mask 与 exec_mask 一致性 |

**关键测试用例:**
```
J1: default active_mask matches exec_mask
J2: active_mask unchanged during divergence
J3: thread exit updates active_mask
J4: active_mask consistent after convergence
J5: active_count matches active_mask bits
```

#### 2.5.4 Branch/Barrier 集成测试

| 测试文件 | 标签 | 测试目的 |
|---------|------|---------|
| `test_handle_branch_integration.cpp` | `branch;integration` | Branch + SIMT stack 集成 |

### 2.6 屏障同步测试

#### 2.6.1 屏障 Reconvergence 测试

| 测试文件 | 标签 | 测试目的 |
|---------|------|---------|
| `test_barrier_reconvergence.cpp` | `barrier;reconvergence` | 屏障后 reconvergence PC 处理 |
| `test_barrier_reconvergence_pc.cpp` | `barrier;reconvergence_pc` | 屏障 reconvergence PC 详细测试 |

#### 2.6.2 Test 3 Deadlock 复现测试

| 测试文件 | 标签 | 测试目的 |
|---------|------|---------|
| `test3_reproduction.cpp` | `test3;deadlock` | CFG post-dominator、屏障转换、分支 divergency、get_lanes_by_pc |
| `test_syncthreads_test3_repro.cpp` | `test3` | 直接 Wbar 操作的 Test 3 复现 |
| `test_syncthreads_test3_isolated.cpp` | `test3` | 完整执行环境的 Test 3 隔离测试 |
| `test_syncthreads_test3_full.cpp` | `test3` | SE→BRA predicate 评估的 Test 3 |
| `test_syncthreads_test3_full_integration.cpp` | `test3;integration` | 完整 GPU 集成的 Test 3 |

#### 2.6.3 屏障交互测试

| 测试文件 | 标签 | 测试目的 |
|---------|------|---------|
| `test_barrier_interaction_integrated.cpp` | `barrier;integration` | I1-I3 hypotheses (屏障交互) |
| `test_exec_integration_h1_h4.cpp` | `exec;integration` | H1-H4 hypotheses (执行层) |
| `test_exec_layer_e1_e3.cpp` | `exec;layer` | E1-E3 hypotheses (执行层) |

### 2.7 其他关键测试

| 测试文件 | 标签 | 测试目的 |
|---------|------|---------|
| `test_pc_management.cpp` | `pc;management` | PC 管理 |
| `test_pc_management_advanced.cpp` | `pc;advanced` | PC 管理高级功能 |
| `test_warp_state.cpp` | `warp;state` | Warp 状态管理 |
| `test_sync_mechanism.cpp` | `sync;mechanism` | 同步机制 |
| `test_barrier_simt_integration.cpp` | `barrier;simt` | 屏障与 SIMT 集成 |
| `test_simt_integration.cpp` | `simt;integration` | SIMT 集成测试 |
| `test_post_barrier_divergence.cpp` | `barrier;divergence` | CTA 屏障后线程 divergence 复现（5→2 TEST_CASE 合并后保留已知问题文档） |
| `test_specific_bugs_unit.cpp` | `bug;unit` | 4个特定 bug 的单元测试 (sync_to_warp_state, CFG barriers, cudaMemset offset, predicate register) |

---

## 3. 独立测试 (非 Catch2)

| 测试文件 | 测试目的 | 关键覆盖 |
|---------|---------|---------|
| `test-ptx.cpp` | PTX 解析端到端测试 | ANTLR4 解析、kernel 提取 |
| `test_printf.cu` | Printf 功能测试 | CUDA printf 到 PTX 模拟 |
| `test_addc_subc_handler.cpp` | Addc/Subc 指令处理 | 进位/借位运算 |
| `instructions/test_ptx_bra.cu` | BRA 指令测试 | 早期退出分支 |

---

## 4. PTX 语法测试 (test_all_ptx.sh)

**位置**: `tests/ptx/test_all_ptx.sh`

**测试方式**:
1. 从真实 CUDA 程序 (cuobjdump) 提取 PTX
2. 使用 ANTLR4 解析
3. 验证解析无错误

**覆盖范围**: 完整的 PTX ISA 语法

---

## 5. Benchmark 测试

### 5.1 Mini 测试 (标签: `mini`)

| 测试名 | 说明 |
|-------|------|
| `dummy` | 最简单的空 kernel |
| `dummy-add` | 加法 kernel |
| `dummy-sub` | 减法 kernel |
| `dummy-mul` | 乘法 kernel |
| `dummy-float` | 浮点运算 |
| `dummy-condition` | 条件分支 |
| `dummy-grid` | Grid/Block 维度 |
| `dummy-args` | Kernel 参数传递 |
| `dummy-long` | 长计算任务 |
| `dummy-sieve` | 素数筛 |
| `dummy-share` | 共享内存使用 |
| `dummy-loop` | 循环结构 |

### 5.2 Basic 测试 (标签: `basic`)

| 测试名 | 说明 |
|-------|------|
| `simpleGEMM-int/float/double` | 矩阵乘法 (各种精度) |
| `simpleCONV-int/float/double` | 卷积运算 |
| `2Dentropy` | 2D 熵计算 |
| `aligned-types` | 对齐类型访问 |
| `all-pairs-distance` | 全对距离计算 |
| `bitonic` | 排序网络 |
| `bfs` | 广度优先搜索 |

### 5.3 Extra 测试 (标签: `extra`)

| 测试名 | 说明 |
|-------|------|
| `dummy-wmma` | WMMA (Tensor Core) 操作 |

### 5.4 Sync 测试 (标签: `sync`)

| 测试名 | 说明 |
|-------|------|
| `test_syncthreads` | `__syncthreads()` 屏障同步 |
| `test_warp_divergence` | Warp 内部分支 divergence |
| `test_shared_memory` | 共享内存访问同步 |
| `test_divergence_sync_standalone` | Divergence 与 sync 集成 |

---

## 6. 测试缺失与改进建议

### 6.1 高优先级缺失

| 类别 | 缺失项 | 建议 |
|------|-------|------|
| **WMMA/Tensor Core** | `test_wmma.cpp` 已存在但被注释 | 需完整实现 wmma/mma 指令 |
| **Atomic 操作** | 无原子操作测试 | 添加 atomicAdd/atomicCAS 等测试 |
| **Event/Stream API** | 无同步测试 | 添加 cudaEvent/cudaStream 测试 |
| **函数调用** | 无递归/函数调用测试 | 添加 call/ret 指令测试 |
| **错误处理** | 边界情况不足 | 添加更多异常情况测试 |

### 6.2 中优先级改进

| 类别 | 现状 | 建议 |
|------|------|------|
| **多 CTA/多 SM** | 基本无 | 添加多 CTA 调度测试 |
| **多 Kernel** | 无 | 添加多 kernel 并发测试 |
| **内存事务** | 仅有基本 ld/st | 添加 memory fence/coherent 测试 |
| **Warp 级原语** | 无 shuffle 测试 | 添加 shfl/bfly 测试 |

### 6.3 低优先级改进

| 类别 | 现状 | 建议 |
|------|------|------|
| **性能测试** | 无 | 添加性能基准测试 |
| **模糊测试** | 无 | 添加随机 PTX 生成测试 |
| **回归测试** | 无自动化 | 添加 git bisect 自动化 |

---

## 7. 测试执行指南

### 7.1 运行所有测试
```bash
cd build && ctest
```

### 7.2 按标签运行
```bash
ctest -L mini           # Mini tests
ctest -L ptx            # PTX instruction tests
ctest -L barrier        # Barrier tests
ctest -L simt_entry     # SIMT stack tests
ctest -L exec_mask      # Exec mask tests
```

### 7.3 运行特定测试
```bash
ctest -R test_memory_manager -V
```

### 7.4 PTX 语法测试 (必须单独运行)
```bash
./tests/ptx/test_all_ptx.sh
```

### 7.5 运行 Benchmark
```bash
make dummy      # 单个 benchmark
make minitest   # 所有 mini 测试
make RAY        # 光线追踪
```

---

## 8. 测试统计

| 指标 | 数量 |
|------|-----|
| Catch2 测试文件 | ~45 |
| 独立测试文件 | ~5 |
| Benchmark 程序 | ~25 |
| PTX 指令类别 | 6 (integer, float, bitwise, cvt, ld_st, cvta) |
| SIMT 相关测试 | 25+ |
| Barrier 相关测试 | 15+ |

---

## 9. 已知限制

| 限制类别 | 状态 | 说明 |
|---------|------|------|
| WMMA/Tensor Core | Stub | 解析但未实现 |
| Atomic 操作 | Stub | 无真正原子性 |
| Hopper (sm_90+) | 不支持 | cluster 抽象未实现 |
| Event/Stream API | Fake | 不同步返回 |
| 函数调用 | 部分 | 未完全实现 |
| Multi-PTX cubins | 单 PTX | 仅第一个 PTX |
| assert(false) | 多处 | 遇未处理代码路径崩溃 |

---

*文档版本: 1.0*
*最后更新: 2026-05-05*