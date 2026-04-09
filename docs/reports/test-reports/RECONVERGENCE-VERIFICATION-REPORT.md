# reconvergence_pc 验证报告

**日期**: 2026-04-11  
**阶段**: Phase 7.1 - reconvergence_pc 深度验证  
**状态**: 进行中

---

## 测试用例 1: 3 层嵌套分支

### PC 布局分析

```
PC=0:  mov.u32 %r1, %tid.x
PC=1:  setp.lt.u32 %p1, %r1, 22
PC=2:  @%p1 bra $L_outer_true         ← outer_bra
PC=3:  add.u32 %r2, %r1, 1000         ← outer_false
PC=4:  bra.uni $L_outer_merge
PC=5:  L_outer_true:
PC=6:  setp.lt.u32 %p2, %r1, 11
PC=7:  @%p2 bra $L_inner1_true        ← inner1_bra
PC=8:  add.u32 %r3, %r1, 100          ← inner1_false
PC=9:  L_inner1_merge:
PC=10: add.u32 %r2, %r3, 0
PC=11: bra.uni $L_outer_merge
PC=12: L_inner1_true:
PC=13: setp.lt.u32 %p3, %r1, 5
PC=14: @%p3 bra $L_inner2_true        ← inner2_bra
PC=15: add.u32 %r4, %r1, 10           ← inner2_false
PC=16: L_inner2_merge:
PC=17: add.u32 %r3, %r4, 0
PC=18: bra.uni $L_inner1_merge
PC=19: L_inner2_true:
PC=20: add.u32 %r4, %r1, 1
PC=21: bra.uni $L_inner2_merge
PC=22: L_outer_merge:                 ← outer_merge (all converge)
PC=23: st.shared.u32 [%shared0], %r2
PC=24: bar.sync 0
PC=25: ret
```

### 预期 reconvergence_pc

| Branch 指令 | PC | 预期 reconvergence_pc | 说明 |
|------------|----|--------------------|------|
| outer_bra | 2 | **22** | L_outer_merge (所有路径汇合) |
| inner1_bra | 7 | **9** | L_inner1_merge (tier 2 汇合) |
| inner2_bra | 14 | **16** | L_inner2_merge (tier 3 汇合) |

### CFG 结构

```
                [PC=0-1]
                   │
                [PC=2] outer_bra
                 /    \
         (lane<22)    (lane>=22)
           /              \
      [PC=5-6]        [PC=3-4]
          │               │
      [PC=7] inner1   [PC=4] bra
       /    \            │
(lane<11) (lane>=11)   [PC=11]
    |        |           │
 [PC=12]  [PC=8-9] ─────┤
    |        │          │
 [PC=14] ───┤          │
   /  \      │          │
...   ...    │          │
     [PC=18]───────────┤
            │          │
         [PC=22] ←─────┘
         L_outer_merge
```

---

## 测试用例 2: 4 路多路分支

### PC 布局分析

```
PC=0:  mov.u32 %r1, %tid.x
PC=1:  setp.lt.u32 %p1, %r1, 8
PC=2:  @%p1 bra $L_path_low            ← path_select
PC=3:  setp.lt.u32 %p2, %r1, 24
PC=4:  @%p2 bra $L_path_mid_high       ← high_check
PC=5:  add.u32 %r2, %r1, 4000          ← path4
PC=6:  bra.uni $L_merge_point
PC=7:  L_path_mid_high:
PC=8:  add.u32 %r2, %r1, 3000          ← path3
PC=9:  bra.uni $L_merge_point
PC=10: L_path_low:
PC=11: setp.lt.u32 %p2, %r1, 16
PC=12: @%p2 bra $L_path_mid_low        ← low_check
PC=13: add.u32 %r2, %r1, 2000          ← path2
PC=14: bra.uni $L_merge_point
PC=15: L_path_mid_low:
PC=16: add.u32 %r2, %r1, 1000          ← path0
PC=17: L_merge_point:                  ← all converge
PC=18: st.shared.u32 [%shared0], %r2
PC=19: bar.sync 0
PC=20: ret
```

### 预期 reconvergence_pc

| Branch 指令 | PC | 预期 reconvergence_pc | 说明 |
|------------|----|--------------------|------|
| path_select | 2 | **17** | L_merge_point (所有路径汇合) |
| high_check | 4 | **17** | L_merge_point (path3/4 汇合) |
| low_check | 12 | **17** | L_merge_point (path0/1 汇合) |

---

## 测试用例 3: 循环结构

### PC 布局分析

```
PC=0:  mov.u32 %r1, 0
PC=1:  L_loop_header:
PC=2:  setp.lt.u32 %p1, %r1, 10
PC=3:  @%p1 bra $L_loop_body           ← loop_cond
PC=4:  bra.uni $L_loop_exit            ← loop_exit_branch
PC=5:  L_loop_body:
PC=6:  add.u32 %r1, %r1, 1
PC=7:  bra.uni $L_loop_header          ← back_edge
PC=8:  L_loop_exit:
PC=9:  st.shared.u32 [%shared0], %r1
PC=10: bar.sync 0
PC=11: ret
```

### 预期 reconvergence_pc

| Branch 指令 | PC | 预期 reconvergence_pc | 说明 |
|------------|----|--------------------|------|
| loop_cond | 3 | **8** | L_loop_exit (循环出口) |
| loop_exit_branch | 4 | **8** | L_loop_exit |
| back_edge | 7 | **1** | L_loop_header (循环头) |

---

## 验证步骤

### Step 1: 编译 PTX 并运行 CFG 分析

```bash
cd build
cmake --build . --target cudart
./bin/test_ptx_ld_st 2>&1 | grep -E "CFG|reconvergence|updated"
```

### Step 2: 对比预期 vs 实际

| 测试用例 | Branch PC | 预期值 | 实际值 | 状态 |
|---------|----------|-------|-------|------|
| nested_3levels | 2 (outer) | 22 | TBD | ⏳ |
| nested_3levels | 7 (inner1) | 9 | TBD | ⏳ |
| nested_3levels | 14 (inner2) | 16 | TBD | ⏳ |
| multipath_4ways | 2 (path_select) | 17 | TBD | ⏳ |
| multipath_4ways | 4 (high_check) | 17 | TBD | ⏳ |
| multipath_4ways | 12 (low_check) | 17 | TBD | ⏳ |
| loop_while | 3 (loop_cond) | 8 | TBD | ⏳ |
| loop_while | 4 (loop_exit) | 8 | TBD | ⏳ |
| loop_while | 7 (back_edge) | 1 | TBD | ⏳ |

### Step 3: 分析偏差

如果实际值与预期值不符：
1. 检查 CFG 构建逻辑
2. 检查 Post-Dominator 算法
3. 检查分支边添加逻辑
4. 修复并重新测试

---

**状态**: 测试用例创建完成，待运行验证  
**下一步**: 编译并运行 CFG 分析，对比预期值
