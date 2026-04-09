# SIMT v2.0 Phase 5-6 实施计划 (Updated with Parser Tests)

**版本**: 3.0 (Updated with Parser Test Strategy)  
**日期**: 2026-04-10  
**状态**: Ready for Execution  
**关键改进**: Parser 级别 reconvergence_pc 测试

---

## 执行摘要

本文档定义了 SIMT v2.0 架构的 Phase 5-6 实施计划，整合了 Parser 级别的 reconvergence_pc 测试方案，实现早期验证和快速反馈。

**核心策略**:
1. ✅ Phase 5.0: 准备与代码审查
2. ✅ Phase 5.1: CFG Builder 编译集成 + Parser 测试验证
3. ✅ Phase 5.2: Kernel 集成 + 端到端验证
4. ✅ Phase 5.3-5.5: 完整测试覆盖
5. ✅ Phase 6: 性能与质量验证

**预计完成时间**: 40 小时 (5 个工作日)  
**测试策略**: Parser 测试 (早期) + Runtime 测试 (完整)

---

## 测试策略对比

| 测试级别 | 测试文件 | 执行时机 | 反馈速度 | 依赖 |
|---------|---------|---------|---------|------|
| **Parser 测试** ⭐ NEW | test_reconvergence_ptx.sh | 解析后 | ~0.1s | CFG Builder |
| **Runtime 测试** | test_ptx_bra, test_syncthreads | Kernel 加载后 | ~2s | 完整运行时 |

**推荐流程**:
```
Phase 5.1: 运行 Parser 测试 (快速反馈)
    ↓
Phase 5.2: Parser 测试 + Runtime 测试
    ↓
Phase 5.3+: 主要运行 Runtime 测试
```

---

## Phase 5: 基础集成 (Updated)

### Phase 5.0: 准备与审查 (2 小时)

**目标**: 确认当前状态，准备实施环境

#### 任务 5.0.1: 代码审查 (30 分钟)

**审查文件**:
- `src/ptx_parser/cfg_builder.h` - 类型定义
- `src/ptx_parser/cfg_builder.cpp` - 实现
- `src/CMakeLists.txt` - 编译配置

**检查清单**:
- [ ] Include 路径正确
- [ ] 类型定义完整
- [ ] 方法签名匹配
- [ ] 无编译语法错误

**输出**: `docs/architecture-upgrade/CODE-REVIEW-CFG.md`

---

#### 任务 5.0.2: 创建备份分支 (15 分钟)

```bash
cd /workspace/project/PTX-EMU
git checkout -b backup/pre-phase5
git push origin backup/pre-phase5
```

---

#### 任务 5.0.3: 记录测试基线 (15 分钟)

```bash
cd build
ctest -R "simt|bra|syncthreads" --output-on-failure > test_baseline.txt
```

**预期基线**: 5/8 tests pass (62.5%)

---

#### 任务 5.0.4: 准备开发环境 (30 分钟)

```bash
# 设置调试标志
export CMAKE_BUILD_TYPE=Debug
export PTX_DEBUG=1
export PTX_SIMT_V2=1

# 验证工具链
which g++ cmake valgrind
```

---

### Phase 5.1: CFG Builder 编译集成 + Parser 测试 (8 小时)

#### 任务 5.1.1: 修复 cfg_builder.h 类型定义 (2 小时)

**验收标准**:
```bash
g++ -c src/ptx_parser/cfg_builder.cpp -I./include -std=c++17
# 预期：0 errors
```

---

#### 任务 5.1.2: 更新 CMakeLists.txt (1 小时)

**文件**: `src/CMakeLists.txt`

**修改**:
```cmake
add_library(ptx_parser SHARED
    ptx_parser/ptx_visitor.cpp
    ptx_parser/cfg_builder.cpp  # ADD THIS
)
```

---

#### 任务 5.1.3: 验证 CFG Builder 编译 (2 小时)

**验收标准**:
```bash
cmake --build build --target ptx_parser
# 预期：100% Built target ptx_parser
```

---

#### 任务 5.1.4: Parser 级别 reconvergence_pc 测试 (3 小时) ⭐ NEW

**文件**: `tests/ptx/test_reconvergence_ptx.sh`

**验收标准**:
```bash
./tests/ptx/test_reconvergence_ptx.sh
# 预期输出：
# ✅ ALL TESTS PASSED
# reconvergence_pc correctly computed for all branches
```

**测试验证**:
- [ ] test_reconvergence.ptx 正确解析
- [ ] CFG Builder 执行成功
- [ ] BranchInstr.reconvergence_pc 非 -1
- [ ] reconvergence_pc > branch_pc

---

### Phase 5.2: Kernel 集成 + 端到端验证 (8 小时)

#### 任务 5.2.1: 在 ptx_interpreter.cpp 中添加 CFG 调用 (3 小时)

**文件**: `src/cudart/ptx_interpreter.cpp`  
**函数**: `PtxInterpreter::setupLabels()`

**代码**:
```cpp
void PtxInterpreter::setupLabels(std::map<std::string, int> &label2pc) {
    // ... existing label setup ...
    
    // NEW: CFG analysis (SIMT v2.0)
    ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(
        kernelContext->kernelStatements, 
        label2pc
    );
    
    auto postDoms = ptx::cfg::CFGBuilder::computePostDominators(cfg);
    
    for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
        auto &stmt = kernelContext->kernelStatements[i];
        if (stmt.type == S_BRA) {
            auto &branch = std::get<BranchInstr>(stmt.data);
            auto it = postDoms.find(i);
            branch.reconvergence_pc = (it != postDoms.end()) ? it->second : i + 1;
        }
    }
}
```

**验收标准**: 编译通过，无链接错误

---

#### 任务 5.2.2: 运行 Parser 测试验证 (2 小时)

```bash
./tests/ptx/test_reconvergence_ptx.sh
# 预期：PASS (所有分支 reconvergence_pc 正确)
```

---

#### 任务 5.2.3: 运行 Runtime 测试 (3 小时)

```bash
cd build
ctest -R "ptx_bra|syncthreads" --output-on-failure -V
# 预期：2/2 tests PASS
```

---

### Phase 5.3: reconvergence_pc 填充验证 (4 小时)

#### 任务 5.3.1: 添加详细日志 (2 小时)

**输出示例**:
```
CFG analysis complete: updated 5 branch instructions
Branch at PC=3: reconvergence_pc=10
Branch at PC=7: reconvergence_pc=10
```

---

#### 任务 5.3.2: 验证多个 PTX 文件 (2 小时)

```bash
for ptx in tests/ptx/*.ptx; do
    ./build/bin/test_reconvergence_parser "$ptx"
done
```

---

### Phase 5.4: 单元测试修复验证 (4 小时)

#### 任务 5.4.1: 运行完整测试套件 (2 小时)

```bash
cd build && ctest --output-on-failure
# 预期：100% tests passed (8/8)
```

**测试清单**:
- [ ] test_ptx_bra ✅
- [ ] test_syncthreads ✅
- [ ] test_simt_thread_pc ✅
- [ ] test_warp_barrier_extended ✅
- [ ] test_scheduler_config ✅
- [ ] test_warp_context ✅
- [ ] test_warp_scheduler ✅
- [ ] test_reconvergence_parser ✅ (NEW)

---

#### 任务 5.4.2: 性能基线对比 (2 小时)

```bash
./bench/test_syncthreads/test_syncthreads > baseline.txt
# Compare with previous runs
```

---

### Phase 5.5: CFG 单元测试 (4 小时)

#### 任务 5.5.1: 创建 test_cfg_builder.cpp (2 小时)

```cpp
TEST_CASE("CFG Builder - Basic Block Identification") {
    // Test block identification
}

TEST_CASE("CFG Builder - Post-Dominator") {
    // Test post-dominator computation
}
```

---

#### 任务 5.5.2: 运行 CFG 测试 (2 小时)

```bash
ctest -R cfg_builder --output-on-failure
# 预期：2/2 tests PASS
```

---

## Phase 6: 完整验证 (Updated)

### Phase 6.1: 分支收敛集成测试 (6 小时)

**测试用例**:
- [ ] 6.1.1 简单 if-else
- [ ] 6.1.2 嵌套分支
- [ ] 6.1.3 分支 + barrier 组合

**验收**: 所有用例通过

---

### Phase 6.2: 性能基准测试 (6 小时)

**基准测试**:
- [ ] 6.2.1 test_syncthreads
- [ ] 6.2.2 test_warp_divergence
- [ ] 6.2.3 2Dentropy

**验收**: 性能回归 < 5%

---

### Phase 6.3: 代码质量审查 (6 小时)

**检查项**:
- [ ] 6.3.1 编译警告 (-Wall -Wextra)
- [ ] 6.3.2 LSP 诊断
- [ ] 6.3.3 内存泄漏 (valgrind)
- [ ] 6.3.4 代码风格 (clang-format)

**验收**: 0 errors, 0 critical warnings

---

### Phase 6.4: 文档完善 (6 小时)

**更新文档**:
- [ ] 6.4.1 SIMT-ARCHITECTURE-V2.md
- [ ] 6.4.2 CFG_ANALYSIS_GUIDE.md (NEW)
- [ ] 6.4.3 PROGRESS-V2.md
- [ ] 6.4.4 RECONVERGENCE_TEST_GUIDE.md (NEW)

---

### Phase 6.5: 最终验收 (6 小时)

**验收清单**:
- [ ] 6.5.1 所有单元测试通过 (100%)
- [ ] 6.5.2 所有集成测试通过 (100%)
- [ ] 6.5.3 性能基准无回归
- [ ] 6.5.4 代码质量审查通过
- [ ] 6.5.5 文档完整

---

## 时间估算 (Updated)

| Phase | 任务 | 原始估算 | Updated 估算 | 说明 |
|-------|------|---------|-------------|------|
| **5.0** | 准备与审查 | - | 2h | 新增 |
| **5.1** | CFG 编译 + Parser 测试 | 4h | 8h | +Parser 测试 |
| **5.2** | Kernel 集成 | 4h | 8h | +端到端验证 |
| **5.3** | reconvergence_pc | 2h | 4h | +多文件验证 |
| **5.4** | 单元测试修复 | 2h | 4h | +完整套件 |
| **5.5** | CFG 单元测试 | 2h | 4h | +覆盖率 |
| **Phase 5 小计** | | 14h | **30h** | +16h |
| | | | | |
| **6.1** | 集成测试 | 4h | 6h | +测试用例 |
| **6.2** | 性能基准 | 4h | 6h | +对比分析 |
| **6.3** | 代码质量 | 4h | 6h | +valgrind |
| **6.4** | 文档完善 | 4h | 6h | +NEW guides |
| **6.5** | 最终验收 | 4h | 6h | +完整测试 |
| **Phase 6 小计** | | 20h | **30h** | +10h |
| | | | | |
| **总计** | | 34h | **60h** | **7.5 工作日** |

---

## 关键里程碑 (Updated)

| 里程碑 | 原计划 | Updated | 交付物 |
|--------|--------|---------|-------|
| M0: Phase 5.0 | - | D0 18:00 | 代码审查报告 |
| M1: Phase 5.1 | D1 12:00 | D1 18:00 | CFG 编译 +Parser 测试 ✅ |
| M2: Phase 5.2 | D1 16:00 | D2 18:00 | Kernel 集成 + Runtime 测试 ✅ |
| M3: Phase 5 完成 | D1 20:00 | D4 18:00 | Phase 5 100% |
| M4: Phase 6 完成 | D2 20:00 | D7 18:00 | Phase 6 100% |
| M5: 合并到 main | D3 | D8 | SIMT v2.0 完整功能 |

---

## 立即行动项 (Updated)

### Phase 5.0.1: 代码审查 (30 分钟) - START NOW

```bash
cd /workspace/project/PTX-EMU
cat src/ptx_parser/cfg_builder.h
cat src/ptx_parser/cfg_builder.cpp
```

**输出**: `docs/architecture-upgrade/CODE-REVIEW-CFG.md`

---

### Phase 5.0.2: 创建备份分支 (15 分钟)

```bash
git checkout -b backup/pre-phase5
```

---

### Phase 5.0.3: 记录测试基线 (15 分钟)

```bash
cd build && ctest -R "simt|bra|syncthreads" --output-on-failure > test_baseline.txt
```

**预期**: 5/8 tests pass (62.5%)

---

### Phase 5.0.4: 准备开发环境 (30 分钟)

```bash
export CMAKE_BUILD_TYPE=Debug
export PTX_DEBUG=1
```

---

## Parser 测试集成点

### Phase 5.1 验收

```bash
# New acceptance criteria
./tests/ptx/test_reconvergence_ptx.sh
# Expected: COMPILATION SUCCESS + reconvergence_pc populated
```

### Phase 5.2 验收

```bash
# Parser test
./tests/ptx/test_reconvergence_ptx.sh
# Expected: ALL TESTS PASSED

# Runtime test
ctest -R "ptx_bra|syncthreads" --output-on-failure
# Expected: 2/2 tests PASSED
```

### Phase 5.3+ 使用

```bash
# Quick validation during development
./tests/ptx/test_reconvergence_ptx.sh
# Expected: Fast feedback (~0.1s)
```

---

## 风险评估 (Updated)

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| CFG 类型不匹配 | 高 | 高 | Phase 5.0 审查 + Phase 5.1 Parser 测试 |
| CMake 链接错误 | 中 | 高 | Phase 5.1.2 仔细配置 |
| reconvergence_pc 错误 | 中 | 高 | Phase 5.1.4 Parser 测试早期发现 |
| 测试仍然失败 | 中 | 中 | Phase 5.4 预留调试时间 |
| 性能回归 | 低 | 中 | Phase 6.2 监控 |

**改进**: Parser 测试使早期发现风险成为可能

---

## 成功标准 (Updated)

### Phase 5 完成

- [ ] CFG Builder 编译通过 ✅
- [ ] Parser 测试通过 ✅
- [ ] Kernel 集成完成 ✅
- [ ] Runtime 测试通过 (8/8) ✅
- [ ] CFG 单元测试通过 (2/2) ✅

### Phase 6 完成

- [ ] 集成测试通过 ✅
- [ ] 性能回归 < 5% ✅
- [ ] 代码质量审查通过 ✅
- [ ] 文档完整 ✅
- [ ] 最终验收通过 ✅

---

## 附录：Parser 测试快速参考

### 运行测试

```bash
# Method 1: Shell script (Recommended)
./tests/ptx/test_reconvergence_ptx.sh

# Method 2: Direct execution
./build/bin/test_reconvergence_parser tests/ptx/test_reconvergence.ptx

# Method 3: CTest
ctest -R reconvergence --output-on-failure -V
```

### 预期输出

```
=== SIMT v2.0 Reconvergence PC Parser Test ===
Step 1: Checking test PTX file... OK
Step 2: Compiling test program... OK
Step 3: Running reconvergence_pc test...
============================================================================
=== Checking test_simple_branch ===
PC=3: bra $L_then, reconvergence_pc=10 ✅ PASS
PC=7: bra.uni $L_merge, reconvergence_pc=10 ✅ PASS

--- Summary for test_simple_branch ---
Total branches: 2
With reconvergence_pc: 2
Invalid reconvergence_pc: 0

=== Overall Result ===
✅ ALL TESTS PASSED
```

### 故障排查

| 错误 | 原因 | 解决 |
|------|------|------|
| compilation failed | CFG 未编译 | Phase 5.1 |
| reconvergence_pc=-1 | CFG 未执行 | Phase 5.2 |
| reconvergence_pc <= branch_pc | Post-Dominator 错误 | Phase 5.3 |

---

**状态**: Ready for Phase 5.0 execution  
**下一步**: 开始 Phase 5.0.1 (代码审查)
