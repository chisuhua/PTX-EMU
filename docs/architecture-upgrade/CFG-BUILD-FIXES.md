# CFG Builder 批量修复清单

**日期**: 2026-04-10  
**状态**: Ready for Bulk Fix  
**修复策略**: 按优先级批量修复

---

## 问题汇总

### 🔴 P0: 关键功能问题 (Blocker)

| ID | 问题 | 文件 | 行号 | 影响 |
|----|------|------|------|------|
| **2.1** | buildEdges 缺少分支目标边 | cfg_builder.cpp | 93-127 | CFG 不完整，reconvergence_pc 错误 |

### 🟡 P1: 健壮性问题 (High)

| ID | 问题 | 文件 | 行号 | 影响 |
|----|------|------|------|------|
| **2.2** | 未验证分支目标存在 | cfg_builder.cpp | 44-55 | 调试困难，错误难以定位 |
| **4.1** | 缺少 CFG 验证 | test_reconvergence_parser.cpp | - | 测试不完整 |

### 🟢 P2: 代码质量问题 (Medium)

| ID | 问题 | 文件 | 行号 | 影响 |
|----|------|------|------|------|
| **1.1** | 缺少 const 方法 | cfg_builder.h | 34-36 | API 不完整 |
| **1.2** | 缺少文档注释 | cfg_builder.h | 全部 | 可维护性差 |
| **3.1** | 缺少预期结果注释 | test_reconvergence.ptx | - | 测试预期不明确 |

---

## 批量修复顺序

### Phase A: P0 修复 (关键)

**文件**: `src/ptx_parser/cfg_builder.cpp`

**修复内容**: buildEdges 函数添加分支目标边

**预计时间**: 30 分钟

**验收**:
```bash
cmake --build build --target ptx_parser
# Expected: 100% Built target ptx_parser
```

---

### Phase B: P1 修复 (健壮性)

**文件 1**: `src/ptx_parser/cfg_builder.cpp`

**修复内容**: 添加分支目标验证和错误日志

**文件 2**: `tests/ptx/test_reconvergence_parser.cpp`

**修复内容**: 添加 CFG 结构验证

**预计时间**: 45 分钟

**验收**:
```bash
./tests/ptx/test_reconvergence_ptx.sh
# Expected: ALL TESTS PASSED with CFG validation
```

---

### Phase C: P2 修复 (代码质量)

**文件 1**: `src/ptx_parser/cfg_builder.h`

**修复内容**: 添加 const 方法和 Doxygen 文档

**文件 2**: `tests/ptx/test_reconvergence.ptx`

**修复内容**: 添加预期结果注释

**预计时间**: 1.5 小时

**验收**:
- 编译无警告
- 文档完整

---

## 修复后验证清单

### 编译验证
- [ ] cmake --build build --target ptx_parser (0 errors)
- [ ] cmake --build build (full build, 0 errors)
- [ ] No compiler warnings (-Wall -Wextra)

### 功能验证
- [ ] ./tests/ptx/test_reconvergence_ptx.sh (ALL PASS)
- [ ] ctest -R reconvergence (PASS)
- [ ] CFG structure validated
- [ ] reconvergence_pc values correct

### 代码质量
- [ ] const 方法完整
- [ ] Doxygen 文档完整
- [ ] 错误日志清晰

---

## 回滚方案

如果批量修复失败：
```bash
# 修复前创建 checkpoint
git checkout -b checkpoint/pre-bulk-fix

# 如果修复失败
git checkout feature/simt-v2-phase5-testing
git branch -D feature/bulk-fix-attempt
```

---

**状态**: Ready to execute Phase A  
**下一步**: 开始 P0 修复 (问题 2.1)
