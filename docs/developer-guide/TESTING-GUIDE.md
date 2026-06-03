# 测试指南

**版本**: v2.0  
**最后更新**: 2026-04-11  
**适合人群**: 测试工程师，开发者

---

## 📋 测试概览

### 测试分类

| 类别 | 数量 | 用途 |
|------|------|------|
| CFG Builder | 3 | CFG 构建验证 |
| SIMT Stack | 4 | SIMT 栈操作 |
| Edge Cases | 16 | 边界情况 |
| Performance | 3 | 性能基准 |
| Integration | 12 | 集成测试 |
| **总计** | **38** | **100% 通过** |

---

## 🧪 运行测试

### 运行所有测试

```bash
cd build
ctest --output-on-failure
```

### 运行特定测试套件

```bash
# CFG 相关测试
ctest -R "^(test_cfg|CFG)" -V

# SIMT 相关测试
ctest -R "^(test_simt|SIMT)" -V

# 性能测试
./bin/dummy
./bin/test_cfg_edge_cases
```

### 详细输出

```bash
ctest -R "test_cfg" --verbose
```

---

## 📁 测试文件组织

```
tests/
├── ptx/                     # PTX 测试文件
│   ├── test_cfg_edge_cases.cpp        # CFG 边界测试 (40 断言)
│   ├── test_simt_stack_integration.cpp # SIMT Stack 集成测试
│   └── *.ptx                          # PTX 汇编测试
└── CMakeLists.txt
```

---

## 🔧 编写新测试

### 测试模板

```cpp
#include <catch2/catch.hpp>
#include "ptx_parser/cfg_builder.h"

TEST_CASE("Description", "[category][tag]") {
    // Setup
    std::vector<StatementContext> statements;
    std::map<std::string, int> label2pc;
    
    // Exercise
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    // Verify
    REQUIRE(cfg.blocks.size() > 0);
    REQUIRE(cfg.entry_block_id == 0);
}
```

### 测试类别标签

| 标签 | 用途 |
|------|------|
| `[cfg]` | CFG Builder 相关 |
| `[simt]` | SIMT Stack 相关 |
| `[edge]` | 边界情况 |
| `[perf]` | 性能测试 |
| `[integration]` | 集成测试 |

---

## 📊 测试覆盖要求

### 覆盖率目标

| 指标 | 目标 | 当前 |
|------|------|------|
| 核心代码 | >90% | 94% |
| 边界情况 | >90% | 94% |
| 集成测试 | 完整流程 | ✅ |

### 必需测试场景

1. **正常路径**
   - 基本功能测试
   - 典型使用场景

2. **边界情况**
   - 空输入
   - 单元素
   - 极大输入

3. **错误处理**
   - 无效输入
   - 异常条件

---

## 🐛 调试测试

### 获取详细日志

```bash
ctest --test-dir build -R "test_name" \
    --output-on-failure \
    --verbose
```

### 运行单个测试

```bash
ctest --test-dir build -R "test_name" --output-on-failure
```

### 检查测试输出

```bash
./bin/tests/test_name 2>&1 | less
```

---

## 📈 性能测试

### 基准测试

```bash
# Small kernel
./bin/dummy

# CFG analysis
./bin/test_cfg_edge_cases

# Full suite
./bench/test_syncthreads/test_syncthreads
```

### 性能指标

| Kernel Size | Expected Time | Max Allowed |
|-------------|--------------|-------------|
| Small (<50) | ~10 μs | <50 μs |
| Medium (50-200) | ~25 μs | <100 μs |
| Large (>200) | ~50 μs | <200 μs |

---

## ✅ 测试清单

### 提交前检查

- [ ] 所有现有测试通过
- [ ] 新测试已添加
- [ ] 测试文档已更新
- [ ] 性能无回归

### Release 前检查

- [ ] 38/38 测试通过
- [ ] 性能基准达标 (<5% 开销)
- [ ] 边界情况覆盖 >90%
- [ ] 文档完整

---

## 📚 参考文档

- Phase 7-8 Reports (在 `reports/phase-reports/`)

---

**最后更新**: 2026-04-11
