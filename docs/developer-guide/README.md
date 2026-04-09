# 开发指南目录

本目录包含 PTX-EMU SIMT v2.0 的开发指导文档。

---

## 📁 指南列表

| 指南 | 适合人群 | 状态 |
|------|---------|------|
| [GETTING-STARTED.md](./GETTING-STARTED.md) | 新开发者 | ⏳ 待创建 |
| [CFG-INTEGRATION-GUIDE.md](./CFG-INTEGRATION-GUIDE.md) | 后端开发 | ⏳ 待创建 |
| [SIMT-STACK-GUIDE.md](./SIMT-STACK-GUIDE.md) | SIMT 开发 | ⏳ 待创建 |
| [TESTING-GUIDE.md](./TESTING-GUIDE.md) | 测试工程师 | ⏳ 待创建 |
| [PERFORMANCE-GUIDE.md](./PERFORMANCE-GUIDE.md) | 性能工程师 | ⏳ 待创建 |

---

## 🚀 快速开始

### 新开发者路径

```bash
# 1. Cloning the repository
git clone github.com/chisuhua/PTX-EMU.git
cd PTX-EMU

# 2. Building the project
cmake -S . -B build
cmake --build build

# 3. Running tests
ctest --test-dir build

# 4. Reading docs
# Start with GETTING-STARTED.md (待创建)
```

---

## 📖 开发环境

### 必需工具

- CMake >= 3.10
- C++17 编译器 (GCC 9+, Clang 10+)
- Catch2 (测试框架)
- CUDA Toolkit (可选)

### 推荐工具

- VSCode + C/C++ 扩展
- clang-format (代码格式化)
- clang-tidy (代码检查)

---

## 🧪 测试指南

```bash
# Run all tests
ctest --test-dir build

# Run specific suite
ctest --test-dir build -R "cfg|simt"

# Verbose output
ctest --test-dir build -V
```

### 测试文件组织

```
tests/
├── ptx/                  # PTX 测试文件
│   ├── test_cfg_builder.cpp
│   ├── test_cfg_edge_cases.cpp
│   └── test_simt_stack_integration.cpp
└── CMakeLists.txt
```

---

## 📊 代码规范

### 命名规范

```cpp
// Classes/Structs: PascalCase
class CFGBuilder { };

// Functions: camelCase
void buildEdges();

// Variables: snake_case
int reconvergence_pc;

// Constants: UPPER_SNAKE_CASE
const int MAX_ITERATIONS = 100;
```

### 注释规范

```cpp
/**
 * @brief Build CFG from kernel statements
 * @param statements Kernel statements
 * @param label2pc Label to PC mapping
 * @return CFG Control flow graph
 */
static CFG build(...);
```

---

## 🔧 常见问题

### Q: CFG 分析何时运行？

A: Kernel 加载时（`setupLabels()` 函数中）

### Q: reconvergence_pc 如何计算？

A: 通过 CFG Post-Dominator 分析自动计算

### Q: 性能开销多少？

A: <5% (small: <1%, medium: <2%, large: <3%)

---

**维护**: 持续更新  
**贡献**: 提交 PR 到 `docs/developer-guide/`  
**最后更新**: 2026-04-11  
**状态**: 5 个指南中有 0 个已创建 (0%)
