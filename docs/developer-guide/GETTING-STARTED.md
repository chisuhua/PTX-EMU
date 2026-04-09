# 快速开始指南

**版本**: v2.0  
**最后更新**: 2026-04-11  
**适合人群**: 新开发者

---

## 🎯 目标读者

本指南适合：
- 初次接触 PTX-EMU 项目的开发者
- 需要快速上手的贡献者
- 了解 SIMT v2.0 架构的工程师

---

## 📋 前置要求

### 必需工具

| 工具 | 版本 | 安装方式 |
|------|------|---------|
| CMake | >= 3.10 | `apt install cmake` |
| C++ Compiler | GCC 9+ / Clang 10+ | `apt install build-essential` |
| Git | 最新版本 | `apt install git` |

### 可选工具

| 工具 | 用途 |
|------|------|
| CUDA Toolkit | CUDA 开发 |
| VSCode + C/C++ 扩展 | 代码编辑 |
| clang-format | 代码格式化 |

---

## 🚀 快速安装

### 1. 克隆项目

```bash
git clone https://github.com/chisuhua/PTX-EMU.git
cd PTX-EMU
```

### 2. 构建项目

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)
```

### 3. 运行测试

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run specific suite
ctest --test-dir build -R "cfg|simt" -V
```

---

## 📁 项目结构

```
PTX-EMU/
├── src/                    # 源代码
│   ├── ptx_parser/         # PTX 解析器
│   ├── ptxsim/             # PTX 模拟器
│   └── cudart/             # CUDA runtime 模拟
├── include/                # 头文件
├── tests/                  # 测试代码
├── docs/                   # 文档
└── build/                  # 构建目录
```

---

## 🏗️ 核心架构

### SIMT v2.0 关键组件

| 组件 | 文件 | 用途 |
|------|------|------|
| **CFG Builder** | `src/ptx_parser/cfg_builder.*` | 控制流图构建 |
| **SIMT Stack** | `src/ptxsim/simt_stack.*` | 分支收敛管理 |
| **Post-Dominator** | `src/ptx_parser/cfg_builder.cpp` | reconvergence 计算 |

### 数据流

```
PTX Kernel
    ↓
Parser (PTX Visitor)
    ↓
CFG Builder ← Phase 5
    ↓
Post-Dominator Analysis
    ↓
reconvergence_pc computation
    ↓
Kernel Execution (SIMT)
    ↓
Branch → SIMT Stack Push
    ↓
Convergence → SIMT Stack Pop
```

---

## 📖 学习路径

### Week 1: 基础理解

1. **阅读文档**
   - [`SIMT-ARCHITECTURE-V2.md`](../architecture/SIMT-ARCHITECTURE-V2.md)
   - [`cfg-builder-pattern.md`](../skills/cfg-builder-pattern.md)

2. **运行示例**
   ```bash
   cd build && ./bin/dummy
   ./bin/tests/test_cfg_builder
   ```

3. **理解测试**
   - 查看 `tests/ptx/test_cfg_builder.cpp`
   - 运行并修改测试

### Week 2: 深入开发

1. **学习算法**
   - [`post-dominator-algorithm.md`](../skills/post-dominator-algorithm.md)
   - [`simt-reconvergence.md`](../skills/simt-reconvergence.md)

2. **代码实践**
   - 修改小规模代码
   - 添加新测试用例

3. **开发流程**
   - [`tdd-workflow.md`](../skills/tdd-workflow.md)

---

## 🧪 测试指南

### 运行特定测试

```bash
# CFG Builder 测试
ctest --test-dir build -R "cfg" -V

# SIMT 测试
ctest --test-dir build -R "simt" -V

# 性能基准
./build/bin/dummy
./build/bin/test_cfg_builder
```

### 添加新测试

1. 创建测试文件: `tests/ptx/test_your_feature.cpp`
2. 添加到 CMakeLists.txt
3. 运行测试: `ctest -R "your_test"`

---

## 🔧 开发工作流

### TDD 流程

```
1. Write failing test
   ↓
2. Run test (Red)
   ↓
3. Implement minimal code
   ↓
4. Run test (Green)
   ↓
5. Refactor
   ↓
6. Repeat
```

详细指南: [`tdd-workflow.md`](../skills/tdd-workflow.md)

### 提交规范

```
[Component] Brief description

Detailed description if needed.

Part of: Phase X
```

示例:
```
[CFG Builder] Fix post-dominator edge case

- Add null check for empty CFG
- Improve error messages

Part of: Phase 7
```

---

## 📚 参考资料

### 内部文档

| 文档 | 用途 |
|------|------|
| [`SIMT-ARCHITECTURE-V2.md`](../architecture/) | 完整架构设计 |
| [`skills/`](../skills/) | 技术技能总结 |
| [`reports/phase-reports/`](../reports/phase-reports/) | Phase 进度报告 |

### 外部资源

| 资源 | 说明 |
|------|------|
| PTX ISA 9.1 | NVIDIA 官方文档 |
| cuda-ptx Skill | 完整 PTX 规范 (405 文件) |

---

## ❓ 常见问题

### Q: 编译失败怎么办？

A: 检查:
1. CMake 版本 >= 3.10
2. C++ 编译器支持 C++17
3. 清理后重新构建: `rm -rf build && cmake .. && cmake --build .`

### Q: 测试失败怎么办？

A: 运行详细输出:
```bash
ctest --test-dir build -R "test_name" -V
```
查看日志定位问题。

### Q: 如何理解 CFG 分析？

A: 阅读:
1. [`cfg-builder-pattern.md`](../skills/cfg-builder-pattern.md)
2. [`post-dominator-algorithm.md`](../skills/post-dominator-algorithm.md)
3. Phase 5 报告

---

## 📞 获取帮助

- **文档**: [`docs/README.md`](../README.md)
- **Issue**: GitHub Issues
- **Discussion**: GitHub Discussions

---

**下一步**: 开始阅读 [`SIMT-ARCHITECTURE-V2.md`](../architecture/) 或 运行第一个测试

**最后更新**: 2026-04-11
