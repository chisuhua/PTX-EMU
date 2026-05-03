---
name: "three-mode-testing"
description: "三模式 PTX 测试生成框架 — 从任意 CUDA 程序自动生成 PTX/IR/执行三种测试模式"
when_to_use: |
  PTX-EMU 项目中出现：
  - "生成测试", "创建 PTX 测试", "添加测试用例"
  - "three-mode", "测试生成", "test generation"
  - "从 CUDA 程序生成测试"
skills_required: []
---

# Three-Mode Testing Framework Skill

## 概述

三模式 PTX 测试生成框架，支持**任意 CUDA 程序**作为输入，自动生成三种测试模式。

## 核心能力

- **任意输入**: 支持 `bench/<name>/`, `bench/<name>.cu`, 编译后的 binary，或任意 `.cu` 源文件
- **自动发现**: 自动查找 CUDA 源码和编译产物
- **修复 Catch2**: 自动包裹 `operator||`/`operator&&` 表达式（避免 static_assert 错误）

## 输入方式

```bash
# 方式 1: bench 目录（自动查找 .cu 和 binary）
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy

# 方式 2: 直接指定 CUDA 源文件
python3 docs/skills/three-mode-testing/generate_tests.py --cuda-source bench/dummy/dummy.cu

# 方式 3: 指定已编译的 binary
python3 docs/skills/three-mode-testing/generate_tests.py --binary build/bin/dummy

# 方式 4: 指定预提取的 PTX 文件
python3 docs/skills/three-mode-testing/generate_tests.py --ptx tests/three_mode_testing/ptx/dummy.ptx
```

## 添加新测试

### 方式 1: 从 bench 目录（推荐）

```bash
# 1. 编译你的 benchmark
cmake --build build --target dummy

# 2. 生成三模式测试
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy

# 3. 构建
cmake --build build --target test_dummy_mode1 test_dummy_mode2 test_dummy_mode3

# 4. 运行
ctest -R "^test_dummy_mode" -V
```

### 方式 2: 从任意 CUDA 源文件

```bash
python3 docs/skills/three-mode-testing/generate_tests.py \
    --cuda-source /path/to/my_kernel.cu \
    --test-name my_kernel
```

### 方式 3: 从已编译 binary

```bash
python3 docs/skills/three-mode-testing/generate_tests.py \
    --binary build/bin/my_kernel \
    --test-name my_kernel
```

## 更新已有测试

```bash
# 重新生成所有模式（智能合并，不重复添加）
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy --force

# 只更新特定模式
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy --mode 2
```

**注意**: `sync_cmake()` 使用 word boundary 检测已存在的测试，不会重复添加。

## Five-Mode Testing Framework

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Mode 1** | cuobjdump 动态提取 PTX | 集成测试、CI/CD |
| **Mode 2** | 预提取 PTX 文件 | 回归测试、版本控制 |
| **Mode 3a** | StatementContext BEFORE CFG | 单元测试，无 reconvergence_pc |
| **Mode 3b** | StatementContext AFTER CFG | 单元测试，reconvergence_pc 已填充 |
| **Mode 3c** | 运行 standalone binary（popen）| 端到端 FAIL 复现测试 |
| **Mode 4** | PTXIR binary serialize/deserialize | 快速加载测试（~5ms vs ~200ms）|

### Mode 3C: Standalone Binary FAIL Reproduction

Mode 3C **不依赖内部 PtxContext/GPUContext 类型**（因为 ptx_parser.h 和 ptx_types.h 之间存在 X-macro 冲突）。
而是通过 `popen()` 直接运行已编译的 standalone binary，检测其输出中的 FAIL 标记。

- **测试目标**: 复现 standalone binary 的 FAIL 行为
- **当前行为**: 检测 `=== Result: FAIL ===` 标记
- **bug 修复后**: 输出会变为 `=== Result: PASS ===`，测试代码需要相应更新
- **运行条件**: 直接调用 standalone binary（使用 `PTX_LOG_LEVEL=error` 减少噪音）

## 完整工作流

```bash
# 1. 编译你的 CUDA 程序
cmake --build build --target dummy

# 2. 生成五模式测试（自动发现源码和 binary）
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy

# 3. 构建并运行
cmake --build build --target test_dummy_mode1 test_dummy_mode2 test_dummy_mode3a test_dummy_mode3b
ctest -R dummy -V
```

## 高级选项

```bash
# 只生成特定模式
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy --mode 3

# 强制覆盖已有文件
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy --force

# 预览（不生成文件）
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy --dry-run

# 指定测试名称（默认从输入自动推断）
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy --test-name my_test
```

## PTX 分析

生成器自动分析 PTX 内容，检测：

- `.entry` 内核入口
- `bar.sync` / `bar.warp.sync` 屏障指令
- `ld.shared` / `st.shared` 共享内存操作
- 条件/无条件分支 (`bra`)
- 共享变量声明

## Mode 3 构造

Mode 3 使用 `test_helpers.hpp` 中的辅助函数构造 StatementContext：

### 语句构造

| 函数 | PTX 指令 |
|------|----------|
| `make_bar_warp_sync(mask, reconv_pc)` | `bar.warp.sync` |
| `make_bar_sync(bar_id)` | `bar.sync` |
| `make_ld_shared(dst, var, offset)` | `ld.shared` |
| `make_st_shared(var, offset, src)` | `st.shared` |
| `make_mov(dst, src)` | `mov` |
| `make_mov_imm(dst, imm)` | `mov` 立即数 |
| `make_add(dst, src1, src2)` | `add` |
| `make_mul(dst, src1, src2)` | `mul` |
| `make_setp_lt(pred, src1, src2)` | `setp.lt` |
| `make_bra(target)` | 无条件分支 |
| `make_bra_pred(target, pred, neg)` | 条件分支 |
| `make_label(name)` | 标签 |
| `make_nop()` | `nop` |
| `make_exit()` | `exit` |

### Warp 设置

| 函数 | 说明 |
|------|------|
| `setup_warp(warp, threads, 32)` | 创建含 32 lane 的 warp |
| `reset_warp(warp)` | 重置所有线程状态 |

### 验证

| 函数 | 说明 |
|------|------|
| `count_active_lanes(warp)` | 统计活跃 lane 数 |
| `count_at_pc(warp, pc)` | 统计特定 PC 处 lane 数 |
| `check_mask(warp, expected)` | 断言 mask 匹配 |
| `get_active_mask(warp)` | 获取当前 active mask |

### 共享内存

| 函数 | 说明 |
|------|------|
| `allocate_shared(n)` | 分配 n 个 uint32_t |
| `write_shared(base, offset, val)` | 写入值 |
| `read_shared(base, offset)` | 读取值 |

## Catch2 注意事项

**重要**: 生成器自动修复 Catch2 的 `operator||`/`operator&&` 限制：

```cpp
// 错误写法（会导致 static_assert 失败）
CHECK(ptx_contains(ptx, "bar.sync") || ptx_contains(ptx, "bar.warp.sync"));

// 正确写法（生成器自动包裹）
CHECK((ptx_contains(ptx, "bar.sync") || ptx_contains(ptx, "bar.warp.sync")));
```

## 调试工作流

```
发现 bug
    ↓
Mode 1: 复现（cuobjdump 提取）
    ↓
Mode 2: 隔离（固定 PTX 文件）
    ↓
Mode 3: 定位（StatementContext 单元测试）
    ↓
修复代码
    ↓
Mode 2: 回归测试
    ↓
Mode 1: 端到端验证
```

## 目录结构

```
tests/three_mode_testing/
├── CMakeLists.txt          # 自动更新
├── test_helpers.hpp        # 共享辅助函数 (StatementContext 构建)
├── test_<name>_mode1.cpp   # Mode 1 测试
├── test_<name>_mode2.cpp   # Mode 2 测试
├── test_<name>_mode3.cpp   # Mode 3 测试
├── ptx/                    # 预提取 PTX
│   └── <name>.ptx
└── golden/                 # 期望值
    └── <name>.expected
```

## 常见问题

### "No tests were found"

使用正则表达式前缀：
```bash
ctest -R "^test_dummy_mode" -V  # ✓ 正确
ctest -R "test_dummy_mode" -V   # ✗ 可能不匹配
```

### "PTX file not found"

确保从项目根目录运行：
```bash
cd /workspace/project/PTX-EMU
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark dummy
```

### "Binary not found"

先编译 benchmark：
```bash
cmake --build build --target dummy
```

### CMakeLists.txt 重复条目

`sync_cmake()` 使用 word boundary 检测，不会重复添加。如果手动修改导致混乱，可重置 CMakeLists.txt 并重新生成。

## 示例

### 为 RAY benchmark 生成测试

```bash
# 编译
cmake --build build --target RAY

# 生成测试
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark RAY

# 构建运行
cmake --build build --target test_RAY_mode1 test_RAY_mode2 test_RAY_mode3
ctest -R RAY -V
```

### 为任意 CUDA 文件生成测试

```bash
# 直接指定源文件
python3 docs/skills/three-mode-testing/generate_tests.py \
    --cuda-source /path/to/my_kernel.cu \
    --test-name my_kernel

# 生成 mode2（使用预提取 PTX）
python3 docs/skills/three-mode-testing/generate_tests.py \
    --ptx tests/three_mode_testing/ptx/my_kernel.ptx \
    --test-name my_kernel \
    --mode 2
```