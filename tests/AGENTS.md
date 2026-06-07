# PTX-EMU Test Suite

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
Catch2-based test suite for PTX simulation, instructions, and memory/register subsystems.

## STRUCTURE

Tests are physically organized into three subdirectories matching the three-type classification in [AGENTS.md § 测试分类规范](../../AGENTS.md#测试分类规范):

```
tests/
├── unit/                 # 类型一：直接单元测试
│   ├── barrier/          #   屏障/Wbar 数据结构
│   ├── simt/             #   SIMT 堆栈、ThreadContext
│   ├── warp/             #   WarpContext、调度器
│   ├── memory/           #   内存管理
│   ├── exec/             #   exec_mask、active_mask
│   ├── pc/               #   PC 管理
│   ├── sync/             #   同步原语
│   ├── common/           #   解析器、调度器配置等通用工具
│   └── ptx/              #   PTX 指令 (.cu)
├── integration/          # 类型二：指令序列集成测试
│   ├── barrier/          #   屏障指令执行流程
│   ├── simt/             #   SIMT 指令执行流程
│   ├── divergence/       #   分歧与 reconvergence
│   ├── exec/             #   exec 指令
│   ├── pc/               #   PC 推进
│   ├── sync/             #   同步指令
│   ├── (cfg/, register/ removed 2026-06; see KNOWN_ISSUES.md §D1.3)
├── e2e/                  # 类型三：CUDA Kernel E2E 测试
│   ├── kernel/           #   完整 .cu kernel
│   └── divergence/       #   Warp 分歧完整 kernel
├── ptx/                  # PTX 语法测试（test_all_ptx.sh）
├── instructions/         # 指令测试（保留）
├── ptxir/                # PTXIR 序列化测试
├── common/               # 通用工具源码
├── archive/              # 历史归档（不再构建）
│   └── three_mode_testing/  # 旧 E2E 实现 → 已迁移至 e2e/kernel/
└── catch_amalgamated.cpp/hpp  # Catch2 测试框架
```

> **历史变更**：2026-06 起的三类测试目录重构（commit `ab55e06`）将原本混在一起的 `.cpp` 文件按类型物理分类到三个子目录。

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| PTX syntax tests | `tests/ptx/test_all_ptx.sh` | Full PTX ISA coverage, NOT ctest |
| 类型一：单元测试 | `tests/unit/<area>/` | 直接实例化类，验证数据结构/算法 |
| 类型二：集成测试 | `tests/integration/<area>/` | 使用 `statement_factory.h` + `execute_warp_instruction()` |
| 类型三：E2E 测试 | `tests/e2e/kernel/` | 真实 CUDA kernel，`cudaLaunchKernel` 拦截 |
| Memory tests | `tests/unit/memory/test_memory_bounds.cpp` | Memory bounds checking |
| Barrier tests | `tests/unit/barrier/test_barrier_*.cpp` | Unit-level barrier data structures |

## TEST FRAMEWORK
- **Framework**: Catch2 (`catch_amalgamated.hpp`)
- **CUDA tests**: Compiled with `-keep` flag to preserve PTX
- **Architecture**: `sm_100` (virtual)

## CONVENTIONS (this dir)

- **Test files**: `test_*.cpp` or `*.cu`（放在 `unit/`、`integration/`、`e2e/` 之一）
- **ctest 命名**: 必须带类型前缀 `unit_` / `integration_` / `e2e_`（commit `ab55e06`）
- **标签格式**: `<type>;<subject>`，例如 `unit;barrier`、`integration;divergence`
- **运行指定测试**: `ctest -R <name>`（注意是带前缀的名称，如 `unit_barrier_module`）
- **PTX 语法测试**: 必须用 `./tests/ptx/test_all_ptx.sh`（**不能用 ctest**）

## COMMANDS

```bash
cd build && ctest                                # Run all tests
ctest -L mini                                    # Mini benchmark tests
ctest -L "unit;barrier"                          # 所有 barrier 相关单元测试
ctest -L integration                             # 所有指令序列集成测试
ctest -L e2e                                     # 所有 E2E 测试
ctest -R unit_barrier_module -V                  # Run specific test
./tests/ptx/test_all_ptx.sh                     # PTX syntax tests (CRITICAL)
```

## ANTI-PATTERNS

- DO NOT use `ctest` for PTX syntax tests - use `test_all_ptx.sh`
- DO NOT commit test changes without verifying all pass
- DO NOT name ctest targets without the `unit_` / `integration_` / `e2e_` prefix
- DO NOT place test files outside the type-appropriate subdirectory
