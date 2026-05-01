# PTX to Statements 直接转换实现文档

> **版本**: v1.0
> **日期**: 2026-04-30
> **状态**: 待实现
> **作者**: PTX-EMU Architecture Team

---

## 1. 概述

### 1.1 背景

当前 `load_ptx_file()` 函数仅返回 PTX 文本字符串 (`std::string`)，用户在使用 Mode 3 测试时需要：
1. 调用 `load_ptx_file()` 获取 PTX 字符串
2. 手动调用 ANTLR 解析流程
3. 从 `PtxContext` 中提取 `kernelStatements`

这个流程繁琐且容易出错。本文档描述如何实现 `load_ptx_statements()` 函数，直接从 PTX 文件返回 `std::vector<StatementContext>`。

### 1.2 目标

| 目标 | 描述 |
|------|------|
| **简化调用** | 一行代码完成文件 → StatementContext 转换 |
| **类型安全** | 直接返回强类型 `std::vector<StatementContext>` |
| **灵活配置** | 支持 kernel 选择、CFG 处理控制 |
| **向后兼容** | 保留原 `load_ptx_file()` 函数 |

### 1.3 非目标

- 不修改现有 PTX 解析逻辑
- 不改变 `StatementContext` 结构
- 不添加新的测试框架

---

## 2. 设计方案

### 2.1 函数接口

#### 2.1.1 主要函数

```cpp
/**
 * @brief 从 PTX 文件加载并解析为 StatementContext 序列
 *
 * @param path PTX 文件路径
 * @param kernel_name 目标 kernel 名称（可选，默认取第一个 .entry kernel）
 * @param apply_cfg 是否应用 CFGBuilder 填充 reconvergence_pc（默认 false，对应 Mode 3a）
 * @return std::vector<StatementContext> 解析后的指令序列
 *
 * @throws std::runtime_error 文件不存在、解析失败、kernel 未找到
 *
 * @example
 * // Mode 3a: 原始解析结果（无 CFG）
 * auto stmts = load_ptx_statements("tests/ptx/simple.ptx");
 *
 * // Mode 3b: 应用 CFG 处理
 * auto stmts = load_ptx_statements("tests/ptx/simple.ptx", "", true);
 *
 * // 指定 kernel 名称
 * auto stmts = load_ptx_statements("tests/ptx/multi.ptx", "my_kernel");
 */
inline std::vector<StatementContext> load_ptx_statements(
    const std::string& path,
    const std::string& kernel_name = "",
    bool apply_cfg = false);
```

#### 2.1.2 辅助函数

```cpp
/**
 * @brief 从 PTX 字符串解析为 StatementContext 序列（用于 Mode 2 → Mode 3 转换）
 *
 * @param ptx_code PTX 源代码字符串
 * @param kernel_name 目标 kernel 名称（可选）
 * @param apply_cfg 是否应用 CFG（默认 false）
 * @return std::vector<StatementContext>
 *
 * @example
 * // 从 Mode 2 字符串转换
 * std::string ptx = load_ptx_file("tests/ptx/simple.ptx");
 * auto stmts = parse_ptx_to_statements(ptx);
 */
inline std::vector<StatementContext> parse_ptx_to_statements(
    const std::string& ptx_code,
    const std::string& kernel_name = "",
    bool apply_cfg = false);
```

#### 2.1.3 CFG 处理函数

```cpp
/**
 * @brief 应用 CFGBuilder 处理 StatementContext 序列
 *
 * @param statements 指令序列（会被修改，reconvergence_pc 会被填充）
 * @param label2pc [输出] Label 到 PC 的映射
 *
 * @example
 * std::map<std::string, int> label2pc;
 * apply_cfg_builder(statements, label2pc);
 */
inline void apply_cfg_builder(
    std::vector<StatementContext>& statements,
    std::map<std::string, int>& label2pc);
```

### 2.2 放置位置

**文件**: `tests/three_mode_testing/test_helpers.hpp`

**理由**:
1. 该文件已包含 Mode 3 测试相关的辅助函数（`make_mov()`, `make_bar_sync()` 等）
2. 与 `load_ptx_file()` 在同一位置，便于查找
3. 符合 Three-Mode Testing Framework 的设计意图

### 2.3 新增 Section

在 `test_helpers.hpp` 中添加新的 section：

```cpp
// ============================================================================
// PTX Parsing Helpers (Mode 2 → Mode 3 转换)
// ============================================================================
```

---

## 3. 实现细节

### 3.1 load_ptx_statements 实现

```cpp
inline std::vector<StatementContext> load_ptx_statements(
    const std::string& path,
    const std::string& kernel_name,
    bool apply_cfg) {

    // Step 1: 加载 PTX 文件
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("PTX file not found: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string ptx_code = ss.str();

    // Step 2: 解析并转换
    return parse_ptx_to_statements(ptx_code, kernel_name, apply_cfg);
}
```

### 3.2 parse_ptx_to_statements 实现

```cpp
inline std::vector<StatementContext> parse_ptx_to_statements(
    const std::string& ptx_code,
    const std::string& kernel_name,
    bool apply_cfg) {

    // Step 1: ANTLR 词法分析
    antlr4::ANTLRInputStream input(ptx_code);
    ptxparser::ptxLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    tokens.fill();

    // Step 2: ANTLR 语法分析
    ptxparser::ptxParser parser(&tokens);

    // Step 3: 创建 PtxListener 并解析
    PtxListener listener;
    antlr4::tree::ParseTreeWalker::defaultWalker().walk(&listener, parser.ast());

    // Step 4: 检查解析错误
    if (parser.getNumberOfSyntaxErrors() > 0) {
        throw std::runtime_error(
            "PTX parsing failed with " +
            std::to_string(parser.getNumberOfSyntaxErrors()) + " errors");
    }

    // Step 5: 查找目标 kernel
    KernelContext* target_kernel = nullptr;

    if (kernel_name.empty()) {
        // 默认取第一个 .entry kernel
        for (auto& kernel : listener.ptxContext.ptxKernels) {
            if (kernel.ifEntryKernel) {
                target_kernel = &kernel;
                break;
            }
        }
        // Fallback: 取第一个 kernel
        if (!target_kernel && !listener.ptxContext.ptxKernels.empty()) {
            target_kernel = &listener.ptxContext.ptxKernels[0];
        }
    } else {
        // 按名称查找
        for (auto& kernel : listener.ptxContext.ptxKernels) {
            if (kernel.kernelName == kernel_name) {
                target_kernel = &kernel;
                break;
            }
        }
    }

    if (!target_kernel) {
        throw std::runtime_error(
            "Kernel not found: " +
            (kernel_name.empty() ? "(first kernel)" : kernel_name));
    }

    // Step 6: 提取 statements（拷贝，避免悬垂引用）
    std::vector<StatementContext> statements = target_kernel->kernelStatements;

    // Step 7: 可选 CFG 处理
    if (apply_cfg) {
        std::map<std::string, int> label2pc;
        apply_cfg_builder(statements, label2pc);
    }

    return statements;
}
```

### 3.3 apply_cfg_builder 实现

```cpp
inline void apply_cfg_builder(
    std::vector<StatementContext>& statements,
    std::map<std::string, int>& label2pc) {

    // Step 1: 构建 label2pc 映射
    for (size_t i = 0; i < statements.size(); ++i) {
        auto& stmt = statements[i];
        if (stmt.type == S_DOLLAR) {
            auto* label = static_cast<StatementContext::DOLLAR*>(stmt.statement);
            label2pc[label->dollorName] = static_cast<int>(i);
        }
    }

    // Step 2: 构建 CFG
    ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(statements, label2pc);

    // Step 3: 计算后支配关系
    ptx::cfg::PostDominatorMap postDoms =
        ptx::cfg::CFGBuilder::computePostDominators(cfg);

    // Step 4: 填充分支指令的 reconvergence_pc
    for (size_t i = 0; i < statements.size(); ++i) {
        auto& stmt = statements[i];

        if (stmt.type == S_BRA) {
            auto* bra = static_cast<StatementContext::BRA*>(stmt.statement);
            auto it = postDoms.find(static_cast<int>(i));
            if (it != postDoms.end() && it->second >= 0) {
                bra->braReconvergencePc = it->second;
            } else {
                bra->braReconvergencePc = static_cast<int>(i + 1);
            }
        }
    }
}
```

---

## 4. 错误处理

### 4.1 异常类型

| 场景 | 异常信息 |
|------|----------|
| 文件不存在 | `PTX file not found: /path/to/file.ptx` |
| PTX 解析失败 | `PTX parsing failed with N errors` |
| Kernel 未找到 | `Kernel not found: kernel_name` 或 `Kernel not found: (first kernel)` |

### 4.2 错误处理策略

```cpp
// 调用示例：使用 try-catch
TEST_CASE("Load PTX with error handling") {
    try {
        auto stmts = load_ptx_statements("nonexistent.ptx");
        FAIL("Should have thrown");
    } catch (const std::runtime_error& e) {
        CHECK(string(e.what()).find("not found") != string::npos);
    }
}
```

---

## 5. 使用示例

### 5.1 Mode 3a: 原始解析（无 CFG）

```cpp
// tests/three_mode_testing/test_mode3a.cpp

#include "test_helpers.hpp"

TEST_CASE("Mode3a: simple branch test") {
    // 直接加载，无需手动解析
    auto stmts = load_ptx_statements(
        "tests/three_mode_testing/ptx/simple_branch.ptx");

    REQUIRE(stmts.size() > 0);

    // 验证 reconvergence_pc 为默认值（-1）
    for (const auto& stmt : stmts) {
        if (stmt.type == S_BRA) {
            const auto& bra = std::get<BranchInstr>(stmt.data);
            CHECK(bra.reconvergence_pc == -1);
        }
    }

    // 执行指令序列
    run_statement_sequence(stmts);
}
```

### 5.2 Mode 3b: 应用 CFG

```cpp
// tests/three_mode_testing/test_mode3b.cpp

#include "test_helpers.hpp"

TEST_CASE("Mode3b: nested branch with CFG") {
    // apply_cfg = true 启用 CFG 处理
    auto stmts = load_ptx_statements(
        "tests/three_mode_testing/ptx/nested_branch.ptx",
        "",  // 使用默认 kernel
        true // 应用 CFG
    );

    // 验证 reconvergence_pc 已填充
    for (const auto& stmt : stmts) {
        if (stmt.type == S_BRA) {
            const auto& bra = std::get<BranchInstr>(stmt.data);
            CHECK(bra.reconvergence_pc >= 0);
        }
    }

    // 执行
    run_statement_sequence(stmts);
}
```

### 5.3 指定 Kernel 名称

```cpp
// 多 kernel PTX 文件
TEST_CASE("Multi-kernel PTX") {
    auto stmts_add = load_ptx_statements(
        "tests/three_mode_testing/ptx/multi_kernel.ptx",
        "add_kernel");  // 指定 kernel

    auto stmts_mul = load_ptx_statements(
        "tests/three_mode_testing/ptx/multi_kernel.ptx",
        "mul_kernel");

    CHECK(stmts_add.size() > 0);
    CHECK(stmts_mul.size() > 0);
}
```

### 5.4 Mode 2 → Mode 3 转换

```cpp
// 从现有 Mode 2 代码迁移
TEST_CASE("Mode 2 to Mode 3 migration") {
    // 原有 Mode 2 代码（仍可使用）
    std::string ptx = load_ptx_file("tests/ptx/simple.ptx");
    INFO("Loaded PTX: " << ptx.size() << " bytes");

    // 新方案：直接转换
    auto stmts = parse_ptx_to_statements(ptx);

    REQUIRE(stmts.size() > 0);
    run_statement_sequence(stmts);
}
```

### 5.5 CFG 处理后验证

```cpp
// 验证 CFG 处理结果
TEST_CASE("CFG builder verification") {
    auto stmts = load_ptx_statements(
        "tests/three_mode_testing/ptx/divergence_test.ptx",
        "",
        true);

    // 查找所有分支指令及其 reconvergence_pc
    for (size_t i = 0; i < stmts.size(); ++i) {
        const auto& stmt = stmts[i];
        if (stmt.type == S_BRA) {
            const auto& bra = std::get<BranchInstr>(stmt.data);
            CAPTURE(i);
            CAPTURE(bra.braTarget);
            CAPTURE(bra.reconvergence_pc);
            CHECK(bra.reconvergence_pc >= 0);
        }
    }
}
```

---

## 6. 测试策略

### 6.1 单元测试

```cpp
// tests/three_mode_testing/test_ptx_load.cpp

#include <catch2/catch.hpp>
#include "test_helpers.hpp"

TEST_CASE("load_ptx_statements: file not found") {
    CHECK_THROWS_AS(load_ptx_statements("/nonexistent/path.ptx"),
                    std::runtime_error);
}

TEST_CASE("load_ptx_statements: kernel not found") {
    auto stmts = load_ptx_statements(
        "tests/three_mode_testing/ptx/simple.ptx",
        "nonexistent_kernel");

    // 应该抛出异常
    CHECK_THROWS_AS(load_ptx_statements(
        "tests/three_mode_testing/ptx/simple.ptx",
        "nonexistent_kernel"), std::runtime_error);
}

TEST_CASE("load_ptx_statements: empty file") {
    CHECK_THROWS_AS(load_ptx_statements(
        "tests/three_mode_testing/ptx/empty.ptx"),
        std::runtime_error);
}

TEST_CASE("load_ptx_statements: basic parsing") {
    auto stmts = load_ptx_statements(
        "tests/three_mode_testing/ptx/simple.ptx");

    CHECK(stmts.size() > 0);

    // 第一个 statement 应该是 .reg 声明或 label
    CHECK(stmts[0].type != S_UNKNOWN);
}

TEST_CASE("load_ptx_statements: with/without CFG") {
    auto stmts_no_cfg = load_ptx_statements(
        "tests/three_mode_testing/ptx/branch.ptx", "", false);
    auto stmts_with_cfg = load_ptx_statements(
        "tests/three_mode_testing/ptx/branch.ptx", "", true);

    // 语句数量应该相同
    CHECK(stmts_no_cfg.size() == stmts_with_cfg.size());

    // CFG 版本应该有 reconvergence_pc
    for (const auto& stmt : stmts_with_cfg) {
        if (stmt.type == S_BRA) {
            const auto& bra = std::get<BranchInstr>(stmt.data);
            CHECK(bra.reconvergence_pc >= 0);
        }
    }
}
```

### 6.2 集成测试

```cpp
TEST_CASE("load_ptx_statements: end-to-end execution") {
    auto stmts = load_ptx_statements(
        "tests/three_mode_testing/ptx/barrier_test.ptx",
        "",
        true);

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads);

    // 执行指令序列
    for (size_t i = 0; i < stmts.size(); ++i) {
        warp.execute_warp_instruction(stmts[i], i);
    }

    // 验证执行结果
    CHECK(count_active_lanes(warp) == 32);
}
```

---

## 7. 向后兼容性

### 7.1 保留的函数

```cpp
// 保持不变，用于 Mode 2 测试
inline std::string load_ptx_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
```

### 7.2 迁移路径

| 现有用法 | 迁移后用法 |
|----------|------------|
| `std::string ptx = load_ptx_file(path);` | `std::string ptx = load_ptx_file(path);` (仍可用) |
| 手动 ANTLR 解析 | `auto stmts = load_ptx_statements(path);` |
| Mode 2 → Mode 3 手动转换 | `auto stmts = parse_ptx_to_statements(ptx);` |

### 7.3 无破坏性变更

- 原 `load_ptx_file()` 函数签名不变
- 现有测试代码无需修改
- 新函数默认参数确保最小侵入

---

## 8. 头文件依赖

### 8.1 现有头文件

```cpp
// test_helpers.hpp 已有
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <map>
```

### 8.2 新增头文件

```cpp
// 需要添加的 ANTLR 头文件
#include "antlr4-runtime.h"
#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptxParserBaseListener.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/kernel_context.h"
#include "ptx_parser/cfg_builder.h"
```

### 8.3 命名空间

```cpp
using namespace antlr4;
using namespace antlr4::tree;
using namespace ptxparser;
```

---

## 9. 实施计划

### 9.1 任务分解

| # | 任务 | 估计时间 | 优先级 |
|---|------|----------|--------|
| 1 | 添加函数声明和文档注释 | 15 min | P0 |
| 2 | 实现 `load_ptx_statements()` | 20 min | P0 |
| 3 | 实现 `parse_ptx_to_statements()` | 30 min | P0 |
| 4 | 实现 `apply_cfg_builder()` | 20 min | P0 |
| 5 | 添加单元测试 | 30 min | P1 |
| 6 | 集成测试验证 | 30 min | P1 |
| 7 | 更新文档 | 15 min | P2 |

### 9.2 实施步骤

**Step 1**: 修改 `tests/three_mode_testing/test_helpers.hpp`
```bash
# 在 test_helpers.hpp 末尾添加新 section
# 添加位置：第 413 行之后
```

**Step 2**: 验证编译
```bash
cd build && cmake --build . --target test_three_mode_testing
```

**Step 3**: 运行测试
```bash
cd build && ctest -R "test_ptx_load" -V
```

---

## 10. 附录

### 10.1 完整代码模板

```cpp
// ============================================================================
// PTX Parsing Helpers (Mode 2 → Mode 3 转换)
// ============================================================================

/**
 * @brief 从 PTX 文件加载并解析为 StatementContext 序列
 * 
 * @param path PTX 文件路径
 * @param kernel_name 目标 kernel 名称（可选，默认取第一个 .entry kernel）
 * @param apply_cfg 是否应用 CFGBuilder（默认 false，对应 Mode 3a）
 * @return std::vector<StatementContext> 解析后的指令序列
 * @throws std::runtime_error 文件不存在、解析失败、kernel 未找到
 */
inline std::vector<StatementContext> load_ptx_statements(
    const std::string& path,
    const std::string& kernel_name = "",
    bool apply_cfg = false);

/**
 * @brief 从 PTX 字符串解析为 StatementContext 序列
 * 
 * @param ptx_code PTX 源代码字符串
 * @param kernel_name 目标 kernel 名称（可选）
 * @param apply_cfg 是否应用 CFG（默认 false）
 * @return std::vector<StatementContext>
 */
inline std::vector<StatementContext> parse_ptx_to_statements(
    const std::string& ptx_code,
    const std::string& kernel_name = "",
    bool apply_cfg = false);

/**
 * @brief 应用 CFGBuilder 处理 StatementContext 序列
 * 
 * @param statements 指令序列（会被修改）
 * @param label2pc [输出] Label 到 PC 的映射
 */
inline void apply_cfg_builder(
    std::vector<StatementContext>& statements,
    std::map<std::string, int>& label2pc);

// ============================================================================
// Implementation
// ============================================================================

inline std::vector<StatementContext> load_ptx_statements(
    const std::string& path,
    const std::string& kernel_name,
    bool apply_cfg) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("PTX file not found: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_ptx_to_statements(ss.str(), kernel_name, apply_cfg);
}

inline std::vector<StatementContext> parse_ptx_to_statements(
    const std::string& ptx_code,
    const std::string& kernel_name,
    bool apply_cfg) {
    // [见 3.2 节实现]
}

inline void apply_cfg_builder(
    std::vector<StatementContext>& statements,
    std::map<std::string, int>& label2pc) {
    // [见 3.3 节实现]
}
```

### 10.2 相关文档

| 文档 | 路径 |
|------|------|
| Three-Mode Testing Guide | `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md` |
| CFG Integration Guide | `docs/developer-guide/CFG-INTEGRATION-GUIDE.md` |
| Test Helpers | `tests/three_mode_testing/test_helpers.hpp` |

---

**最后更新**: 2026-04-30
**维护者**: PTX-EMU Architecture Team
