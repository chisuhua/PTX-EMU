# ptx-visitor-dup-include-cleanup — Spec

## Purpose

`#include "ptx_visitor_warp.cpp"` 在 ptx_visitor.cpp:917 和 :922 **重复出现两次**（验证：`grep -c '#include "ptx_visitor_warp.cpp"' src/ptx_parser/ptx_visitor.cpp` = 2）。本 spec 定义删除 :922 重复 include 的需求。

## ADDED Requirements

### Requirement: 删除重复 include

The system MUST delete the duplicate `#include "ptx_visitor_warp.cpp"` at ptx_visitor.cpp:922.

#### Scenario: 重复 include 计数为 1

- **WHEN** 删除后运行 `grep -c '#include "ptx_visitor_warp.cpp"' src/ptx_parser/ptx_visitor.cpp`
- **THEN** 命中 1（仅保留 :917）

#### Scenario: :917 的 include 保留

- **GIVEN** 原文件 :917 处的 `#include "ptx_visitor_warp.cpp"`
- **WHEN** 删除 :922 重复
- **THEN** :917 仍存在
- **AND** `ptx_visitor_warp.cpp` 内容被 include 一次

### Requirement: 行为不变性

The system MUST maintain identical compile-time and runtime behavior.

#### Scenario: 无 ODR 违规

- **WHEN** 编译运行 `cmake --build build`
- **THEN** 无 ODR warning
- **AND** 链接期无重定义错误

#### Scenario: tcgen05 warp 行为不变

- **GIVEN** `ptx_visitor_warp.cpp` 包含 tcgen05 warp 相关 visitor override
- **WHEN** 删除重复 include 后解析含 tcgen05 warp 指令的 PTX
- **THEN** IR 输出与删除前字节级一致
- **AND** 所有 warp 测试通过

### Requirement: 验证

The system MUST pass all verification.

#### Scenario: 全量测试通过

- **WHEN** 运行 `./tests/ptx/test_all_ptx.sh && ctest --output-on-failure`
- **THEN** 所有测试通过
- **AND** 零回归

## 关联

- `src/ptx_parser/ptx_visitor.cpp:917, :922` — 重复 include 位置
- `src/ptx_parser/ptx_visitor_warp.cpp` — 被 include 的文件
- `improvements/split-ptx-visitor-god-class.md` — 完整提案