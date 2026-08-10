# 已批准提案索引

> guide-arch 批准的提案索引。guide-plan propose 从此文件读取链接。
>
> **2026-08-03 数据完整性说明**（Tier 1 reopen 决策）：
> - `god-class-refactor-sm-context` / `reduce-thread-context-includes` / `reduce-memory-test-utils-includes` 三项原 2026-07-31 标记 `## 已实施`，但当前代码状态显示任务未真正完成（sm_context.cpp 862 行 vs <250；thread_context.h 25 include vs 21 基线；memory_test_utils.h 18 include 未变）
> - 2026-08-02 debt-audit 重新审计并把它们标记为 `✅ 已批准`，但 sweep 机制会把所有同名 archive 目录的提案自动移回 `## 已实施`，导致 reopen 决策无法持久化到本文件的活跃段
> - **解决方案**：plan 阶段用 option **i**（手动输入）创建**新名** change，例如 `god-class-refactor-sm-context-phase3` / `reduce-thread-context-includes-v2` / `reduce-memory-test-utils-includes-v2`，绕过 sweep 自动归档机制

## 已批准提案

| 提案 | 优先级 | 来源 | 批准日期 | 批准人 |
|------|--------|------|----------|--------|
| [implement-ptxir-cubin-embed-extension](improvements/implement-ptxir-cubin-embed-extension.md) | P1 | ADR-0024 (Accepted 2026-08-06, commit `18ad58cb`) | 2026-08-06 | Oracle (architecture review 2nd-pass APPROVED) |

| [ptxir-driver-api-front-door](improvements/ptxir-driver-api-front-door.md) | P0 | 2026-08-10 | guide-arch |

| [multi-kernel-manifest-adr-0028](improvements/multi-kernel-manifest-adr-0028.md) | P1 | 2026-08-10 | guide-arch |

| [hal-extension-ptxemu-usrlinu-emu-taskrunner](improvements/hal-extension-ptxemu-usrlinu-emu-taskrunner.md) | P1 | 2026-08-10 | guide-arch |

## 已实施

| 提案 | 优先级 | 完成时间 |
|------|--------|----------|
| [god-class-refactor-sm-context](improvements/god-class-refactor-sm-context.md) | P2 | 2026-07-31 |
| [strengthen-pod-tests](improvements/strengthen-pod-tests.md) | P3 | 2026-07-31 |
| [split-ptx-visitor-god-class](improvements/split-ptx-visitor-god-class.md) | P2 | 2026-07-31 |
| [replace-assert-false-with-throw](improvements/replace-assert-false-with-throw.md) | P3 | 2026-07-31 |
| [refactor-warp-context](improvements/refactor-warp-context.md) | P2 | 2026-07-31 |
| [refactor-ptxir-writer](improvements/refactor-ptxir-writer.md) | P2 | 2026-07-31 |
| [reduce-thread-context-includes](improvements/reduce-thread-context-includes.md) | P3 | 2026-07-31 |
| [reduce-memory-test-utils-includes](improvements/reduce-memory-test-utils-includes.md) | P3 | 2026-07-31 |
| [merge-arithmetic-handlers](improvements/merge-arithmetic-handlers.md) | P3 | 2026-07-31 |
| [god-class-refactor-thread-context](improvements/god-class-refactor-thread-context.md) | ? | 2026-07-31 |
| [expand-e2e-divergence-coverage](improvements/expand-e2e-divergence-coverage.md) | P3 | 2026-07-31 |
| [dedupe-ptx-op-def-format](improvements/dedupe-ptx-op-def-format.md) | P3 | 2026-07-31 |
| [consolidate-sub-agents-md](improvements/consolidate-sub-agents-md.md) | P3 | 2026-07-31 |
| [complete-x-macro-dispatch](improvements/complete-x-macro-dispatch.md) | P3 | 2026-07-31 |
| [cmake-use-glob-for-sources](improvements/cmake-use-glob-for-sources.md) | P3 | 2026-07-31 |
| [add-cmake-options](improvements/add-cmake-options.md) | P3 | 2026-07-31 |
