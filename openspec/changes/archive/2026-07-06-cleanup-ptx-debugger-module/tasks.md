# Tasks: Cleanup PTXDebugger Module

> **Type**: 1-Phase 纯删除
> **Risk**: 🟢 极低（0 调用方，grep 验证）
> **Status**: ✅ **COMPLETED** via commit `fc11e99`

---

## Phase 1: 删除 PTXDebugger 模块 ✅ DONE

- [x] 1.1 二次验证 0 调用方
  ```bash
  grep -rn "PTXDebugger\|ptx_debugger\|PerfStats" src/ include/ tests/ \
    | grep -v "ptx_debugger\.\(h\|cpp\)" | grep -v "src/ptxsim/debug/"
  # 期望: 3 行（CMakeLists.txt, ptx_debug.h, thread_context.cpp 注释）
  ```
- [x] 1.2 删除头文件
- [x] 1.3 删除整个 debug/ 目录
- [x] 1.4 修改 src/CMakeLists.txt
- [x] 1.5 修改 include/ptxsim/ptx_debug.h
- [x] 1.6 修改 src/ptxsim/core/thread_context.cpp
- [x] 1.7 验证编译通过 + ctest 100% PASS
- [x] 1.8 Commit Fix #1 (`fc11e99`)

---

## Phase 2: 文档同步（DEFERRED to separate change）

文档同步（AGENTS.md + audit + roadmap）已记录在 debt-audit-2026-07-02.md 中需要更新。
实际同步留待 future `docs-update-after-debt-cleanup` change（本次未实施以保持 change 范围聚焦）。

---

## 风险缓解验证

| 风险 | 缓解任务 | 验证 |
|------|---------|------|
| R1: 隐藏依赖 | 1.1 + 1.7 | grep + ctest 100% |