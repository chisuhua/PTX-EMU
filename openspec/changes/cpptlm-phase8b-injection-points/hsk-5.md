# HSK-5: PTX-EMU exe_once() 3-step injection implementation complete

> **状态**: ✅ **已发出（待 CppTLM 确认 / rebase / CI 验证）**

**发送记录**: 2026-07-17, Phase 4 (PTX-6) 完成, 伴随 commits `367fd6a5` (sm_context.cpp) + `921b4542` (tasks.md)

---

## 📤 准备发给 CppTLM 团队的完整消息

```
Subject: [HSK-5] PTX-EMU exe_once() 3-step injection complete — please rebase to commit 367fd6a5

Cc: CppTLM Team (#cpptlm-integration Slack)

CppTLM Team,

PTX-EMU Phase 4 (PTX-6) exe_once() 3-step injection 实施完成。请 CppTLM 端 rebase 并验证。

======================== 关键事实 ========================

- PTX-EMU repo:  https://github.com/chisuhua/PTX-EMU
- Commit hash (PTX-6 impl):  367fd6a5
- Commit hash (tasks.md done):  921b4542
- ABI path:  src/ptxsim/core/sm_context.cpp
- Phase 0-3 已 completed (commits 8acfd2d1 / 9e7361b9 / 463038e0 / 1217f67d / 834b6f3b / 34620770 / 9bb61db8 / 42f91988 / d993be0d / a53508c2 / 43860824 / a72425b9 / 93726f62 / a55b45b2)

======================== 3 个注入点 ========================

Step A: Scoreboard hazard check
  - Tick + has_free_entry + allocate(dest_regs, warp_id) for all dest regs
  - On failure: rollback allocated_so_far + goto warp_done
  - nullptr scoreboard_ = skip (byte-identical to pre-injection)

Step B: Latency query (priority chain)
  - 1. pipeline_provider_->get_fractional_cycles_by_type(stmt.type, PipelineId)
  - 2. tensor_core_timing_->get_latency(TcPrecision) [if TC instruction]
  - 3. ptxsim::getLatency(stmt.type).cycles [fallback]
  - Result: next_warp->set_blocked_cycles_for_active(latency)
  - Only runs in execution path (not skip path) — Oracle 2026-07-17 BUG-2 fix

Step C: Scoreboard release (GATED by warp_executed)
  - scoreboard_->release(dest_reg, warp_id) for all dest regs
  - Guard prevents releasing unallocated entries when Step A failed
  - Oracle 2026-07-17 BUG-3 critical fix (prevents scoreboard state corruption)

======================== Critical Control Flow (Oracle 2026-07-17) ========================

- goto warp_done label is BEFORE next_warp->set_scheduled(false)
  (not after, as originally designed in design.md §7.1)
- BUG-1: original goto skip_warp_execution skipped set_scheduled(false)
- BUG-2: original Step B ran in skip path, causing false blocking
- BUG-3: original Step C ran in skip path, releasing unallocated regs

All 3 bugs fixed in commit 367fd6a5.

======================== 3 Static Helper Methods (Public for testing) ========================

```cpp
class SMContext {
public:
    static bool         is_tensor_core_instruction(const StatementContext&);
    static PipelineId   map_instruction_to_pipeline(const StatementContext&);
    static TcPrecision  map_instruction_to_tc_precision(const StatementContext&);
};
```

Implemented in include/ptxsim/sm_context.h. Tests in
tests/unit/sm/test_exe_once_helpers.cpp (27 assertions, 9 test cases, all PASS).

======================== 3 File-Local Helper Functions ========================

In src/ptxsim/core/sm_context.cpp anonymous namespace:
- step_a_scoreboard_check(scoreboard, warp, stmt) -> bool
- step_b_set_blocked_cycles(pipeline, tc, warp, stmt) -> void
- step_c_release_scoreboard(scoreboard, warp, stmt) -> void

All three respect nullptr semantics: nullptr injector = byte-identical
to pre-injection behavior.

======================== 验证 ========================

PTX-EMU 端验证 (commit 367fd6a5):
- ptxsim build PASS
- unit_sm_exe_once_helpers: 27/27 PASS
- unit_barrier + integration_barrier: 13/13 PASS (0 regression)
- nullptr fallback: byte-identical to pre-change (4 injectors all nullptr)

CppTLM 端需验证 (你们):
- CppTLM MemoryBridge adapter 编译期 static_assert PipelineId/TcPrecision
- CppTLM MemoryBridge 实现 IScoreboard 接口（4 方法）
- CppTLM MemoryBridge 实现 IPipelineLatencyProvider 接口（2 方法）
- CppTLM MemoryBridge 实现 ITensorCoreTiming 接口（3 方法）
- End-to-end integration test: PTX-EMU + CppTLM 协同

======================== 下一步 ========================

1. CppTLM 端 rebase 到 PTX-EMU commit 367fd6a5
2. 实现 MemoryBridge adapter 编译期 + runtime
3. 回复确认 ✓

Please confirm receipt and alignment.
```

---

## 验证清单

- [x] 3 注入点 (Step A / B / C) 已实施
- [x] `warp_executed` 守卫保护 Step C
- [x] `goto warp_done` label 在 set_scheduled(false) 之前
- [x] 3 匿名namespace helper functions
- [x] 3 public static helper methods
- [x] nullptr fallback 字节级兼容
- [x] ptxsim build PASS
- [x] unit_sm_exe_once_helpers 27/27 PASS
- [x] barrier tests 13/13 PASS
- [x] 发送日期: **2026-07-17 已发出**

**最后更新**: 2026-07-17 (Phase 4 PTX-6 complete, 3 注入点 + 3 helpers 实施 + 27/27 tests PASS + 0 regression)
