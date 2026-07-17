# HSK-4: PTX-EMU 3 纯虚接口头文件首发 commit hash

> **状态**: ✅ **已发出（待 CppTLM 确认 / Adapter 实现 / 枚举 static_assert 验证）**

**发送记录**: 2026-07-17, commits: `8acfd2d1` (IScoreboard) / `9e7361b9` (IPipelineLatencyProvider) / `463038e0` (ITensorCoreTiming), 伴随 Phase 1 完成
> **回传目标**: CppTLM Team
> **承诺时间**: Phase 1 完成后立即
> **形式**: 3 个 git commit hash + 接口签名快照
> **PTX-EMU 侧**: 3 接口已实现 + Phase 1 tasks.md 已更新

---

## 📤 准备发给 CppTLM 团队的完整消息

```
Subject: [HSK-4] PTX-EMU 3 injection interfaces ready — IScoreboard / IPipelineLatencyProvider / ITensorCoreTiming

CppTLM Team,

PTX-EMU 已完成 Phase 1 三个纯虚接口头文件首发。请确认接口签名与 CppTLM 端 RFC-P1-001~004 对齐。

======================== 关键事实 ========================

- PTX-EMU repo:  https://github.com/chisuhua/PTX-EMU
- Commit hash IScoreboard:              `8acfd2d1` (Phase 8.B PTX-1)
- Commit hash IPipelineLatencyProvider:  `9e7361b9` (Phase 8.B PTX-2)
- Commit hash ITensorCoreTiming:         `463038e0` (Phase 8.B PTX-3)
- ABI path prefix:                       include/ptxsim/
- Zero external deps:                    仅 <cstdint> / <cstdint>+<string>
- Phase 1 test:                          21/21 assertions PASS (commit `1217f67d`)

======================== 接口签名（3 个纯虚接口）========================

=== IScoreboard (include/ptxsim/scoreboard_interface.h) ===

class IScoreboard {
public:
    virtual ~IScoreboard() = default;
    virtual bool has_free_entry() const = 0;
    virtual bool allocate(uint32_t reg_id, uint32_t warp_id) = 0;
    virtual bool release(uint32_t reg_id, uint32_t warp_id) = 0;
    virtual void tick() = 0;
};

=== IPipelineLatencyProvider (include/ptxsim/pipeline_interface.h) ===

enum class PipelineId : uint32_t {
    P0_INT_FP32 = 0, V_SIMD = 1, P1_FP64 = 2,
    P2_SFU = 3, P3_LSU = 4, P4_TC = 5
};
// ⬆️ MUST match CppTLM tlm::PipelineId 0-5 (RFC-P1-003 §3.1)

class IPipelineLatencyProvider {
public:
    virtual ~IPipelineLatencyProvider() = default;
    virtual double get_fractional_cycles(
        const std::string& instruction, PipelineId pipe_id) const = 0;
    virtual double get_fractional_cycles_by_type(
        int statement_type, PipelineId pipe_id) const = 0;
};

=== ITensorCoreTiming (include/ptxsim/tensor_core_interface.h) ===

enum class TcPrecision : uint32_t {
    FP4 = 0, FP6 = 1, FP8 = 2, FP16 = 3, BF16 = 4, TF32 = 5
};
// ⬆️ MUST match CppTLM tlm::TcPrecision 0-5 (RFC-P1-003 §3.2)

class ITensorCoreTiming {
public:
    virtual ~ITensorCoreTiming() = default;
    virtual uint32_t get_latency(TcPrecision prec) const = 0;
    virtual uint32_t get_throughput_cycles(TcPrecision prec) const = 0;
    virtual uint32_t get_latency_mnk(
        TcPrecision prec, uint32_t M, uint32_t N, uint32_t K) const {
        return get_latency(prec);  // default degeneracy
    }
};

======================== 枚举一致性 ========================

- PipelineId 0-5: 双端 RFC-P1-003 §3.1 已锁定 ✅
- TcPrecision 0-5: 双端 RFC-P1-003 §3.2 已锁定 ✅
- CppTLM Adapter 需 static_assert 验证编译期一致性

======================== 下一步 ========================

1. CppTLM 端 rebase 到 PTX-EMU main (commit `463038e0` 或 later)
2. 实现 IScoreboard / IPipelineLatencyProvider / ITensorCoreTiming 的 CppTLM 端 adapter
3. 验证 21 个 ABI 测试在 CppTLM 侧通过
4. 回复确认 ✓

Please confirm receipt and alignment.
```

---

## 验证清单

- [x] 3 个接口头文件已创建 (`include/ptxsim/`)
- [x] 每个头文件独立 commit（分 Phase 可回退）
- [x] 21/21 ABI 测试断言 PASS
- [x] PipelineId 0-5 枚举值与 CppTLM 一致
- [x] TcPrecision 0-5 枚举值与 CppTLM 一致
- [x] `ptxsim` target 编译通过
- [x] 发送日期: **2026-07-17 已发出**（3 commit hash 已锁定：`8acfd2d1` / `9e7361b9` / `463038e0`）

**最后更新**: 2026-07-17（已发出 — 3 接口已实现 + 21 测试 PASS + Phase 2 SMContext 扩展进行中；待 CppTLM 确认）