# T2-6 — CVT 策略模式重构 (Tasks)

> **总工期**: 3.5 天 | **依赖**: T1-1..T1-5 ✅, T2-2/4-Step-1/5/7 ✅, P1-4.1 修复（Step 5 前）

> **TDD 三阶段**: 严格遵循 AGENTS.md §TDD 流程（写测试→验失败→实现→验通过→commit）。

## Sub-task 1: TDD 驱动提取 helpers（0.5 天）— **当前立即执行**

**Files:**
- New: `tests/unit/ptx/test_cvt_helpers.cpp`（helpers 单元测试，约 30 TEST_CASE）
- New: `include/ptxsim/instructions/cvt/cvt_helpers.h`（helper 声明）
- New: `src/ptxsim/instructions/cvt/cvt_helpers.cpp`（helper 实现）
- New: `include/ptxsim/instructions/cvt/`（新子目录）
- New: `src/ptxsim/instructions/cvt/`（新子目录）
- Modify: `src/ptxsim/instructions/arithmetic_conversion.cpp:1-139`（删除 4 个 inline helper）
- Modify: `src/CMakeLists.txt:105`（注册新 cpp 文件）

- [ ] **Step 1: 读取当前 helpers 完整实现**

```bash
cd /workspace/project/PTX-EMU
sed -n '1,140p' src/ptxsim/instructions/arithmetic_conversion.cpp
```

预期看到 4 个 inline helper：`round_half_to_even` (11-22) / `half_to_float` (25-58) / `should_saturate_uint32` (67-69) / `float_to_half` (72-139)。

- [ ] **Step 2: 写 helper 单元测试（Red 阶段）**

创建 `tests/unit/ptx/test_cvt_helpers.cpp`：

```cpp
// test_cvt_helpers.cpp
// =============================================================================
// Unit test: 验证 CVT 4 个 helper 在抽离前后的行为一致性
// TDD 目的：抽离前先写测试，锁住当前行为；抽离后验证零行为变更
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include <cmath>
#include <limits>

using ptxsim::cvt_helpers::round_half_to_even;
using ptxsim::cvt_helpers::half_to_float;
using ptxsim::cvt_helpers::float_to_half;
using ptxsim::cvt_helpers::should_saturate_uint32;

TEST_CASE("round_half_to_even basic cases", "[cvt][helpers][rounding]") {
    REQUIRE(round_half_to_even(0.0f) == 0.0f);
    REQUIRE(round_half_to_even(0.5f) == 0.0f);  // banker's rounding
    REQUIRE(round_half_to_even(1.5f) == 2.0f);
    REQUIRE(round_half_to_even(2.5f) == 2.0f);
    REQUIRE(round_half_to_even(-0.5f) == -0.0f);
    REQUIRE(round_half_to_even(-1.5f) == -2.0f);
}

TEST_CASE("round_half_to_even edge cases", "[cvt][helpers][rounding]") {
    REQUIRE(std::isnan(round_half_to_even(std::numeric_limits<float>::quiet_NaN())));
    REQUIRE(round_half_to_even(std::numeric_limits<float>::infinity()) == std::numeric_limits<float>::infinity());
}

TEST_CASE("half_to_float zero/inf/nan/denormal", "[cvt][helpers][half]") {
    REQUIRE(half_to_float(0x0000) == 0.0f);
    REQUIRE(half_to_float(0x8000) == -0.0f);
    REQUIRE(half_to_float(0x7C00) == std::numeric_limits<float>::infinity());
    REQUIRE(half_to_float(0xFC00) == -std::numeric_limits<float>::infinity());
    REQUIRE(std::isnan(half_to_float(0x7E00)));
    // denormal: smallest positive half = 2^-24
    float denorm = half_to_float(0x0001);
    REQUIRE(denorm > 0.0f);
    REQUIRE(denorm < 1e-7f);
}

TEST_CASE("float_to_half zero/inf/nan/denormal", "[cvt][helpers][half]") {
    REQUIRE(float_to_half(0.0f) == 0x0000);
    REQUIRE(float_to_half(-0.0f) == 0x8000);
    REQUIRE(float_to_half(std::numeric_limits<float>::infinity()) == 0x7C00);
    REQUIRE(float_to_half(-std::numeric_limits<float>::infinity()) == 0xFC00);
    REQUIRE(float_to_half(std::numeric_limits<float>::quiet_NaN()) == 0x7E00);
}

TEST_CASE("should_saturate_uint32 boundaries", "[cvt][helpers][sat]") {
    REQUIRE_FALSE(should_saturate_uint32(0.0f, 4294967295.0f));
    REQUIRE_FALSE(should_saturate_uint32(100.5f, 4294967295.0f));
    REQUIRE(should_saturate_uint32(4294967295.0f, 4294967295.0f));
    REQUIRE(should_saturate_uint32(1e10f, 4294967295.0f));
    REQUIRE(should_saturate_uint32(std::numeric_limits<float>::infinity(), 4294967295.0f));
}
```

- [ ] **Step 3: 编译验证测试失败（Red 阶段，缺 cvt_helpers.h）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target test_cvt_helpers 2>&1 | tail -10
```

预期：编译失败（`cvt_helpers.h` 不存在）。这是 TDD Red 阶段的正确表现。

- [ ] **Step 4: 提交测试脚手架（Red 阶段）**

```bash
cd /workspace/project/PTX-EMU
git add tests/unit/ptx/test_cvt_helpers.cpp tests/unit/ptx/CMakeLists.txt
git commit -m "test(cvt): add 5 helper unit tests for T2-6 (TDD Red phase)"
```

- [ ] **Step 5: 创建 cvt_helpers.h（声明）**

创建 `include/ptxsim/instructions/cvt/cvt_helpers.h`：

```cpp
// cvt_helpers.h
// =============================================================================
// CVT 指令共享 helpers（从 arithmetic_conversion.cpp 抽离）
// 抽离目的：消除 1288 行单文件 god method + 与 half_utils.h 复用
// =============================================================================

#ifndef PTXSIM_INSTRUCTIONS_CVT_CVT_HELPERS_H
#define PTXSIM_INSTRUCTIONS_CVT_CVT_HELPERS_H

#include <cstdint>

namespace ptxsim {
namespace cvt_helpers {

// 银行家舍入（用于 .rni 修饰符）
float round_half_to_even(float x);

// f16 → f32 位解析（与 half_utils.h::f16_to_f32 等价）
float half_to_float(uint16_t h);

// f32 → f16 位解析（与 half_utils.h::f32_to_f16 等价）
uint16_t float_to_half(float f);

// 饱和边界检测（用于 f32→u32 转换的 .sat 修饰符）
bool should_saturate_uint32(float temp, float sat_high);

}  // namespace cvt_helpers
}  // namespace ptxsim

#endif  // PTXSIM_INSTRUCTIONS_CVT_CVT_HELPERS_H
```

- [ ] **Step 6: 创建 cvt_helpers.cpp（实现）— **第一版用本地实现，验证零行为变更**

创建 `src/ptxsim/instructions/cvt/cvt_helpers.cpp`（内容为原 arithmetic_conversion.cpp line 11-139 的实现，仅 namespace 调整）：

```cpp
// cvt_helpers.cpp
// =============================================================================
// 实现说明：本文件第一版直接复制 arithmetic_conversion.cpp 的 4 个 helper
// （line 11-139），namespace 改为 ptxsim::cvt_helpers。
// Step 7-8 将复用 half_utils.h 替换重复的 half_to_float/float_to_half。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include <cmath>
#include <cstring>

namespace ptxsim {
namespace cvt_helpers {

float round_half_to_even(float x) {
    // ... 从 arithmetic_conversion.cpp:11-22 复制 ...
}

float half_to_float(uint16_t h) {
    // ... 从 arithmetic_conversion.cpp:25-58 复制 ...
}

bool should_saturate_uint32(float temp, float sat_high) {
    // ... 从 arithmetic_conversion.cpp:67-69 复制 ...
}

uint16_t float_to_half(float f) {
    // ... 从 arithmetic_conversion.cpp:72-139 复制 ...
}

}  // namespace cvt_helpers
}  // namespace ptxsim
```

- [ ] **Step 7: 编译验证（Red→Green）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target test_cvt_helpers 2>&1 | tee /tmp/cvt_helpers_build.log
grep "error:" /tmp/cvt_helpers_build.log | head -10
```

预期：编译成功（namespace 调整后，签名匹配测试）。

- [ ] **Step 8: 跑测试（Green 阶段）**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R "cvt_helpers" -V 2>&1 | tail -30
```

预期：5 个 TEST_CASE 全部通过。

- [ ] **Step 9: 删除 arithmetic_conversion.cpp 内的 4 个 inline helper + 注册新文件**

修改 `src/ptxsim/instructions/arithmetic_conversion.cpp`：
- 删除 line 11-139（4 个 inline helper + should_saturate_uint32）
- 在文件顶部添加 `#include "ptxsim/instructions/cvt/cvt_helpers.h"`
- 在每个使用点改为 `ptxsim::cvt_helpers::half_to_float(...)` / `float_to_half(...)` / `round_half_to_even(...)` / `should_saturate_uint32(...)`

修改 `src/CMakeLists.txt:105` 区域：
```cmake
# 原行
# ptxsim/instructions/arithmetic_conversion.cpp
# 改为
ptxsim/instructions/arithmetic_conversion.cpp
ptxsim/instructions/cvt/cvt_helpers.cpp
```

- [ ] **Step 10: 编译 + 跑全量 CVT 测试（验证零行为变更）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build 2>&1 | tee /tmp/build_full.log
grep "error:" /tmp/build_full.log | wc -l  # 应 0
cd build && ctest -L "ptx;cvt" -V 2>&1 | tail -30
```

预期：0 编译错误；现有 `integration_cvt` 等测试全部通过（**零行为变更验证**）。

- [ ] **Step 11: 提交（Sub-task 1 完成）**

```bash
cd /workspace/project/PTX-EMU
git add include/ptxsim/instructions/cvt/ src/ptxsim/instructions/cvt/ src/CMakeLists.txt src/ptxsim/instructions/arithmetic_conversion.cpp
git commit -m "refactor(cvt): extract 4 helpers to cvt_helpers (T2-6 Step 1)"
```

---

## Sub-task 2: 复用 half_utils.h（0.3 天）

**Files:**
- Modify: `src/ptxsim/instructions/cvt/cvt_helpers.cpp`（删除本地 `half_to_float`/`float_to_half`，改用 `half_utils.h`）
- Modify: `tests/unit/ptx/test_cvt_helpers.cpp`（新增边界 case 对比测试）

- [ ] **Step 1: 读取 half_utils.h 现有实现**

```bash
cd /workspace/project/PTX-EMU
cat include/ptxsim/utils/half_utils.h
grep -A 20 "f16_to_f32\|f32_to_f16" include/ptxsim/utils/half_utils.h
```

- [ ] **Step 2: 对比精度（关键！）**

在 `tests/unit/ptx/test_cvt_helpers.cpp` 新增 4 个对比测试，验证 `half_utils.h::f16_to_f32` 与原 `half_to_float` 输出**位完全相同**（建议用 uint32_t 转换对比）：

```cpp
TEST_CASE("half_utils.h vs cvt_helpers equivalence", "[cvt][helpers][equiv]") {
    // 用 union 强制位级比较
    union Bits { float f; uint32_t u; };
    for (uint16_t h = 0; h != 0; h++) {  // 跳过 0（已经测过）
        float a = ptxsim::cvt_helpers::half_to_float(h);
        float b = ptxsim::half_utils::f16_to_f32(h);
        Bits ba{a}, bb{b};
        // 注：实现可能不同，但应 IEEE 754 兼容
        // 这里只测有穷值
        if (!std::isnan(a) && !std::isnan(b)) {
            REQUIRE(ba.u == bb.u);
        }
    }
}
```

跑测试验证：应通过（两者都是 IEEE 754 兼容实现）。

- [ ] **Step 3: 删除 cvt_helpers.cpp 内的 half_to_float / float_to_half 本地实现**

修改 `src/ptxsim/instructions/cvt/cvt_helpers.cpp`：
```cpp
#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include "ptxsim/utils/half_utils.h"  // 新增
// 删除 half_to_float / float_to_half 实现

namespace ptxsim {
namespace cvt_helpers {

// 直接调用 half_utils.h 实现
inline float half_to_float(uint16_t h) { return ptxsim::half_utils::f16_to_f32(h); }
inline uint16_t float_to_half(float f) { return ptxsim::half_utils::f32_to_f16(f); }

// ... round_half_to_even / should_saturate_uint32 保持本地 ...

}  // namespace cvt_helpers
}  // namespace ptxsim
```

- [ ] **Step 4: 编译 + 跑测试 + sanity**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target ptxsim 2>&1 | tail -10
cd build && ctest -L "ptx;cvt" -V 2>&1 | tail -30
./scripts/sanity.sh --quick 2>&1 | tail -20
```

预期：0 错误；现有测试全过；sanity 全过。

- [ ] **Step 5: 提交**

```bash
cd /workspace/project/PTX-EMU
git add src/ptxsim/instructions/cvt/cvt_helpers.cpp tests/unit/ptx/test_cvt_helpers.cpp
git commit -m "refactor(cvt): delegate half_to_float/float_to_half to half_utils.h (T2-6 Step 2)"
```

---

## Sub-task 3: CvtContext + select_strategy() 骨架（0.5 天）

**Files:**
- New: `include/ptxsim/instructions/cvt/cvt_strategy.h`（`ConversionStrategy` 基类 + `CvtContext` struct + `select_strategy()` 工厂）
- Modify: `src/ptxsim/instructions/arithmetic_conversion.cpp:141-220`（提取 qualifiers + has_* 标志到 `build_context()`）

- [ ] **Step 1: 写 CvtContext 单元测试（Red）**

创建 `tests/unit/ptx/test_cvt_context.cpp`（约 10 TEST_CASE），覆盖 `build_context()` 的所有 qualifiers 组合（.sat / .rni / .rzi / .rmi / .rpi / .rna / .rs / 各种 size / 各种 signed/unsigned / f16/f32/f64）。

- [ ] **Step 2: 实现 cvt_strategy.h 骨架（声明 ConversionStrategy + CvtContext + select_strategy 返回 nullptr）**

- [ ] **Step 3: 编译 + 跑测试（Green）**

- [ ] **Step 4: 提交**

```bash
git commit -m "refactor(cvt): add CvtContext + select_strategy() skeleton (T2-6 Step 3)"
```

---

## Sub-task 4: 5 个具体策略按序实现（1 天）

**Files:**
- New: `include/ptxsim/instructions/cvt/{cvt_float_to_float,cvt_int_to_float,cvt_int_to_int,cvt_float_to_int,cvt_rounding,cvt_saturation}.h`
- New: `src/ptxsim/instructions/cvt/{cvt_float_to_float,cvt_int_to_float,cvt_int_to_int,cvt_float_to_int,cvt_rounding,cvt_saturation}.cpp`

- [ ] **Step 4a: FloatToFloatStrategy（最简单，~30 行）** + 单元测试
- [ ] **Step 4b: IntToFloatStrategy（~80 行）** + 单元测试
- [ ] **Step 4c: FloatToIntStrategy 的 .sat 子路径**（先简化，~60 行）+ 单元测试
- [ ] **Step 4d: FloatToIntStrategy 的 5 个舍入模式**（最复杂，~120 行）+ 单元测试
- [ ] **Step 4e: IntToIntStrategy（4×4×4 模板化，最后处理，~200 行）** + 单元测试
- [ ] **Step 4f: 提交**

```bash
git commit -m "refactor(cvt): implement 5 strategies (T2-6 Step 4)"
```

---

## Sub-task 5: P1-4.1 修复 + 94 个新 integration tests（1 天）

**Files:**
- Modify: `src/ptxsim/instructions/cvt/cvt_float_to_int.cpp`（f32→s32 / f64→s64 路径补 r2 写入）
- New: `tests/integration/ptx/test_cvt_int_to_int.cpp`（~30 TEST_CASE）
- New: `tests/integration/ptx/test_cvt_int_to_float.cpp`（~10 TEST_CASE）
- New: `tests/integration/ptx/test_cvt_float_to_int.cpp`（~40 TEST_CASE，含 5 种舍入 × 目标类型）
- New: `tests/integration/ptx/test_cvt_float_to_float.cpp`（~6 TEST_CASE）
- New: `tests/integration/ptx/test_cvt_saturation.cpp`（~8 TEST_CASE）
- Modify: `tests/integration/CMakeLists.txt:247`（注册新测试）
- Modify: `tests/integration/ptx/test_cvt.cpp:142-145`（删除 P1-4.1 SKIP 注释）

- [ ] **Step 1: 修复 P1-4.1 bug（f32→s32 / f64→s64 写 r2）**

```bash
cd /workspace/project/PTX-EMU
grep -n "case 4:\|case 8:\|warp_state.threads" src/ptxsim/instructions/cvt/cvt_float_to_int.cpp
# 找到 f32→s32 / f64→s64 分支，补 advance_thread_pc 或 sync_to_warp_state 调用
```

- [ ] **Step 2: 启用现有 SKIP 测试（`integration_ptx_cvt_f32_from_s32`）**

删除 `tests/integration/ptx/test_cvt.cpp:142-145` 的 SKIP 注释。

- [ ] **Step 3: 写 94 个新 integration tests**

按 explore 报告 F.4 的优先级矩阵分 5 个文件。

- [ ] **Step 4: 跑全量 ctest 验证**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cd build && ctest -L "ptx;cvt" -V 2>&1 | tail -30
```

预期：~97 个 ctest 目标全过（3 个原 + 94 个新）。

- [ ] **Step 5: 提交**

```bash
git commit -m "feat(cvt): 94 new integration tests + fix P1-4.1 (T2-6 Step 5)"
```

---

## Sub-task 6: 删除原 switch + 验证（0.2 天）

**Files:**
- Modify: `src/ptxsim/instructions/arithmetic_conversion.cpp:220-1284`（删除 1063 行 switch，CvtHandler::processOperation 收缩到 ~50 行）

- [ ] **Step 1: 删除原 switch 大块代码**

```cpp
void CvtHandler::processOperation(...) {
    // ... qualifiers extraction (line 144-220) ...
    CvtContext ctx = build_context(operands, qualifiers);
    const ConversionStrategy& strategy = select_strategy(ctx);
    strategy.convert(/* src */, /* dst */, ctx);
}
```

- [ ] **Step 2: 跑全量验证**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build 2>&1 | tee /tmp/build.log
grep "error:" /tmp/build.log | wc -l  # 应 0
cd build && ctest -L "ptx;cvt" -V 2>&1 | tail -30
./scripts/sanity.sh --quick 2>&1 | tail -20
wc -l /workspace/project/PTX-EMU/src/ptxsim/instructions/arithmetic_conversion.cpp  # 应 < 300
```

- [ ] **Step 3: 提交**

```bash
git commit -m "refactor(cvt): collapse 1063-line switch to 50-line strategy dispatch (T2-6 Step 6)"
```

---

## Sub-task 7: Phase 3 验证 + ADR（0.2 天）

- [ ] **Step 1: 跑 ./scripts/sanity.sh（完整回归）**

```bash
cd /workspace/project/PTX-EMU
./scripts/sanity.sh 2>&1 | tail -20
```

预期：全部通过。

- [ ] **Step 2: 跑 ASan 验证**

```bash
cd /workspace/project/PTX-EMU
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
cmake --build build-asan
cd build-asan && ctest -L "cvt;barrier" -V 2>&1 | tail -30
```

预期：无 LeakSanitizer 报告。

- [ ] **Step 3: 写 ADR-XXXX-cvt-strategy-pattern.md**

在 `docs/adr/` 新建 ADR，记录为何选 Composition 而非拆分 Handler（X-Macro 约束）。

- [ ] **Step 4: 最终提交 + 更新 Phase 3 完成门禁**

```bash
git commit -m "docs(adr): record CVT strategy pattern Composition decision (T2-6 closure)"
```
