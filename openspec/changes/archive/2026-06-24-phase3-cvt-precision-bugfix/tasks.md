# CVT Precision Bugfix (Tasks)

> **总工期**: 0.3-0.5 天 | **依赖**: T2-6 Step 1 ✅ (commit `d3c77b5`)

> **TDD 三阶段**: 严格遵循 AGENTS.md §TDD 流程（红→绿→refactor）

---

## Task 1: 修复 `half_to_float` denormal 路径

**Files:**
- Modify: `src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float`
- Modify: `tests/unit/ptx/test_cvt_helpers.cpp`（更新 Step 1 故意宽松的断言）
- Reference: `include/ptxsim/utils/half_utils.h::f16_to_f32`（正确实现参考）

- [ ] **Step 1: 读取当前 `half_to_float` 实现 + `half_utils.h::f16_to_f32` 对比**

```bash
cd /workspace/project/PTX-EMU
sed -n '/^float half_to_float/,/^}/p' src/ptxsim/instructions/cvt/cvt_helpers.cpp
cat include/ptxsim/utils/half_utils.h
grep -A 30 "f16_to_f32" include/ptxsim/utils/half_utils.h src/ptxsim/utils/half_utils.cpp 2>/dev/null
```

预期：发现 `cvt_helpers.cpp::half_to_float` 的 denormal 路径 bug，对比 `half_utils.h::f16_to_f32` 找到正确算法。

- [ ] **Step 2: 写 denormal 正确性测试（Red 阶段）**

修改 `tests/unit/ptx/test_cvt_helpers.cpp`，**移除** Step 1 中"故意宽松"的注释，恢复严格 IEEE 754 行为验证：

```cpp
// 替换 Step 1 中"故意宽松"的 denormal 断言
TEST_CASE("half_to_float denormal smallest positive", "[cvt][helpers][half]") {
    // smallest positive denormal half = 2^-24
    float result = half_to_float(0x0001);
    REQUIRE(std::isfinite(result));
    REQUIRE(result > 0.0f);
    // 正确值：2^-24 = 5.9604644775390625e-08
    REQUIRE(result == Approx(5.9604644775390625e-08f).epsilon(0.01f));
}

TEST_CASE("half_to_float denormal largest", "[cvt][helpers][half]") {
    // largest denormal half = 0x03FF = (1 - 2^-10) * 2^-14 ≈ 6.097e-05
    float result = half_to_float(0x03FF);
    REQUIRE(std::isfinite(result));
    REQUIRE(result > 0.0f);
    REQUIRE(result == Approx(6.0975551605224609e-05f).epsilon(0.01f));
}

TEST_CASE("half_to_float negative denormal", "[cvt][helpers][half]") {
    // negative denormal = 0x8001
    float result = half_to_float(0x8001);
    REQUIRE(std::isfinite(result));
    REQUIRE(result < 0.0f);
    REQUIRE(result == Approx(-5.9604644775390625e-08f).epsilon(0.01f));
}
```

- [ ] **Step 3: 验证测试失败（Red）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target unit_cvt_helpers 2>&1 | tail -5
cd build && ctest -R "cvt_helpers" -V 2>&1 | tail -30
```

预期：3 个新 denormal 测试全部失败（因 `half_to_float` 仍 buggy）。

- [ ] **Step 4: 修复 `cvt_helpers.cpp::half_to_float` denormal 路径**

按 IEEE 754 half precision 规范重写 denormal 处理：

```cpp
// 替换原 denormal 分支（行号基于 Step 1 commit d3c77b5）
if (exp == 0) {
    if (mantissa == 0) {
        // ±0
        f = sign << 31;
    } else {
        // Denormal: value = (-1)^sign × 2^-14 × (0.mantissa)
        // 即 = mantissa × 2^-24 （mantissa 是 10-bit，0x001 到 0x3FF）
        // 转换为 float32: mantissa × 2^(127-14) = mantissa × 2^113 在 float32 表示
        // 但 mantissa 是 10-bit（最多 1023），所以结果 < 2^113 × 1024 = 2^123
        // 正确转换：
        while ((mantissa & 0x400) == 0) {
            mantissa <<= 1;
        }
        // 现在 mantissa 最高位是 bit 10（0x400），前面计算已对齐
        // 移除最高位并归一化
        mantissa &= 0x3ff;  // 10-bit mantissa
        // value = mantissa × 2^-24 = mantissa / 2^24
        // float32 指数 = 127 - 24 = 103, 尾数 = mantissa × 2^(23-10) = mantissa << 13
        f = (sign << 31) | ((127 - 24 + 1) << 23) | (mantissa << 13);
        // 注：上述循环实现在原代码已有，但最后指数+1 缺失导致结果 × 2
    }
}
```

更清晰的正确实现（参考 `half_utils.h::f16_to_f32`）：

```cpp
if (exp == 0) {
    if (mantissa == 0) {
        f = sign << 31;  // ±0
    } else {
        // Subnormal: value = mantissa × 2^-24
        // float32: 指数 = 127-24 = 103, 尾数 = mantissa × 2^13
        // 排除最高位：mantissa 是 10-bit，循环归一化后剩余 10-bit
        // 直接用整数转换为 float 更简单：
        union { uint32_t u; float f; } bits;
        bits.u = (sign << 31) | (103 << 23) | (mantissa << 13);
        f = bits.f;
    }
}
```

> **关键差异**：原代码 denormal 路径指数计算有 bug（用 `127 - 15 = 112` + 循环 10 次 = 102 + 127 = 229，导致结果 = 2^102）。正确实现：denormal 固定指数 = 127 - 24 = 103，尾数 = 10-bit mantissa × 2^13 = 23-bit float mantissa。

- [ ] **Step 5: 编译 + 跑测试（Green）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target ptxsim 2>&1 | tail -5
cd build && ctest -R "cvt_helpers" -V 2>&1 | tail -30
```

预期：所有 cvt_helpers 测试 + 新 denormal 测试通过。

- [ ] **Step 6: 跑全量 CVT/PTX 回归**

```bash
cd /workspace/project/PTX-EMU/build
ctest -L "ptx;cvt" -V 2>&1 | tail -30
```

预期：16/16 现有测试全过（denormal bug 仅在特殊输入触发，常规回归不受影响）。

- [ ] **Step 7: 提交**

```bash
cd /workspace/project/PTX-EMU
git add src/ptxsim/instructions/cvt/cvt_helpers.cpp tests/unit/ptx/test_cvt_helpers.cpp
git commit -m "fix(cvt): correct half_to_float denormal path (PTX cvt.f32.f16 edge case)"
```

---

## Task 2: 修复 `should_saturate_uint32` 边界

**Files:**
- Modify: `src/ptxsim/instructions/cvt/cvt_helpers.cpp::should_saturate_uint32`
- Modify: `tests/unit/ptx/test_cvt_helpers.cpp`（更新 Step 1 故意宽松的断言）

- [ ] **Step 1: 写边界正确性测试（Red）**

```cpp
TEST_CASE("should_saturate_uint32 boundary equality", "[cvt][helpers][sat]") {
    // 4294967295.0f 在 float32 中 = 4294967296.0f
    // 当 sat_high = 4294967295.0f 时，temp == sat_high 应饱和
    REQUIRE(should_saturate_uint32(4294967295.0f, 4294967295.0f));
    // 严格大于 sat_high
    REQUIRE_FALSE(should_saturate_uint32(1e10f, 4294967295.0f));
    // 低于边界
    REQUIRE_FALSE(should_saturate_uint32(1e9f, 4294967295.0f));
}
```

- [ ] **Step 2: 验证测试失败（Red）**

```bash
cd build && ctest -R "cvt_helpers" -V 2>&1 | tail -20
```

预期：第 1 个断言（`4294967295.0f == 4294967295.0f` 应返回 true）失败。

- [ ] **Step 3: 修复 `should_saturate_uint32`**

```cpp
// 修改 src/ptxsim/instructions/cvt/cvt_helpers.cpp
// 原: return temp >= 4294967295.0f && temp < sat_high;
// 改为: return temp >= 4294967295.0f && temp <= sat_high;
bool should_saturate_uint32(float temp, float sat_high) {
    return temp >= 4294967295.0f && temp <= sat_high;
}
```

- [ ] **Step 4: 编译 + 跑测试（Green）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target ptxsim
cd build && ctest -R "cvt_helpers" -V 2>&1 | tail -20
```

预期：所有测试通过。

- [ ] **Step 5: 跑全量 CVT/PTX 回归 + sanity**

```bash
cd /workspace/project/PTX-EMU/build
ctest -L "ptx;cvt" -V 2>&1 | tail -30
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --quick 2>&1 | tail -20
```

预期：0 回归。

- [ ] **Step 6: 提交**

```bash
cd /workspace/project/PTX-EMU
git add src/ptxsim/instructions/cvt/cvt_helpers.cpp tests/unit/ptx/test_cvt_helpers.cpp
git commit -m "fix(cvt): should_saturate_uint32 use <= for boundary inclusion (PTX cvt.u32.f32.sat)"
```

---

## Task 3: 新增 PTX 端到端边界测试

**Files:**
- New: `tests/integration/ptx/test_cvt_edge_cases.cpp`（denormal + sat 边界 e2e 验证）
- Modify: `tests/integration/CMakeLists.txt`（注册）

- [ ] **Step 1: 写 PTX 测试（Red）**

```cpp
// test_cvt_edge_cases.cpp
// =============================================================================
// Integration test: 验证 cvt.f32.f16 denormal + cvt.u32.f32.sat 边界
// 修复 CVT Precision Bugfix 后的端到端正确性
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/cta_context.h"
#include <cmath>
#include <cstdint>

TEST_CASE("cvt.f32.f16 denormal round-trip", "[cvt][edge-case][denormal]") {
    // Setup: half denormal → float → half 应保持
    // 0x0001 (smallest positive denormal half = 2^-24) → f32 → f16 应保持 0x0001
    SMContext sm(4, 128, 4096, 0);
    // ... 构造指令序列 ...
    // 验证: dst half 值 == 0x0001
    REQUIRE(true);  // 占位 - 详细实现在 TDD 阶段填
}

TEST_CASE("cvt.u32.f32.sat boundary equality", "[cvt][edge-case][sat]") {
    // Setup: f32 = 4294967295.0f → .sat → u32 应 = 0xFFFFFFFF
    // 修复前: 严格 < 导致 0x00000000（不饱和）
    // 修复后: <= 边界 → 0xFFFFFFFF
    SMContext sm(4, 128, 4096, 0);
    // ... 构造指令序列 ...
    // 验证: dst u32 值 == 0xFFFFFFFF
    REQUIRE(true);  // 占位
}

TEST_CASE("cvt.u32.f32.sat above boundary clamps to max", "[cvt][edge-case][sat]") {
    // Setup: f32 = 1e10f → .sat → u32 应 = 0xFFFFFFFF（已在 u32 表示范围外）
    SMContext sm(4, 128, 4096, 0);
    // ... 构造指令序列 ...
    // 验证: dst u32 值 == 0xFFFFFFFF
    REQUIRE(true);  // 占位
}
```

- [ ] **Step 2: 填充占位实现（具体指令构造）**

参考 `tests/integration/ptx/test_cvt.cpp` 的现有 3 个 case 实现风格，构造真实的 CVT 指令序列。

- [ ] **Step 3: 编译 + 跑测试**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build 2>&1 | tail -5
cd build && ctest -R "cvt_edge_cases" -V 2>&1 | tail -30
```

- [ ] **Step 4: 注册到 CMakeLists.txt**

```cmake
# tests/integration/CMakeLists.txt
add_catch_test(integration_cvt_edge_cases
    ptx/test_cvt_edge_cases.cpp
)
set_tests_properties(integration_cvt_edge_cases PROPERTIES LABELS "integration;cvt;edge_case")
```

- [ ] **Step 5: 跑全量 sanity**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --quick 2>&1 | tail -20
```

- [ ] **Step 6: 提交**

```bash
cd /workspace/project/PTX-EMU
git add tests/integration/ptx/test_cvt_edge_cases.cpp tests/integration/CMakeLists.txt
git commit -m "test(cvt): add edge case integration tests for denormal + sat boundary"
```

---

## 验证门禁

- [ ] Task 1 + 2 + 3 全部完成（3 commits）
- [ ] `ctest -L "ptx;cvt"` 全过（含新 3+ 个 e2e 测试）
- [ ] `./scripts/sanity.sh --quick` 全过
- [ ] 0 行为变更（除 denormal + 边界 bug 修复外）
- [ ] `docs/audits/HEALTH-AUDIT-2026-06-21.md` 不需更新（CVT 精度未在审计范围）
- [ ] T2-6 Step 2 可继续（前提：half_to_float 修复后，Step 2 复用 half_utils.h 时 denormal 行为应一致）
