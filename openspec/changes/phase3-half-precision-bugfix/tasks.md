# Half Precision Bugfix (Tasks)

> **总工期**: 0.3 天 | **依赖**: `phase3-cvt-precision-bugfix` (commit `32ce8a0`)

> **TDD 三阶段**: 严格遵循 AGENTS.md §TDD 流程

---

## Task 1: 调研 + 修复 `f16_to_f32` denormal 路径

**Files:**
- Read: `include/ptxsim/utils/half_utils.h`（声明）
- Modify: `src/ptxsim/utils/half_utils.cpp`（实现 — 修复 denormal 分支）
- Reference: `src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float`（commit `32ce8a0`，正确算法参考）

- [ ] **Step 1: 读取当前 `half_utils.h` 和实现**

```bash
cd /workspace/project/PTX-EMU
cat include/ptxsim/utils/half_utils.h
ls src/ptxsim/utils/half_utils.cpp 2>/dev/null && cat src/ptxsim/utils/half_utils.cpp
grep -A 30 "f16_to_f32\|f32_to_f16" src/ptxsim/utils/half_utils.cpp 2>/dev/null
```

预期：找到 `f16_to_f32` 和 `f32_to_f16` 实现；denormal 分支 buggy。

- [ ] **Step 2: 调研 `f32_to_f16` 是否也有同源 bug**

对比 `f32_to_f16` 实现与 `cvt_helpers.cpp::float_to_half`（如有）。如确认有 bug，一并记录到 proposal。

- [ ] **Step 3: 写 denormal 正确性测试（Red 阶段）**

创建 `tests/unit/utils/test_half_utils.cpp`：

```cpp
// test_half_utils.cpp
// =============================================================================
// Unit test: 验证 half_utils.h f16 ↔ f32 转换的 denormal 路径正确性
// 修复 half-precision-bugfix 后，half_utils 与 cvt_helpers 应行为一致
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxsim/utils/half_utils.h"
#include <cmath>
#include <cstdint>
#include <limits>

using ptxsim::utils::f16_to_f32;
using ptxsim::utils::f32_to_f16;

TEST_CASE("f16_to_f32 zero/inf/nan", "[half][utils]") {
    REQUIRE(f16_to_f32(0x0000) == 0.0f);
    REQUIRE(f16_to_f32(0x8000) == -0.0f);
    REQUIRE(f16_to_f32(0x7C00) == std::numeric_limits<float>::infinity());
    REQUIRE(f16_to_f32(0xFC00) == -std::numeric_limits<float>::infinity());
    REQUIRE(std::isnan(f16_to_f32(0x7E00)));
}

TEST_CASE("f16_to_f32 denormal smallest positive", "[half][utils][denormal]") {
    // smallest positive denormal half = 2^-24
    REQUIRE(f16_to_f32(0x0001) == Catch::Approx(5.9604644775390625e-08f).epsilon(0.01f));
}

TEST_CASE("f16_to_f32 denormal largest", "[half][utils][denormal]") {
    // largest denormal half = 0x03FF ≈ 6.097e-5
    REQUIRE(f16_to_f32(0x03FF) == Catch::Approx(6.0975551605224609e-05f).epsilon(0.01f));
}

TEST_CASE("f16_to_f32 denormal negative", "[half][utils][denormal]") {
    REQUIRE(f16_to_f32(0x8001) == Catch::Approx(-5.9604644775390625e-08f).epsilon(0.01f));
}

TEST_CASE("f16_to_f32 normal boundary", "[half][utils]") {
    // 0x3C00 = 1.0 in half
    REQUIRE(f16_to_f32(0x3C00) == 1.0f);
    // 0x4000 = 2.0
    REQUIRE(f16_to_f32(0x4000) == 2.0f);
    // 0x7BFF = largest normal ≈ 65504
    REQUIRE(f16_to_f32(0x7BFF) == Catch::Approx(65504.0f).epsilon(0.001f));
}

// f32_to_f16 tests (Task 1 Step 2 调研后如确认 bug，添加对应测试)
TEST_CASE("f32_to_f16 zero/inf/nan", "[half][utils]") {
    REQUIRE(f32_to_f16(0.0f) == 0x0000);
    REQUIRE(f32_to_f16(-0.0f) == 0x8000);
    REQUIRE(f32_to_f16(std::numeric_limits<float>::infinity()) == 0x7C00);
    REQUIRE(f32_to_f16(-std::numeric_limits<float>::infinity()) == 0xFC00);
    REQUIRE(f32_to_f16(std::numeric_limits<float>::quiet_NaN()) == 0x7E00);
}

TEST_CASE("f32_to_f16 normal boundary", "[half][utils]") {
    REQUIRE(f32_to_f16(1.0f) == 0x3C00);
    REQUIRE(f32_to_f16(2.0f) == 0x4000);
    REQUIRE(f32_to_f16(65504.0f) == 0x7BFF);
    REQUIRE(f32_to_f16(65536.0f) == 0x7C00);  // overflows to +Inf
}
```

- [ ] **Step 4: 验证测试失败（Red）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target unit_half_utils 2>&1 | tail -5
cd build && ctest -R "unit_half_utils" -V 2>&1 | tail -30
```

预期：denormal 测试失败（与 `cvt_helpers.cpp::half_to_float` Step 1 Red 阶段类似症状）。

- [ ] **Step 5: 修复 `f16_to_f32` denormal 路径**

**重要：必须参考 `src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float`（commit `32ce8a0`）已验证正确的算法**，不要重新发明：

```cpp
// 替换原 denormal 分支（参考 cvt_helpers.cpp 已修算法）
// IEEE 754 half denormal: value = mantissa × 2^-24
// In float32: exp_f = 103 + p (p = high bit position of mantissa)
// frac = (mantissa - 2^p) << (23 - p)
uint32_t m = mantissa;
int p = 0;
while ((m & 0x200) == 0) {
    m <<= 1;
    p++;
}
uint32_t frac = (m & 0x1FF) << 14;
uint32_t exp_f = 103 + p;
f32 = (sign << 31) | (exp_f << 23) | frac;
```

> **如 Task 1 Step 2 调研发现 `f32_to_f16` 也有 bug**：同时修复（参考 `cvt_helpers.cpp::float_to_half` 正确算法）

- [ ] **Step 6: 编译 + 跑测试（Green）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target ptxsim
cd build && ctest -R "unit_half_utils" -V 2>&1 | tail -30
```

预期：所有测试通过。

- [ ] **Step 7: 跑全量 CVT/PTX 回归（确保无副作用）**

```bash
cd /workspace/project/PTX-EMU/build
ctest -L "ptx;cvt" -V 2>&1 | tail -30
```

预期：16/16 现有测试全过。

- [ ] **Step 8: 跑 sanity**

```bash
cd /workspace/project/PTX-EMU
./scripts/sanity.sh --quick 2>&1 | tail -10
```

预期：ALL TESTS PASS。

- [ ] **Step 9: 提交**

```bash
cd /workspace/project/PTX-EMU
git add include/ptxsim/utils/half_utils.h src/ptxsim/utils/half_utils.cpp tests/unit/utils/test_half_utils.cpp tests/unit/utils/CMakeLists.txt
git commit -m "fix(half-utils): correct f16_to_f32 denormal path (PTX cvt.f16 precision)"
```

---

## Task 2: 验证 `half_utils.h` 与 `cvt_helpers.cpp` 行为一致

**Files:**
- New: `tests/unit/utils/test_half_utils_consistency.cpp`（对比测试）

- [ ] **Step 1: 写对比测试（验证 `half_utils::f16_to_f32` 与 `cvt_helpers::half_to_float` 一致）**

```cpp
// test_half_utils_consistency.cpp
// 验证两个实现的 bit-perfect 一致
TEST_CASE("half_utils vs cvt_helpers f16_to_f32 equivalence", "[half][utils][cvt][consistency]") {
    union Bits { float f; uint32_t u; };
    for (uint32_t h = 0; h <= 0xFFFF; h++) {
        float a = ptxsim::utils::f16_to_f32(static_cast<uint16_t>(h));
        float b = ptxsim::cvt_helpers::half_to_float(static_cast<uint16_t>(h));
        Bits ba{a}, bb{b};
        if (!std::isnan(a) && !std::isnan(b)) {
            REQUIRE(ba.u == bb.u);
        } else {
            // NaN 也要 bit-perfect（避免 quiet vs signaling 不一致）
            REQUIRE(ba.u == bb.u);
        }
    }
}
```

- [ ] **Step 2: 跑对比测试**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R "half_utils_consistency" -V 2>&1 | tail -20
```

预期：65536 个 case 全过（bit-perfect 一致）。

- [ ] **Step 3: 提交**

```bash
cd /workspace/project/PTX-EMU
git add tests/unit/utils/test_half_utils_consistency.cpp tests/unit/utils/CMakeLists.txt
git commit -m "test(half-utils): verify half_utils.h consistent with cvt_helpers.cpp"
```

---

## 验证门禁

- [ ] Task 1 完成：f16_to_f32 denormal 修复 + 全量 sanity
- [ ] Task 2 完成：half_utils.h ↔ cvt_helpers.cpp bit-perfect 一致（65536 case）
- [ ] T2-6 Step 2 即可解封（half_utils.h 复用为安全）
- [ ] `docs/audits/HEALTH-AUDIT-2026-06-21.md` 不需更新（half precision 未在审计范围）
