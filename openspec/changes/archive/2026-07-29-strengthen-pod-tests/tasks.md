# strengthen-pod-tests - Tasks

## Task List

### Phase 1: 深化 test_warp_context.cpp（40 min）

- [ ] 1.1 MUST 新增 SECTION 验证 `set_active_mask()` overwrite 语义（非 OR 合并）：
  ```cpp
  SECTION("set_active_mask overwrites, not ORs") {
      warp.set_active_mask(0x0000000F);
      warp.set_active_mask(0x000000F0);
      REQUIRE(warp.get_active_mask() == 0x000000F0);  // overwrite, not 0xFF
  }
  ```
- [ ] 1.2 MUST 新增 SECTION 验证边界条件：`active_mask = 0x0`（全 inactive）、`0xFFFFFFFF`（全 active）、`0x1`（仅 lane 0）
- [ ] 1.3 MUST 新增 SECTION 验证错误路径：`is_lane_active(32)` 越界、`get_warp_thread_id(32)` 越界
- [ ] 1.4 MUST 确保文件 ≥ 150 行有效断言
- [ ] 1.5 MUST 检查 `tests/integration/exec/` 或 `tests/integration/warp/` 是否有对应 `execute_warp_instruction` 集成测试
- [ ] 1.6 如缺失: MUST 新增对应 integration 测试（`step_warp()` 驱动 active_mask 变化场景）

### Phase 2: 深化 test_sm_context.cpp（40 min）

- [ ] 2.1 MUST 新增 SECTION 验证 SM warp 调度状态转换（idle -> running -> done）
- [ ] 2.2 MUST 新增 SECTION 验证边界条件：0 warp SM、最大 warp 数 SM
- [ ] 2.3 MUST 新增 SECTION 验证错误路径：越界 warp index 访问
- [ ] 2.4 MUST 确保文件 ≥ 180 行有效断言
- [ ] 2.5 MUST 检查对应 integration 测试存在性
- [ ] 2.6 如缺失: MUST 新增对应 integration 测试

### Phase 3: 深化 test_cvt_context.cpp（40 min）

- [ ] 3.1 MUST 新增 SECTION 验证 f16 边界值转换（min/max/denormalized）
- [ ] 3.2 MUST 新增 SECTION 验证 NaN/Inf 转换行为
- [ ] 3.3 MUST 新增 SECTION 验证错误路径：不支持的类型组合（如 f16 -> s64 with .sat）
- [ ] 3.4 MUST 确保文件 ≥ 330 行有效断言
- [ ] 3.5 MUST 检查对应 integration 测试存在性（`tests/integration/ptx/`）
- [ ] 3.6 如缺失: MUST 新增对应 integration 测试

### Phase 4: 深化 test_smcontext_injection.cpp（40 min）

- [ ] 4.1 MUST 新增 SECTION 验证 null bridge 注入错误路径
- [ ] 4.2 MUST 新增 SECTION 验证 detach 后操作错误路径
- [ ] 4.3 MUST 新增 SECTION 验证 attach -> detach -> reattach 生命周期
- [ ] 4.4 MUST 确保文件 ≥ 420 行有效断言
- [ ] 4.5 MUST 检查对应 integration 测试存在性（`tests/integration/cpptlm/`）
- [ ] 4.6 如缺失: MUST 新增对应 integration 测试

### Phase 5: 验证（20 min）

- [ ] 5.1 MUST 验证编译：`. env.sh && cmake --build build`
- [ ] 5.2 MUST 验证 unit 全绿：`cd build && ctest -L unit --output-on-failure`
- [ ] 5.3 MUST 验证 integration 全绿：`cd build && ctest -L integration --output-on-failure`
- [ ] 5.4 MUST 验证行数：每个文件 ≥ 50 行有效断言（目标行数见 design.md）
- [ ] 5.5 MUST 验证全量回归：`cd build && ctest --output-on-failure`

### Phase 6: 提交

- [ ] 6.1 git commit -m "test(unit): strengthen context behavioral tests (state/boundary/error)"
- [ ] 6.2 MUST 运行 `openspec validate strengthen-pod-tests --strict`
- [ ] 6.3 MUST 通过所有验证后 archive 此 change

## 验收

- 每个 context 测试文件 ≥ 50 行有效断言（目标：warp ≥150, sm ≥180, cvt ≥330, smcontext_injection ≥420）
- 新增测试覆盖至少 1 个错误路径
- 对应 integration 测试存在且通过
- `ctest -L unit` 全绿
- `ctest -L integration` 全绿

## 关键约束（MUST/MUST NOT）

- MUST 遵循 test-coverage-enforcer：新增 unit 测试必须有对应 `execute_warp_instruction` 集成测试
- MUST 使用 Catch2 框架
- MUST 测试标签格式 `<type>;<subject>`（如 `[unit;warp]`）
- MUST NOT 修改被测源码（`src/ptxsim/`）
- MUST NOT 新建测试目录结构
- MUST NOT 重复已有 integration/e2e 测试覆盖的场景
