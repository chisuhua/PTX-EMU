## 1. 测试框架搭建

- [x] 1.1 创建 `tests/unit/cudart/test_cudart_sim.cpp` 测试文件，引入 Catch2 和必要的 cudart 头文件
- [x] 1.2 在 `tests/unit/CMakeLists.txt` 中添加 `add_catch_test(unit_cudart_sim cudart/test_cudart_sim.cpp)`，设置标签 `"unit;cudart"`，链接 `cudart` 库
- [x] 1.3 验证编译通过：`cmake --build build --target unit_cudart_sim`

## 2. cudaLaunchKernel 单元测试

- [x] 2.x **已确认**：cudaLaunchKernel 需要 PTX context（func2name、g_ptx_interpreter），无法无 `__cudaRegisterFatBinary` 直接测试。推迟到 integration/E2E 覆盖

## 3. cudaStreamSynchronize 单元测试

- [x] 3.2 测试 `cudaStreamSynchronize` 默认 stream（nullptr → cudaSuccess）
- [x] 3.3 测试 `cudaStreamSynchronize` 无 kernel launch（返回 cudaSuccess）
- [x] 3.x **备注**："after launch" 需要 cudaLaunchKernel，推迟到 integration/E2E

## 4. 验证与回归

- [x] 4.1 运行 `ctest -R unit_cudart_sim -V`，确认 2/2 测试绿色通过
- [x] 4.2 运行 `./scripts/sanity.sh --quick`，确认无回归
- [x] 4.3 更新 `proposal-suggestions.md`：将 `add-cudart-unit-tests` 条目 status 改为 `已完成`