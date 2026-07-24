## 1. 测试框架搭建

- [ ] 1.1 创建 `tests/unit/cudart/test_cudart_sim.cpp` 测试文件，引入 Catch2 和必要的 cudart 头文件
- [ ] 1.2 在 `tests/unit/cudart/CMakeLists.txt` 中添加 `add_catch_test(unit_cudart_sim cudart/test_cudart_sim.cpp)`，设置标签 `"unit;cudart"`，链接 `cudart` 库
- [ ] 1.3 验证空测试文件编译通过：`cmake --build build --target unit_cudart_sim`

## 2. cudaLaunchKernel 单元测试

- [ ] 2.1 测试 `cudaLaunchKernel` 正常路径：构造 `GPUContext`，调用 `cudaLaunchKernel`，验证 kernel 被注册且参数正确
- [ ] 2.2 测试 `cudaLaunchKernel` 错误路径：传入空函数指针，验证返回 `cudaErrorInvalidDeviceFunction`
- [ ] 2.3 测试 `cudaLaunchKernel` 多次 launch：连续调用两次不同 kernel，验证均正确注册

## 3. cudaStreamSynchronize 单元测试

- [ ] 3.1 测试 `cudaStreamSynchronize` 正常路径：launch kernel 后调用 synchronize，验证返回 `cudaSuccess`
- [ ] 3.2 测试 `cudaStreamSynchronize` 默认 stream：验证默认 stream（0）同步正确
- [ ] 3.3 测试 `cudaStreamSynchronize` 无 kernel launch：调用 synchronize 但不 launch，验证返回 `cudaSuccess`

## 4. 验证与回归

- [ ] 4.1 运行 `ctest -R unit_cudart_sim -V`，确认所有测试绿色通过
- [ ] 4.2 运行 `./scripts/sanity.sh --quick`，确认无回归
- [ ] 4.3 更新 `proposal-suggestions.md`：将 `add-cudart-unit-tests` 条目 status 改为 `已完成`