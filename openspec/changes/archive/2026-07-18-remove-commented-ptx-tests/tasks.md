# 实施任务

## Phase 1: 删除注释块

- [x] Task 1.1: 删除 tests/unit/CMakeLists.txt 第 535-580 行（7 个被注释的 add_catch_test 块 + 注释头）
- [x] Task 1.2: 确认被注释块引用的 .cu 源文件（tests/unit/ptx/test_ptx_integer.cu 等）在其他 CMakeLists.txt 中无引用
- [x] Task 1.3: cmake --build build 确认编译通过

## Phase 2: 验证

- [x] Task 2.1: cd build && ctest -L "unit" -V（所有单元测试通过，无 CMake 配置错误）
- [x] Task 2.2: cd build && ctest -L "integration;ptx" -V（integration 等价覆盖的测试全部通过）
- [x] Task 2.3: git diff --stat 确认仅 tests/unit/CMakeLists.txt 被修改（-46 行）
