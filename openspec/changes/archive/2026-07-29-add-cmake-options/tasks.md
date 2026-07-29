# add-cmake-options - Tasks

## 1. Phase 1: 添加 option 定义和 sanitizer 逻辑（30 min）

- [ ] 1.1 MUST 在根 `CMakeLists.txt` 的 `USE_DETAILED_LOGGING` option 块之后（约 line 37）添加新 option 定义：
  ```cmake
  # Sanitizers and warnings
  option(ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
  option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer for UB detection" OFF)
  option(ENABLE_WERROR "Treat compiler warnings as errors" OFF)
  ```
- [ ] 1.2 MUST 添加 ASAN 条件块：
  ```cmake
  if(ENABLE_ASAN)
      add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
      add_link_options(-fsanitize=address)
      message(STATUS "AddressSanitizer: enabled")
  endif()
  ```
- [ ] 1.3 MUST 添加 UBSAN 条件块：
  ```cmake
  if(ENABLE_UBSAN)
      add_compile_options(-fsanitize=undefined)
      add_link_options(-fsanitize=undefined)
      message(STATUS "UndefinedBehaviorSanitizer: enabled")
  endif()
  ```
- [ ] 1.4 MUST 添加 WERROR 条件块（仅 CXX）：
  ```cmake
  if(ENABLE_WERROR)
      add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
      message(STATUS "Warnings as errors: enabled")
  ```
- [ ] 1.5 MUST 验证：`cmake -L ..` 显示所有新 option（ENABLE_ASAN, ENABLE_UBSAN, ENABLE_WERROR）
- [ ] 1.6 MUST 验证：默认构建（`cmake -S . -B build && cmake --build build`）行为不变
- [ ] 1.7 MUST 验证：`ctest` 全绿（默认构建）
- [ ] 1.8 git commit -m "feat(cmake): add ENABLE_ASAN/ENABLE_UBSAN/ENABLE_WERROR options"

## 2. Phase 2: 验证各选项独立工作（20 min）

- [ ] 2.1 MUST 验证 ASAN 构建：
  ```bash
  cmake -DENABLE_ASAN=ON -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug
  cmake --build build-asan
  cd build-asan && ctest --output-on-failure
  ```
- [ ] 2.2 MUST 验证 UBSAN 构建：
  ```bash
  cmake -DENABLE_UBSAN=ON -S . -B build-ubsan -DCMAKE_BUILD_TYPE=Debug
  cmake --build build-ubsan
  cd build-ubsan && ctest --output-on-failure
  ```
- [ ] 2.3 SHOULD 验证 WERROR 构建（如暴露已有警告，记录但不阻塞）：
  ```bash
  cmake -DENABLE_WERROR=ON -S . -B build-werror
  cmake --build build-werror
  ```

## 3. Phase 3: 验证选项组合（15 min）

- [ ] 3.1 MUST 验证 ASAN + UBSAN 同时启用：
  ```bash
  cmake -DENABLE_ASAN=ON -DENABLE_UBSAN=ON -S . -B build-san -DCMAKE_BUILD_TYPE=Debug
  cmake --build build-san
  cd build-san && ctest --output-on-failure
  ```
- [ ] 3.2 MUST 验证默认构建无行为变化（不传任何 option）：
  ```bash
  cmake -S . -B build-default && cmake --build build-default
  cd build-default && ctest --output-on-failure
  ```

## 4. Phase 4: 可选 - BUILD_TESTS / BUILD_BENCH 开关（15 min）

- [ ] 4.1 SHOULD 添加 `BUILD_TESTS` option（默认 ON）：
  ```cmake
  option(BUILD_TESTS "Build test targets" ON)
  if(BUILD_TESTS)
      enable_testing()
      add_subdirectory(tests)
  endif()
  ```
- [ ] 4.2 SHOULD 添加 `BUILD_BENCH` option（默认 OFF）：
  ```cmake
  option(BUILD_BENCH "Build benchmark targets" OFF)
  if(BUILD_BENCH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/cute/CMakeLists.txt)
      add_subdirectory(bench/cute)
  endif()
  ```
- [ ] 4.3 SHOULD 验证 `BUILD_TESTS=OFF` 跳过测试编译
- [ ] 4.4 git commit -m "feat(cmake): add BUILD_TESTS/BUILD_BENCH options"（如实施）

## 5. 应用阶段

- [ ] 5.1 MUST 运行 `openspec validate add-cmake-options --strict`
- [ ] 5.2 MUST 通过所有验证后 archive 此 change

## 验收

- `cmake -DENABLE_ASAN=ON ..` 构建通过且测试可运行
- `cmake -DENABLE_UBSAN=ON ..` 构建通过且测试可运行
- ASAN 和 UBSAN 可同时启用
- 默认构建（无选项）行为不变
- 选项列表通过 `cmake -L` 可见（4+ option）

## 关键约束（MUST/MUST NOT）

- MUST 所有新选项默认 OFF（不破坏现有构建）
- MUST ASAN 和 UBSAN 可同时启用
- SHOULD 在 CMakeLists.txt 中添加选项说明注释
- MUST NOT 修改 CI 配置（CI 可后续集成）
- MUST NOT 改变默认构建行为
