# cmake-use-glob-for-sources - Tasks

## 1. Phase 0: 准备工作

- [ ] 1.1 MUST 记录基线：`cmake -S . -B build && cmake --build build && cd build && ctest` 全绿
- [ ] 1.2 MUST 记录当前手动源文件数量：`git ls-files 'src/**/*.cpp' 'src/*.cpp' | wc -l`（应为 ~77 个手动 + 生成文件）

## 2. Phase 1: 替换 SOURCES (cudart) 列表为 GLOB（15 min）

- [ ] 2.1 MUST 将 `src/CMakeLists.txt` line 41-50 的 `set(SOURCES ...)` 替换为：
  ```cmake
  file(GLOB CUDART_SOURCES CONFIGURE_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/cudart/*.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/cudart/cpptlm_bridge/*.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ptxsim/atomic/*.cpp
  )
  set(SOURCES ${CUDART_SOURCES} ${ANTLR4_GENERATED_SOURCES})
  ```
- [ ] 2.2 MUST 验证：`cmake -S . -B build && cmake --build build` 通过
- [ ] 2.3 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 2.4 git commit -m "refactor(cmake): replace manual SOURCES list with GLOB for cudart"

## 3. Phase 2: 替换 ptx_ir 库文件列表为 GLOB（10 min）

- [ ] 3.1 MUST 将 `src/CMakeLists.txt` line 58-64 的 `add_library(ptx_ir SHARED ...)` 替换为：
  ```cmake
  file(GLOB PTX_IR_SOURCES CONFIGURE_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/ptx_ir/*.cpp
  )
  add_library(ptx_ir SHARED ${PTX_IR_SOURCES})
  ```
- [ ] 3.2 MUST 验证：`cmake -S . -B build && cmake --build build` 通过
- [ ] 3.3 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 3.4 git commit -m "refactor(cmake): replace manual ptx_ir sources with GLOB"

## 4. Phase 3: 替换 ptxsim 库文件列表为 GLOB_RECURSE（15 min）

- [ ] 4.1 MUST 将 `src/CMakeLists.txt` line 69-157 的 `add_library(ptxsim SHARED ...)` 替换为：
  ```cmake
  file(GLOB_RECURSE PTXSIM_SOURCES CONFIGURE_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/ptxsim/*.cpp
  )
  add_library(ptxsim SHARED ${PTXSIM_SOURCES})
  ```
- [ ] 4.2 MUST 验证：`cmake -S . -B build && cmake --build build` 通过
- [ ] 4.3 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 4.4 MUST 验证：GLOB_RECURSE 结果与原手动列表文件数一致
- [ ] 4.5 git commit -m "refactor(cmake): replace manual ptxsim sources with GLOB_RECURSE"

## 5. Phase 4: 替换 ptx_parser 库文件列表为 GLOB（10 min）

- [ ] 5.1 MUST 将 `src/CMakeLists.txt` line 160-163 的 `add_library(ptx_parser SHARED ...)` 替换为：
  ```cmake
  file(GLOB PTX_PARSER_SOURCES CONFIGURE_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/ptx_parser/*.cpp
  )
  add_library(ptx_parser SHARED ${PTX_PARSER_SOURCES})
  ```
- [ ] 5.2 MUST 验证：`cmake -S . -B build && cmake --build build` 通过
- [ ] 5.3 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 5.4 git commit -m "refactor(cmake): replace manual ptx_parser sources with GLOB"

## 6. Phase 5: 新增文件自动检测验证（10 min）

- [ ] 6.1 MUST 创建临时测试文件验证 GLOB 自动检测：
  ```bash
  echo 'void _glob_test_func() {}' > src/ptxsim/_glob_test.cpp
  cmake --build build  # 应自动编译新文件
  # 验证编译产物中包含新文件
  nm build/lib/libptxsim.so | grep _glob_test_func
  rm src/ptxsim/_glob_test.cpp
  cmake --build build  # 应自动移除引用
  ```
- [ ] 6.2 MUST 验证 CONFIGURE_DEPENDS 增量检测工作（删除文件后增量构建不报错）

## 7. Phase 6: 最终验证（10 min）

- [ ] 7.1 MUST 验证：`cmake -S . -B build && cmake --build build` 全量编译通过
- [ ] 7.2 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 7.3 SHOULD 验证：GLOB 结果与 git 跟踪文件一致：
  ```bash
  grep -o 'src/[^"]*\.cpp' build/compile_commands.json | sort -u > /tmp/globbed.txt
  git ls-files 'src/**/*.cpp' 'src/*.cpp' | sort > /tmp/tracked.txt
  diff /tmp/globbed.txt /tmp/tracked.txt
  ```

## 8. 应用阶段

- [ ] 8.1 MUST 运行 `openspec validate cmake-use-glob-for-sources --strict`
- [ ] 8.2 MUST 通过所有验证后 archive 此 change

## 验收

- 新增 .cpp 文件无需修改 CMakeLists.txt 即可编译
- 全量编译通过
- ctest 全绿
- GLOB 结果与原手动列表文件数一致
- CONFIGURE_DEPENDS 增量检测工作正常

## 关键约束（MUST/MUST NOT）

- MUST 使用 `CONFIGURE_DEPENDS` 确保增量构建正确
- MUST 排除不应编译的文件（如有，当前无）
- SHOULD 添加注释说明 GLOB 策略
- MUST NOT 修改 `tests/CMakeLists.txt`（测试文件需显式控制）
- MUST NOT 修改 `src/ptx_ir/CMakeLists.txt`（1-2 文件，GLOB 无收益）
- MUST NOT 修改 `src/ptxir/CMakeLists.txt`（1 文件，GLOB 无收益）
- MUST NOT 改变编译选项或链接配置
- MUST NOT 影响 `build.sh` 或 `env.sh`
