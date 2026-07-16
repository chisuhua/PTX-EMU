# HSK-3 草稿：libcpptlm_cudart.so CMake ExternalProject_Add 暴露方式草案

> **生成时间**: 2026-07-16  
> **来源**: PTX-EMU `cpptlm-d1-full` change（OpenSpec / ADR-0021 §D-PTX-6, commit `d0803a09`）  
> **目标**: CppTLM 团队 (`chisuhua/CppTLM`)  
> **上下文**: F12b-LD MemoryBridge 集成第三阶段交付

---

## 📧 Send-to (待用户填写)

- **C.C.**: CppTLM Build 团队
- **Channel**: GitHub issue `@chisuhua/CppTLM` + Slack `#cpptlm-ptxemu-bridge`
- **Subject**: `[CppTLM D1-Full] HSK-3: libcpptlm_cudart.so CMake 暴露方式 — ExternalProject_Add + D5 EOD CPPTLM_COMMIT_HASH`

---

## 📋 Message Body

### 1. 状态锁定

PTX-EMU 端 `libcpptlm_cudart.so` 集成 CMake 已通过 **ExternalProject_Add**（commit `d0803a09`）落地；OpenSpec `design.md §7.1` 与 `spec.md` 已与实现对齐（B5 修复 commit `0456418e`）。

### 2. CMake 集成草案（推荐 — 选项 1）

PTX-EMU 端 CMake 引用 CppTLM 的模式：

```cmake
# PTX-EMU 端 CMakeLists.txt（commit d0803a09 实现）
include(ExternalProject)

ExternalProject_Add(cpptlm
    GIT_REPOSITORY "https://github.com/chisuhua/CppTLM.git"
    GIT_TAG        ${CPPTLM_COMMIT_HASH}             # D5 EOD 时锁定
    PREFIX         "${CMAKE_BINARY_DIR}/_deps/cpptlm"
    BUILD_IN_SOURCE 1
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
        # 由 sibling CMakeLists 把 -DHAVE_PTXEMU_CPPTLM_BRIDGE=1 传给 CppTLM
    PATCH_COMMAND   ""
    UPDATE_COMMAND  ""
    BUILD_COMMAND   ${CMAKE_COMMAND} --build .
    INSTALL_COMMAND ""
)

# 暴露给 src/cudart/cudart_sim.cpp 用 ABI 头文件
ExternalProject_Get_Property(cpptlm BINARY_DIR)
target_include_directories(cudart PRIVATE
    ${cpptlm_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
)

# 链接 PTX-EMU cudart.so（libcpptlm_cudart 是新库）
add_library(cpptlm_cudart SHARED
    src/cudart_cpptlm_bridge_shim.cpp
)
target_link_libraries(cpptlm_cudart
    PRIVATE
        cpptlm-cpp-bridge      # 来自 CppTLM ExternalProject_Add
        PTXEMU::cpptlm_bridge_headers  # 来自 include/cudart/cpptlm_bridge.h
)
target_compile_definitions(cpptlm_cudart PRIVATE
    CPPTLMBRIDGE_VERSION=1     # 与 ABI 头文件对齐
)
```

### 3. CppTLM 端 4 步接入

```cmake
# CppTLM 端 CMakeLists.txt（待 CppTLM 修改）

# Step 1: 提供 ExternalProject_Add 友好的目标名
add_library(cpptlm-cpp-bridge SHARED
    src/cpputlm/memory_bridge.cpp
)
target_include_directories(cpptlm-cpp-bridge PUBLIC include/)

# Step 2: 提供 ABI 真值源的 include 目录
install(DIRECTORY include/cudart DESTINATION include)

# Step 3: 暴露 `cpptlm-cpp-bridge-config.cmake` 供 PTX-EMU find_package
install(TARGETS cpptlm-cpp-bridge EXPORT cpptlm-cpp-bridge-targets)
install(EXPORT cpptlm-cpp-bridge-targets
    FILE cpptlm-cpp-bridge-config.cmake
    NAMESPACE cpptlm::
    DESTINATION cmake
)

# Step 4: ABI 真值源版本检查
if(NOT TARGET cpptlm::cpptlm_bridge_headers)
    add_library(cpptlm::cpptlm_bridge_headers INTERFACE)
    target_compile_definitions(cpptlm::cpptlm_bridge_headers INTERFACE
        CPPTLMBRIDGE_VERSION=1
    )
endif()
```

### 4. PTX-EMU 端验证清单

```bash
# 1. include 路径正确指向 ABI 真值源
grep -n "cpptlm_bridge.h" build/_deps/cpptlm-src/include/cudart/cpptlm_bridge.h
# 期望：找到（ExternalProject_Add 拉取成功）

# 2. CPPTLM_COMMIT_HASH 锁定
git -C build/_deps/cpptlm-src log --oneline -1
# 期望：d5eod 时锁定的具体 commit hash + "HSK-3 anchor" 标签

# 3. 编译期 ABI 断言
nm -D build/lib/libcpptlm_cudart.so 2>&1 | grep -E "T cpptlm_(attach|detach)_bridge"
# 期望：T cpptlm_attach_bridge + T cpptlm_detach_bridge

# 4. 完整链路：libcpptlm_cudart.so 调用 ABI 后转发到 PTX-EMU cudart_sim.cpp
LD_LIBRARY_PATH=build/lib ldd /usr/bin/nvcc-driver-using-pemt
# 期望：看到 libptxemu_cudart + libcpptlm_cudart 同时链接

# 5. 性能：-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake -S . -B build_lto -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build_lto --target cpptlm_cudart
time ./build/bin/cpptlm_smoke_test
# 期望：< 100 cycles / global_access call（per D-PTX-6 性能预算）
```

### 5. CPPTLM_COMMIT_HASH 锁定流程（D5 EOD = 2026-07-16）

PTX-EMU 端会在 D5 EOD（=今天，2026-07-16）锁定 CPPTLM_COMMIT_HASH:

```bash
# PTX-EMU 端在今天 EOD 执行：
CPPT_HASH=$(git -C /path/to/CppTLM rev-parse HEAD)
sed -i "s/set(CPPTLM_COMMIT_HASH \".*\")/set(CPPTLM_COMMIT_HASH \"${CPPT_HASH}\")/" CMakeLists.txt
git add CMakeLists.txt
git commit -m "build(cmake): lock CPPTLM_COMMIT_HASH=${CPPT_HASH} (HSK-3 D5 EOD anchor)"
git push origin main
```

CppTLM 端会收到的 commit hash 在 PTX-EMU 仓库的 `CMakeLists.txt:125` 行找到（`set(CPPTLM_COMMIT_HASH "...")` ）。

### 6. 跨 PTX-EMU-Phase 9+ 演进

当前 HSK-3 草案支持 **F12b-LD 阶段的同步场景**。后续 Phase 9+ 演进（async IAsyncCompletion）需要：

```cmake
# Future: Phase 9+ async bridge
ExternalProject_Add(cpptlm_async
    GIT_REPOSITORY "https://github.com/chisuhua/CppTLM.git"
    GIT_TAG        ${CPPTLM_COMMIT_HASH_ASYNC}
    ...
    CMAKE_ARGS
        -DENABLE_ASYNC_COMPLETION=ON
)
```

不在本 HSK-3 范围。后续 HSK-3b 处理。

---

## 📎 交叉引用

- PTX-EMU 端 commit `d0803a09`（HSK-3 + D-PTX-6 libcpptlm_cudart 集成）: https://github.com/chisuhua/PTX-EMU/commit/d0803a09
- PTX-EMU 端 ADR-0021 §D-PTX-6: https://github.com/chisuhua/PTX-EMU/blob/380a8b6a/docs/adr/0021-cpptlm-d1-full-integration.md#决策-d-ptx-6-性能预算vtable-优化-编译期内联
- PTX-EMU 端 design.md §7.1（B5 修复 commit `0456418e` 后）: https://github.com/chisuhua/PTX-EMU/blob/380a8b6a/openspec/changes/cpptlm-d1-full/design.md
- PTX-EMU 端 spec.md（B5 修复 commit `0456418e` 后）: https://github.com/chisuhua/PTX-EMU/blob/380a8b6a/openspec/changes/cpptlm-d1-full/specs/cpptlm-d1-full/spec.md
- 综合任务书: https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #4

---

## ⏱️ 等待 CppTLM 端的反馈

- **期望反馈类型**: PR → `chisuhua/CppTLM main` + 提供 `cpptlm-cpp-bridge-config.cmake`
- **本 PR 应包含**:
  - `cpptlm-cpp-bridge-config.cmake` 安装到 `${PREFIX}/cmake/`
  - target name 严格使用 `cpptlm::cpptlm_bridge_headers` + `cpptlm-cpp-bridge`
  - 编译期断言 `CPPTLMBRIDGE_VERSION=1`
- **不在本 PR 范围**:
  - async IAsyncCompletion — 推迟 F12c
  - 网表/CCL/IPC fake — 推迟 F12c

---

**发送方**: PTX-EMU Architecture Team  
**ADR-0021 状态**: Active (2026-07-16)  
**本 HSK 版本**: HSK-3 v1  
**签发**: ⏳ 待 PTX-EMU Architecture Team 发出（您审核后手动 send）
