# HSK-3: libcpptlm_cudart.so CMake 暴露方式草案

> **状态**: ✅ **已发出（待 CppTLM 确认 / CI ON-path 验证 / ExternalProject_Add end-to-end）**

**发送记录**: 2026-07-17, CPPTLM_COMMIT_HASH=73e5422, 伴随 commit `93726f62` (archive + spec 晋升)
> **回传目标**: CppTLM Team (`#cpptlm-integration` Slack 频道 / PR comment)
> **承诺时间**: D5 EOD 前
> **形式**: CMake 草案（3 选项对比，默认 ExternalProject_Add）
> **锁定信息**: 选项 1 (ExternalProject_Add) + CPPTLM_COMMIT_HASH=`73e5422` + GIT_REPOSITORY=`https://github.com/chisuhua/CppTLM.git`

---

## 📤 准备发给 CppTLM 团队的完整消息

```
Subject: [HSK-3] libcpptlm_cudart.so CMake 暴露方式草案 — 选择倾向

Cc: CppTLM Team (#cpptlm-integration Slack)

CppTLM Team,

PTX-EMU 端 libcpptlm_cudart.so 的 CMake 暴露方式草案如下，请 review 后反馈偏好。

======================== 3 选项对比 ========================

### 选项 1: ExternalProject_Add（PTX-EMU 默认倾向 ⭐）

```cmake
# PTX-EMU 根 CMakeLists.txt
include(ExternalProject)

ExternalProject_Add(cpptlm
    GIT_REPOSITORY  https://github.com/chisuhua/CppTLM.git
    GIT_TAG         73e5422  # P0 main merge: MemoryBridge + KernelLaunchTLM ext + 12 [f12b] tests + smoke
    CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/cpptlm-install
                    -DBUILD_TESTING=OFF
                    -DCPPTLM_BUILD_CUDART_BRIDGE=ON
    UPDATE_DISCONNECTED TRUE  # 不自动 fetch，避免 ABI 漂移
)

add_subdirectory(src/cudart/cpptlm_bridge)
target_link_libraries(cpptlm_cudart_lib PRIVATE cpptlm)
target_include_directories(cpptlm_cudart_lib PRIVATE
    ${CMAKE_BINARY_DIR}/cpptlm-install/include
)
```

**优点**:
- ✅ 版本 pin（git tag 锁定 CppTLM commit，零 ABI 漂移）
- ✅ build 隔离（CppTLM 构建产物在独立子目录）
- ✅ 与 libcpptlm_cudart.so 静态链接时简化依赖
- ✅ PTX-EMU 主分支 commit 与 CppTLM commit 一一对应

**缺点**:
- ⚠️ 首次 build 需访问 CppTLM 仓库（CI/CD 时需配置 credentials）
- ⚠️ UPDATE_DISCONNECTED 需要手动 bump CppTLM commit hash（release 时更新）

### 选项 2: find_library + 环境变量

```cmake
# PTX-EMU 根 CMakeLists.txt
find_library(CPPTLM_CUDART_LIB
    NAMES cpptlm_cudart
    HINTS $ENV{CPPTLM_PTXEMU_LIBDIR}
)
find_path(CPPTLM_INCLUDE_DIR
    NAMES cudart/cpptlm_bridge.h
    HINTS $ENV{CPPTLM_PTXEMU_INCDIR}
)

if(CPPTLM_CUDART_LIB AND CPPTLM_INCLUDE_DIR)
    message(STATUS "Found libcpptlm_cudart.so: ${CPPTLM_CUDART_LIB}")
    target_link_libraries(ptxemu_runtime PRIVATE ${CPPTLM_CUDART_LIB})
    target_include_directories(ptxemu_runtime PRIVATE ${CPPTLM_INCLUDE_DIR})
else()
    message(STATUS "libcpptlm_cudart.so NOT found — independent mode (bridge == nullptr)")
endif()
```

**用户配置**:
```bash
export CPPTLM_PTXEMU_LIBDIR=/path/to/cpptlm/install/lib
export CPPTLM_PTXEMU_INCDIR=/path/to/cpptlm/install/include
cmake --build build
```

**优点**:
- ✅ build 时完全解耦（不需要网络访问）
- ✅ 适合容器化部署（CppTLM.so 预先打包进镜像）

**缺点**:
- ⚠️ 需手动管理 CppTLM 库安装路径
- ⚠️ 版本不强制（多个 CppTLM 版本共存时易混淆）

### 选项 3: pkg-config

```cmake
# PTX-EMU 根 CMakeLists.txt
find_package(PkgConfig REQUIRED)
pkg_check_modules(CPPTLM REQUIRED IMPORTED_TARGET cpptlm)

target_link_libraries(ptxemu_runtime PRIVATE PkgConfig::CPPTLM)
target_include_directories(ptxemu_runtime PRIVATE ${CPPTLM_INCLUDE_DIRS})
```

**用户配置**:
```bash
# CppTLM 端提供 .pc 文件
PKG_CONFIG_PATH=/path/to/cpptlm/install/lib/pkgconfig:$PKG_CONFIG_PATH
pkg-config --modversion cpptlm  # 验证
```

**优点**:
- ✅ 标准 Linux 工具集成（pkg-config 已成熟）
- ✅ 跨包管理一致（deb/rpm/brew/pacman）

**缺点**:
- ❌ Windows/macOS 需替代方案（vcpkg/homebrew 各自机制）
- ❌ PTX-EMU 当前主要目标 Linux x86_64，可接受但非完美

======================== PTX-EMU 倾向 ========================

**首选: 选项 1 (ExternalProject_Add)**

理由：
1. **CI/CD 简单**: PTX-EMU CI 拉取 CppTLM 仓库 + build libcpptlm_cudart.so 自动化
2. **版本 pin 强制**: CppTLM ABI 变更强制走 git tag 更新
3. **build 隔离**: 避免污染 PTX-EMU 主 build 目录
4. **single source of truth**: cpptlm_bridge.h + MemoryBridge + libcpptlm_cudart.so 来自同一 commit

但是 — 我们也理解 CppTLM 可能有不同偏好：

- 如果 CppTLM 已有现成 libcpptlm_cudart.so 安装 → 选项 2 更快
- 如果 CppTLM 重视发行版生态 → 选项 3 更标准

======================== 决策时间窗口 ========================

**D5 EOD 前**（即 PTX-EMU Phase 5 完成时）：
- CppTLM 团队 review 此草案
- 反馈首选选项
- PTX-EMU 实施 Phase 6 时按选定方案实施

**CPPTLM_COMMIT_HASH** 已锁定: `73e5422` (P0 main merge, 776/776 用例 / 15562 断言 + 12/12 [f12b] 测试)
- 含 MemoryBridge + KernelLaunchTLM extension + `--f12b-ld` flag + vector_add 烟雾测试 + AsyncCompletionAdapter 占位
- 不推荐 `main` (生产前必须固定 SHA,避免 ABI 漂移)

======================== 引用 ========================

- ADR-0021 D-PTX-6 (性能预算 + 暴露方式)
- 综合任务书 §2.1 Task #5
- openspec/changes/cpptlm-d1-full/tasks.md Phase 6

======================== 请求 ========================

请 CppTLM 团队：

1. 评审 3 选项
2. 反馈首选方案
3. 如果选 1：CppTLM GIT_REPOSITORY URL 是否 = `https://github.com/chisuhua/CppTLM.git`
4. 如果选 2：CppTLM 是否提供标准安装脚本
5. 如果选 3：CppTLM 是否提供 `.pc` 文件

确认收到后回复。

— PTX-EMU Architecture Team
```

---

## 🔧 使用方法（PTX-EMU 内部）

实施 Phase 6（任务 6.1-6.4）后：

1. 修改 `CMakeLists.txt`（使用选项 1 默认值）
2. 创建 `src/cudart/cpptlm_bridge/` 子目录 + CMakeLists.txt stub
3. 验证编译：
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build  # 默认 OFF 路径
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_LIB_CPPTLM_CUDART=ON
   cmake --build build  # ON 路径（需 mock find_package）
   ```
4. 提交：
   ```bash
   git add CMakeLists.txt src/cudart/cpptlm_bridge/
   git commit -m "build(cmake): libcpptlm_cudart integration via ExternalProject_Add (HSK-3 + D-PTX-6)

   Default BUILD_LIB_CPPTLM_CUDART=OFF preserves zero-regression.
   ON path enables ExternalProject_Add-to-cpptlm integration.

   Refs:
   - ADR-0021 (D-PTX-6)
   - CppTLM docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #5
   - openspec/changes/cpptlm-d1-full"
   ```
5. 发送给 CppTLM（替换 `<CPPTLM_COMMIT_HASH>` 占位符为 D5 实际值）

---

## 🔍 验证清单（发出前）

- [x] Phase 6 完成：CMakeLists.txt 修改 + cpptlm_bridge 子目录（commit `d0803a09`）
- [x] OFF 路径：现有 202 测试中 188 通过（14 环境性 CUDA SEGFAIL 需 GPU 环境，sandbox 无 GPU）
- [x] ON 路径：ExternalProject_Add 草案完整（见上方 cmake 代码块）
- [x] 选项 1 ExternalProject_Add 完整注释（其他开发者可读）
- [x] CPPTLM_COMMIT_HASH 占位符已替换 → `73e5422` (2026-07-17)

> **修正**: 原始声明"600+ 测试零回归"不准确——实际注册 202 个测试。188/202 通过 = 93.1% 基线。14 个 SEGFAIL 全部为 CUDA runtime 限制（`cudaLaunchKernel` 需真实 GPU device），与代码变更无关，详见 `docs/superpowers/specs/2026-07-15-phase05-baseline-report.md` 报告。

---

## 📋 跟踪

发送后请更新本文件：
- [x] 发送日期: **2026-07-17 已发出**（CPPTLM_COMMIT_HASH 已锁定 `73e5422`）
- [ ] 发送渠道: 用户手动复制（无 Slack/邮件）
- [ ] CppTLM 确认收到:
- [x] CppTLM 首选方案: **已确认选项 1**（见 CppTLM docs/superpowers/specs/2026-07-15-cpptlm-hsk-response.md）
- [x] CPPTLM_COMMIT_HASH 锁定: `73e5422`（2026-07-17 锁定,P0 main merge）
- [x] CppTLM 提供 GIT_REPOSITORY URL 确认: `https://github.com/chisuhua/CppTLM.git`（**已确认**）

---

## 🔄 选项决策流程图

```
PTX-EMU 倾向选项 1
       │
       ▼
CppTLM 反馈 ──────────┐
       │              │
       ├─ 同意选项 1  ──→ 确认 CPPTLM_COMMIT_HASH + GIT_REPOSITORY ✅
       │              │
       ├─ 偏好选项 2  ──→ PTX-EMU 实施 ON 路径用 find_library
       │              │
       └─ 偏好选项 3  ──→ PTX-EMU 实施 ON 路径用 pkg-config
```

---

**最后更新**: 2026-07-17（已发出 — CPPTLM_COMMIT_HASH=`73e5422` + 选项 1 ExternalProject_Add + GIT_REPOSITORY 确认；待 CppTLM 确认）
