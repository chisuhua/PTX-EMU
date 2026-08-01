# cmake-use-glob-for-sources

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-9
**阶段**: default | **分类**: infra-setup
**类型**: infra

## 架构依据

- `src/CMakeLists.txt` 手动维护 83 个 .cpp 文件列表
- 新增源文件需手动添加到 CMakeLists.txt，容易遗漏导致链接错误
- GLOB 可自动发现新文件，配合 CONFIGURE_DEPENDS 实现增量检测

## 范围

- **In Scope**:
  - 将手动 `set(SOURCES ...)` 替换为 `file(GLOB ... CONFIGURE_DEPENDS)`
  - 添加 CI 检查确保 GLOB 结果与预期一致
- **Out Scope**:
  - 不改变编译选项或链接配置
  - 不修改 tests/ 目录的 CMakeLists.txt（测试文件通常需显式控制）
  - 不影响 build.sh 或 env.sh

## 关键场景

- GIVEN GLOB 替换后, WHEN 新增 src/*.cpp 文件, THEN cmake 自动检测并编译
- GIVEN CI 检查, WHEN GLOB 结果与 git 跟踪文件不一致, THEN CI 报错

## 技术约束

- MUST 使用 `CONFIGURE_DEPENDS` 确保增量构建正确
- MUST 排除不应编译的文件（如有）
- SHOULD 添加注释说明 GLOB 策略

## 验收标准

- 新增 .cpp 文件无需修改 CMakeLists.txt 即可编译
- 全量编译通过
- ctest 全绿
