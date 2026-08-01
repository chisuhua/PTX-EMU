# add-cmake-options

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-10
**阶段**: default | **分类**: infra-setup
**类型**: infra

## 架构依据

- 根 `CMakeLists.txt` 仅有 **1 个 cmake option**
- 缺少常用开发选项：ASAN/UBSAN 内存检测、编译警告级别、构建类型快捷方式等
- 内存安全检测对模拟器项目尤为重要（并发 warp 调度 + 共享内存模拟）

## 范围

- **In Scope**:
  - 添加 `ENABLE_ASAN` — AddressSanitizer 支持
  - 添加 `ENABLE_UBSAN` — UndefinedBehaviorSanitizer 支持
  - 添加 `ENABLE_WERROR` — 将警告视为错误
  - 可选：添加 `BUILD_TESTS` / `BUILD_BENCH` 开关
- **Out Scope**:
  - 不改变默认构建行为（所有新选项默认 OFF）
  - 不修改 CI 配置（CI 可后续集成）

## 关键场景

- GIVEN `-DENABLE_ASAN=ON`, WHEN 构建并运行测试, THEN ASAN 报告内存错误
- GIVEN `-DENABLE_UBSAN=ON`, WHEN 构建并运行测试, THEN UBSAN 报告未定义行为
- GIVEN 默认构建, WHEN 不传递任何选项, THEN 行为与之前完全一致

## 技术约束

- MUST 所有新选项默认 OFF（不破坏现有构建）
- MUST ASAN 和 UBSAN 可同时启用
- SHOULD 在 CMakeLists.txt 中添加选项说明注释

## 验收标准

- `cmake -DENABLE_ASAN=ON ..` 构建通过且测试可运行
- `cmake -DENABLE_UBSAN=ON ..` 构建通过且测试可运行
- 默认构建（无选项）行为不变
- 选项列表通过 `cmake -L` 可见
