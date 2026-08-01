# reduce-thread-context-includes

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-7
**阶段**: default | **分类**: core-impl
**类型**: refactor

## 架构依据

- `include/ptxsim/thread_context.h` 头部有 **21 个 #include**
- 过多 include 导致编译依赖膨胀，任何被包含头文件变更触发全部依赖者重编译
- 部分 include 可能仅需前向声明（指针/引用参数）

## 范围

- **In Scope**:
  - 分析 21 个 include 的使用方式（值类型 vs 指针/引用）
  - 对仅需指针/引用的类型改为前向声明
  - 将实现特有的 include 移到 .cpp 文件
- **Out Scope**:
  - 不改变 ThreadContext 的 public API
  - 不删除实际需要的 include
  - 不引入新的头文件

## 关键场景

- GIVEN include 精简后, WHEN 编译项目, THEN 编译通过且无新增 warning
- GIVEN include 精简后, WHEN 修改被前向声明替代的头文件, THEN 不触发 thread_context.h 依赖者重编译

## 技术约束

- MUST 编译通过（所有使用 thread_context.h 的文件）
- MUST NOT 将值类型参数/成员的 include 改为前向声明
- SHOULD 保持 include 分组（标准库 / 项目 / 第三方）

## 验收标准

- include 数量减少 ≥ 30%（21 → ≤ 15）
- 全量编译通过
- 无新增编译 warning
