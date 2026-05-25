# Tasks: add-sm90-100-bsync-interleave

## 任务清单

### Phase 1: BSSY/BSYNC 状态机 (3 tasks)

#### Task 1.1: 创建 bsync_state.h
**时间**: 30 分钟
**验证**: 编译通过

MUST:
- 在 `include/ptxsim/` 创建 `bsync_state.h`
- 定义 `BsyncState` 结构体：barrier_id, waiting_threads mask, suspended_pc
- 定义 `BsyncManager` 管理所有屏障状态
- 提供 `bssy()`, `bsync()`, `check_release()`, `release()` 接口

#### Task 1.2: 创建 bsync_state.cpp
**时间**: 45 分钟
**验证**: 单元测试通过

MUST:
- 在 `src/ptxsim/core/` 创建 `bsync_state.cpp`
- 实现 `BsyncManager::bssy()` - 设置屏障点
- 实现 `BsyncManager::bsync()` - 线程到达挂起
- 实现 `BsyncManager::check_release()` - 检测是否所有线程到达
- 实现 `BsyncManager::release()` - 释放屏障，唤醒所有线程

#### Task 1.3: 集成到 bar.warp.sync handler
**时间**: 30 分钟
**验证**: 相关测试通过

MUST:
- 修改 `barrier.cpp` 中的 `BarWarpSyncHandler::processOperation()`
- 用 BsyncManager 替代简单的 is_blocked 标记
- 添加 `blocked_cycles_remaining` 递减逻辑

---

### Phase 2: 动态线程迁移调度器 (3 tasks)

#### Task 2.1: 修改 WarpScheduler
**时间**: 45 分钟
**验证**: 编译通过

MUST:
- 在 `warp_scheduler.cpp` 添加动态交错逻辑
- 实现 `schedule_with_migration()` - 允许部分线程挂起后切换执行其他组

#### Task 2.2: 修改 exe_once() 分叉处理
**时间**: 45 分钟
**验证**: 测试通过

MUST:
- 在 `sm_context.cpp` 中添加 "交错执行模式" 开关
- 实现 `select_next_group()` - 随机选择可执行组（实现非确定性）
- 实现 `suspend_and_switch()` - 挂起当前组，切换到另一组

#### Task 2.3: 实现非确定性执行顺序
**时间**: 30 分钟
**验证**: 三种模式可切换

MUST:
- 添加配置项 `divergence_execution_mode`: "sequential" | "interleaved" | "shortest_first"
- 在交错模式下使用随机选择或最短路径启发式

---

### Phase 3: 生命周期管理 (2 tasks)

#### Task 3.1: 添加 blocked_cycles 递减
**时间**: 30 分钟
**验证**: 编译 + 测试通过

MUST:
- 在 `thread_state.h` 添加 `blocked_cycles_remaining` 字段 (已有部分)
- 在 `sm_context.cpp` exe_once() 中递减
- 当递减到 0 时自动清除 is_blocked

#### Task 3.2: 完善屏障释放逻辑
**时间**: 30 分钟
**验证**: 相关测试通过

MUST:
- 修改 `check_reconvergence()` 处理 BsyncManager 状态
- 确保 barrier 释放时所有 suspended 线程恢复执行

---

### Phase 4: 测试验证 (4 tasks)

#### Task 4.1: 单元测试 - BsyncState
**时间**: 30 分钟
**验证**: 测试通过

MUST:
- 创建 `tests/test_bsync_state.cpp`
- 测试 bssy/bsync/release 生命周期
- 测试多线程同时到达屏障

#### Task 4.2: 集成测试 - 动态交错
**时间**: 45 分钟
**验证**: 测试通过

MUST:
- 创建 `tests/test_divergence_interleaved.cpp`
- 验证场景 2：Path A 执行一部分后切换到 Path B

#### Task 4.3: 集成测试 - 短路径优先
**时间**: 45 分钟
**验证**: 测试通过

MUST:
- 创建 `tests/test_shortest_path_first.cpp`
- 验证场景 3：短路径被优先调度

#### Task 4.4: 端到端测试
**时间**: 30 分钟
**验证**: 所有相关测试通过

MUST:
- 修改 `test_divergence_sync_standalone` 支持三种模式
- 验证所有线程最终在 reconvergence point 汇合

---

### Phase 5: 文档更新 (2 tasks)

#### Task 5.1: 更新架构文档
**时间**: 30 分钟
**验证**: 文档已更新

MUST:
- 更新 `docs/architecture/simt_emulation.md` 说明新的执行模型
- 添加 BSSY/BSYNC 语义说明
- 添加动态交错执行流程图

#### Task 5.2: 更新注释
**时间**: 15 分钟
**验证**: 代码注释已更新

MUST:
- 在关键代码位置添加 TODO 注释指向此计划
- 更新 AGENTS.md 中的限制说明