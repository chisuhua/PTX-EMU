# ADR-0005: MemoryRegion 注册机制

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-03 |
| **关联任务** | T11.1.5-T11.1.7 |
| **作者** | PTX-EMU Team |

## 上下文

PTX-EMU 的内存管理之前缺乏边界检查机制：

- `SimpleMemory::read/write` 直接使用偏移量，不验证是否在合法范围内
- `HardwareMemoryManager::access` 没有验证地址是否在已注册的内存区域内
- 越界访问会导致未定义行为（segfault、数据损坏）

## 决策驱动因素

1. **内存安全**：越界访问必须被检测并报告，而非静默崩溃
2. **区域管理**：需要知道哪些地址范围是合法的
3. **与异常体系集成**：越界应抛出 InvalidMemoryAccessException

## 考虑的替代方案

### 方案 A: 每次访问时计算边界

**描述**: read/write 函数中直接检查 offset < size

**优点**:
- 简单，无需额外数据结构

**缺点**:
- 只能检查单个缓冲区的边界
- 无法处理复杂场景（如多个不连续区域）
- 检查逻辑分散在各个访问函数中

### 方案 B: MemoryRegion 注册机制 (✅ 选中)

**描述**: 在 HardwareMemoryManager 中注册合法的内存区域，每次访问时验证地址是否在某个已注册区域内

**优点**:
- 支持多个不连续区域的统一管理
- 可以查询地址属于哪个区域
- 检查逻辑集中在 access() 函数中
- 与异常体系集成（越界抛异常）

**缺点**:
- 需要维护区域注册表（map 查找，O(log N)）
- 注册/注销 API 增加了使用复杂度

**选择理由**: PTX-EMU 模拟的 GPU 有多个独立的内存空间（global、shared、local、constant），每个空间可能包含多个不连续的区域。注册机制提供了统一的管理方式。

### 方案 C: 使用 guard page 和 signal handler

**描述**: 在内存区域后设置不可访问的 guard page，通过 signal handler 捕获越界

**优点**:
- 零运行时检查开销

**缺点**:
- 平台依赖性强
- 调试信息有限
- 不适用于模拟环境（需要精确的错误报告）

## 决策内容

### 设计原则

1. **注册优先**：内存使用前必须注册区域
2. **集中检查**：所有访问通过 HardwareMemoryManager::access() 验证
3. **快速失败**：越界立即抛异常，而非静默损坏

### 实现要点

```cpp
// MemoryRegion 结构
struct MemoryRegion {
    uint64_t base_addr;
    size_t size;
    MemoryType type;
    
    bool contains(uint64_t addr, size_t access_size) const {
        return addr >= base_addr && (addr + access_size) <= (base_addr + size);
    }
};

// HardwareMemoryManager 注册 API
class HardwareMemoryManager {
    std::map<uint64_t, MemoryRegion> regions_;
    std::mutex mutex_;
    
    void register_region(uint64_t base, size_t size, MemoryType type);
    void unregister_region(uint64_t base);
    const MemoryRegion* get_region(uint64_t addr) const;
    
    // 集中检查
    void access(uint64_t addr, size_t size, AccessType type) {
        std::lock_guard<std::mutex> lock(mutex_);
        // 直接遍历 regions_ 查找（避免调用 get_region 导致死锁）
        bool found = false;
        for (const auto& [key, region] : regions_) {
            if (region.contains(addr, size)) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw InvalidMemoryAccessException(addr, size);
        }
    }
};
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/memory/hardware_memory_manager.h` | 修改 | 添加 MemoryRegion 和注册 API |
| `src/memory/hardware_memory_manager.cpp` | 修改 | access() 中增加边界检查 |
| `include/memory/simple_memory.h` | 修改 | 添加 validate_offset() |
| `tests/test_memory_bounds.cpp` | 新增 | 内存安全单元测试 |

## 后果

### 正面影响

- 所有越界访问被检测并报告
- 错误消息包含越界地址和区域信息，便于调试
- 内存区域管理清晰

### 负面影响

- 每次访问需要遍历区域注册表（可通过优化数据结构改进）
- 需要先注册区域才能访问，增加了初始化步骤

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 区域注册表查找性能瓶颈 | 低 | 低 | 区域数量通常很少（<10），O(log N) 可忽略 |
| 忘记注册区域导致访问失败 | 中 | 高 | 初始化时注册所有必需区域，测试验证 |
| 死锁（嵌套获取 mutex） | 低 | 高 | access() 直接遍历 regions_，不调用 get_region() |

## 合规检查

后续相关开发应检查：

- [ ] 新增内存区域时必须调用 register_region()
- [ ] 不在 access() 之外的地方进行边界检查（避免重复和遗漏）
- [ ] 单元测试覆盖越界访问场景

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-03 | 初始版本 | PTX-EMU Team |

## 参考

- [任务计划](../reports/task-plan.md#sprint-111-异常体系--内存安全day-1-4)
