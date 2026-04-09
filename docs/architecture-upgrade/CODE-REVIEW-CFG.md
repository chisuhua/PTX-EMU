# CFG Builder 代码审查报告

**日期**: 2026-04-10  
**审查者**: PTX-EMU Architecture Team  
**范围**: Phase 5.1 CFG Builder 编译集成  
**状态**: Ready for Implementation

---

## 当前状态

### 文件缺失

```bash
$ ls -la src/ptx_parser/cfg_builder.*
cfg_builder files not found in src/ptx_parser/
```

**问题**: Phase 3 实验性代码已被移除

**原因**: 之前的实现存在编译链接问题，需要重新创建

---

## 需要的文件

### 1. cfg_builder.h

**位置**: `src/ptx_parser/cfg_builder.h`

**关键组件**:
- BasicBlock 结构体
- CFG 结构体
- CFGBuilder 类
- PostDominatorMap 类型

**依赖**:
- `<vector>`, `<map>`, `<set>`, `<string>`
- `ptx_ir/statement_context.h`

---

### 2. cfg_builder.cpp

**位置**: `src/ptx_parser/cfg_builder.cpp`

**关键函数**:
- `CFGBuilder::build()`
- `CFGBuilder::computePostDominators()`
- `CFGBuilder::identifyBasicBlocks()`
- `CFGBuilder::findBranchTargets()`
- `CFGBuilder::buildEdges()`
- `CFGBuilder::findImmediatePostDominator()`

---

### 3. CMakeLists.txt 更新

**位置**: `src/CMakeLists.txt`

**修改**:
```cmake
add_library(ptx_parser SHARED
    ptx_parser/ptx_visitor.cpp
    ptx_parser/cfg_builder.cpp  # ADD THIS
)
```

---

## 实施计划

### Phase 5.1 任务

1. **5.1.1**: 创建 cfg_builder.h (2 小时)
2. **5.1.2**: 创建 cfg_builder.cpp (4 小时)
3. **5.1.3**: 更新 CMakeLists.txt (1 小时)
4. **5.1.4**: 验证编译 (1 小时)
5. **5.1.5**: 运行 Parser 测试 (2 小时)

**总计**: 10 小时

---

## 验收标准

### 编译验证

```bash
cmake --build build --target ptx_parser
# Expected: 100% Built target ptx_parser
```

### Parser 测试

```bash
./tests/ptx/test_reconvergence_ptx.sh
# Expected: ALL TESTS PASSED
```

---

## 风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 类型定义错误 | 中 | 高 | 仔细审查 include 路径 |
| 方法签名不匹配 | 中 | 高 | 保持 header/source 一致 |
| CMake 配置错误 | 低 | 高 | 复制现有库配置模式 |

---

**状态**: 审查完成，Ready for implementation  
**下一步**: 开始 Phase 5.1.1 (创建 cfg_builder.h)
