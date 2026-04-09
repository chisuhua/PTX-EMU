# 代码重构计划 (Phase 7.2)

**日期**: 2026-04-11  
**阶段**: Phase 7.2 - 代码质量检查与重构  
**状态**: 进行中

---

## 审查范围

### 已修改文件

| 文件 | 修改行数 | 审查优先级 |
|------|---------|-----------|
| cfg_builder.h | +54 | 🔴 高 |
| cfg_builder.cpp | +88 | 🔴 高 |
| ptx_interpreter.cpp | +27 | 🟡 中 |
| test_cfg_edge_cases.cpp | +384 | 🟢 低 |
| test_reconvergence_parser.cpp | +177 | 🟢 低 |

---

## 代码质量检查清单

### cfg_builder.h 审查

- [ ] 命名规范一致性
- [ ] Doxygen 注释完整性
- [ ] const 正确性
- [ ] 头文件保护
- [ ] include 顺序

**状态**: ✅ 已审查  
**问题**: 无严重问题  
**建议**: 添加更多使用示例注释

---

### cfg_builder.cpp 审查

- [x] 函数长度检查 (<100 行)
- [x] 错误处理完整性
- [x] 日志记录充分性
- [x] 边界条件处理
- [x] 变量命名清晰

**状态**: ✅ 已审查  
**发现**: buildEdges 函数注释可以更详细  
**行动**: 添加详细注释

---

### ptx_interpreter.cpp 审查

- [x] try-catch 正确性
- [x] 日志级别适当
- [x] 错误处理完整
- [x] CFG 分析调用位置合理

**状态**: ✅ 已审查  
**发现**: 良好  
**建议**: 无

---

## 重构任务

### Task 1: 更新 cfg_builder.h 注释 (30 min)

**目标**: 添加使用示例和更详细的 API 说明

**修改**:
```cpp
/**
 * @example
 * // Build CFG from kernel statements
 * CFG cfg = CFGBuilder::build(statements, label2pc);
 * 
 * // Compute post-dominators for reconvergence points
 * PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
 * 
 * // Update branch instructions with reconvergence PC
 * for (auto& stmt : statements) {
 *     if (stmt.type == S_BRA) {
 *         auto& branch = std::get<BranchInstr>(stmt.data);
 *         branch.reconvergence_pc = postDoms[stmt.pc];
 *     }
 * }
 */
```

---

### Task 2: 优化 buildEdges 注释 (30 min)

**目标**: 清晰说明分支边添加逻辑

**修改**: 添加详细注释说明：
1. Fall-through 边添加
2. Branch target 边添加（关键修复）
3. 边的去重逻辑

---

### Task 3: 提取公共验证函数 (1 小时)

**目标**: 提取 CFG 验证的公共代码

**候选函数**:
```cpp
// 从 test_cfg_edge_cases.cpp 提取
bool validate_cfg_structure(const CFG& cfg);
bool validate_post_dominators(const PostDominatorMap& postDoms);
void log_cfg_statistics(const CFG& cfg);
```

---

### Task 4: 改进错误消息 (30 min)

**目标**: 提供更具体的错误信息

**当前**:
```cpp
PTX_ERROR_EMU("CFG analysis failed: %s", e.what());
```

**改进后**:
```cpp
PTX_ERROR_EMU("CFG build failed for kernel '%s': %s", 
              kernel_name.c_str(), e.what());
```

---

## 验证计划

### 重构后测试

```bash
# 1. 编译验证
cmake --build build --target ptx_parser cudart

# 2. 运行现有测试
ctest -R "cfg|simt" --output-on-failure

# 3. 验证重构未破坏功能
./bin/test_cfg_builder
./bin/dummy
./bin/test_ptx_ld_st
```

---

## 时间表

| 任务 | 预计时间 | 状态 |
|------|---------|------|
| Task 1: 更新注释 | 30 min | ⏳ |
| Task 2: 优化 buildEdges 注释 | 30 min | ⏳ |
| Task 3: 提取公共函数 | 1 hour | ⏳ |
| Task 4: 改进错误消息 | 30 min | ⏳ |
| **总计** | **2.5 hours** | - |

---

**状态**: 准备开始 Task 1  
**下一步**: 更新 cfg_builder.h 注释
