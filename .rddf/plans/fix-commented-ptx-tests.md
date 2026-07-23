# Plan: fix-commented-ptx-tests

> **生成时间**: 2026-07-23
> **执行模式**: ⚡ 轻量模式 (分支: openspec/fix-commented-ptx-tests)

## 1. 恢复 CMake 注册

### Task 1.1: 恢复 add_catch_test 行

- [x] Step 1: `tests/unit/CMakeLists.txt` 中找到被注释的 7 个 `add_catch_test` 行
- [x] Step 2: 逐行取消注释
- [x] Step 3: 确认测试源文件在对应路径存在

### Task 1.2: 编译验证

- [x] Step 1: `cmake -S . -B build && cmake --build build --target GenerateParser`
- [x] Step 2: 编译 7 个测试目标
- [x] Step 3: 记录编译错误

## 2. 修复测试代码

### Task 2.1: 逐一修复 API 适配

- [x] Step 1: 更新测试代码适配当前 `ptxsim::testing` 命名空间
- [x] Step 2: 更新 `StatementContext` 构造签名
- [x] Step 3: 修复所有 7 个测试的编译错误

### Task 2.2: 编译通过

- [x] Step 1: 全量编译通过，无错误

## 3. 验证测试

### Task 3.1: 逐个运行测试

- [x] Step 1: `ctest -R unit_ptx_integer -V` ✅
- [x] Step 2: `ctest -R unit_ptx_float -V` ✅
- [x] Step 3: `ctest -R unit_ptx_extended -V` ✅
- [x] Step 4: `ctest -R unit_ptx_bitwise -V` ✅
- [x] Step 5: `ctest -R unit_ptx_cvt -V` ✅
- [x] Step 6: `ctest -R unit_ptx_ld_st -V` ✅
- [x] Step 7: `ctest -R unit_ptx_cvta -V` ✅

### Task 3.2: 全量回归

- [x] Step 1: `./scripts/sanity.sh` 无回归

## 4. 提交与清理

### Task 4.1: 提交变更

- [x] Step 1: git add + commit
- [x] Step 2: 更新 `proposal-suggestions.md`

### Task 4.2: 归档

- [x] Step 1: 回到 main 分支
- [x] Step 2: 归档 change