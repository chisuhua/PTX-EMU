## 1. 恢复 CMake 注册

- [x] 1.1 定位 `tests/unit/CMakeLists.txt` 中被注释的 7 个 `add_catch_test` 行（line 432-472），逐行取消注释
- [x] 1.2 确认测试源文件路径有效：`ptx/integer/test_ptx_integer.cpp`、`ptx/float/test_ptx_float.cpp`、`ptx/extended/test_ptx_extended.cpp`、`ptx/bitwise/test_ptx_bitwise.cpp`、`ptx/cvt/test_ptx_cvt.cpp`、`ptx/ld_st/test_ptx_ld_st.cpp`、`ptx/cvta/test_ptx_cvta.cpp`

## 2. 修复编译错误

- [x] 2.1 构建并编译所有 7 个恢复的测试，记录编译错误
- [x] 2.2 更新测试代码以适配当前 API（如 `ptxsim::testing` 命名空间工具、`StatementContext` 签名变化）
- [x] 2.3 逐个修复编译错误直到 7 个测试全部编译通过

## 3. 验证测试通过

- [x] 3.1 运行 `ctest -R unit_ptx_integer -V` 确认通过
- [x] 3.2 运行 `ctest -R unit_ptx_float -V` 确认通过
- [x] 3.3 运行 `ctest -R unit_ptx_extended -V` 确认通过
- [x] 3.4 运行 `ctest -R unit_ptx_bitwise -V` 确认通过
- [x] 3.5 运行 `ctest -R unit_ptx_cvt -V` 确认通过
- [x] 3.6 运行 `ctest -R unit_ptx_ld_st -V` 确认通过
- [x] 3.7 运行 `ctest -R unit_ptx_cvta -V` 确认通过
- [x] 3.8 运行 `./scripts/sanity.sh` 全量验证无回归

## 4. 提交与清理

- [x] 4.1 更新 `proposal-suggestions.md`：标记 `fix-commented-ptx-tests` 状态为 `已完成`
- [x] 4.2 提交所有变更（commit message 标注 Fix #N）