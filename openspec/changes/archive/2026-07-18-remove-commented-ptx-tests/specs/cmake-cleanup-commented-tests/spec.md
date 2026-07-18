# cmake-cleanup-commented-tests

## ADDED Requirements

### Requirement: tests/unit/CMakeLists.txt 不含被注释的废弃测试目标
被 Oracle A2 决策禁用且有 integration 等价覆盖的测试目标，其注释块应从 CMakeLists.txt 中删除（非永久注释）。
