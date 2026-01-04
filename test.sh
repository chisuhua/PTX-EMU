#!/bin/bash

# PTX-EMU 测试脚本
# 用于运行不同类型的单元测试

set -e  # 遇到错误时退出

# 默认测试类型
TEST_TYPE="all"

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -t, --type TYPE     指定测试类型 (all, warp, basic, ptx_ir, ptxsim)"
    echo "                      all: 运行所有测试 (默认)"
    echo "                      warp: 运行warp相关测试"
    echo "                      basic: 运行基本功能测试"
    echo "                      ptx_ir: 运行PTX IR相关测试"
    echo "                      ptxsim: 运行PTX模拟器相关测试"
    echo ""
    echo "示例:"
    echo "  $0                  # 运行所有测试"
    echo "  $0 -t warp          # 只运行warp测试"
    echo "  $0 --type basic     # 只运行基本功能测试"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "测试类型: $TEST_TYPE"

# 检查构建目录是否存在
if [ ! -d "build" ]; then
    echo "错误: build目录不存在，请先运行build.sh构建项目"
    exit 1
fi

# 进入构建目录
cd build

# 根据测试类型运行测试
case $TEST_TYPE in
    all)
        echo "运行所有测试..."
        ctest --output-on-failure
        ;;
    warp)
        echo "运行warp相关测试..."
        ctest -R warp --output-on-failure
        ;;
    basic)
        echo "运行基本功能测试..."
        ctest -R basic --output-on-failure
        ;;
    ptx_ir)
        echo "运行PTX IR相关测试..."
        ctest -R ptx_ir --output-on-failure
        ;;
    ptxsim)
        echo "运行PTX模拟器相关测试..."
        ctest -R ptxsim --output-on-failure
        ;;
    *)
        echo "错误: 未知的测试类型: $TEST_TYPE"
        echo "支持的类型: all, warp, basic, ptx_ir, ptxsim"
        exit 1
        ;;
esac

echo "测试完成！"