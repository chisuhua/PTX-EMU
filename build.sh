#!/bin/bash

# PTX-EMU 构建脚本
# 用于构建项目并生成可执行文件

set -e  # 遇到错误时退出

source env.sh

# 检查CUDA_PATH环境变量
if [ -z "$CUDA_PATH" ]; then
    echo "错误: 未设置CUDA_PATH环境变量"
    echo "请先执行: export CUDA_PATH=/usr/local/cuda 或指向你的CUDA安装路径"
    exit 1
fi

if [ ! -d "$CUDA_PATH" ]; then
    echo "错误: CUDA_PATH目录不存在: $CUDA_PATH"
    exit 1
fi

echo "CUDA_PATH: $CUDA_PATH"

# 创建构建目录
mkdir -p build

# 进入构建目录
cd build

# 配置项目
echo "配置项目..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 构建项目
echo "构建项目..."
make -j$(nproc)

echo "构建完成！"
echo "可执行文件位置: build/ptx_emu"