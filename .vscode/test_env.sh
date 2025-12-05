#!/bin/bash

# 测试环境设置脚本
echo "Testing environment setup..."

# 检查工作目录
echo "Working directory: $(pwd)"

# 检查PTX_EMU_PATH
echo "PTX_EMU_PATH: $PTX_EMU_PATH"

# 检查LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 检查库文件是否存在
if [ -f "$PTX_EMU_PATH/lib/libcudart.so" ]; then
    echo "Found libcudart.so in $PTX_EMU_PATH/lib/"
else
    echo "ERROR: libcudart.so not found in $PTX_EMU_PATH/lib/"
fi

# 检查可执行文件是否存在
if [ -f "$PTX_EMU_PATH/bin/dummy-add" ]; then
    echo "Found dummy-add executable in $PTX_EMU_PATH/bin/"
else
    echo "ERROR: dummy-add executable not found in $PTX_EMU_PATH/bin/"
fi

echo "Environment test completed."