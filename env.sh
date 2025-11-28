#!/bin/bash

# 检查是否已经设置了基本环境变量
if [ -n "$PTX_EMU_ENV_SET" ]; then
    echo "PTX-EMU 环境已设置"
    return 0
fi

# 检查Java环境
if ! which java > /dev/null; then
    echo "错误：未找到Java环境，请先安装并设置Java"
    return 1
fi

# 检查CMake环境
if ! which cmake > /dev/null; then
    echo "错误：未找到CMake，请先安装CMake"
    return 1
fi

# 检查CUDA_PATH环境变量
if [ -z "$CUDA_PATH" ]; then
    echo "错误：未设置CUDA_PATH环境变量，请先设置CUDA_PATH"
    return 1
fi

# 设置CLASSPATH和别名
export CLASSPATH=".:$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/antlr4/antlr-4.11.1-complete.jar:$CLASSPATH"
alias antlr4='java -Xmx500M -cp "$CLASSPATH" org.antlr.v4.Tool'
alias grun='java -Xmx500M -cp "$CLASSPATH" org.antlr.v4.gui.TestRig'

# 设置PTX-EMU相关环境变量
export PTX_EMU_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$PTX_EMU_PATH/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 创建输出目录
[ ! -d "$PTX_EMU_PATH/bin" ] && mkdir "$PTX_EMU_PATH/bin"
[ ! -d "$PTX_EMU_PATH/lib" ] && mkdir "$PTX_EMU_PATH/lib"

# 标记环境已设置
export PTX_EMU_ENV_SET=1

echo "PTX-EMU 环境设置成功！"
