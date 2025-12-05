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

if command -v nvcc &> /dev/null; then
    # 获取nvcc的绝对路径
    NVCC_PATH=$(which nvcc)
    echo "找到 nvcc: $NVCC_PATH"
    
    # 获取CUDA安装路径（nvcc通常在$CUDA_PATH/bin/nvcc）
    CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))
    echo "设置 CUDA_PATH 为: $CUDA_PATH"
    
    # 设置环境变量
    export CUDA_PATH=$CUDA_PATH
    
    # 可选：将CUDA的库和bin目录添加到相应环境变量
    #export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH
    #export PATH=$CUDA_PATH/bin:$PATH
    
    echo "CUDA环境变量设置成功。"
else
    echo "未找到 nvcc。CUDA可能未安装或不在PATH中。"
    
    # 尝试在常见位置查找CUDA
    if [ -d "/usr/local/cuda" ]; then
        echo "在/usr/local/cuda找到CUDA安装"
        export CUDA_PATH="/usr/local/cuda"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH"
        export PATH="/usr/local/cuda/bin:$PATH"
    else
        echo "无法找到CUDA安装目录。"
        return 1
    fi
fi

# 检查CUDA_PATH环境变量
#if [ -z "$CUDA_PATH" ]; then
#    echo "错误：未设置CUDA_PATH环境变量，请先设置CUDA_PATH"
#    return 1
#fi

# 设置CLASSPATH和别名
export CLASSPATH=".:$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/antlr4/antlr-4.11.1-complete.jar:$CLASSPATH"
alias antlr4='java -Xmx500M -cp "$CLASSPATH" org.antlr.v4.Tool'
alias grun='java -Xmx500M -cp "$CLASSPATH" org.antlr.v4.gui.TestRig'

# 设置PTX-EMU相关环境变量
export PTX_EMU_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
#export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$PTX_EMU_PATH/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 创建输出目录
[ ! -d "$PTX_EMU_PATH/bin" ] && mkdir "$PTX_EMU_PATH/bin"
[ ! -d "$PTX_EMU_PATH/lib" ] && mkdir "$PTX_EMU_PATH/lib"

# 标记环境已设置
export PTX_EMU_ENV_SET=1

echo "PTX-EMU 环境设置成功！"
