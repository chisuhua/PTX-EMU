#!/bin/bash
# PTX-EMU 调试运行快捷脚本
# 用法：./debug-run.sh <配置> <程序> [参数...]
# 配置选项：release, debug, trace, memory, instruction, perf

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}PTX-EMU 调试运行脚本${NC}"
    echo ""
    echo -e "${YELLOW}用法:${NC} $0 <配置> <程序> [参数...]"
    echo ""
    echo -e "${YELLOW}配置选项:${NC}"
    echo "  release      - 生产环境配置（最小日志，最佳性能）"
    echo "  debug        - 开发调试配置（平衡的日志级别）"
    echo "  trace        - 详细跟踪配置（最高详细级别）"
    echo "  memory       - 内存调试配置（专注内存操作）"
    echo "  instruction  - 指令调试配置（专注指令执行）"
    echo "  perf         - 性能分析配置（最小开销）"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0 debug ./build/bin/dummy-args"
    echo "  $0 trace ./build/bin/dummy-args"
    echo "  $0 memory ./build/bin/dummy-args 256 256"
    echo "  $0 perf ./build/bin/RAY 512 512"
    echo ""
}

# 检查参数
if [ $# -lt 2 ]; then
    show_help
    exit 1
fi

CONFIG_NAME=$1
shift
PROGRAM=$@

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIGS_DIR="$PROJECT_ROOT/configs"

# 映射配置名到配置文件
case $CONFIG_NAME in
    release)
        CONFIG_FILE="$CONFIGS_DIR/config.ini"
        ;;
    debug|dev)
        CONFIG_FILE="$CONFIGS_DIR/dev_debug_config.ini"
        ;;
    trace|verbose)
        CONFIG_FILE="$CONFIGS_DIR/verbose_trace_config.ini"
        ;;
    memory)
        CONFIG_FILE="$CONFIGS_DIR/memory_debug_config.ini"
        ;;
    instruction|instr)
        CONFIG_FILE="$CONFIGS_DIR/instruction_debug_config.ini"
        ;;
    perf|performance)
        CONFIG_FILE="$CONFIGS_DIR/perf_config.ini"
        ;;
    help|-h|--help)
        show_help
        exit 0
        ;;
    *)
        echo -e "${RED}错误：未知的配置 '$CONFIG_NAME'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误：配置文件不存在：$CONFIG_FILE${NC}"
    exit 1
fi

# 检查程序是否存在
if [ ! -f "$PROGRAM" ]; then
    echo -e "${RED}错误：程序不存在：$PROGRAM${NC}"
    exit 1
fi

# 复制配置文件到工作目录
WORK_CONFIG="$PROJECT_ROOT/ptx_debug.conf"
cp "$CONFIG_FILE" "$WORK_CONFIG"

# 获取配置描述
case $CONFIG_NAME in
    release)
        DESC="生产环境配置 - 最小日志，最佳性能"
        ;;
    debug|dev)
        DESC="开发调试配置 - 平衡的日志级别"
        ;;
    trace|verbose)
        DESC="详细跟踪配置 - 最高详细级别（可能产生大量日志）"
        ;;
    memory)
        DESC="内存调试配置 - 专注内存操作跟踪"
        ;;
    instruction|instr)
        DESC="指令调试配置 - 专注指令执行跟踪"
        ;;
    perf|performance)
        DESC="性能分析配置 - 最小开销"
        ;;
esac

# 显示配置信息
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}PTX-EMU 调试运行${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "${YELLOW}配置:${NC} $CONFIG_NAME"
echo -e "${YELLOW}说明:${NC} $DESC"
echo -e "${YELLOW}程序:${NC} $PROGRAM"
echo -e "${YELLOW}配置文件:${NC} $CONFIG_FILE"
echo -e "${BLUE}================================${NC}"
echo ""

# 运行程序
echo -e "${GREEN}开始运行...${NC}"
echo ""

cd "$PROJECT_ROOT"
$PROGRAM

# 显示日志文件位置
echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${YELLOW}运行完成${NC}"

# 检查是否生成了日志文件
LOG_FILE="$PROJECT_ROOT/ptx_emu_debug.log"
TRACE_FILE="$PROJECT_ROOT/ptx_emu_trace.log"
MEM_DEBUG_FILE="$PROJECT_ROOT/ptx_emu_memory_debug.log"
INSTR_DEBUG_FILE="$PROJECT_ROOT/ptx_emu_instr_debug.log"

if [ -f "$TRACE_FILE" ]; then
    echo -e "${YELLOW}日志文件:${NC} $TRACE_FILE"
    echo -e "  查看最新日志：${GREEN}tail -f $TRACE_FILE${NC}"
elif [ -f "$MEM_DEBUG_FILE" ]; then
    echo -e "${YELLOW}日志文件:${NC} $MEM_DEBUG_FILE"
    echo -e "  查看最新日志：${GREEN}tail -f $MEM_DEBUG_FILE${NC}"
elif [ -f "$INSTR_DEBUG_FILE" ]; then
    echo -e "${YELLOW}日志文件:${NC} $INSTR_DEBUG_FILE"
    echo -e "  查看最新日志：${GREEN}tail -f $INSTR_DEBUG_FILE${NC}"
elif [ -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}日志文件:${NC} $LOG_FILE"
    echo -e "  查看最新日志：${GREEN}tail -f $LOG_FILE${NC}"
else
    echo -e "${YELLOW}未生成日志文件（可能使用了 console 输出）${NC}"
fi

echo -e "${BLUE}================================${NC}"
