#!/bin/bash
# PTX-EMU 回归测试 — 标准同步模式（默认）
#
# 同一份 binary，默认走同步路径（g_cpptlm_bridge == nullptr）
# EMU_COSIM=1 激活协同仿真 → 见 scripts/regression-cosim.sh
#
# 用法:
#   ./scripts/regression.sh              # 自动构建 + 全量回归
#   ./scripts/regression.sh --no-build   # 跳过构建
#   ./scripts/regression.sh --quick      # 快速模式（跳过 benchmarks）

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
JOBS=$(nproc)
NO_BUILD=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build) NO_BUILD=true; shift ;;
        --quick)    QUICK=true; shift ;;
        -j*)        JOBS="${1#-j}"; shift ;;
        --help|-h)
            cat <<HELP
用法: $0 [选项]

选项:
  --no-build    跳过构建（使用现有 build 目录）
  --quick       快速模式（跳过 benchmarks）
  -j N          并行构建线程数（默认: nproc）

说明:
  默认同步模式 — 不设置任何环境变量
  g_cpptlm_bridge == nullptr → cudaLaunchKernel 走同步执行路径
HELP
            exit 0 ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} PTX-EMU 回归测试 — 同步模式（默认）${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if [[ "$NO_BUILD" == "true" ]]; then
    echo -e "${GREEN}[SKIP]${NC} 跳过构建"
else
    echo -e "${CYAN}[BUILD]${NC} 构建 PTX-EMU..."
    cd "$ROOT_DIR"
    . env.sh > /dev/null 2>&1
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    cmake --build "$BUILD_DIR" -j"$JOBS" > /dev/null 2>&1
    echo -e "${GREEN}[BUILD]${NC} 构建完成"
fi
echo ""

unset EMU_COSIM

FAILED=0
run_test() {
    local desc="$1"; shift
    echo -n "  $desc ... "
    if ctest --test-dir "$BUILD_DIR" "$@" -Q 2>&1; then
        echo -e "${GREEN}PASS${NC}"
    else
        echo -e "${RED}FAIL${NC}"
        FAILED=$((FAILED + 1))
    fi
}

echo -e "${CYAN}[TEST] 单元测试 (88)${NC}"
run_test "unit" -L unit

echo -e "\n${CYAN}[TEST] 集成测试${NC}"
run_test "integration" -L integration

echo -e "\n${CYAN}[TEST] E2E 测试 (同步路径)${NC}"
run_test "e2e (excl. SingletonGuard)" -L e2e -E 'e2e_divergence$'

echo -e "\n${CYAN}[TEST] PTX 语法测试${NC}"
echo -n "  test_all_ptx.sh ... "
if bash "$ROOT_DIR/tests/ptx/test_all_ptx.sh" 2>&1 | tail -1 | grep -q "全部通过"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    FAILED=$((FAILED + 1))
fi

if [[ "$QUICK" != "true" ]]; then
    echo -e "\n${CYAN}[TEST] Benchmark 测试${NC}"
    run_test "mini"  -L mini
    run_test "cute"  -L cute
fi

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} 标准回归全部通过 ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED} 失败: ${FAILED} 项${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi