#!/bin/bash
# PTX-EMU 回归测试 — 同步执行模式
#
# 同一份 binary (BUILD_LIB_CPPTLM_CUDART=ON)
# 设置 EMU_NO_BRIDGE=1 → g_cpptlm_bridge == nullptr → 同步路径
#
# 用法: ./scripts/regression.sh [--no-build] [--quick]

set -e
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
JOBS=$(nproc)
NO_BUILD=false; QUICK=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build) NO_BUILD=true; shift ;;
        --quick) QUICK=true; shift ;;
        -j*) JOBS="${1#-j}"; shift ;;
        --help) echo "Standard (sync) regression test (EMU_NO_BRIDGE=1)"; exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} PTX-EMU 标准回归 (同步路径 EMU_NO_BRIDGE=1)${NC}"
echo -e "${CYAN}========================================${NC}"

[[ "$NO_BUILD" == "false" ]] && {
    cd "$ROOT_DIR"; . env.sh > /dev/null 2>&1
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_LIB_CPPTLM_CUDART=ON > /dev/null 2>&1
    cmake --build "$BUILD_DIR" -j"$JOBS" > /dev/null 2>&1
}
export EMU_NO_BRIDGE=1  # 禁用 bridge → 同步路径
FAILED=0
run_test() { local d="$1"; shift; echo -n "  $d ... "
    if EMU_NO_BRIDGE=1 ctest --test-dir "$BUILD_DIR" "$@" -Q 2>&1; then echo -e "${GREEN}PASS${NC}"
    else echo -e "${RED}FAIL${NC}"; FAILED=$((FAILED+1)); fi; }

echo -e "${CYAN}[TEST] 单元测试${NC}"; run_test unit -L unit
echo -e "${CYAN}[TEST] 集成测试${NC}"; run_test integration -L integration
echo -e "${CYAN}[TEST] E2E (同步路径)${NC}"; run_test e2e -L e2e -E 'e2e_divergence$'
echo -e "${CYAN}[TEST] PTX语法${NC}"
echo -n "  test_all_ptx.sh ... "
bash "$ROOT_DIR/tests/ptx/test_all_ptx.sh" 2>&1 | tail -1 | grep -q "全部通过" && echo -e "${GREEN}PASS${NC}" || { echo -e "${RED}FAIL${NC}"; FAILED=$((FAILED+1)); }
[[ "$QUICK" != "true" ]] && { echo -e "${CYAN}[TEST] Benchmarks${NC}"; run_test mini -L mini; run_test cute -L cute; }

echo ""
[[ $FAILED -eq 0 ]] && echo -e "${GREEN}标准回归全部通过 ✓${NC}" || { echo -e "${RED}失败: $FAILED 项${NC}"; exit 1; }
