#!/bin/bash
# PTX-EMU 回归测试 — CppTLM 协同仿真模式（默认模式）
#
# 同一份 binary: BUILD_LIB_CPPTLM_CUDART=ON
# 默认自动 StubBridge + auto-advance
# 设置 EMU_NO_BRIDGE=1 回退到同步路径
#
# 用法: ./scripts/regression-cosim.sh [--rebuild] [--quick] [-j N]

set -e
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
JOBS=$(nproc)
REBUILD=false; QUICK=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild) REBUILD=true; shift ;;
        --quick) QUICK=true; shift ;;
        -j*) JOBS="${1#-j}"; shift ;;
        --help) echo "Co-sim regression test (default StubBridge mode)"; exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} PTX-EMU 协同仿真回归 (默认 bridge 路径)${NC}"
echo -e "${CYAN}========================================${NC}"

[[ "$REBUILD" == "true" ]] || [[ ! -f "$BUILD_DIR/bin/ptxsim" ]] && {
    cd "$ROOT_DIR"; . env.sh > /dev/null 2>&1
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_LIB_CPPTLM_CUDART=ON > /dev/null 2>&1
    cmake --build "$BUILD_DIR" -j"$JOBS" > /dev/null 2>&1
}
export EMU_NO_BRIDGE=  # unset: default co-sim mode
FAILED=0
run_test() { local d="$1"; shift; echo -n "  $d ... "
    if ctest --test-dir "$BUILD_DIR" "$@" -Q 2>&1; then echo -e "${GREEN}PASS${NC}"
    else echo -e "${RED}FAIL${NC}"; FAILED=$((FAILED+1)); fi; }

echo -e "\n${CYAN}[TEST] 单元 + 集成测试${NC}"
run_test unit -L unit; run_test integration -L integration
echo -e "\n${CYAN}[TEST] E2E (bridge路径)${NC}"
run_test "e2e_cosim_vector_add" -R e2e_cosim_vector_add
run_test "e2e (其他)" -L e2e -E 'e2e_divergence$'
echo -e "\n${CYAN}[TEST] PTX语法${NC}"
echo -n "  test_all_ptx.sh ... "
bash "$ROOT_DIR/tests/ptx/test_all_ptx.sh" 2>&1 | tail -1 | grep -q "全部通过" && echo -e "${GREEN}PASS${NC}" || { echo -e "${RED}FAIL${NC}"; FAILED=$((FAILED+1)); }
[[ "$QUICK" != "true" ]] && { echo -e "\n${CYAN}[TEST] Benchmarks${NC}"; run_test mini -L mini; run_test cute -L cute; }

echo ""
[[ $FAILED -eq 0 ]] && echo -e "${GREEN}协同仿真回归全部通过 ✓${NC}" || { echo -e "${RED}失败: $FAILED 项${NC}"; exit 1; }
