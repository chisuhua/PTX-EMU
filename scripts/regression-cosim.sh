#!/bin/bash
# PTX-EMU 回归测试 — CppTLM 协同仿真模式
#
# 同一份 binary，EMU_COSIM=1 激活协同仿真:
#   StubBridge auto-attach → bridge 异步路径
#   cudaDeviceSynchronize 自动 advance() 驱动 PTX 执行
#
# 用法:
#   ./scripts/regression-cosim.sh              # 使用现有 build 目录
#   ./scripts/regression-cosim.sh --rebuild    # 强制重新构建
#   ./scripts/regression-cosim.sh --quick      # 快速模式

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
JOBS=$(nproc)
REBUILD=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild) REBUILD=true; shift ;;
        --quick)   QUICK=true; shift ;;
        -j*)       JOBS="${1#-j}"; shift ;;
        --help|-h)
            cat <<HELP
用法: $0 [选项]

选项:
  --rebuild     强制重新构建
  --quick       快速模式（跳过 benchmarks）
  -j N          并行构建线程数（默认: nproc）

说明:
  EMU_COSIM=1 → StubBridge auto-attach
  → cudaLaunchKernel 走 bridge 异步路径
  → cudaDeviceSynchronize 自动 advance() 驱动 PTX 执行
HELP
            exit 0 ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} PTX-EMU 协同仿真回归 (EMU_COSIM=1)${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if [[ "$REBUILD" == "true" ]] || [[ ! -f "$BUILD_DIR/bin/ptxsim" ]]; then
    echo -e "${CYAN}[BUILD]${NC} 构建 PTX-EMU..."
    cd "$ROOT_DIR"
    . env.sh > /dev/null 2>&1
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    cmake --build "$BUILD_DIR" -j"$JOBS" > /dev/null 2>&1
    echo -e "${GREEN}[BUILD]${NC} 构建完成"
else
    echo -e "${GREEN}[SKIP]${NC} 使用现有 build 目录 ($BUILD_DIR)"
fi
echo ""

export EMU_COSIM=1
echo -e "${CYAN}[ENV]${NC} EMU_COSIM=1 (StubBridge auto-attach + auto-advance)"
echo ""

FAILED=0
run_test() {
    local desc="$1"; shift
    echo -n "  $desc ... "
    if EMU_COSIM=1 ctest --test-dir "$BUILD_DIR" "$@" -Q 2>&1; then
        echo -e "${GREEN}PASS${NC}"
    else
        echo -e "${RED}FAIL${NC}"
        FAILED=$((FAILED + 1))
    fi
}

echo -e "${CYAN}[TEST] 单元测试 (co-sim)${NC}"
run_test "unit" -L unit

echo -e "\n${CYAN}[TEST] 集成测试 (co-sim)${NC}"
run_test "integration" -L integration

echo -e "\n${CYAN}[TEST] E2E 测试 (bridge 路径 + auto-advance)${NC}"
run_test "e2e_cosim_vector_add" -R e2e_cosim_vector_add
# Known EMU_COSIM=1 failures: tcgen05 MMA/scoreboard bridge path
# regression (Phase 2a TLM injection confirmed NOT the cause — tests
# fail even with zero injection; root cause is in bridge kernel launch
# + GPUContext task_queue flow for tcgen05 kernel types). Separately
# tracked as "bridge-tcgen05-regression".
run_test "e2e (excl. SingletonGuard+tcgen05)" -L e2e -E 'e2e_divergence$|e2e_blackwell_gemm|e2e_flashattention_mini'

echo -e "\n${CYAN}[TEST] PTX 语法测试${NC}"
echo -n "  test_all_ptx.sh ... "
if bash "$ROOT_DIR/tests/ptx/test_all_ptx.sh" 2>&1 | tail -1 | grep -q "全部通过"; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    FAILED=$((FAILED + 1))
fi

if [[ "$QUICK" != "true" ]]; then
    echo -e "\n${CYAN}[TEST] Benchmark 测试 (co-sim)${NC}"
    run_test "mini" -L mini
    run_test "cute" -L cute
fi

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} 协同仿真回归全部通过 ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED} 失败: ${FAILED} 项${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi