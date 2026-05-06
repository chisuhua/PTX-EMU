#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

BUILD_DIR="${BUILD_DIR:-build}"
VERBOSE=false
QUICK=false
FULL=false
PTX_ONLY=false

FAILED_TESTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK=true; shift ;;
        --full) FULL=true; shift ;;
        --ptx) PTX_ONLY=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --quick    Quick check (critical bugs only)"
            echo "  --full     Full check (includes benchmark)"
            echo "  --ptx      PTX syntax test only"
            echo "  --verbose  Verbose output"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

print_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

print_test() { echo -e "${BLUE}[TEST]${NC} $1"; }
print_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_section() { echo -e "\n${YELLOW}--- $1 ---${NC}"; }

check_build() {
    if [[ ! -d "$BUILD_DIR" ]]; then
        print_fail "Build dir not found: $BUILD_DIR"
        exit 1
    fi
    if [[ ! -f "$BUILD_DIR/bin/ptxsim" ]]; then
        print_warn "ptxsim not built, building..."
        cmake --build "$BUILD_DIR" || { print_fail "Build failed"; exit 1; }
    fi
}

run_regex_tests() {
    local regex="$1"
    local description="$2"

    print_test "$description (regex: $regex)"

    if [[ "$VERBOSE" == "true" ]]; then
        ctest --test-dir "$BUILD_DIR" -R "$regex" --output-on-failure 2>&1
    else
        ctest --test-dir "$BUILD_DIR" -R "$regex" -Q 2>&1
    fi

    if [[ $? -eq 0 ]]; then
        print_pass "$description"
        return 0
    else
        print_fail "$description"
        FAILED_TESTS+=("$description")
        return 1
    fi
}

run_label_tests() {
    local label="$1"
    local description="$2"

    print_test "$description (label: $label)"

    if [[ "$VERBOSE" == "true" ]]; then
        ctest --test-dir "$BUILD_DIR" -L "$label" --output-on-failure 2>&1
    else
        ctest --test-dir "$BUILD_DIR" -L "$label" -Q 2>&1
    fi

    if [[ $? -eq 0 ]]; then
        print_pass "$description"
        return 0
    else
        print_fail "$description"
        FAILED_TESTS+=("$description")
        return 1
    fi
}

run_ptx_syntax_test() {
    local ptx_script="$(dirname "$0")/../tests/ptx/test_all_ptx.sh"
    if [[ ! -f "$ptx_script" ]]; then
        print_fail "PTX script not found: $ptx_script"
        return 1
    fi

    print_test "PTX syntax test (test_all_ptx.sh)"
    if [[ "$VERBOSE" == "true" ]]; then
        bash "$ptx_script"
    else
        bash "$ptx_script" 2>&1 | tail -30
    fi

    if [[ $? -eq 0 ]]; then
        print_pass "PTX syntax test"
        return 0
    else
        print_fail "PTX syntax test"
        FAILED_TESTS+=("PTX syntax test")
        return 1
    fi
}

print_header "PTX-EMU Sanity Test Suite"
echo "Build: $BUILD_DIR | Mode: quick=$QUICK full=$FULL ptx=$PTX_ONLY"
echo ""

check_build

if [[ "$QUICK" == "true" ]]; then
    print_header "Quick: Critical Bug Verification"
    run_regex_tests "test_exec_mask" "BUG-001: exec_mask restore"
    run_regex_tests "test_simt_stack_entry" "BUG-002: SIMT stack exit handling"
    run_regex_tests "test_active_mask_consistency" "ISSUE-004: active_mask consistency"
    run_regex_tests "test_divergence_sync_standalone" "DIVERGENCE: divergence + barrier sync"
    print_header "Quick check done. Failed: ${#FAILED_TESTS[@]}"
    exit ${#FAILED_TESTS[@]}
fi

if [[ "$PTX_ONLY" == "true" ]]; then
    run_ptx_syntax_test
    exit ${#FAILED_TESTS[@]}
fi

print_header "1. Critical Bug Fixes"
run_regex_tests "test_exec_mask" "BUG-001: exec_mask restore"
run_regex_tests "test_simt_stack_entry" "BUG-002: SIMT stack exit"
run_regex_tests "test_active_mask_consistency" "ISSUE-004: active_mask consistency"
run_regex_tests "test_specific_bugs_unit" "Specific bugs unit tests"

print_header "2. SIMT Execution Model"
run_regex_tests "test_simt" "SIMT Stack tests"
run_regex_tests "test_handle_branch" "Branch + SIMT integration"
run_regex_tests "test_warp_state" "Warp state"

print_header "3. Barrier Sync"
run_regex_tests "test_barrier_reconvergence" "Barrier reconvergence"
#run_regex_tests "test_syncthreads_test3" "Test 3 deadlock reproduction"
run_regex_tests "test_barrier_simt" "Barrier SIMT integration"
run_regex_tests "test_full_barrier" "Full barrier execution"
run_regex_tests "test_barrier_scenarios" "Barrier scenarios (warp_sync/divergence/PC protection)"
run_regex_tests "test_divergence_sync_standalone" "Divergence + barrier sync (standalone)"

print_header "4. Memory Management"
run_regex_tests "test_memory_manager" "Memory manager"
run_regex_tests "test_memory_bounds" "Memory bounds"

print_header "5. PTX Instructions"
run_label_tests "ptx" "PTX instructions (integer/float/bitwise/cvt/ld_st/cvta)"

print_header "6. Warp Scheduling"
run_regex_tests "test_warp_context" "WarpContext"
run_regex_tests "test_warp_scheduler" "WarpScheduler"
run_regex_tests "test_sm_context" "SMContext"

print_header "7. PTX Syntax Test"
run_ptx_syntax_test

if [[ "$FULL" == "true" ]]; then
    print_header "8. Benchmark Tests"
    run_label_tests "mini" "Mini test suite"
    run_label_tests "basic" "Basic test suite (GEMM/CONV)"
fi

print_header "Summary"
if [[ ${#FAILED_TESTS[@]} -eq 0 ]]; then
    print_pass "All tests passed!"
    exit 0
else
    print_fail "${#FAILED_TESTS[@]} test(s) failed:"
    for t in "${FAILED_TESTS[@]}"; do echo -e "  ${RED}x${NC} $t"; done
    exit 1
fi
