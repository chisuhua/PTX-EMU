#!/bin/bash
# PTX-EMU Sanity Test Suite
#
# Tiered test execution (simple -> complex):
#   Tier 1:  Smoke Tests                  - build verification
#   Tier 2:  Pure Data Structures         - memory, register (no scheduling)
#   Tier 3:  Single Instruction Tests     - simple PTX ops
#   Tier 4:  Warp/PC Scheduling Unit      - context, config (no execution)
#   Tier 5:  SIMT Execution Model         - exec_mask, SIMT stack, state
#   Tier 6:  Multi-Instruction Flows      - barrier, sync mechanisms
#   Tier 7:  Divergence & Reconvergence   - divergence scenarios
#   Tier 8:  Cross-Component Integration  - full warp flows
#   Tier 9:  PTX Syntax Validation        - static PTX parsing
#   Tier 10: Shared Memory 专项           - ld/st/cvta/barrier/dynamic
#   Tier 11: Benchmarks (--full only)     - mini/basic/cute/etc
#
# Flags:
#   --quick     Tiers 1-5 (smoke + simple + critical)
#   --full      All tiers 1-10 (includes benchmarks)
#   --ptx       Tier 9 only (PTX syntax)
#   --tier N    Run only tier N (1-10)
#   --verbose   Show full ctest output

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
TIER=""

FAILED_TESTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK=true; shift ;;
        --full) FULL=true; shift ;;
        --ptx) PTX_ONLY=true; shift ;;
        --tier) TIER="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift ;;
        --help)
            cat <<EOF
Usage: $0 [options]

Tiered test execution (simple -> complex):
  Tier 1:  Smoke Tests                  (build verification)
  Tier 2:  Pure Data Structures         (memory, register)
  Tier 3:  Single Instruction Tests     (PTX ops)
  Tier 4:  Warp/PC Scheduling Unit      (context, config)
  Tier 5:  SIMT Execution Model         (exec_mask, SIMT stack)
  Tier 6:  Multi-Instruction Flows      (barrier, sync)
  Tier 7:  Divergence & Reconvergence
  Tier 8:  Cross-Component Integration
  Tier 9:  PTX Syntax Validation
  Tier 10: Shared Memory 专项           (ld/st/cvta/barrier/dynamic)
  Tier 11: Benchmarks                   (--full only)

Flags:
  --quick         Run Tiers 1-5 (smoke + simple + critical)
  --full          Run all Tiers 1-10 (adds benchmarks)
  --ptx           Run Tier 9 only (PTX syntax)
  --tier N        Run only tier N (1-10)
  --verbose       Show full ctest output

Examples:
  $0 --quick                    # Tiers 1-5 (fast smoke check)
  $0                            # Tiers 1-9 (default)
  $0 --full                     # Tiers 1-10 (includes benchmarks)
  $0 --ptx                      # Tier 9 only
  $0 --tier 5                   # SIMT Execution Model only
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
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
        print_warn "ptxsim not built, building core libs..."
        cmake --build "$BUILD_DIR" --target ptxsim --target cudart --target ptx_parser -- -j$(nproc) 2>/dev/null || {
            cmake --build "$BUILD_DIR" --target ptxsim --target cudart -- -j$(nproc) || {
                print_fail "Build failed"; exit 1;
            }
        }
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

# Determine tier range based on flags
if [[ -n "$TIER" ]]; then
    if ! [[ "$TIER" =~ ^[0-9]+$ ]] || [[ $TIER -lt 1 ]] || [[ $TIER -gt 11 ]]; then
        echo "Invalid --tier value: $TIER (must be 1-11)"
        exit 1
    fi
    MIN_TIER=$TIER
    MAX_TIER=$TIER
elif [[ "$PTX_ONLY" == "true" ]]; then
    MIN_TIER=9
    MAX_TIER=9
elif [[ "$QUICK" == "true" ]]; then
    MIN_TIER=1
    MAX_TIER=5
else
    MIN_TIER=1
    MAX_TIER=10
fi

if [[ "$FULL" == "true" ]]; then
    MAX_TIER=11
fi

# Helper: returns 0 (true) if the given tier number is out of range
skip_tier() {
    local n=$1
    [[ $n -lt $MIN_TIER || $n -gt $MAX_TIER ]]
}

print_header "PTX-EMU Sanity Test Suite"
echo "Build: $BUILD_DIR | Mode: quick=$QUICK full=$FULL ptx=$PTX_ONLY tier=${TIER:-all}"
echo "Tiers: $MIN_TIER-$MAX_TIER"
echo ""

check_build

# Tier 1: Smoke Tests
if ! skip_tier 1; then
    print_header "Tier 1: Smoke Tests (build verification)"
    print_pass "Build directory verified: $BUILD_DIR"
fi

# Tier 2: Pure Data Structures
if ! skip_tier 2; then
    print_header "Tier 2: Pure Data Structures (memory, register)"
    run_regex_tests "test_memory_manager" "Memory manager"
    run_regex_tests "test_memory_bounds" "Memory bounds"
fi

# Tier 3: Single Instruction Tests
if ! skip_tier 3; then
    print_header "Tier 3: Single Instruction Tests (PTX ops)"
    run_label_tests "ptx" "PTX instructions (integer/float/bitwise/cvt/ld_st/cvta)"
    run_regex_tests "test_addc_subc_handler|test_ptx_bra" "Standalone instruction tests (ADD/SUBC/BRA)"
    run_regex_tests "test-ptx|test_printf" "Standalone PTX tests (generic + printf)"
fi

# Tier 4: Warp/PC Scheduling Unit
if ! skip_tier 4; then
    print_header "Tier 4: Warp/PC Scheduling Unit (context, config)"
    run_regex_tests "test_pc_management|test_pc_management_integrated|test_pc_management_advanced" "PC management (unit/advanced/integrated)"
    run_regex_tests "test_scheduler_config" "Scheduler config"
    run_regex_tests "test_warp_context" "WarpContext"
    run_regex_tests "test_warp_scheduler" "WarpScheduler"
    run_regex_tests "test_sm_context" "SMContext"
fi

# Tier 5: SIMT Execution Model
if ! skip_tier 5; then
    print_header "Tier 5: SIMT Execution Model (exec_mask, SIMT stack, state)"
    run_regex_tests "test_exec_mask" "BUG-001: exec_mask restore"
    run_regex_tests "test_simt_stack_entry" "BUG-002: SIMT stack exit"
    run_regex_tests "test_active_mask_consistency" "ISSUE-004: active_mask consistency"
    run_regex_tests "test_specific_bugs_unit" "Specific bugs unit tests"
    run_regex_tests "test_simt|test_simt_stack_entry_integrated|test_simt_thread_pc|test_simt_thread_pc_integrated|unit_simt_integration" "SIMT Stack tests (core + integrated)"
    run_regex_tests "unit_handle_branch" "Branch + SIMT integration (unit)"
    run_regex_tests "test_warp_state|test_warp_state_integrated" "Warp state (unit + integrated)"
    run_regex_tests "unit_exec_layer_e1_e3|unit_exec_integration_h1_h4" "Execution layer hypotheses (E1-E3/H1-H4, unit)"
    run_regex_tests "unit_ret_handler_divergent" "BUG-RETHANG: ret handler on divergent warp"
fi

# Tier 6: Multi-Instruction Flows (barrier, sync)
if ! skip_tier 6; then
    print_header "Tier 6: Multi-Instruction Flows (barrier, sync)"
    run_regex_tests "test_barrier_reconvergence" "Barrier reconvergence"
    run_regex_tests "unit_syncthreads_direction" "Syncthreads direction (unit)"
    run_regex_tests "unit_barrier_simt" "Barrier SIMT integration (unit)"
    run_regex_tests "test_barrier_scenarios|test_barrier_scenarios_integrated" "Barrier scenarios (unit + integrated)"
    run_regex_tests "test_barrier_verification|test_barrier_verification_integrated" "Barrier verification (unit + integrated)"
    run_regex_tests "test_barrier_pc" "Barrier PC overwrite protection"
    run_regex_tests "test_barrier_active_mask" "Barrier active_mask preserved"
    run_regex_tests "test_warp_barrier_integrated|test_warp_barrier_extended|test_post_barrier_divergence|unit_barrier_interaction" "Barrier warp/interaction (integrated + unit)"
    run_regex_tests "test_sync_mechanism|test_sync_mechanism_integrated" "Sync mechanism (unit + integrated)"
fi

# Tier 7: Divergence & Reconvergence
if ! skip_tier 7; then
    print_header "Tier 7: Divergence & Reconvergence"
    run_regex_tests "test_divergence_sync_standalone" "Divergence + barrier sync (standalone)"
fi

# Tier 8: Cross-Component Integration
# (added 2026-06-07 per docs/superpowers/specs/2026-06-07-ptx-emu-tier8-design.md)
if ! skip_tier 8; then
    print_header "Tier 8: Cross-Component Integration (full warp flows)"
    run_regex_tests "integration_barrier_full_lifecycle" "Barrier full lifecycle (init/arrive/release/reset)"
fi

# Tier 9: PTX Syntax Validation
if ! skip_tier 9; then
    print_header "Tier 9: PTX Syntax Validation"
    run_ptx_syntax_test
fi

# Tier 10: Shared Memory 专项
if ! skip_tier 10; then
    print_header "Tier 10: Shared Memory 专项 (ld/st/cvta/barrier/dynamic)"
    run_label_tests "shared_memory" "Shared Memory (ld/st/cvta/barrier/dynamic)"
fi

# Tier 11: Benchmarks (--full only)
if ! skip_tier 11; then
    print_header "Tier 11: Benchmarks (--full only)"
    run_label_tests "mini" "Mini test suite"
    run_label_tests "basic" "Basic test suite (GEMM/CONV)"
    run_label_tests "cute" "CuTE test suite"
    run_label_tests "three_mode" "Three-mode test suite"
    run_regex_tests "test_syncthreads$|test_warp_divergence$|test_shared_memory$" "Sync bench tests (CUDA)"
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
