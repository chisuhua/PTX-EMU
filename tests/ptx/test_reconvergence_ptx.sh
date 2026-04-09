#!/bin/bash
# ============================================================================
# SIMT v2.0 Reconvergence PC Test Script
# ============================================================================
# This script tests CFG analysis and reconvergence_pc computation
# during the PTX parser phase (before kernel loading)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BUILD_DIR="$PROJECT_ROOT/build"
TEST_PTX="$SCRIPT_DIR/test_reconvergence.ptx"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================================"
echo "SIMT v2.0 Reconvergence PC Parser Test"
echo "============================================================================"
echo ""

# Step 1: Check if test PTX file exists
echo -n "Step 1: Checking test PTX file... "
if [ ! -f "$TEST_PTX" ]; then
    echo -e "${RED}FAILED${NC}"
    echo "Error: Test PTX file not found: $TEST_PTX"
    exit 1
fi
echo -e "${GREEN}OK${NC}"
echo "  File: $TEST_PTX"

# Step 2: Compile test program
echo ""
echo -n "Step 2: Compiling test program... "
TEST_CPP="$SCRIPT_DIR/test_reconvergence_parser.cpp"
TEST_BIN="$BUILD_DIR/test_reconvergence_parser"

if [ ! -f "$TEST_CPP" ]; then
    echo -e "${RED}FAILED${NC}"
    echo "Error: Test C++ file not found: $TEST_CPP"
    exit 1
fi

# Compile the test
cd "$PROJECT_ROOT"
cmake --build "$BUILD_DIR" --target test_reconvergence_parser 2>&1 | tail -5

if [ $? -eq 0 ] && [ -f "$TEST_BIN" ]; then
    echo -e "${GREEN}OK${NC}"
    echo "  Binary: $TEST_BIN"
else
    echo -e "${YELLOW}COMPILATION NEEDED${NC}"
    echo "Attempting to compile manually..."
    g++ -std=c++17 \
        -I"$PROJECT_ROOT/include" \
        -I"$BUILD_DIR/antlr4_generated" \
        -L"$BUILD_DIR/lib" \
        "$TEST_CPP" \
        -lptx_parser -lptx_ir -lantlr4-runtime \
        -o "$TEST_BIN" 2>&1
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}COMPILATION FAILED${NC}"
        echo "This is expected if CFG Builder is not yet integrated."
        echo "Please complete Phase 5.1 (CFG Builder compilation) first."
        exit 1
    fi
    echo -e "${GREEN}OK${NC}"
fi

# Step 3: Run test
echo ""
echo "Step 3: Running reconvergence_pc test..."
echo "============================================================================"
"$TEST_BIN" "$TEST_PTX"
TEST_RESULT=$?
echo "============================================================================"

# Step 4: Report result
echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}=== TEST PASSED ===${NC}"
    echo "CFG analysis successfully computed reconvergence_pc for all branches."
    exit 0
else
    echo -e "${RED}=== TEST FAILED ===${NC}"
    echo "CFG analysis failed to compute correct reconvergence_pc."
    echo ""
    echo "Possible causes:"
    echo "1. CFG Builder not compiled (check Phase 5.1)"
    echo "2. CFG not integrated into parser (check Phase 5.2)"
    echo "3. Post-Dominator analysis bug"
    echo ""
    echo "Next step: Run 'Phase 5.1' first to ensure CFG Builder compiles."
    exit 1
fi
