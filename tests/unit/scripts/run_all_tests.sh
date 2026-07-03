#!/usr/bin/env bash
# run_all_tests.sh — Aggregate runner for tests/unit/scripts/
#
# Iterates over all test_check_*.py files, runs each as subprocess.
# Exits 0 only if ALL pass.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0
FAILED_TESTS=()

for test_file in "$SCRIPT_DIR"/test_check_*.py; do
    name=$(basename "$test_file" .py)
    echo "Running $name ..."
    if python3 "$test_file"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$name")
    fi
done

echo ""
echo "=== Aggregate Results ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"
if [[ $FAIL -gt 0 ]]; then
    echo "FAILED: ${FAILED_TESTS[*]}"
    exit 1
fi
echo "All tests passed"
exit 0
