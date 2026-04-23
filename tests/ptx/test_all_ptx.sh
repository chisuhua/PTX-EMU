#!/bin/bash
# PTX 批量解析测试脚本
# 测试 tests/ptx/ 目录下所有 .ptx 文件

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTX_DIR="${SCRIPT_DIR}"
TEST_BIN="${PTX_DIR}/../../build/bin/test-ptx"

echo "========================================"
echo "PTX 批量解析测试"
echo "========================================"
echo "测试目录：${PTX_DIR}"
echo "测试程序：${TEST_BIN}"
echo ""

# 检查 test-ptx 是否存在
if [ ! -x "${TEST_BIN}" ]; then
    echo "[ERROR] test-ptx 不存在或未构建"
    echo "请先执行：cd build && cmake --build ."
    exit 1
fi

# 检查 preprocessor 是否存在
PREPROCESSOR="${SCRIPT_DIR}/ptx_preprocess.py"
if [ ! -x "${PREPROCESSOR}" ]; then
    echo "[ERROR] ptx_preprocess.py 不存在"
    exit 1
fi

# 统计
TOTAL=0
PASSED=0
FAILED=0
FAILED_FILES=()

# 遍历所有 PTX 文件
for ptx_file in "${PTX_DIR}"/*.ptx; do
    if [ ! -f "${ptx_file}" ]; then
        echo "[WARN] 未找到 PTX 文件"
        exit 0
    fi
    
    TOTAL=$((TOTAL + 1))
    filename=$(basename "${ptx_file}")
    
    echo -n "测试 ${filename} ... "
    
    # 预处理 PTX 文件
    preprocessed_file=$(mktemp "/tmp/ptx_preprocessed_XXXXXX.ptx")
    python3 "${PREPROCESSOR}" "${ptx_file}" "${preprocessed_file}" 2>/dev/null
    
    # 捕获输出以检测 segfault
    output=$(PTX_EMU_PATH="${PTX_DIR}/.." "${TEST_BIN}" "${preprocessed_file}" 2>&1) || true
    
    rm -f "${preprocessed_file}"
    
    if echo "${output}" | grep -q "PASS"; then
        echo "[PASS]"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL]"
        FAILED=$((FAILED + 1))
        FAILED_FILES+=("${filename}")
        echo "  错误输出:"
        echo "${output}" | head -20 | sed 's/^/    /'
    fi
done

echo ""
echo "========================================"
echo "测试结果汇总"
echo "========================================"
echo "总数：${TOTAL}"
echo "通过：${PASSED}"
echo "失败：${FAILED}"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "失败文件列表:"
    for f in "${FAILED_FILES[@]}"; do
        echo "  - ${f}"
    done
    echo ""
    echo "[FAIL] 测试失败！"
    exit 1
else
    echo ""
    echo "[SUCCESS] 全部通过！"
    exit 0
fi
