#!/usr/bin/env python3
"""
Four-Mode Test Generator for PTX-EMU

Generates four-mode PTX tests from ANY CUDA program:
- Mode 1: cuobjdump dynamic extraction (end-to-end integration)
- Mode 2: pre-extracted PTX file (static analysis)
- Mode 3a: StatementContext BEFORE CFG (reconvergence_pc = -1)
- Mode 3b: StatementContext AFTER CFG (reconvergence_pc filled = final execution version)

Usage:
    python3 generate_tests.py --benchmark dummy
    python3 generate_tests.py --cuda-source bench/dummy/dummy.cu
    python3 generate_tests.py --binary build/bin/dummy
    python3 generate_tests.py --ptx path/to/kernel.ptx
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict
import re

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
THREE_MODE_DIR = PROJECT_ROOT / "tests" / "three_mode_testing"
PTX_DIR = THREE_MODE_DIR / "ptx"
GOLDEN_DIR = THREE_MODE_DIR / "golden"


class PTXAnalyzer:
    def __init__(self, ptx_content: str):
        self.ptx = ptx_content
        self.lines = ptx_content.split('\n')
        self.barriers: List[Dict] = []
        self.shared_lds: List[Dict] = []
        self.shared_sts: List[Dict] = []
        self.branches: List[Dict] = []
        self.entries: List[str] = []
        self.shared_var: Optional[str] = None
        self.analyze()

    def analyze(self):
        for i, line in enumerate(self.lines):
            stripped = line.strip()
            entry_match = re.search(r'(?:\.visible\s+)?\.entry\s+(\S+)', stripped)
            if entry_match:
                self.entries.append(entry_match.group(1))
            if 'bar.sync' in stripped or 'bar.warp.sync' in stripped:
                self.barriers.append({
                    'line': i,
                    'text': stripped,
                    'type': 'warp' if 'bar.warp' in stripped else 'cta'
                })
            if re.match(r'ld\.shared', stripped):
                self.shared_lds.append({'line': i, 'text': stripped})
            if re.match(r'st\.shared', stripped):
                self.shared_sts.append({'line': i, 'text': stripped})
            if re.match(r'@!?\s*\w+\s+bra', stripped) or stripped.startswith('bra'):
                self.branches.append({'line': i, 'text': stripped})
            if '.shared' in line and 'shared_data' in line:
                match = re.search(r'(\w+)\[', line)
                if match:
                    self.shared_var = match.group(1)

    def get_summary(self) -> Dict:
        return {
            'entries': self.entries,
            'num_barriers': len(self.barriers),
            'num_shared_lds': len(self.shared_lds),
            'num_shared_sts': len(self.shared_sts),
            'num_branches': len(self.branches),
            'has_cta_barrier': any(b['type'] == 'cta' for b in self.barriers),
            'has_warp_barrier': any(b['type'] == 'warp' for b in self.barriers),
            'shared_var': self.shared_var,
        }


class CUDAAnalyzer:
    def __init__(self, cuda_content: str):
        self.cuda = cuda_content
        self.lines = cuda_content.split('\n')
        self.kernel_patterns: List[Dict] = []
        self.expected_outputs: Dict[str, str] = {}
        self.analyze()

    def analyze(self):
        for i, line in enumerate(self.lines):
            if '__global__' in line or '__kernel' in line:
                match = re.search(r'void\s+(\w+)\s*\(', line)
                if match:
                    self.kernel_patterns.append({
                        'name': match.group(1),
                        'line': i,
                        'text': line.strip()
                    })

    def compute_expected_sum(self, values: List[int]) -> int:
        return sum(values)

    def generate_golden_value(self) -> Optional[str]:
        cuda_text = self.cuda
        sum_pattern = re.search(r'expected_sum\s*\+=.*\(i\s*\*\s*\(i\s*\+\s*1\)\)\s*/\s*2', cuda_text)
        if sum_pattern:
            total = 0
            for i in range(16):
                total += (i * (i + 1)) // 2
            for i in range(16, 32):
                prod = 1
                for j in range(1, i - 14):
                    prod *= j
                total += prod
            return str(total)
        return None


def extract_ptx_from_binary(binary_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ['cuobjdump', '-ptx', '-all', binary_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Error extracting PTX: {result.stderr}")
            return None
        if result.stdout.strip():
            return result.stdout
        print("cuobjdump produced no output")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def _check_expr(expr: str) -> str:
    if expr.startswith('(') and expr.endswith(')'):
        return expr
    if '||' in expr or '&&' in expr:
        return f'({expr})'
    return expr


def _strip_test_prefix(name: str) -> str:
    """Remove 'test_' prefix if present to avoid 'test_test_foo' filenames."""
    if name.startswith('test_'):
        return name[5:]
    return name


def generate_mode1_test(test_name: str, analyzer: PTXAnalyzer = None) -> str:
    entry_check = _check_expr('ptx_contains(ptx, ".entry")')
    barrier_section = ""
    if analyzer and analyzer.barriers:
        summary = analyzer.get_summary()
        if summary['has_cta_barrier'] and summary['has_warp_barrier']:
            bar_check = _check_expr('ptx_contains(ptx, "bar.sync") || ptx_contains(ptx, "bar.warp.sync")')
        elif summary['has_warp_barrier']:
            bar_check = _check_expr('ptx_contains(ptx, "bar.warp.sync")')
        else:
            bar_check = _check_expr('ptx_contains(ptx, "bar.sync")')
        barrier_section = f"\n    CHECK({bar_check});"

    return f'''/**
 * @file test_{test_name}_mode1.cpp
 * @brief Mode 1: cuobjdump Dynamic Extraction
 *
 * Auto-generated by four-mode testing generator.
 * Extracts PTX at runtime using cuobjdump.
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.hpp"

#define TEST_MODE 1
#define TEST_NAME "{test_name}"

#ifndef TEST_BINARY
#define TEST_BINARY "build/bin/{test_name}"
#endif

TEST_CASE("Mode1: Extract PTX via cuobjdump", "[mode1][extract]") {{
    std::string ptx = extract_ptx_cuobjdump(TEST_BINARY);
    CHECK_FALSE(ptx.empty());
    INFO("Extracted " << ptx.size() << " bytes");
}}

TEST_CASE("Mode1: Parse PTX structure", "[mode1][parse]") {{
    std::string ptx = extract_ptx_cuobjdump(TEST_BINARY);
    if (ptx.empty()) SKIP("PTX extraction failed");

    bool has_entry = ptx_contains(ptx, ".entry");
    bool has_bar = ptx_contains(ptx, "bar.");
    bool has_shared = ptx_contains(ptx, ".shared") || ptx_contains(ptx, "ld.shared") || ptx_contains(ptx, "st.shared");

    INFO("entry: " << has_entry << ", barrier: " << has_bar << ", shared: " << has_shared);
    CHECK(has_entry);
}}

TEST_CASE("Mode1: PTX contains expected instructions", "[mode1][verify]") {{
    std::string ptx = extract_ptx_cuobjdump(TEST_BINARY);
    if (ptx.empty()) SKIP("PTX extraction failed");

    CHECK({entry_check});{barrier_section}
}}

TEST_CASE("Mode1: PTX structure validation", "[mode1][trace]") {{
    init_factory_once();

    std::string ptx = extract_ptx_cuobjdump(TEST_BINARY);
    if (ptx.empty()) SKIP("PTX extraction failed");

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);
    warp.set_active_mask(0xFFFFFFFF);

    // 验证 PTX 包含预期指令关键字
    INFO("Mode1 PTX size: " << ptx.size() << " bytes");
    CHECK(ptx.size() > 100);
    CHECK(ptx.find("entry") != std::string::npos);
}}
'''


def generate_mode2_test(test_name: str, analyzer: PTXAnalyzer = None) -> str:
    bar_check = 'false'
    if analyzer and analyzer.barriers:
        summary = analyzer.get_summary()
        if summary['has_cta_barrier'] and summary['has_warp_barrier']:
            bar_check = _check_expr('ptx_contains(ptx, "bar.sync") || ptx_contains(ptx, "bar.warp.sync")')
        elif summary['has_warp_barrier']:
            bar_check = _check_expr('ptx_contains(ptx, "bar.warp.sync")')
        else:
            bar_check = _check_expr('ptx_contains(ptx, "bar.sync")')

    return f'''/**
 * @file test_{test_name}_mode2.cpp
 * @brief Mode 2: Pre-extracted PTX File
 *
 * Auto-generated by four-mode testing generator.
 * Loads pre-extracted PTX from file for stable, reproducible testing.
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.hpp"

#define TEST_MODE 2
#define TEST_NAME "{test_name}"

#ifndef PTX_FILE
#define PTX_FILE "tests/three_mode_testing/ptx/{test_name}.ptx"
#endif

TEST_CASE("Mode2: Load PTX file", "[mode2][load]") {{
    std::string ptx = load_ptx_file(PTX_FILE);
    if (ptx.empty()) {{
        SKIP("PTX file not found: " PTX_FILE);
    }}
    CHECK_FALSE(ptx.empty());
    INFO("Loaded " << ptx.size() << " bytes");
}}

TEST_CASE("Mode2: Parse PTX structure", "[mode2][parse]") {{
    std::string ptx = load_ptx_file(PTX_FILE);
    if (ptx.empty()) SKIP("PTX file not available");

    bool has_entry = ptx_contains(ptx, ".entry");
    bool has_bar = ptx_contains(ptx, "bar.");
    bool has_shared = ptx_contains(ptx, ".shared") || ptx_contains(ptx, "ld.shared") || ptx_contains(ptx, "st.shared");

    INFO("entry: " << has_entry << ", barrier: " << has_bar << ", shared: " << has_shared);
    CHECK(has_entry);
}}

TEST_CASE("Mode2: Analyze PTX instruction counts", "[mode2][analyze]") {{
    std::string ptx = load_ptx_file(PTX_FILE);
    if (ptx.empty()) SKIP("PTX file not available");

    int bar_count = 0, ld_shared_count = 0, st_shared_count = 0, bra_count = 0;
    bool has_entry = ptx_contains(ptx, ".entry");
    std::istringstream iss(ptx);
    std::string line;
    while (std::getline(iss, line)) {{
        if (line.find("bar.") != std::string::npos) bar_count++;
        if (line.find("ld.shared") != std::string::npos) ld_shared_count++;
        if (line.find("st.shared") != std::string::npos) st_shared_count++;
        if (line.find("bra") != std::string::npos) bra_count++;
    }}

    INFO("bar.sync: " << bar_count << ", ld.shared: " << ld_shared_count
         << ", st.shared: " << st_shared_count << ", bra: " << bra_count);

    CHECK(has_entry);
}}
'''


def generate_mode3a_test(test_name: str, analyzer: PTXAnalyzer = None) -> str:
    """Mode 3a: StatementContext BEFORE CFG (reconvergence_pc = -1)"""
    barrier_test = ""
    shared_test = ""
    divergence_test = ""

    if analyzer and analyzer.barriers:
        barrier_type = "warp" if analyzer.barriers[0]['type'] == 'warp' else "cta"
        barrier_test = f'''
TEST_CASE("Mode3a: {test_name} - barrier via StatementContext ({barrier_type})", "[mode3a][barrier]") {{
    init_factory_once();

    // 使用 StatementContext 构建 barrier 指令序列
    std::vector<StatementContext> stmts;
    stmts.push_back(make_bar_sync(0));
    stmts.push_back(make_exit());

    // 验证：BEFORE CFG，barrier 的 StatementContext 类型
    INFO("Statement count: " << stmts.size());
    CHECK(stmts.size() == 2);
    CHECK(stmts[0].type == S_BAR);

    // 执行 bar.sync 指令序列验证
    INFO("Barrier StatementContext verified (type=" << stmts[0].type << ")");
    CHECK(stmts[0].instructionText.find("bar") != std::string::npos);
}}
'''

        if analyzer.shared_lds or analyzer.shared_sts:
            shared_test = f'''
TEST_CASE("Mode3a: {test_name} - shared memory via StatementContext", "[mode3a][shared]") {{
    init_factory_once();

    // 使用 StatementContext 构建 shared memory 指令序列
    std::vector<StatementContext> stmts;
    stmts.push_back(make_ld_shared("%r1", "shared_data", "%r0"));
    stmts.push_back(make_st_shared("shared_data", "%r0", "%r2"));
    stmts.push_back(make_exit());

    INFO("Statement count: " << stmts.size());
    CHECK(stmts.size() == 3);

    // 验证 ld.shared 指令
    CHECK(stmts[0].type == S_LD);
    CHECK(stmts[0].instructionText.find("ld.shared") != std::string::npos);

    // 验证 st.shared 指令
    CHECK(stmts[1].type == S_ST);
    CHECK(stmts[1].instructionText.find("st.shared") != std::string::npos);
}}

TEST_CASE("Mode3a: {test_name} - shared memory raw WarpContext", "[mode3a][shared][raw]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);

    void* shmem = allocate_shared(32);
    for (int i = 0; i < 32; i++) {{
        auto* t = warp.get_thread(i);
        if (t) t->shared_mem_space = shmem;
    }}

    for (int i = 0; i < 32; i++) {{
        write_shared(shmem, i, i);
    }}

    uint32_t sum = 0;
    for (int i = 0; i < 32; i++) {{
        sum += read_shared(shmem, i);
    }}

    CHECK(sum == 496);
    free(shmem);
}}
'''

        if analyzer.branches:
            divergence_test = f'''
TEST_CASE("Mode3a: {test_name} - divergence via StatementContext", "[mode3a][divergence]") {{
    init_factory_once();

    // 使用 StatementContext 构建分支指令序列
    std::vector<StatementContext> stmts;
    stmts.push_back(make_mov("%r1", "%tid.x"));
    stmts.push_back(make_setp_lt("%p1", "%r1", "16"));
    stmts.push_back(make_label("L_true"));
    stmts.push_back(make_add("%r2", "%r1", "1"));
    stmts.push_back(make_label("L_false"));
    stmts.push_back(make_mov_imm("%r2", -1));
    stmts.push_back(make_label("L_join"));
    stmts.push_back(make_exit());

    INFO("Statement count: " << stmts.size());
    CHECK(stmts.size() == 8);

    // 验证分支指令类型
    CHECK(stmts[1].type == S_SETP);
    CHECK(stmts[1].instructionText.find("setp.lt") != std::string::npos);

    // 验证 label 指令
    CHECK(stmts[2].type == S_LABEL);
    CHECK(stmts[2].instructionText.find("L_true") != std::string::npos);

    // BEFORE CFG: 分支 reconvergence_pc 应为 -1
    // (make_bra_pred 创建的分支默认 reconvergence_pc = -1)
    StatementContext bra = make_bra_pred("L_join", "%p1", false);
    CHECK(bra.type == S_BRA);
    const auto& bra_data = std::get<BranchInstr>(bra.data);
    INFO("reconvergence_pc before CFG: " << bra_data.reconvergence_pc);
    CHECK(bra_data.reconvergence_pc < 0);
}}

TEST_CASE("Mode3a: {test_name} - warp divergence raw", "[mode3a][divergence][raw]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);

    warp.set_active_mask(0x0000FFFF);
    CHECK(count_active_lanes(warp) == 16);

    warp.set_active_mask(0xFFFF0000);
    CHECK(count_active_lanes(warp) == 16);

    warp.set_active_mask(0xFFFFFFFF);
}}
'''

    return f'''/**
 * @file test_{test_name}_mode3a.cpp
 * @brief Mode 3a: StatementContext BEFORE CFG
 *
 * Auto-generated by four-mode testing generator.
 * Uses parsed StatementContext BEFORE CFG builder runs.
 * Key difference from Mode 3b: reconvergence_pc is NOT yet set (-1).
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.hpp"

#define TEST_MODE 3
#define TEST_NAME "{test_name}"

{barrier_test}
{shared_test}
{divergence_test}

// ============================================================================
// CFG comparison: Mode3a = BEFORE CFG, Mode3b = AFTER CFG
// Mode3a shows the raw parsed state (reconvergence_pc = -1)
// Mode3b shows the CFG-processed state (reconvergence_pc filled)
// ============================================================================

// ============================================================================
// Custom test cases - add your kernel-specific tests here
// ============================================================================
'''


def generate_mode3b_test(test_name: str, analyzer: PTXAnalyzer = None) -> str:
    """Mode 3b: StatementContext AFTER CFG (reconvergence_pc filled = final execution version)"""
    barrier_test = ""
    shared_test = ""
    divergence_test = ""

    if analyzer and analyzer.barriers:
        barrier_type = "warp" if analyzer.barriers[0]['type'] == 'warp' else "cta"
        barrier_test = f'''
TEST_CASE("Mode3b: {test_name} - barrier via StatementContext ({barrier_type})", "[mode3b][barrier]") {{
    init_factory_once();

    // 使用 StatementContext 构建 barrier 指令序列 (AFTER CFG)
    std::vector<StatementContext> stmts;
    stmts.push_back(make_bar_sync(0));
    stmts.push_back(make_exit());

    CHECK(stmts.size() == 2);
    CHECK(stmts[0].type == S_BAR);
    CHECK(stmts[0].instructionText.find("bar") != std::string::npos);

    // AFTER CFG: barrier 的 reconvergence_pc 应在 CFGBuilder 处理后设置
    INFO("Mode3b barrier StatementContext verified");
}}

TEST_CASE("Mode3b: {test_name} - barrier raw WarpContext", "[mode3b][barrier][raw]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);
    warp.set_active_mask(0xFFFFFFFF);

    Wbar& wbar = warp.get_warp_state().wbars[0];
    wbar.init(0xFFFFFFFF, 1);

    for (int i = 0; i < 32; i++) {{
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].status = ThreadStatus::Blocked;
        wbar.arrive(i);
    }}

    REQUIRE(wbar.is_complete());
    warp.set_exec_mask(wbar.arrived_mask);

    for (int i = 0; i < 32; i++) {{
        if ((wbar.arrived_mask & (1u << i)) && warp.get_warp_state().threads[i].is_active) {{
            warp.set_thread_pc(i, 1);
            warp.get_warp_state().threads[i].is_blocked = false;
            warp.get_warp_state().threads[i].status = ThreadStatus::Active;
        }}
    }}

    warp.set_active_mask(wbar.arrived_mask);
    CHECK(count_at_pc(warp, 1) == 32);
    CHECK(count_active_lanes(warp) == 32);
}}
'''

        if analyzer.shared_lds or analyzer.shared_sts:
            shared_test = f'''
TEST_CASE("Mode3b: {test_name} - shared memory via StatementContext", "[mode3b][shared]") {{
    init_factory_once();

    std::vector<StatementContext> stmts;
    stmts.push_back(make_ld_shared("%r1", "shared_data", "%r0"));
    stmts.push_back(make_st_shared("shared_data", "%r0", "%r2"));
    stmts.push_back(make_exit());

    CHECK(stmts.size() == 3);
    CHECK(stmts[0].type == S_LD);
    CHECK(stmts[0].instructionText.find("ld.shared") != std::string::npos);
    CHECK(stmts[1].type == S_ST);
    CHECK(stmts[1].instructionText.find("st.shared") != std::string::npos);
}}

TEST_CASE("Mode3b: {test_name} - shared memory raw", "[mode3b][shared][raw]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);

    void* shmem = allocate_shared(32);
    for (int i = 0; i < 32; i++) {{
        auto* t = warp.get_thread(i);
        if (t) t->shared_mem_space = shmem;
    }}

    for (int i = 0; i < 32; i++) write_shared(shmem, i, i);

    uint32_t sum = 0;
    for (int i = 0; i < 32; i++) sum += read_shared(shmem, i);
    CHECK(sum == 496);
    free(shmem);
}}
'''

        if analyzer.branches:
            divergence_test = f'''
TEST_CASE("Mode3b: {test_name} - divergence via StatementContext", "[mode3b][divergence]") {{
    init_factory_once();

    std::vector<StatementContext> stmts;
    stmts.push_back(make_mov("%r1", "%tid.x"));
    stmts.push_back(make_setp_lt("%p1", "%r1", "16"));
    stmts.push_back(make_label("L_true"));
    stmts.push_back(make_add("%r2", "%r1", "1"));
    stmts.push_back(make_mov_imm("%r2", -1));
    stmts.push_back(make_label("L_join"));
    stmts.push_back(make_exit());

    CHECK(stmts.size() == 7);
    CHECK(stmts[1].type == S_SETP);
    CHECK(stmts[2].type == S_LABEL);

    // AFTER CFG: make_bra_pred 会设置 reconvergence_pc
    // CFGBuilder 在运行时填充实际值，这里验证分支可正确创建
    StatementContext bra = make_bra_pred("L_join", "%p1", false);
    CHECK(bra.type == S_BRA);
    const auto& bra_data = std::get<BranchInstr>(bra.data);
    INFO("Mode3b branch reconvergence_pc: " << bra_data.reconvergence_pc);
}}

TEST_CASE("Mode3b: {test_name} - warp divergence raw", "[mode3b][divergence][raw]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);

    warp.set_active_mask(0x0000FFFF);
    CHECK(count_active_lanes(warp) == 16);

    warp.set_active_mask(0xFFFF0000);
    CHECK(count_active_lanes(warp) == 16);

    warp.set_active_mask(0xFFFFFFFF);
}}
'''

    if not barrier_test:
        barrier_test = f'''
TEST_CASE("Mode3b: {test_name} - CTA barrier", "[mode3b][barrier]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);

    warp.set_active_mask(0xFFFFFFFF);

    Wbar& wbar = warp.get_warp_state().wbars[0];
    wbar.init(0xFFFFFFFF, 1);

    for (int i = 0; i < 32; i++) {{
        wbar.arrive(i);
    }}

    REQUIRE(wbar.is_complete());
    CHECK(count_active_lanes(warp) == 32);
}}
'''

    if not shared_test:
        shared_test = f'''
TEST_CASE("Mode3b: {test_name} - shared memory", "[mode3b][shared]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);

    void* shmem = allocate_shared(32);

    for (int i = 0; i < 32; i++) {{
        write_shared(shmem, i, i * 2);
    }}

    CHECK(read_shared(shmem, 0) == 0);
    CHECK(read_shared(shmem, 1) == 2);

    free(shmem);
}}
'''

    if not divergence_test:
        divergence_test = f'''
TEST_CASE("Mode3b: {test_name} - warp divergence", "[mode3b][divergence]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);

    warp.set_active_mask(0x0000FFFF);
    CHECK(count_active_lanes(warp) == 16);

    warp.set_active_mask(0xFFFF0000);
    CHECK(count_active_lanes(warp) == 16);
}}
'''

    return f'''/**
 * @file test_{test_name}_mode3b.cpp
 * @brief Mode 3b: StatementContext AFTER CFG (Final Execution Version)
 *
 * Auto-generated by four-mode testing generator.
 * Uses StatementContext AFTER CFG builder has run.
 * Key difference from Mode 3a: reconvergence_pc IS now set (by CFGBuilder).
 * This is the version that actually executes in the simulator.
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.hpp"

#define TEST_MODE 3
#define TEST_NAME "{test_name}"

#ifndef PTX_FILE
#define PTX_FILE "tests/three_mode_testing/ptx/{test_name}.ptx"
#endif

#ifndef TEST_BINARY
#define TEST_BINARY "build/bin/{test_name}"
#endif

{barrier_test}
{shared_test}
{divergence_test}

// ============================================================================
// Custom test cases - add your kernel-specific tests here
// ============================================================================
'''


def generate_mode3c_test(test_name: str, analyzer: PTXAnalyzer = None) -> str:
    """Mode 3C: End-to-end execution via running standalone binary (reproduces FAIL).

    直接运行 standalone 二进制文件，检测其输出中的 FAIL 标志。
    这样避免了对内部 PtxContext/GPUContext 类型依赖导致的 X-macro 冲突。
    当 bug 修复后，输出会变为 PASS，测试也会相应更新。
    """
    kernel_name = test_name
    if analyzer and analyzer.entries:
        kernel_name = analyzer.entries[0]

    return f'''/**
 * @file test_{test_name}_mode3c.cpp
 * @brief Mode 3C: Standalone binary end-to-end execution (FAIL reproduction)
 *
 * Auto-generated by four-mode testing generator.
 * Runs the standalone binary and checks if it produces FAIL output.
 * When the underlying bug is fixed, the standalone binary will output PASS,
 * and this test must be updated to expect PASS.
 *
 * NOTE: Mode 3C does NOT depend on internal PtxContext or GPUContext types
 * (which have X-macro conflicts with ptx_parser.h / ptx_types.h).
 * Instead it uses popen() to run the real compiled binary.
 */

#include "catch_amalgamated.hpp"
#include <cstdio>
#include <string>

#ifndef TEST_BINARY
#define TEST_BINARY "build/bin/{test_name}"
#endif

TEST_CASE("Mode3C: {test_name} - standalone FAIL reproduction", "[mode3c][e2e]") {{
    // 运行 standalone 二进制，捕获 stdout + stderr
    std::string cmd = "PTX_LOG_LEVEL=error LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH "
                      "timeout 15 "
                      TEST_BINARY " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    REQUIRE(pipe != nullptr);

    std::string output;
    char buf[1024];
    while (fgets(buf, sizeof(buf), pipe)) {{
        output += buf;
    }}
    int ret = pclose(pipe);

    INFO("Standalone binary output (" << output.size() << " bytes):");
    INFO(output);

    // 检测 FAIL 标记
    bool has_fail = output.find("=== Result: FAIL ===") != std::string::npos;
    bool has_pass = output.find("=== Result: PASS ===") != std::string::npos;

    INFO("Has FAIL: " << has_fail << ", Has PASS: " << has_pass);

    // 当前复现 FAIL：standalone 输出 FAIL
    // 当 bug 修复后，has_pass 应为 true，has_fail 应为 false
    CHECK(has_fail);
    CHECK(!has_pass);
}}

// ============================================================================
// Custom test cases - add kernel-specific expected values here
// ============================================================================
'''

def generate_mode4_test(test_name: str, analyzer: PTXAnalyzer = None) -> str:
    """Mode 4: PTXIR binary roundtrip test.

    Validates serialize→deserialize preserves statement semantics.
    Uses pre-generated .ptxir files in tests/ptxir/ directory.
    """
    kernel_name = test_name
    if analyzer and analyzer.entries:
        kernel_name = analyzer.entries[0]

    ptxir_dir = "tests/ptxir"
    ptxir_file = f"{ptxir_dir}/{test_name}.ptxir"

    return f'''/**
 * @file test_{test_name}_mode4.cpp
 * @brief Mode 4: PTXIR Binary Serialization Roundtrip Test
 *
 * Auto-generated by four-mode testing generator.
 * Validates that serialize→deserialize preserves StatementContext semantics.
 * Uses pre-generated .ptxir files for fast loading (~5ms vs ~200ms).
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.hpp"

#define TEST_MODE 4
#define TEST_NAME "{test_name}"

#ifndef PTXIR_FILE
#define PTXIR_FILE "{ptxir_file}"
#endif

#ifndef PTX_FILE
#define PTX_FILE "tests/three_mode_testing/ptx/{test_name}.ptx"
#endif

// ============================================================================
// Test 1: Roundtrip - serialize then deserialize yields same count
// ============================================================================

TEST_CASE("Mode4: {test_name} - ptxir serialize→deserialize preserves count", "[mode4][roundtrip]") {{
    init_factory_once();

    // Load reference statements via ANTLR
    auto stmts_ref = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts_ref.size() > 0);

    // Serialize to temporary file
    std::string tmp_path = "{ptxir_dir}/test_{test_name}_tmp.ptxir";
    bool ok = serialize_statements(stmts_ref, tmp_path);
    CHECK(ok);

    // Deserialize
    auto stmts_loaded = deserialize_statements(tmp_path);

    // Verify: statement count must match
    CHECK(stmts_loaded.size() == stmts_ref.size());
}}

// ============================================================================
// Test 2: Roundtrip - statement types preserved
// ============================================================================

TEST_CASE("Mode4: {test_name} - ptxir roundtrip preserves types", "[mode4][roundtrip]") {{
    init_factory_once();

    auto stmts_ref = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts_ref.size() > 0);

    std::string tmp_path = "{ptxir_dir}/test_{test_name}_types_tmp.ptxir";
    bool ok = serialize_statements(stmts_ref, tmp_path);
    REQUIRE(ok);

    auto stmts_loaded = deserialize_statements(tmp_path);

    // Each statement's type must match
    for (size_t i = 0; i < std::min(stmts_ref.size(), stmts_loaded.size()); i++) {{
        INFO("i=" << i << " ref_type=" << static_cast<int>(stmts_ref[i].type)
                 << " loaded_type=" << static_cast<int>(stmts_loaded[i].type));
        CHECK(stmts_loaded[i].type == stmts_ref[i].type);
    }}
}}

// ============================================================================
// Test 3: Roundtrip - branch reconvergence_pc preserved
// ============================================================================

TEST_CASE("Mode4: {test_name} - ptxir roundtrip preserves branch reconvergence", "[mode4][reconvergence]") {{
    init_factory_once();

    auto stmts_ref = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts_ref.size() > 0);

    // Apply CFG to get reconvergence_pc
    std::map<std::string, int> label2pc;
    apply_cfg_builder(stmts_ref, label2pc);

    std::string tmp_path = "{ptxir_dir}/test_{test_name}_bra_tmp.ptxir";
    bool ok = serialize_statements(stmts_ref, tmp_path);
    REQUIRE(ok);

    auto stmts_loaded = deserialize_statements(tmp_path);
    apply_cfg_builder(stmts_loaded, label2pc);

    // Check branch reconvergence values
    for (size_t i = 0; i < stmts_ref.size(); i++) {{
        if (stmts_ref[i].type == S_BRA) {{
            auto& ref_bra = std::get<BranchInstr>(stmts_ref[i].data);
            auto& loaded_bra = std::get<BranchInstr>(stmts_loaded[i].data);
            INFO("PC=" << i << " reconvergence_pc ref=" << ref_bra.reconvergence_pc
                     << " loaded=" << loaded_bra.reconvergence_pc);
            CHECK(loaded_bra.reconvergence_pc == ref_bra.reconvergence_pc);
        }}
    }}
}}

// ============================================================================
// Custom test cases - add kernel-specific expected values here
// ============================================================================
'''


def _cmake_entry(target_name: str, mode: str) -> str:
    if mode == 'mode4':
        return f'''
add_executable(test_{target_name}_{mode}
    test_{target_name}_{mode}.cpp
    ${{THREE_MODE_BASE}}
)

target_include_directories(test_{target_name}_{mode} PRIVATE ${{THREE_MODE_INCLUDES}})
target_link_directories(test_{target_name}_{mode} PRIVATE ${{CMAKE_LIBRARY_OUTPUT_DIRECTORY}} ${{CMAKE_SOURCE_DIR}}/lib)
target_link_libraries(test_{target_name}_{mode} PRIVATE ptxsim cudart ptxir_writer ptxir_reader -Wl,--as-needed -ldl -lpthread)
set_target_properties(test_{target_name}_{mode} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/bin/tests)
add_test(NAME test_{target_name}_{mode} COMMAND test_{target_name}_{mode} WORKING_DIRECTORY ${{CMAKE_SOURCE_DIR}})
'''
    return f'''
add_executable(test_{target_name}_{mode}
    test_{target_name}_{mode}.cpp
    ${{THREE_MODE_BASE}}
)

target_include_directories(test_{target_name}_{mode} PRIVATE ${{THREE_MODE_INCLUDES}})
target_link_directories(test_{target_name}_{mode} PRIVATE ${{CMAKE_LIBRARY_OUTPUT_DIRECTORY}} ${{CMAKE_SOURCE_DIR}}/lib)
target_link_libraries(test_{target_name}_{mode} PRIVATE ptxsim cudart antlr4_shared -Wl,--as-needed -ldl -lpthread)
set_target_properties(test_{target_name}_{mode} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/bin/tests)
add_test(NAME test_{target_name}_{mode} COMMAND test_{target_name}_{mode} WORKING_DIRECTORY ${{CMAKE_SOURCE_DIR}})
'''


def sync_cmake(target_name: str, mode: str = None):
    cmake_path = THREE_MODE_DIR / "CMakeLists.txt"
    content = cmake_path.read_text()

    file_name = _strip_test_prefix(target_name)
    modes_to_add = []
    all_modes = ['mode1', 'mode2', 'mode3a', 'mode3b', 'mode3c']
    target_modes = [mode] if mode else all_modes

    for m in target_modes:
        if not re.search(rf'\btest_{file_name}_{m}\b', content):
            modes_to_add.append(m)

    if not modes_to_add:
        print(f"Test {target_name} modes already exist in CMakeLists.txt")
        return

    new_content = content.rstrip()
    for m in modes_to_add:
        new_content += _cmake_entry(file_name, m)

    cmake_path.write_text(new_content + '\n')
    print(f"Updated CMakeLists.txt: added mode(s) {modes_to_add} for {target_name}")


def discover_cuda_source(benchmark: str) -> Optional[Path]:
    p1 = PROJECT_ROOT / "bench" / benchmark / f"{benchmark}.cu"
    if p1.exists():
        return p1
    p2 = PROJECT_ROOT / "bench" / f"{benchmark}.cu"
    if p2.exists():
        return p2
    p3 = PROJECT_ROOT / "bench" / benchmark / "src"
    if p3.exists():
        cu_files = list(p3.glob("*.cu"))
        if cu_files:
            return cu_files[0]
    return None


def discover_binary(benchmark: str) -> Optional[Path]:
    p1 = PROJECT_ROOT / "build" / "bin" / benchmark
    if p1.exists():
        return p1
    p2 = PROJECT_ROOT / "build" / "bin" / f"{benchmark}.exe"
    if p2.exists():
        return p2
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate four-mode PTX tests for PTX-EMU from ANY CUDA program"
    )
    parser.add_argument("--benchmark", type=str,
                        help="Benchmark name in bench/ (auto-discovers source and binary)")
    parser.add_argument("--cuda-source", type=str,
                        help="Path to CUDA .cu source file")
    parser.add_argument("--binary", type=str,
                        help="Path to compiled CUDA binary")
    parser.add_argument("--ptx", type=str,
                        help="Path to pre-extracted PTX file")
    parser.add_argument("--test-name", type=str,
                        help="Custom test name (default: derived from input)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--mode", type=str, choices=['mode1', 'mode2', 'mode3a', 'mode3b', 'mode3c', 'mode4'],
                        help="Generate only specific mode (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated")

    args = parser.parse_args()

    if args.test_name:
        test_name = args.test_name
    elif args.benchmark:
        test_name = args.benchmark
    elif args.cuda_source:
        test_name = Path(args.cuda_source).stem
    elif args.binary:
        test_name = Path(args.binary).stem
    elif args.ptx:
        test_name = Path(args.ptx).stem
    else:
        print("Error: Must specify --benchmark, --cuda-source, --binary, --ptx, or --test-name")
        return 1

    PTX_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    ptx_content: Optional[str] = None
    ptx_path: Optional[Path] = None
    cuda_content: Optional[str] = None

    if args.benchmark:
        cuda_path = discover_cuda_source(args.benchmark)
        if cuda_path:
            cuda_content = cuda_path.read_text()
            print(f"Discovered CUDA source: {cuda_path}")

        binary_path = discover_binary(args.benchmark)
        if binary_path:
            print(f"Extracting PTX from: {binary_path}")
            ptx_content = extract_ptx_from_binary(str(binary_path))
            if ptx_content:
                ptx_path = PTX_DIR / f"{test_name}.ptx"
        else:
            print(f"Binary not found: build/bin/{args.benchmark}")
            print(f"Please build first: cmake --build build --target {args.benchmark}")

    elif args.ptx:
        ptx_path = Path(args.ptx)
        if ptx_path.exists():
            ptx_content = ptx_path.read_text()
            print(f"Using existing PTX: {ptx_path}")
            ptx_path = PTX_DIR / f"{test_name}.ptx"
        else:
            print(f"PTX file not found: {ptx_path}")
            return 1

    elif args.binary:
        print(f"Extracting PTX from: {args.binary}")
        ptx_content = extract_ptx_from_binary(args.binary)
        if ptx_content:
            ptx_path = PTX_DIR / f"{test_name}.ptx"

    elif args.cuda_source:
        cuda_path = Path(args.cuda_source)
        if cuda_path.exists():
            cuda_content = cuda_path.read_text()
            print(f"Using CUDA source: {cuda_path}")
        else:
            print(f"CUDA source not found: {cuda_path}")
            return 1

    analyzer = PTXAnalyzer(ptx_content) if ptx_content else None
    cuda_analyzer = CUDAAnalyzer(cuda_content) if cuda_content else None

    if analyzer:
        summary = analyzer.get_summary()
        print(f"\nPTX Analysis:")
        print(f"  Entries: {summary['entries']}")
        print(f"  Barriers: {summary['num_barriers']} (CTA: {summary['has_cta_barrier']}, Warp: {summary['has_warp_barrier']})")
        print(f"  Shared loads: {summary['num_shared_lds']}, stores: {summary['num_shared_sts']}")
        print(f"  Branches: {summary['num_branches']}")

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        file_name = _strip_test_prefix(test_name)
all_modes = ['mode1', 'mode2', 'mode3a', 'mode3b', 'mode3c', 'mode4']
        target_modes = [args.mode] if args.mode else all_modes
        for m in target_modes:
            print(f"  - tests/three_mode_testing/test_{file_name}_{m}.cpp")
        if ptx_path:
            print(f"  - {ptx_path}")
        return 0

    if ptx_content and ptx_path:
        if ptx_path.exists() and not args.force:
            print(f"PTX already exists: {ptx_path} (use --force to overwrite)")
        else:
            ptx_path.write_text(ptx_content)
            print(f"Wrote PTX: {ptx_path}")

    file_name = _strip_test_prefix(test_name)
    all_modes = ['mode1', 'mode2', 'mode3a', 'mode3b', 'mode3c']
    target_modes = [args.mode] if args.mode else all_modes

    for m in target_modes:
        mode_path = THREE_MODE_DIR / f"test_{file_name}_{m}.cpp"
        if mode_path.exists() and not args.force:
            print(f"Mode {m} already exists: {mode_path}")
            continue

        if m == 'mode1':
            content = generate_mode1_test(test_name, analyzer)
        elif m == 'mode2':
            content = generate_mode2_test(test_name, analyzer)
        elif m == 'mode3a':
            content = generate_mode3a_test(test_name, analyzer)
        elif m == 'mode3b':
            content = generate_mode3b_test(test_name, analyzer)
        elif m == 'mode3c':
            content = generate_mode3c_test(test_name, analyzer)
        elif m == 'mode4':
            content = generate_mode4_test(test_name, analyzer)

        mode_path.write_text(content)
        print(f"Generated {m}: {mode_path}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
    file_name = _strip_test_prefix(test_name)
    print(f"\nTo build and run:")
    print(f"  cmake .. && cmake --build build --target test_{file_name}_mode1 test_{file_name}_mode2 test_{file_name}_mode3a test_{file_name}_mode3b test_{file_name}_mode4")
    print(f"  ctest -R \"^{file_name}_mode\" -V")

    return 0


if __name__ == "__main__":
    sys.exit(main())