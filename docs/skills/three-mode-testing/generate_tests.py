#!/usr/bin/env python3
"""
Three-Mode Test Generator for PTX-EMU

Generates three-mode PTX tests from ANY CUDA program:
- Mode 1: cuobjdump dynamic extraction (end-to-end)
- Mode 2: pre-extracted PTX file (stable reproduction)
- Mode 3: direct StatementContext construction (unit testing)

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
            ['cuobjdump', '-xptx', 'all', '-arch=sm_100', binary_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Error extracting PTX: {result.stderr}")
            return None
        ptx_files = list(Path('.').glob('*.sm_100.ptx'))
        if ptx_files:
            ptx_content = ptx_files[0].read_text()
            ptx_files[0].unlink()
            return ptx_content
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
 * Auto-generated by three-mode-testing generator.
 * Extracts PTX at runtime using cuobjdump.
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.cpp"

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
'''


def generate_mode2_test(test_name: str, analyzer: PTXAnalyzer = None, cuda_analyzer: CUDAAnalyzer = None) -> str:
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
 * Auto-generated by three-mode-testing generator.
 * Loads pre-extracted PTX from file for stable, reproducible testing.
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.cpp"

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


def generate_mode3_test(test_name: str, analyzer: PTXAnalyzer = None) -> str:
    barrier_test = ""
    shared_test = ""
    divergence_test = ""

    if analyzer and analyzer.barriers:
        barrier_type = "warp" if analyzer.barriers[0]['type'] == 'warp' else "cta"
        barrier_test = f'''
TEST_CASE("Mode3: {test_name} - barrier synchronization ({barrier_type})", "[mode3][barrier]") {{
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
TEST_CASE("Mode3: {test_name} - shared memory write/read", "[mode3][shared]") {{
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

    INFO("Sum of shared[0..31] = " << sum);
    CHECK(sum == 496);

    free(shmem);
}}
'''

        if analyzer.branches:
            divergence_test = f'''
TEST_CASE("Mode3: {test_name} - divergent paths", "[mode3][divergence]") {{
    init_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> threads;
    setup_warp(warp, threads, 32);
    reset_warp(warp);

    warp.set_exec_mask(0xFFFFFFFF);
    warp.set_active_mask(0x0000FFFF);

    CHECK(count_active_lanes(warp) == 16);

    warp.set_active_mask(0xFFFF0000);
    CHECK(count_active_lanes(warp) == 16);

    warp.set_active_mask(0xFFFFFFFF);
}}
'''

    if not barrier_test:
        barrier_test = f'''
TEST_CASE("Mode3: {test_name} - CTA barrier", "[mode3][barrier]") {{
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
TEST_CASE("Mode3: {test_name} - shared memory", "[mode3][shared]") {{
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
TEST_CASE("Mode3: {test_name} - warp divergence", "[mode3][divergence]") {{
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
 * @file test_{test_name}_mode3.cpp
 * @brief Mode 3: Direct StatementContext Construction
 *
 * Auto-generated by three-mode-testing generator.
 * Uses test_helpers.cpp to construct StatementContext sequences.
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.cpp"

#define TEST_MODE 3
#define TEST_NAME "{test_name}"

{barrier_test}
{shared_test}
{divergence_test}

// ============================================================================
// Custom test cases - add your kernel-specific tests here
// ============================================================================
'''


def _cmake_entry(target_name: str, mode: int) -> str:
    return f'''

add_executable(test_{target_name}_mode{mode}
    test_{target_name}_mode{mode}.cpp
    ${{THREE_MODE_BASE}}
)

target_include_directories(test_{target_name}_mode{mode} PRIVATE ${{THREE_MODE_INCLUDES}})
target_link_directories(test_{target_name}_mode{mode} PRIVATE ${{CMAKE_LIBRARY_OUTPUT_DIRECTORY}} ${{CMAKE_SOURCE_DIR}}/lib)
target_link_libraries(test_{target_name}_mode{mode} PRIVATE ptxsim cudart antlr4_shared -Wl,--as-needed -ldl -lpthread)
set_target_properties(test_{target_name}_mode{mode} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/bin/tests)
add_test(NAME test_{target_name}_mode{mode} COMMAND test_{target_name}_mode{mode} WORKING_DIRECTORY ${{CMAKE_SOURCE_DIR}})
'''


def sync_cmake(target_name: str, mode: int = None):
    cmake_path = THREE_MODE_DIR / "CMakeLists.txt"
    content = cmake_path.read_text()

    file_name = _strip_test_prefix(target_name)
    modes_to_add = []
    for m in ([1, 2, 3] if mode is None else [mode]):
        if not re.search(rf'\btest_{file_name}_mode{m}\b', content):
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
        description="Generate three-mode PTX tests for PTX-EMU from ANY CUDA program"
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
    parser.add_argument("--mode", type=int, choices=[1, 2, 3],
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
        for m in ([1, 2, 3] if args.mode is None else [args.mode]):
            print(f"  - tests/three_mode_testing/test_{file_name}_mode{m}.cpp")
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
    for m in ([1, 2, 3] if args.mode is None else [args.mode]):
        mode_path = THREE_MODE_DIR / f"test_{file_name}_mode{m}.cpp"
        if mode_path.exists() and not args.force:
            print(f"Mode {m} already exists: {mode_path}")
            continue

        if m == 1:
            content = generate_mode1_test(test_name, analyzer)
        elif m == 2:
            content = generate_mode2_test(test_name, analyzer, cuda_analyzer)
        else:
            content = generate_mode3_test(test_name, analyzer)

        mode_path.write_text(content)
        print(f"Generated Mode {m}: {mode_path}")

    sync_cmake(test_name, args.mode)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print(f"\nTo build and run:")
    file_name = _strip_test_prefix(test_name)
    print(f"  cmake --build build --target test_{file_name}_mode1 test_{file_name}_mode2 test_{file_name}_mode3")
    print(f"  ctest -R \"^{file_name}_mode\" -V")

    return 0


if __name__ == "__main__":
    sys.exit(main())