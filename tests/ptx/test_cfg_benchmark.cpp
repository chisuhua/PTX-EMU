/**
 * CFG Performance Benchmark
 *
 * Measures CFG analysis overhead for different kernel sizes by:
 *   1. Generating synthetic StatementContext vectors of varying sizes
 *   2. Building the CFG with CFGBuilder::build()
 *   3. Computing post-dominators with CFGBuilder::computePostDominators()
 *   4. Averaging elapsed time over N iterations (after warm-up)
 *   5. Comparing against per-size latency budgets
 *
 * Why synthetic statements (not the .ptx files in this directory)?
 *   - The PtxVisitor API has known issues (see tests/CMakeLists.txt
 *     comments for test_cfg_debug.cpp) that prevent reliable parsing
 *   - Synthetic statements give us deterministic, reproducible CFGs
 *     with controllable size and branch density
 *   - The .ptx files (test_cfg_perf_small/medium/large.ptx) remain
 *     available for future use once the parser API is stabilized
 *
 * CTest integration: outputs "[PASS]" / "[FAIL]" so the existing
 *   PASS_REGULAR_EXPRESSION "PASS" matcher picks it up.
 */

#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <algorithm>

using namespace ptx;
using namespace ptx::cfg;
using clk = std::chrono::high_resolution_clock;
using nsec = std::chrono::nanoseconds;

// ---------------------------------------------------------------------------
// Synthetic kernel generator
// Produces stmt_count statements with branches inserted at regular intervals.
// The last statement is forced to S_RET so the CFG has a well-defined exit.
// ---------------------------------------------------------------------------
static std::pair<std::vector<StatementContext>, std::map<std::string, int>>
generateSyntheticKernel(int stmt_count, int branch_count) {
    if (stmt_count < 1) stmt_count = 1;
    if (branch_count < 0) branch_count = 0;
    if (branch_count > stmt_count - 1) branch_count = stmt_count - 1;

    std::vector<StatementContext> statements(stmt_count);
    std::map<std::string, int> label2pc;

    // Place `branch_count` unconditional forward branches evenly spaced.
    // Each branch targets a label placed a few statements ahead.
    if (branch_count > 0) {
        int step = std::max(1, stmt_count / (branch_count + 1));
        for (int b = 0; b < branch_count; b++) {
            int branch_pc = std::min(b * step, stmt_count - 2);
            int target_pc = std::min(branch_pc + step, stmt_count - 1);
            std::string label = "L_" + std::to_string(b);

            BranchInstr br;
            br.target = label;
            br.predicate = "%p" + std::to_string(b);  // conditional -> fallthrough
            br.predicate_negated = false;
            br.reconvergence_pc = -1;
            statements[branch_pc] = StatementContext(S_BRA, std::move(br));

            // Place a S_DOLLOR at the target so the label is "declared" inline
            // (the CFG builder only needs label2pc to resolve targets, but
            // this keeps the synthetic stream structurally realistic).
            DollarNameInstr dn;
            dn.name = label;
            statements[target_pc] = StatementContext(S_DOLLOR, std::move(dn));

            label2pc[label] = target_pc;
        }
    }

    // Fill remaining / override with S_RET to ensure at least one exit.
    for (auto& s : statements) {
        if (s.type != S_BRA && s.type != S_DOLLOR) {
            s.type = S_RET;
        }
    }
    if (statements.back().type != S_RET) {
        statements.back().type = S_RET;
    }

    return {std::move(statements), std::move(label2pc)};
}

// ---------------------------------------------------------------------------
// One benchmark measurement: build CFG + compute post-dominators,
// averaged over `iterations` runs after `warmup` warm-up runs.
// Returns per-iteration averages in microseconds.
// ---------------------------------------------------------------------------
struct BenchResult {
    int stmt_count;
    int block_count;
    double build_us;      // avg CFG build time per iteration
    double postdom_us;    // avg post-dominator computation time per iteration
    double total_us;      // build + postdom
};

static BenchResult runBenchmark(int stmt_count, int branch_count,
                                int warmup, int iterations) {
    auto [statements, label2pc] = generateSyntheticKernel(stmt_count, branch_count);

    // Warm-up: prime the instruction cache and the allocator.
    for (int i = 0; i < warmup; i++) {
        CFG c = CFGBuilder::build(statements, label2pc);
        PostDominatorMap pd = CFGBuilder::computePostDominators(c);
        (void)pd;
    }

    double build_sum_us = 0.0;
    double postdom_sum_us = 0.0;
    int last_block_count = 0;

    for (int i = 0; i < iterations; i++) {
        auto t1 = clk::now();
        CFG cfg = CFGBuilder::build(statements, label2pc);
        auto t2 = clk::now();
        PostDominatorMap pd = CFGBuilder::computePostDominators(cfg);
        auto t3 = clk::now();

        build_sum_us += std::chrono::duration_cast<nsec>(t2 - t1).count() / 1000.0;
        postdom_sum_us += std::chrono::duration_cast<nsec>(t3 - t2).count() / 1000.0;
        last_block_count = static_cast<int>(cfg.blocks.size());
    }

    BenchResult r;
    r.stmt_count = stmt_count;
    r.block_count = last_block_count;
    r.build_us = build_sum_us / iterations;
    r.postdom_us = postdom_sum_us / iterations;
    r.total_us = r.build_us + r.postdom_us;
    return r;
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== CFG Performance Benchmark ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Builds synthetic PTX kernels of varying sizes, runs CFG" << std::endl;
    std::cout << "analysis (block identification + edge construction + post-" << std::endl;
    std::cout << "dominator fixed-point), and reports average per-iteration cost." << std::endl;
    std::cout << std::endl;

    struct TestCase {
        std::string name;
        int stmt_count;
        int branch_count;
        int warmup;
        int iterations;
        double budget_us;
    };

    // Budgets are generous: the CFG builder is O(n) and runs in low microseconds
    // for these sizes on any modern host. A real regression would show up as
    // a 10x+ blow-up, not a 2x one.
    std::vector<TestCase> test_cases = {
        {"Small  Kernel (<50 stmts,  ~3 branches)",   30,  3,  200, 5000,   500.0},
        {"Medium Kernel (50-200 stmts, ~10 branches)", 100, 10, 200, 2000,  2000.0},
        {"Large  Kernel (>200 stmts, ~30 branches)",  300, 30, 100, 1000, 10000.0},
    };

    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Build:    CFGBuilder::build(statements, label2pc)" << std::endl;
    std::cout << "  - Analyze:  CFGBuilder::computePostDominators(cfg)" << std::endl;
    std::cout << "  - Units:    microseconds (us), per-iteration average" << std::endl;
    std::cout << "  - Method:   warm-up + averaged repeated runs" << std::endl;
    std::cout << std::endl;

    std::cout << "Results:" << std::endl;
    std::cout << "--------" << std::endl;

    int passed = 0;
    int failed = 0;
    std::vector<std::string> recommendations;

    for (const auto& tc : test_cases) {
        BenchResult r = runBenchmark(tc.stmt_count, tc.branch_count,
                                     tc.warmup, tc.iterations);

        std::cout << tc.name << std::endl;
        std::cout << "  Statements:   " << r.stmt_count << std::endl;
        std::cout << "  Basic blocks: " << r.block_count << std::endl;
        std::cout << "  Iterations:   " << tc.iterations
                  << " (warmup=" << tc.warmup << ")" << std::endl;
        std::cout << "  Build time:   " << std::fixed << std::setprecision(3)
                  << r.build_us << " us" << std::endl;
        std::cout << "  PostDom time: " << std::fixed << std::setprecision(3)
                  << r.postdom_us << " us" << std::endl;
        std::cout << "  Total time:   " << std::fixed << std::setprecision(3)
                  << r.total_us << " us" << std::endl;
        std::cout << "  Budget:       " << std::fixed << std::setprecision(3)
                  << tc.budget_us << " us" << std::endl;

        if (r.total_us <= tc.budget_us) {
            std::cout << "  Status:       PASS (within budget)" << std::endl;
            passed++;
        } else {
            std::cout << "  Status:       FAIL (exceeds budget by "
                      << std::fixed << std::setprecision(3)
                      << (r.total_us - tc.budget_us) << " us)" << std::endl;
            failed++;

            // Targeted recommendation based on which phase is slow
            if (r.build_us > r.postdom_us) {
                recommendations.push_back(
                    "[" + tc.name + "] CFG build dominates; inspect "
                    "CFGBuilder::identifyBasicBlocks / buildEdges for O(n^2) "
                    "loops (e.g., linear successor/predecessor searches).");
            } else {
                recommendations.push_back(
                    "[" + tc.name + "] Post-dominator computation dominates; "
                    "postDomSets uses std::set<int> with O(n log n) ops per "
                    "intersection -- consider bitset for fixed block-ID ranges.");
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Summary:" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << "  Passed:   " << passed << "/" << test_cases.size() << std::endl;
    std::cout << "  Failures: " << failed << "/" << test_cases.size() << std::endl;
    std::cout << std::endl;

    if (failed == 0) {
        std::cout << "[PASS] All CFG benchmarks within budget" << std::endl;
        std::cout << "       (no optimization actions required)" << std::endl;
        return 0;
    }

    std::cout << "[FAIL] " << failed << " benchmark(s) exceeded budget" << std::endl;
    std::cout << std::endl;
    std::cout << "Optimization Recommendations:" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    if (recommendations.empty()) {
        std::cout << "  - Investigate the timing of build_us vs postdom_us" << std::endl;
    } else {
        for (const auto& rec : recommendations) {
            std::cout << "  - " << rec << std::endl;
        }
    }
    std::cout << "  - Note: synthetic kernels only; real PTX via" << std::endl;
    std::cout << "    test_cfg_perf_*.ptx is blocked by PtxVisitor API gaps." << std::endl;
    return 1;
}
