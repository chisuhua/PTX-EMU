# cluster-context-tcgen05-wiring Specification

## Purpose
TBD - created by archiving change wire-cluster-context-to-tcgen05. Update Purpose after archive.

## ADDED Requirements

### Requirement: ClusterContext-Wired-To-TCGen05 MUST

The `ClusterContext` infrastructure (defined in `src/ptxsim/cluster/cluster_context.h`) MUST be wired into the tcgen05 instruction execution path via opt-in pattern.

Specifically:
1. `CTAContext::init_cluster_context()` MUST be callable from `GPUContext` after CTA creation
2. `tcgen05.commit` handler (`execute_tcgen05_commit` in `src/ptxsim/instructions/wmma.cpp`) MUST call `cta->cluster_context().cta_cluster_arrive(cta_id)` IF `cta->has_cluster_context()` returns true
3. `tcgen05.wait` handler (`execute_tcgen05_wait`) MUST call `cta->cluster_context().cta_cluster_wait(cta_id)` IF `cta->has_cluster_context()` returns true

The wiring MUST use opt-in pattern (guard via `has_cluster_context()`) so existing `cta_group::1` tests continue to pass without modification.

#### Scenario: Cluster-Context-Initialized-Arrives
- **WHEN** a CTA has cluster context initialized AND tcgen05.commit is executed
- **THEN** `cta_cluster_arrive(cta_id)` is called
- **AND** the arrive registration succeeds
- **AND** PTX_DEBUG_EMU log message is emitted

#### Scenario: Cluster-Context-Not-Initialized-Skips
- **WHEN** a CTA does NOT have cluster context initialized AND tcgen05.commit is executed
- **THEN** `cta_cluster_arrive` is NOT called (opt-in skip)
- **AND** no exception is thrown
- **AND** the handler completes successfully

#### Scenario: TCGen05-Wait-Cluster-Sync
- **WHEN** a CTA has cluster context initialized AND tcgen05.wait is executed
- **THEN** `cta_cluster_wait(cta_id)` is called
- **AND** the wait returns when all peer CTAs have arrived (or immediately if `num_ctas==1`)

### Requirement: ClusterContext-Init-Conditional MUST

`CTAContext::init_cluster_context()` MUST only be called when:
1. `KernelContext::usesClusterScope` is true
2. `KernelContext::clusterDimX > 1`

This MUST be enforced in `GPUContext` after CTA creation to avoid unnecessary ClusterContext overhead for non-cluster kernels.

#### Scenario: Non-Cluster-Kernel-Skips-Init
- **WHEN** a kernel is launched without cluster scope
- **THEN** `cta->has_cluster_context()` MUST return false
- **AND** the opt-in check in tcgen05 handlers MUST skip cluster calls

#### Scenario: Cluster-Kernel-Initializes
- **WHEN** a kernel is launched with `usesClusterScope=true` and `clusterDimX=4`
- **THEN** `cta->init_cluster_context(cta_id_in_cluster, 4)` MUST be called
- **AND** `cta->has_cluster_context()` returns true

### Requirement: ClusterContext-Wiring-Verified MUST

The wiring MUST be verified by oracle tests:

1. New test file `tests/unit/cluster/test_cluster_tcgen05_integration.cpp` with 2 scenarios:
   - cluster context initialized → arrive/wait API works
   - cluster context not initialized → opt-in skip verified
2. All pre-existing tests MUST pass (`ctest --output-on-failure` 100% PASS)
3. PTX syntax tests MUST pass (`./tests/ptx/test_all_ptx.sh` 100% PASS)

#### Scenario: Oracle-Test-Passes
- **WHEN** running `ctest -R unit_cluster_tcgen05_integration`
- **THEN** 2 scenarios MUST pass
- **AND** the test MUST be labeled `unit;cluster;tcgen05`

#### Scenario: Zero-Regression
- **WHEN** comparing baseline ctest output vs post-integration ctest output
- **THEN** the only difference MUST be the addition of `unit_cluster_tcgen05_integration` test
- **AND** zero existing tests MUST fail