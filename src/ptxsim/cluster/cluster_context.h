// src/ptxsim/cluster/cluster_context.h
// Phase 0.3 (Fix #7): cluster arrive/wait synchronization primitives.
//
// 1-8 CTA cluster identifier + arrive/wait primitives for cta_group::1
// (Phase 1-3). NO distributed_smem — DEFERRED to cta_group::2 per Oracle
// simplification.
//
// Design:
//   - cta_cluster_arrive() marks a CTA as arrived within the cluster.
//   - cta_cluster_wait() blocks until all peer CTAs have arrived.
//   - Per ptx-lessons-learned §2 (recursive locking), public methods hold
//     the mutex and no public method calls another public method.
//   - std::mutex + std::condition_variable for blocking semantics
//     (real atomic sem, NOT busy loops).
//   - Error rejection: invalid cta_id, duplicate arrive, wait-before-arrive
//     all throw std::runtime_error.

#ifndef PTXSIM_CLUSTER_CLUSTER_CONTEXT_H
#define PTXSIM_CLUSTER_CLUSTER_CONTEXT_H

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <set>

class ClusterContext {
public:
    using cta_id_t = uint32_t;
    using cluster_size_t = uint8_t;

    static constexpr cluster_size_t kMinClusterSize = 1;
    static constexpr cluster_size_t kMaxClusterSize = 8;

    ClusterContext(cta_id_t cluster_root_id, cluster_size_t num_ctas);
    ~ClusterContext();

    void cta_cluster_arrive(cta_id_t cta_id);
    void cta_cluster_wait(cta_id_t cta_id);

    bool validate_cta_id(cta_id_t id) const;
    cluster_size_t size() const;

    ClusterContext(const ClusterContext&) = delete;
    ClusterContext& operator=(const ClusterContext&) = delete;

private:
    cta_id_t cluster_root_id_;
    cluster_size_t num_ctas_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::set<cta_id_t> arrived_set_;
};

#endif  // PTXSIM_CLUSTER_CLUSTER_CONTEXT_H