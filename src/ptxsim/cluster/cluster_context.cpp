// src/ptxsim/cluster/cluster_context.cpp
// Phase 0.3 (Fix #7): cluster arrive/wait synchronization implementation.
//
// Per ptx-lessons-learned §2（递归锁死锁）: 每个 public 方法独立持锁，
// 且不调用任何其他 public 方法。arrive 使用 lock_guard，wait 使用
// unique_lock（配合 condition_variable）。

#include "ptxsim/cluster/cluster_context.h"

#include <stdexcept>
#include <string>

namespace {

void throw_error(const char* msg) { throw std::runtime_error(msg); }

void throw_error(const std::string& msg) { throw std::runtime_error(msg); }

}  // namespace

ClusterContext::ClusterContext(cta_id_t cluster_root_id,
                               cluster_size_t num_ctas)
    : cluster_root_id_(cluster_root_id)
    , num_ctas_(num_ctas)
{
    if (num_ctas_ < kMinClusterSize || num_ctas_ > kMaxClusterSize) {
        throw_error("ClusterContext: num_ctas must be in [1, 8]");
    }
    if (cluster_root_id_ >= kMaxClusterSize) {
        throw_error("ClusterContext: cluster_root_id must be in [0, 7]");
    }
}

ClusterContext::~ClusterContext() = default;

bool ClusterContext::validate_cta_id(cta_id_t id) const {
    return id < num_ctas_;
}

ClusterContext::cluster_size_t ClusterContext::size() const {
    return num_ctas_;
}

void ClusterContext::cta_cluster_arrive(cta_id_t cta_id) {
    if (!validate_cta_id(cta_id)) {
        throw_error("ClusterContext::cta_cluster_arrive: cta_id " +
                    std::to_string(cta_id) +
                    " out of range [0, " + std::to_string(num_ctas_) + ")");
    }

    std::lock_guard<std::mutex> lock(mu_);

    if (arrived_set_.count(cta_id)) {
        throw_error("ClusterContext::cta_cluster_arrive: cta_id " +
                    std::to_string(cta_id) + " has already arrived");
    }

    arrived_set_.insert(cta_id);

    if (arrived_set_.size() == num_ctas_) {
        cv_.notify_all();
    }
}

void ClusterContext::cta_cluster_wait(cta_id_t cta_id) {
    if (!validate_cta_id(cta_id)) {
        throw_error("ClusterContext::cta_cluster_wait: cta_id " +
                    std::to_string(cta_id) +
                    " out of range [0, " + std::to_string(num_ctas_) + ")");
    }

    std::unique_lock<std::mutex> lock(mu_);

    if (!arrived_set_.count(cta_id)) {
        throw_error("ClusterContext::cta_cluster_wait: cta_id " +
                    std::to_string(cta_id) +
                    " has not called cta_cluster_arrive yet");
    }

    cv_.wait(lock, [this]() {
        return arrived_set_.size() == num_ctas_;
    });
}