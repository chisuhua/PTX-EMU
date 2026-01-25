#include "ptxsim/gpu_context.h"
#include "memory/hardware_memory_manager.h" // 添加硬件内存管理器头文件
#include "memory/resource_manager.h"
#include "memory/simple_memory.h" // 添加SimpleMemory头文件
#include <fstream>
#include <future>
#include <inipp/inipp.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

GPUContext::GPUContext(const std::string &config_path)
    : gpu_state(RUN), config(GPUConfig()) {
    if (!config_path.empty()) {
        // 尝试加载JSON配置文件
        if (config_path.substr(config_path.find_last_of(".") + 1) == "json") {
            load_json_config(config_path);
        }
        // 如果不是JSON，暂时也尝试加载（为了向后兼容）
        else {
            load_json_config(config_path);
        }
    } else {
        // 使用默认配置
        config = GPUConfig();
    }

    std::cout << "GPU context created." << std::endl;
}

void GPUContext::init() {
    // 创建SimpleMemory实例作为成员变量
    device_memory = std::make_unique<SimpleMemory>(config.global_mem_size);

    // 设置HardwareMemoryManager使用的SimpleMemory实例
    HardwareMemoryManager::instance().set_simple_memory(device_memory.get());

    // 初始化 ResourceManager
    ResourceManager::instance().initialize(config.num_sms,
                                           config.shared_mem_size_per_sm);

    // 创建SMs
    sms.clear();
    sms.reserve(config.num_sms);
    for (int i = 0; i < config.num_sms; i++) {
        auto sm = std::make_unique<SMContext>(config.max_warps_per_sm,
                                              config.max_threads_per_sm,
                                              config.shared_mem_size_per_sm,
                                              i); // 传递SM ID
        // SMContext现在在构造时完成初始化
        sms.push_back(std::move(sm));
    }

    std::cout << "Initialized GPU with " << config.num_sms << " SMs"
              << std::endl;
}

bool GPUContext::load_json_config(const std::string &config_path) {
    try {
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            std::cerr << "Error: Could not open config file: " << config_path
                      << std::endl;
            return false;
        }

        nlohmann::json j;
        config_file >> j;

        // 从JSON中加载配置参数
        if (j.contains("num_sms")) {
            config.num_sms = j["num_sms"];
        }
        if (j.contains("max_warps_per_sm")) {
            config.max_warps_per_sm = j["max_warps_per_sm"];
        }
        if (j.contains("max_threads_per_sm")) {
            config.max_threads_per_sm = j["max_threads_per_sm"];
        }
        if (j.contains("shared_mem_size_per_sm")) {
            config.shared_mem_size_per_sm = j["shared_mem_size_per_sm"];
        }
        if (j.contains("registers_per_sm")) {
            config.registers_per_sm = j["registers_per_sm"];
        }
        if (j.contains("max_blocks_per_sm")) {
            config.max_blocks_per_sm = j["max_blocks_per_sm"];
        }
        if (j.contains("warp_size")) {
            config.warp_size = j["warp_size"];
        }
        if (j.contains("global_mem_size")) {
            config.global_mem_size = j["global_mem_size"];
        }

        std::cout << "GPU configuration loaded from: " << config_path
                  << std::endl;
        std::cout << "  num_sms: " << config.num_sms << std::endl;
        std::cout << "  max_warps_per_sm: " << config.max_warps_per_sm
                  << std::endl;
        std::cout << "  max_threads_per_sm: " << config.max_threads_per_sm
                  << std::endl;
        std::cout << "  shared_mem_size_per_sm: "
                  << config.shared_mem_size_per_sm << std::endl;
        std::cout << "  registers_per_sm: " << config.registers_per_sm
                  << std::endl;
        std::cout << "  max_blocks_per_sm: " << config.max_blocks_per_sm
                  << std::endl;
        std::cout << "  warp_size: " << config.warp_size << std::endl;
        std::cout << "  global_mem_size: " << config.global_mem_size
                  << std::endl;

        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

void GPUContext::submit_kernel_request(KernelLaunchRequest &&request) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        // 为请求分配唯一ID
        request.request_id = next_request_id++;
        task_queue.emplace(std::forward<KernelLaunchRequest>(request));
    }
    task_cv.notify_one(); // 通知执行线程有新任务
}

bool GPUContext::execute_kernel_internal(
    void **args, Dim3 &gridDim, Dim3 &blockDim,
    std::vector<StatementContext> &statements,
    std::map<std::string, Symtable *> &name2Sym,
    std::map<std::string, int> &label2pc,
    const KernelLaunchRequest &request) {
    // 计算总的CTA数量
    int ctaNum = gridDim.x * gridDim.y * gridDim.z;

    // 为每个CTA创建上下文并尝试添加到SM
    for (int i = 0; i < ctaNum; i++) {
        Dim3 blockIdx;
        blockIdx.z = i / (gridDim.x * gridDim.y);
        blockIdx.y = i % (gridDim.x * gridDim.y) / (gridDim.x);
        blockIdx.x = i % (gridDim.x * gridDim.y) % (gridDim.x);

        // 创建CTAContext
        auto cta = std::make_unique<CTAContext>();
        cta->init(gridDim, blockDim, blockIdx, statements, &name2Sym, label2pc,
                  request.local_memory_base, request.local_mem_per_thread);

        // 尝试将CTA添加到一个可用的SM
        bool added = false;
        for (auto &sm : sms) {
            if (sm->add_block(std::move(cta))) { // 使用std::move转移所有权
                added = true;
                break;
            }
        }

        if (!added) {
            std::cerr << "Error: Could not add block " << i << " to any SM"
                      << std::endl;
            return false;
        }
    }

    std::cout << "Launched kernel with " << ctaNum << " CTAs" << std::endl;

    return true;
}

/**
 * @brief 执行GPU模拟器的一个时钟周期。
 *
 * 该方法是整个GPU模拟器的核心驱动循环。它在一个时间片内完成以下主要工作：
 * 1. **任务调度**:
 * 检查所有流式多处理器（SM）是否都处于空闲状态（IDLE/EXIT）。如果是，并且任务队列中有待处理的核函数请求，
 *    则从队列中取出一个请求并调用 `execute_kernel_internal`
 * 将其启动。这实现了核函数的按序、非抢占式调度。
 * 2. **SM执行**: 遍历所有SM，对每一个当前状态为 `RUN` 的SM调用其 `exe_once()``
 * 方法，让它们各自向前执行一个模拟周期。
 * 3. **状态管理**:
 * 在所有SM都完成了一个周期的执行后，检查它们的整体状态。如果所有SM都已退出（`EXIT`）当前核函数的执行，
 *    并且任务队列中没有更多待处理的任务，则将整个GPU上下文的状态 `gpu_state`
 * 设置为 `EXIT`，表示模拟结束。 否则，将 `gpu_state` 保持为
 * `RUN`，以便下一次调用 `exe_once` 继续执行。
 *
 * 通过反复调用此方法，可以逐步推进GPU上所有核函数的执行，直到所有任务完成。
 *
 * @return EXE_STATE
 * 返回当前GPU的整体执行状态。当所有任务队列为空且所有SM都已完成时返回
 * EXIT，否则返回 RUN。
 */
EXE_STATE GPUContext::exe_once() {
    // 检查任务队列，如果有新任务且当前没有正在运行的kernel，则启动它
    bool all_sm_idle = true;
    for (const auto &sm : sms) {
        if (sm->get_state() == RUN) {
            all_sm_idle = false;
            break;
        }
    }

    // 如果所有SM都处于空闲状态且有任务等待执行，则启动新任务
    if (all_sm_idle) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (!task_queue.empty()) {
            // 启动新任务
            auto request = std::move(task_queue.front());
            task_queue
                .pop(); // 现在可以安全地移除，因为我们要将它标记为正在执行

            // 将请求添加到正在执行的映射中
            executing_requests[request.request_id] = request;

            // 执行任务分配，将kernel分配给各个SM
            execute_kernel_internal(request.args, request.gridDim,
                                    request.blockDim, *request.statements,
                                    *request.name2Sym, *request.label2pc,
                                    request);
        }
    }

    // 执行每个SM的一个周期
    for (auto &sm : sms) {
        if (sm->get_state() == RUN) {
            sm->exe_once();
        }
    }

    // 检查是否所有SM都已完成当前kernel，清理已完成的请求
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto it = executing_requests.begin();
         it != executing_requests.end();) {
        bool request_complete = true;

        // 检查所有SM的状态，判断这个特定请求是否完成
        for (const auto &sm : sms) {
            // 如果SM状态不是EXIT，则说明还有任务在执行
            // 这里我们假设如果SM状态是EXIT，意味着它已经完成了之前分配的请求
            if (sm->get_state() != EXIT) {
                request_complete = false;
                break;
            }
        }

        if (request_complete) {
            // 请求已完成，如果存在完成回调，则执行它
            if (it->second.on_complete) {
                it->second.on_complete();
            }

            // 请求已完成，从执行映射中移除
            it = executing_requests.erase(it);
        } else {
            ++it;
        }
    }

    // 检查是否所有SM都已完成当前kernel
    bool all_finished = true;
    for (const auto &sm : sms) {
        if (sm->get_state() != EXIT) {
            all_finished = false;
            break;
        }
    }

    // 如果所有SM都完成了当前kernel
    if (all_finished && task_queue.empty() && executing_requests.empty()) {
        gpu_state = EXIT; // 没有任务了，设置为EXIT状态
    } else {
        gpu_state = RUN; // 还有任务要处理或当前kernel还在运行
    }

    // 每次执行后增加GPU时钟
    gpu_clock++;

    return gpu_state;
}

bool GPUContext::has_pending_tasks() const {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return !task_queue.empty() || !executing_requests.empty();
}

void GPUContext::wait_for_completion() {
    EXE_STATE state;
    do {
        state = exe_once();
    } while (state != EXIT);
}