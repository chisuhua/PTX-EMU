#include "ptxsim/gpu_context.h"
#include "ptxsim/interpreter.h"
#include <fstream>
#include <future>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

GPUContext::GPUContext(const std::string &config_path)
    : gpu_state(RUN), statements(nullptr), name2Sym(nullptr),
      label2pc(nullptr) {
    if (!config_path.empty()) {
        load_config(config_path);
    } else {
        // 使用默认配置
        config = GPUConfig();
    }
}

bool GPUContext::load_config(const std::string &config_path) {
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

        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

void GPUContext::init(Dim3 &gridDim, Dim3 &blockDim,
                      std::vector<StatementContext> &statements,
                      std::map<std::string, Symtable *> &name2Sym,
                      std::map<std::string, int> &label2pc) {
    this->gridDim = gridDim;
    this->blockDim = blockDim;
    this->statements = &statements;
    this->name2Sym = &name2Sym;
    this->label2pc = &label2pc;

    // 创建SMs
    sms.clear();
    sms.reserve(config.num_sms);
    for (int i = 0; i < config.num_sms; i++) {
        auto sm = std::make_unique<SMContext>(config.max_warps_per_sm,
                                              config.max_threads_per_sm,
                                              config.shared_mem_size_per_sm);
        sm->init(gridDim, blockDim, statements, name2Sym, label2pc);
        sms.push_back(std::move(sm));
    }

    std::cout << "Initialized GPU with " << config.num_sms << " SMs"
              << std::endl;
}

bool GPUContext::execute_kernel_internal(
    void **args, Dim3 &gridDim, Dim3 &blockDim,
    std::vector<StatementContext> &statements,
    std::map<std::string, Symtable *> &name2Sym,
    std::map<std::string, int> &label2pc) {
    // 计算总的CTA数量
    int ctaNum = gridDim.x * gridDim.y * gridDim.z;

    // 清理之前的CTA
    active_ctas.clear();

    // 为每个CTA创建上下文并尝试添加到SM
    for (int i = 0; i < ctaNum; i++) {
        Dim3 blockIdx;
        blockIdx.z = i / (gridDim.x * gridDim.y);
        blockIdx.y = i % (gridDim.x * gridDim.y) / (gridDim.x);
        blockIdx.x = i % (gridDim.x * gridDim.y) % (gridDim.x);

        // 创建CTAContext
        auto cta = std::make_unique<CTAContext>();
        cta->init(gridDim, blockDim, blockIdx, statements, name2Sym, label2pc);

        // 尝试将CTA添加到一个可用的SM
        bool added = false;
        for (auto &sm : sms) {
            if (sm->add_block(cta.get())) {
                // 如果添加成功，将CTA加入活跃列表
                active_ctas.push_back(std::move(cta));
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

    // 执行直到所有CTA完成
    while (get_state() == RUN) {
        exe_once();
    }

    return true;
}

bool GPUContext::launch_kernel(void **args, Dim3 &gridDim, Dim3 &blockDim,
                               std::vector<StatementContext> &statements,
                               std::map<std::string, Symtable *> &name2Sym,
                               std::map<std::string, int> &label2pc) {
    return execute_kernel_internal(args, gridDim, blockDim, statements,
                                   name2Sym, label2pc);
}

std::future<EXE_STATE>
GPUContext::launch_kernel_async(void **args, Dim3 &gridDim, Dim3 &blockDim,
                                std::vector<StatementContext> &statements,
                                std::map<std::string, Symtable *> &name2Sym,
                                std::map<std::string, int> &label2pc) {
    // 创建一个promise和future
    auto promise = std::make_shared<std::promise<EXE_STATE>>();
    auto future = promise->get_future();

    // 在新线程中执行kernel
    std::thread([this, promise, args, &gridDim, &blockDim, &statements, &name2Sym,
                 &label2pc]() {
        try {
            bool success = execute_kernel_internal(
                args, gridDim, blockDim, statements, name2Sym, label2pc);
            if (success) {
                promise->set_value(get_state());
            } else {
                promise->set_value(EXIT);
            }
        } catch (...) {
            promise->set_value(EXIT);
        }
    }).detach();

    return future;
}

EXE_STATE GPUContext::exe_once() {
    if (gpu_state != RUN) {
        return gpu_state;
    }

    // 检查是否所有SM都已完成
    bool all_finished = true;
    for (const auto &sm : sms) {
        if (sm->get_state() != EXIT) {
            all_finished = false;
            break;
        }
    }

    if (all_finished) {
        gpu_state = EXIT;
        return gpu_state;
    }

    // 执行每个SM的一个周期
    for (auto &sm : sms) {
        if (sm->get_state() == RUN) {
            sm->exe_once();
        }
    }

    return gpu_state;
}

bool GPUContext::has_pending_tasks() const {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return !task_queue.empty();
}