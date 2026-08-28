#include "ptxsim/register_access_layer.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/ptx_exceptions.h"
#include <stdexcept>

RegisterAccessLayer::RegisterAccessLayer(
    std::shared_ptr<RegisterBankManager> bank_mgr, int warp_id, int lane_id,
    const Dim3 &tIdx, const Dim3 &bIdx, const Dim3 &gDim, const Dim3 &bDim)
    : register_bank_manager_(std::move(bank_mgr)), warp_id_(warp_id),
      lane_id_(lane_id), thread_idx_(tIdx), block_idx_(bIdx), grid_dim_(gDim),
      block_dim_(bDim) {}

void *RegisterAccessLayer::acquire_register(const RegOperand &reg,
                                            std::vector<ptxemu::ir::Qualifier> qualifier) {
// Special registers (per-thread identifiers)
if (reg.name.find('.') != std::string::npos) {
    if (reg.name == "tid.x")
        return const_cast<uint32_t *>(&thread_idx_.x);
    if (reg.name == "tid.y")
        return const_cast<uint32_t *>(&thread_idx_.y);
    if (reg.name == "tid.z")
        return const_cast<uint32_t *>(&thread_idx_.z);
    if (reg.name == "ctaid.x")
        return const_cast<uint32_t *>(&block_idx_.x);
    if (reg.name == "ctaid.y")
        return const_cast<uint32_t *>(&block_idx_.y);
    if (reg.name == "ctaid.z")
        return const_cast<uint32_t *>(&block_idx_.z);
    if (reg.name == "nctaid.x")
        return const_cast<uint32_t *>(&grid_dim_.x);
    if (reg.name == "nctaid.y")
        return const_cast<uint32_t *>(&grid_dim_.y);
    if (reg.name == "nctaid.z")
        return const_cast<uint32_t *>(&grid_dim_.z);
    if (reg.name == "ntid.x")
        return const_cast<uint32_t *>(&block_dim_.x);
    if (reg.name == "ntid.y")
        return const_cast<uint32_t *>(&block_dim_.y);
    if (reg.name == "ntid.z")
        return const_cast<uint32_t *>(&block_dim_.z);
}

    if (!register_bank_manager_) {
        throw std::runtime_error(
            "RegisterBankManager is required but not set");
    }

    std::string combinedName = reg.fullName();
    void *reg_data =
        register_bank_manager_->get_register(combinedName, warp_id_, lane_id_);

    if (reg_data == nullptr) {
        throw InvalidMemoryAccessException(
            0, 0, "null register data",
            "Register not found in bank manager: " + combinedName);
    }
    return reg_data;
}