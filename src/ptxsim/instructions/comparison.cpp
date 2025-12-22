#include "ptxsim/instruction_handlers_decl.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void SETP::process_operation(ThreadContext *context, void *op[3],
                             std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    Qualifier cmpOp = getCmpOpQualifier(qualifiers);
    Qualifier dtype = getDataQualifier(qualifiers);
    uint8_t result;

    SET_P_COMPARE(cmpOp, dtype, &result, src1, src2);

    *static_cast<uint8_t *>(dst) = result;
}

void SELP::process_operation(ThreadContext *context, void *op[4],
                             std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    void *pred = op[3];
    int bytes = TypeUtils::get_bytes(qualifiers);

    // 根据用户建议的实现方式
    const void *selected = *(uint8_t *)pred ? src1 : src2;
    std::memcpy(dst, selected, bytes);
}
