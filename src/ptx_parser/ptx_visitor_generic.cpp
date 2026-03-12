// GENERIC_INSTR 类别的实现
#define VISITOR_GENERIC_INSTR(openum, opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opname##Inst(ptxparser::ptxParser::opname##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                                 \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = openum;                                                 \
                                                                                 \
    GenericInstr instr;                                                        \
                                                                                 \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                                 \
    /* 提取操作数 */                                                           \
    auto operands = ctx->getRuleContexts<ptxparser::ptxParser::OperandContext>(); \
    for (size_t i = 0; i < std::min(operands.size(), (size_t)opcount); ++i) { \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
                                                                               \
    auto normalizeBaseText = [](std::string text) {                            \
        while (!text.empty() && text.front() == '[') {                         \
            text.erase(text.begin());                                          \
        }                                                                      \
        while (!text.empty() && text.back() == ']') {                          \
            text.pop_back();                                                   \
        }                                                                      \
        if (!text.empty() && (text.front() == '%' || text.front() == '$')) {   \
            text.erase(text.begin());                                          \
        }                                                                      \
        return text;                                                           \
    };                                                                         \
                                                                               \
    auto parseRegFromText = [&](const std::string &raw, RegOperand &regOut) { \
        std::string name = normalizeBaseText(raw);                             \
        if (name.empty()) {                                                    \
            return false;                                                      \
        }                                                                      \
        size_t pos = 0;                                                        \
        while (pos < name.size() &&                                            \
               std::isalpha(static_cast<unsigned char>(name[pos]))) {          \
            ++pos;                                                             \
        }                                                                      \
        if (pos == 0 || pos >= name.size()) {                                  \
            return false;                                                      \
        }                                                                      \
        for (size_t i = pos; i < name.size(); ++i) {                           \
            if (!std::isdigit(static_cast<unsigned char>(name[i]))) {          \
                return false;                                                  \
            }                                                                  \
        }                                                                      \
        regOut.name = name.substr(0, pos);                                     \
        regOut.index = std::stoi(name.substr(pos));                            \
        return true;                                                           \
    };                                                                         \
                                                                               \
    auto buildAddrFromExpr =                                                   \
        [&](ptxparser::ptxParser::AddressExprContext *addressExprCtx) {        \
            AddrOperand addr;                                                  \
            addr.space = AddrOperand::Space::GLOBAL;                           \
            addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;              \
            addr.immediateOffset = "0";                                        \
                                                                               \
            if (!addressExprCtx || !addressExprCtx->operand()) {               \
                return addr;                                                   \
            }                                                                  \
                                                                               \
            auto baseOp = createOperandFromContext(addressExprCtx->operand()); \
            if (baseOp.kind() == OperandKind::VAR) {                           \
                const auto &var = std::get<VariableOperand>(baseOp.data);      \
                RegOperand reg;                                                \
                if (parseRegFromText(var.name, reg)) {                         \
                    addr.baseSymbol = reg.fullName();                          \
                    addr.id = reg.fullName();                                  \
                    addr.offsetType = AddrOperand::OffsetType::REGISTER;       \
                    addr.registerOffset =                                      \
                        std::make_shared<OperandContext>(OperandContext{reg}); \
                } else {                                                       \
                    auto n = normalizeBaseText(var.name);                      \
                    addr.baseSymbol = n;                                       \
                    addr.id = n;                                               \
                }                                                              \
            } else if (baseOp.kind() == OperandKind::REG) {                    \
                const auto &reg = std::get<RegOperand>(baseOp.data);           \
                addr.baseSymbol = reg.fullName();                              \
                addr.id = reg.fullName();                                      \
                addr.offsetType = AddrOperand::OffsetType::REGISTER;           \
                addr.registerOffset = std::make_shared<OperandContext>(baseOp);\
            } else if (baseOp.kind() == OperandKind::ADDR) {                   \
                const auto &inner = std::get<AddrOperand>(baseOp.data);        \
                auto n = normalizeBaseText(                                    \
                    inner.id.empty() ? inner.baseSymbol : inner.id);           \
                addr.baseSymbol = n;                                           \
                addr.id = n;                                                   \
                addr.offsetType = inner.offsetType;                            \
                addr.immediateOffset = inner.immediateOffset;                  \
                addr.registerOffset = inner.registerOffset;                    \
            } else {                                                           \
                auto rawBase = addressExprCtx->getText();                      \
                RegOperand reg;                                                \
                if (parseRegFromText(rawBase, reg)) {                          \
                    addr.baseSymbol = reg.fullName();                          \
                    addr.id = reg.fullName();                                  \
                    addr.offsetType = AddrOperand::OffsetType::REGISTER;       \
                    addr.registerOffset =                                      \
                        std::make_shared<OperandContext>(OperandContext{reg}); \
                } else {                                                       \
                    auto n = normalizeBaseText(rawBase);                       \
                    addr.baseSymbol = n;                                       \
                    addr.id = n;                                               \
                }                                                              \
            }                                                                  \
                                                                               \
            if (addressExprCtx->immediate()) {                                 \
                auto *imm = addressExprCtx->immediate();                       \
                addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;          \
                addr.registerOffset.reset();                                   \
                addr.immediateOffset = imm->MINUS()                            \
                    ? ("-" + imm->IMMEDIATE()->getText())                     \
                    : imm->IMMEDIATE()->getText();                             \
            }                                                                  \
            return addr;                                                       \
        };                                                                     \
                                                                               \
    auto *ldCtx = dynamic_cast<ptxparser::ptxParser::LdInstContext *>(ctx);   \
    if (ldCtx && ldCtx->addressExpr() && ldCtx->addressExpr()->operand()) {    \
        if (instr.operands.size() > 1) {                                       \
            instr.operands.erase(instr.operands.begin() + 1,                   \
                                instr.operands.end());                         \
        }                                                                      \
        if (ldCtx->operand()) {                                                \
            instr.operands.push_back(createOperandFromContext(ldCtx->operand())); \
            if (instr.operands.size() > 1) {                                   \
                instr.operands.erase(instr.operands.begin() + 1);              \
            }                                                                  \
        }                                                                      \
                                                                               \
        if (instr.operands.empty() && ldCtx->operand()) {                      \
            instr.operands.push_back(createOperandFromContext(ldCtx->operand())); \
        }                                                                      \
                                                                               \
        if (!instr.operands.empty()) {                                         \
            AddrOperand addr = buildAddrFromExpr(ldCtx->addressExpr());        \
            instr.operands.push_back(OperandContext{addr});                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    auto *stCtx = dynamic_cast<ptxparser::ptxParser::StInstContext *>(ctx);   \
    if (stCtx && stCtx->addressExpr() && stCtx->addressExpr()->operand()) {    \
        if (stCtx->operand()) {                                                \
            instr.operands.clear();                                            \
            AddrOperand addr = buildAddrFromExpr(stCtx->addressExpr());        \
            instr.operands.push_back(OperandContext{addr});                    \
            instr.operands.push_back(createOperandFromContext(stCtx->operand())); \
        }                                                                      \
    }                                                                          \
                                                                                 \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                                 \
    return nullptr;                                                            \
}

// X-Macro展开
