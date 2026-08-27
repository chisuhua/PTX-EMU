#ifndef STATEMENT_CONTEXT_H
#define STATEMENT_CONTEXT_H

#include <ptxemu/ir/statement.h>
#include <ptxemu/ir/operand_context.h>
#include "ptx_types.h"
#include "ptxsim/execution_types.h"

namespace ptx_ir = ::ptxemu::ir;

using ::ptxemu::ir::Qualifier;
using ::ptxemu::ir::StatementType;
using ::ptxemu::ir::OperandType;
using ::ptxemu::ir::OperandKind;
using ::ptxemu::ir::OperandContext;
using ::ptxemu::ir::DeclarationInstr;
using ::ptxemu::ir::DollarNameInstr;
using ::ptxemu::ir::PragmaInstr;
using ::ptxemu::ir::LabelInstr;
using ::ptxemu::ir::VoidInstr;
using ::ptxemu::ir::BranchInstr;
using ::ptxemu::ir::BarrierInstr;
using ::ptxemu::ir::MembarInstr;
using ::ptxemu::ir::FenceInstr;
using ::ptxemu::ir::ReduxSyncInstr;
using ::ptxemu::ir::MbarrierInstr;
using ::ptxemu::ir::CallInstr;
using ::ptxemu::ir::PredicatePrefix;
using ::ptxemu::ir::GenericInstr;
using ::ptxemu::ir::Tcgen05OpKind;
using ::ptxemu::ir::Tcgen05Dtype;
using ::ptxemu::ir::Tcgen05Instr;
using ::ptxemu::ir::AtomInstr;
using ::ptxemu::ir::VoteInstr;
using ::ptxemu::ir::ShflInstr;
using ::ptxemu::ir::ActivemaskInstr;
using ::ptxemu::ir::BarWarpSyncInstr;
using ::ptxemu::ir::TextureInstr;
using ::ptxemu::ir::SurfaceInstr;
using ::ptxemu::ir::ReductionInstr;
using ::ptxemu::ir::PrefetchInstr;
using ::ptxemu::ir::AbiDirective;
using ::ptxemu::ir::CpAsyncInstr;
using ::ptxemu::ir::InstrVariant;
using ::ptxemu::ir::StatementContext;

#endif  // STATEMENT_CONTEXT_H
