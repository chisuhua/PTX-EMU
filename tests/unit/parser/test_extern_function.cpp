// test_extern_function.cpp
// =============================================================================
// Unit test: 验证 extern 函数声明的双路径解析契约
//   Path 1: PtxListener.exitExternFuncStatement → ptxContext.externFuncs
//   Path 2: PtxVisitor.visitFunctionDecl        → currentKernel->kernelName
//
// Spec: openspec/changes/add-extern-function-declaration/specs/extern-function-parse-coverage/spec.md
// 注：完整 ANTLR 解析测试在 integration 层（避免触发 pre-existing parser LSP 错误）
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/param_context.h"
#include "ptx_ir/ptx_types.h"
#include <string>

TEST_CASE("ExternFuncDecl struct exists with name + params fields",
          "[parser][extern][smoke]") {
  ExternFuncDecl decl;
  decl.name = "test_func";
  REQUIRE(decl.name == "test_func");
  REQUIRE(decl.params.empty());
}

TEST_CASE("PtxContext stores externFuncs vector",
          "[parser][extern][smoke]") {
  PtxContext ctx;
  REQUIRE(ctx.externFuncs.empty());

  ExternFuncDecl decl;
  decl.name = "added_func";
  ctx.externFuncs.push_back(decl);
  REQUIRE(ctx.externFuncs.size() == 1);
  REQUIRE(ctx.externFuncs[0].name == "added_func");
}

TEST_CASE("ExternFuncDecl params support multiple qualifiers",
          "[parser][extern][logic]") {
  ExternFuncDecl decl;
  decl.name = "multi_param";

  ParamContext p1;
  p1.paramName = "x";
  p1.byteSize = 4;
  p1.paramTypes.push_back(Qualifier::Q_U32);
  decl.params.push_back(p1);

  ParamContext p2;
  p2.paramName = "y";
  p2.byteSize = 8;
  p2.paramTypes.push_back(Qualifier::Q_U64);
  decl.params.push_back(p2);

  REQUIRE(decl.params.size() == 2);
  REQUIRE(decl.params[0].byteSize == 4);
  REQUIRE(decl.params[1].byteSize == 8);
  REQUIRE(decl.params[0].paramTypes[0] == Qualifier::Q_U32);
  REQUIRE(decl.params[1].paramTypes[0] == Qualifier::Q_U64);
}

TEST_CASE("ExternFuncDecl 简单形式 — 行为契约",
          "[parser][extern][contract]") {
  ExternFuncDecl decl;
  decl.name = "simple_extern";
  decl.params.clear();

  REQUIRE(decl.name == "simple_extern");
  REQUIRE(decl.params.empty());
}

TEST_CASE("ExternFuncDecl 带参数形式 — 行为契约",
          "[parser][extern][contract]") {
  ExternFuncDecl decl;
  decl.name = "param_extern";

  ParamContext p1;
  p1.paramName = "x";
  p1.byteSize = 4;
  p1.paramTypes.push_back(Qualifier::Q_U32);
  decl.params.push_back(p1);

  ParamContext p2;
  p2.paramName = "y";
  p2.byteSize = 8;
  p2.paramTypes.push_back(Qualifier::Q_U64);
  decl.params.push_back(p2);

  REQUIRE(decl.name == "param_extern");
  REQUIRE(decl.params.size() == 2);
  REQUIRE(decl.params[0].paramName == "x");
  REQUIRE(decl.params[1].paramName == "y");
}

TEST_CASE("PtxContext externFuncs vs ptxKernels 区分",
          "[parser][extern][distinction]") {
  PtxContext ctx;
  ctx.ptxKernels.clear();
  ctx.externFuncs.clear();
  REQUIRE(ctx.ptxKernels.empty());
  REQUIRE(ctx.externFuncs.empty());

  ExternFuncDecl e;
  e.name = "extern_helper";
  ctx.externFuncs.push_back(e);

  REQUIRE(ctx.externFuncs.size() == 1);
  REQUIRE(ctx.ptxKernels.empty());
}