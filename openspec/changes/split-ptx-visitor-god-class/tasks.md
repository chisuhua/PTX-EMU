# split-ptx-visitor-god-class — Tasks

## 1. Phase 0: 准备工作

- [ ] 1.1 建立 baseline worktree（Checklist B：避免污染 main 分支）
- [ ] 1.2 运行 `wc -l src/ptx_parser/ptx_visitor.cpp` 记录基线行数（应为 1067）
- [ ] 1.3 运行 `./tests/ptx/test_all_ptx.sh` 记录基线测试结果（应为 47/47）
- [ ] 1.4 运行 `ctest --output-on-failure` 记录基线 ctest 全绿
- [ ] 1.5 MUST 验证：grep -c '#include "ptx_visitor_warp.cpp"' = 2（确认 :922 重复 include）

## 2. Phase 1: 删除 :922 重复 include（30 min）

- [ ] 2.1 MUST 删除 ptx_visitor.cpp:922 的 `#include "ptx_visitor_warp.cpp"`
- [ ] 2.2 MUST 验证：grep -c '#include "ptx_visitor_warp.cpp"' = 1
- [ ] 2.3 MUST 验证：cmake --build build 无 ODR warning
- [ ] 2.4 MUST 验证：./tests/ptx/test_all_ptx.sh 全绿
- [ ] 2.5 MUST 验证：ctest 全绿
- [ ] 2.6 git commit -m "fix(ptx_visitor): remove duplicate #include ptx_visitor_warp.cpp at :922"

## 3. Phase 2: 提取 tcgen05 visitor（45 min）

- [ ] 3.1 MUST 新建 src/ptx_parser/ptx_visitor_tcgen05.cpp
- [ ] 3.2 MUST 移动 ptx_visitor.cpp:841-902（visitTcgen05Inst）至新文件
- [ ] 3.3 MUST 行级随迁 :863-877 的 C3 fix parse-tree walk 注释（lessons-learned §1）
- [ ] 3.4 MUST 行级随迁 :866-877 的 cta_group IMMEDIATE 提取逻辑（lessons-learned §13）
- [ ] 3.5 MUST 在 ptx_visitor.cpp include 聚合区添加 #include "ptx_visitor_tcgen05.cpp"
- [ ] 3.6 MUST 更新 src/ptx_parser/CMakeLists.txt 添加新源文件
- [ ] 3.7 MUST 验证：cmake --build build 通过
- [ ] 3.8 MUST 验证：./tests/ptx/test_all_ptx.sh 全绿（含 tcgen05 测试）
- [ ] 3.9 MUST 验证：ctest 全绿
- [ ] 3.10 MUST 验证：grep -c 'visitTcgen05Inst' src/ptx_parser/ptx_visitor*.cpp 总和 = 1（仅在新文件定义）
- [ ] 3.11 git commit -m "refactor(ptx_visitor): extract visitTcgen05Inst to ptx_visitor_tcgen05.cpp"

## 4. Phase 3: 提取 operand/qualifier helpers（60 min）

- [ ] 4.1 MUST 新建 src/ptx_parser/ptx_visitor_operands.cpp
- [ ] 4.2 MUST 移动 ptx_visitor.cpp:138-154（tokenToQualifier）至新文件
- [ ] 4.3 MUST 移动 ptx_visitor.cpp:155-184（extractQualifiersFromContext）至新文件
- [ ] 4.4 MUST 移动 ptx_visitor.cpp:185-317（createOperandFromContext）至新文件
- [ ] 4.5 MUST 移动 ptx_visitor.cpp:937-1067（visitOperand/visitSpecialRegister/visitRegister/visitImmediate/visitAddress）至新文件
- [ ] 4.6 MUST 保持 extractQualifiersFromContext 签名不变（19 个调用点依赖）
- [ ] 4.7 MUST 保持所有 visitor override 签名与 ANTLR 兼容
- [ ] 4.8 MUST 在 ptx_visitor.cpp include 聚合区添加 #include "ptx_visitor_operands.cpp"
- [ ] 4.9 MUST 更新 src/ptx_parser/CMakeLists.txt 添加新源文件
- [ ] 4.10 MUST 验证：cmake --build build 通过
- [ ] 4.11 MUST 验证：./tests/ptx/test_all_ptx.sh 全绿
- [ ] 4.12 MUST 验证：ctest 全绿（特别是 operand 相关测试）
- [ ] 4.13 MUST 验证：grep -c 'extractQualifiersFromContext' src/ptx_parser/ptx_visitor.cpp = 0（已移出主文件）
- [ ] 4.14 git commit -m "refactor(ptx_visitor): extract operand/qualifier helpers to ptx_visitor_operands.cpp"

## 5. Phase 4: 提取 X-Macro dispatch + include 聚合区（30 min）

- [ ] 5.1 MUST 新建 src/ptx_parser/ptx_visitor_dispatch.cpp
- [ ] 5.2 MUST 移动 ptx_visitor.cpp:905-925（10 文件 include 聚合区）至新文件
- [ ] 5.3 MUST 移动 ptx_visitor.cpp:927-931（X-Macro dispatch）至新文件
- [ ] 5.4 MUST 移动 VISITOR_<struct_kind> 宏定义展开部分至新文件
- [ ] 5.5 MUST 在 ptx_visitor.cpp 中将原 :905-931 区块替换为 #include "ptx_visitor_dispatch.cpp"
- [ ] 5.6 MUST 更新 src/ptx_parser/CMakeLists.txt 添加新源文件
- [ ] 5.7 MUST 验证：cmake --build build 通过
- [ ] 5.8 MUST 验证：./tests/ptx/test_all_ptx.sh 全绿
- [ ] 5.9 MUST 验证：ctest 全绿
- [ ] 5.10 MUST 验证：grep -c '#include "ptx_visitor_' src/ptx_parser/ptx_visitor.cpp = 1（仅 dispatch）
- [ ] 5.11 git commit -m "refactor(ptx_visitor): extract X-Macro dispatch + include aggregation"

## 6. Phase 5: 最终验证（15 min）

- [ ] 6.1 MUST 验证：wc -l src/ptx_parser/ptx_visitor.cpp ≤ 700
- [ ] 6.2 MUST 验证：./tests/ptx/test_all_ptx.sh 47/47 全绿
- [ ] 6.3 MUST 验证：ctest 全绿
- [ ] 6.4 MUST 验证：grep -c '#include "ptx_visitor_warp.cpp"' src/ptx_parser/ptx_visitor.cpp = 1
- [ ] 6.5 SHOULD 更新 src/ptx_parser/AGENTS.md 文档化 3 个新子文件
- [ ] 6.6 git commit -m "docs(ptx_parser): document 3 new ptx_visitor sub-files in AGENTS.md"（如更新 AGENTS.md）

## 7. 应用阶段

- [ ] 7.1 MUST 运行 openspec validate split-ptx-visitor-god-class --strict
- [ ] 7.2 MUST 通过所有验证后 archive 此 change 至 openspec/changes/archive/

## 验收

- ptx_visitor.cpp ≤ 700 行（从 1067 行）
- grep -c '#include "ptx_visitor_warp.cpp"' src/ptx_parser/ptx_visitor.cpp == 1
- ./tests/ptx/test_all_ptx.sh 全绿（47/47 目标）
- parser 单测 + CFG post-dominator 集成测试零回归（ADR-0007）
- 所有 5 个 Phase commit 独立可 revert
- ptx-lessons-learned Checklist H 全部勾选

## 关键约束（MUST/MUST NOT）

- MUST 执行 Checklist H 实证验证（lessons-learned §7, SKILL.md:539-555）
- MUST 保持 IR 输出字节级一致
- MUST 保持 extractQualifiersFromContext 签名不变（19 调用点）
- MUST visitTcgen05Inst 内 C3 fix parse-tree walk 注释 (:863-877) 行级随迁（§1）
- MUST NOT 修改 .g4 grammar 文件
- MUST NOT 修改 IR 数据结构
- MUST NOT 拆分 visitFunctionDecl（另立提案）
- SHOULD 新文件命名与现有 ptx_visitor_<category>.cpp 模式一致