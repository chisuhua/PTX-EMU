# consolidate-sub-agents-md - Tasks

## 1. Phase 1: 根文件 SSOT 声明（10 min）

- [ ] 1.1 MUST 在根 `AGENTS.md` 顶部（标题下第一行）添加 SSOT 声明：
  ```markdown
  > **本文件是 PTX-EMU 项目规范的单一真值源（SSOT）。子目录 AGENTS.md 仅包含目录特有导航信息。**
  ```
- [ ] 1.2 MUST 确认根 `AGENTS.md` 包含完整 STRUCTURE 段
- [ ] 1.3 MUST 确认根 `AGENTS.md` 包含完整 COMMANDS 段（含 `. env.sh`、`cmake`、`ctest`、`test_all_ptx.sh`、`sanity.sh`、`regression.sh`）
- [ ] 1.4 MUST 确认根 `AGENTS.md` 包含完整 CONVENTIONS 段
- [ ] 1.5 MUST 确认根 `AGENTS.md` 包含完整 ANTI-PATTERNS 段
- [ ] 1.6 MUST 确认根 `AGENTS.md` 包含完整 CODE MAP 段
- [ ] 1.7 MUST 确认根 `AGENTS.md` 包含完整 CHILD AGENTS.MD 段
- [ ] 1.8 git commit -m "docs(agents-md): add SSOT declaration to root AGENTS.md"

## 2. Phase 2: 精简子文件 - 第一批（30 min）

- [ ] 2.1 MUST 精简 `src/ptxsim/core/AGENTS.md`（142 -> ~30 行）：
  - 添加引用声明首行
  - 保留 WHERE TO LOOK 表（该目录符号定位）
  - 保留目录特有的执行层次说明（GPU->SM->CTA->Warp->Thread 调用链细节）
  - 删除重复的 COMMANDS、CONVENTIONS、ANTI-PATTERNS
  - 保留 CODE MAP 中该目录符号子集
- [ ] 2.2 MUST 精简 `src/ptx_parser/AGENTS.md`（85 -> ~25 行）：
  - 添加引用声明首行
  - 保留 WHERE TO LOOK 表
  - 保留 PTX 解析特有信息（PtxVisitor、CFGBuilder 等）
  - 删除重复的公共规范
- [ ] 2.3 MUST 精简 `src/grammar/AGENTS.md`（84 -> ~25 行）：
  - 添加引用声明首行
  - 保留 WHERE TO LOOK 表
  - 保留 ANTLR4 语法文件特有信息
  - 删除重复的公共规范
- [ ] 2.4 MUST 精简 `src/ptxsim/instructions/AGENTS.md`（88 -> ~25 行）：
  - 添加引用声明首行
  - 保留 WHERE TO LOOK 表
  - 保留指令 handler 特有信息
  - 删除重复的公共规范
- [ ] 2.5 MUST 验证：每个精简后的文件首行包含引用声明
- [ ] 2.6 MUST 验证：每个精简后的文件保留 WHERE TO LOOK 表
- [ ] 2.7 git commit -m "docs(agents-md): deduplicate first batch of sub-agents-md (core, parser, grammar, instructions)"

## 3. Phase 3: 精简子文件 - 第二批（25 min）

- [ ] 3.1 MUST 精简 `src/ptxsim/AGENTS.md`（81 -> ~20 行）：
  - 添加引用声明首行
  - 保留 WHERE TO LOOK 表
  - 保留执行引擎概览特有的模块导航
  - 删除重复的公共规范
- [ ] 3.2 MUST 精简 `src/ptx_ir/AGENTS.md`（54 -> ~20 行）：
  - 添加引用声明首行
  - 保留 WHERE TO LOOK 表
  - 保留 IR 类型 + X-Macro + PTXIR 特有信息
  - 删除重复的公共规范
- [ ] 3.3 MUST 精简 `src/cudart/AGENTS.md`（46 -> ~15 行）：
  - 添加引用声明首行
  - 保留 WHERE TO LOOK 表
  - 保留 CUDA runtime 拦截特有信息
  - 删除重复的公共规范
- [ ] 3.4 MUST 精简 `src/ptxsim/barrier/AGENTS.md`（47 -> ~15 行）：
  - 添加引用声明首行
  - 保留 WHERE TO LOOK 表
  - 保留屏障状态机特有信息
  - 删除重复的公共规范
- [ ] 3.5 MUST 验证：每个精简后的文件首行包含引用声明
- [ ] 3.6 MUST 验证：每个精简后的文件保留 WHERE TO LOOK 表
- [ ] 3.7 git commit -m "docs(agents-md): deduplicate second batch of sub-agents-md (ptxsim, ptx_ir, cudart, barrier)"

## 4. Phase 4: 最终验证（10 min）

- [ ] 4.1 MUST 验证：子文件总行数减少 ≥ 50%
  ```bash
  # 精简前: 727 行
  # 精简后目标: ≤ 363 行
  cat src/*/AGENTS.md src/ptxsim/*/AGENTS.md | wc -l
  ```
- [ ] 4.2 MUST 验证：所有 8 个子文件首行包含 `> **公共规范**` 引用声明
  ```bash
  grep -rl "公共规范" src/*/AGENTS.md src/ptxsim/*/AGENTS.md | wc -l  # 应为 8
  ```
- [ ] 4.3 MUST 验证：所有 8 个子文件保留 WHERE TO LOOK 表
  ```bash
  grep -rl "WHERE TO LOOK" src/*/AGENTS.md src/ptxsim/*/AGENTS.md | wc -l  # 应为 8
  ```
- [ ] 4.4 MUST 验证：根 `AGENTS.md` 保留所有公共规范段落（STRUCTURE、COMMANDS、CONVENTIONS、ANTI-PATTERNS、CODE MAP、CHILD AGENTS.MD）
- [ ] 4.5 MUST 逐文件确认：删除的内容均在根文件中存在（无信息丢失）

## 5. 应用阶段

- [ ] 5.1 MUST 运行 `openspec validate consolidate-sub-agents-md --strict`
- [ ] 5.2 MUST 通过所有验证后 archive 此 change

## 验收

- 子 AGENTS.md 总行数减少 ≥ 50%（727 -> ≤ 363 行）
- 所有子文件首行包含 SSOT 引用声明
- 所有子文件保留 WHERE TO LOOK 表
- 根 AGENTS.md 包含完整公共规范定义
- 无规范内容丢失

## 关键约束（MUST/MUST NOT）

- MUST 保留每个子 AGENTS.md 的 WHERE TO LOOK 表
- MUST 保留根 AGENTS.md 的完整规范定义
- MUST 子文件首行添加引用声明
- MUST NOT 删除任何子 AGENTS.md 文件
- MUST NOT 修改 `.opencode/skills/` 下技能文件
- SHOULD 子文件引用声明格式统一
