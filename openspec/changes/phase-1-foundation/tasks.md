## 1. 准备与审计

- [ ] 1.1 创建隔离 worktree：`git worktree add ../ptx-emu-foundation -b feat/phase-1-foundation`；后续所有 commit 在 worktree 中进行
- [ ] 1.2 在 worktree 中 `. ./env.sh` 并验证环境：`which nvcc java cmake ninja`；记录 CUDA 版本（期望 `11.4.4`）
- [ ] 1.3 全项目 grep 关键路径建立审计清单：
  - `grep -rn "compile_commands" CMakeLists.txt build/ 2>/dev/null`
  - `grep -rn "CMAKE_EXPORT_COMPILE_COMMANDS" CMakeLists.txt`
  - `ls -la compile_commands.json build/compile_commands.json 2>&1`
  - `ls .github/workflows/`
  - `grep "TODO\|implement" .github/workflows/generate-ptxir.yml`
  - 审计结果存入 `/tmp/phase1_audit.txt` 用于 Task 2/3/4 决策
- [ ] 1.4 跑基线构建确认能正常 build：`. ./env.sh && rm -rf build && cmake -S . -B build && cmake --build build`；记录 build 时间（CI baseline 对比用）；NOTE：如失败，立即停止调查（不是本 change 范围）
- [ ] 1.5 跑基线 ctest 确认状态：`. ./env.sh && cd build && ctest -N` 记录 131 个测试目标；`ctest --output-on-failure -E "Disabled" 2>&1 | tail -50` 记录 baseline pass/fail 状态

## 2. T0-1：修复 compile_commands.json 生成

- [ ] 2.1 检查 `CMakeLists.txt` 是否设置 `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)`：`grep -n "CMAKE_EXPORT_COMPILE_COMMANDS" CMakeLists.txt`；记录行号和上下文
- [ ] 2.2 **MUST**：如设置缺失，在 `CMakeLists.txt` 顶层（CMake `cmake_minimum_required` 之后、`project()` 之前）添加 `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)`；NOTE：必须顶层而非子目录，避免传递性覆盖
- [ ] 2.3 检查根目录错误符号链：`ls -la compile_commands.json 2>/dev/null`；如指向不存在的路径（`/workspace/.../build/...` 等），记录符号链目标后删除
- [ ] 2.4 删除根目录错误符号链（如果存在）：`rm compile_commands.json`；NOTE：仅删除符号链，不删除文件本身（如有真实文件）
- [ ] 2.5 重新生成 build 让 cmake 自动输出：`rm -rf build && cmake -S . -B build`；验证 `build/compile_commands.json` 存在
- [ ] 2.6 **VERIFICATION**：执行所有 4 个验证命令：
  ```bash
  test -f build/compile_commands.json && echo "OK: exists" || echo "FAIL: missing"
  wc -l build/compile_commands.json  # 应 > 100 行（每个 TU 一条命令）
  python3 -c "import json; data = json.load(open('build/compile_commands.json')); print(f'TU count: {len(data)}')"  # 应 > 100
  test -L compile_commands.json && echo "FAIL: stale symlink at root" || echo "OK: no stale symlink"
  ```
- [ ] 2.7 **VERIFICATION**：用 clangd 验证（如果本地已装）：`clangd --check=build/compile_commands.json --query-driver=/usr/bin/clang++`；无解析错误即可
- [ ] 2.8 提交：`git add CMakeLists.txt build/compile_commands.json && git commit -m "fix(build): enable compile_commands.json generation (T0-1)"`；NOTE：如 `CMakeLists.txt` 未修改，仅提交 `build/` 目录生成物时，需确保 `build/` 不在 `.gitignore`（审计无此信息，需 `cat .gitignore | grep build`）

## 3. T0-3：存档 baseline + 发布审计 Errata

- [ ] 3.1 跑 sanity.sh 完整输出存档：`. ./env.sh && ./scripts/sanity.sh 2>&1 | tee docs/audits/baseline-2026-06-21.log`；NOTE：如 sanity.sh 不存在，先 `ls scripts/` 确认；如存在，stdout+stderr 必须都被捕获
- [ ] 3.2 手动汇总 ctest 三态追加到 baseline：
  ```bash
  cd build && ctest -N 2>&1 | tail -200 >> ../docs/audits/baseline-2026-06-21.log
  echo "--- DISABLED TESTS ---" >> ../docs/audits/baseline-2026-06-21.log
  cd build && ctest -N 2>&1 | grep -i "disabled" >> ../docs/audits/baseline-2026-06-21.log
  echo "--- PASS/FAIL SUMMARY ---" >> ../docs/audits/baseline-2026-06-21.log
  cd build && ctest --output-on-failure -E "Disabled" 2>&1 | tail -10 >> ../docs/audits/baseline-2026-06-21.log
  ```
- [ ] 3.3 验证 baseline 文件：`wc -l docs/audits/baseline-2026-06-21.log` 应 > 100 行；`head -30 docs/audits/baseline-2026-06-21.log` 确认有 sanity 输出；`grep -c "Disabled\|PASS\|FAIL" docs/audits/baseline-2026-06-21.log` 应有命中
- [ ] 3.4 决定 baseline 大小管理策略（决策点 Q2）：
  - 选项 A：直接提交（接受 10-50 MB 仓库膨胀）
  - 选项 B：git-lfs track `*.log`
  - 选项 C：压缩为 `baseline-2026-06-21.log.gz` 后提交
  - **推荐**：B（如已配 git-lfs）或 C（兼容性好）；实施对应 `.gitattributes` 或 `gzip` 命令
- [ ] 3.5 创建 Errata 文档骨架：`touch docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md`
- [ ] 3.6 填写 Errata 文档头部 + 8 项事实错误（参考 `docs/roadmap/README.md §6.2`）：
  - 1.1 ThreadContext public 字段：108 → 81（虚增 33%；grep 实证）
  - 1.2 Symtable 泄漏：5 → 7（漏 `src/ptxsim/core/cta_context.cpp:74,104` 2 处）
  - 1.3 `ptx_visiter` 影响文件：14 → 18（grep 实证）
  - 1.4 H2 反向依赖：🔴 H → 🟡 M（`execution_types.h:8` 是 4 值枚举）
  - 1.5 P0-1 membar 工作量：2 d → 2-3 d（未计 DUAL STATE 修复）
  - 1.6 Phase 1 顺序：P0-1→P0-2→P0-3 → **P0-4→P0-3→P0-2→P0-1**
  - 1.7 cudaStream_t 性质：漏写 delete → **destroy 是 STUB**（`cudart_sim.cpp:688,717`）
  - 1.8 PTX 8.7+ 选项 C 现状被低估（IMPLEMENT_SIMPLE_HANDLER 静默失败）
- [ ] 3.7 填写 Errata 严重遗漏章节（1 项）：`BarWarpSyncHandler` 仍用 deprecated `warp_state.wbars[]`（per AGENTS.md Phase 5 deferred）；影响：阻塞 Phase 2 T1-4；建议：作为隐藏 P0 加入 Phase 2
- [ ] 3.8 填写 Errata 优先级调整建议章节（采纳用户决策 2026-06-22）：
  - Phase 1 顺序：P0-4→P0-3→P0-2→P0-1
  - PTX 8.7+ 占位：A + PTX_WARN
  - CI 首次失败：xfail 不阻塞 PR
  - H2 反向依赖降为 M
- [ ] 3.9 在 Errata 末尾添加决策日志（5 条 2026-06-22 决策）
- [ ] 3.10 **VERIFICATION**：
  ```bash
  test -f docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md && echo "OK"
  wc -l docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md  # 应 > 100 行
  grep -c "1\.[1-8]" docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md  # 应 = 8
  grep -c "BarWarpSyncHandler" docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md  # 应 >= 1
  ```
- [ ] 3.11 在原审计 `HEALTH-AUDIT-2026-06-21.md` 末尾（报告元信息前）添加 Errata 引用链接；NOTE：仅追加引用，不修改审计内容（保持历史快照完整性）
- [ ] 3.12 提交：
  ```bash
  git add docs/audits/baseline-2026-06-21.log docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md docs/audits/HEALTH-AUDIT-2026-06-21.md .gitattributes 2>/dev/null
  git commit -m "docs(audit): archive baseline + publish errata (T0-3)"
  ```

## 4. T0-2：创建 CI workflow

- [ ] 4.1 创建 workflow 文件：`touch .github/workflows/build-test.yml`
- [ ] 4.2 决定 xfail 机制（决策点 Q1）：
  - 选项 A：Catch2 `[!mayfail]` tag（catch_amalgamated.hpp 支持）
  - 选项 B：CTestCustom.cmake.in 配置 `WILL_FAIL TRUE`
  - 选项 C：解析 ctest 输出后用 GitHub Actions `continue-on-error`
  - **推荐**：A（最轻量）；如 Catch2 不支持则降级到 C；NOTE：xfail 仅首次启用后标记，新测试不应直接 xfail
- [ ] 4.3 编写 `.github/workflows/build-test.yml` 完整内容：
  ```yaml
  name: build-and-test
  on:
    push:
      branches: [main]
    pull_request:
      branches: [main]
  jobs:
    build:
      runs-on: ubuntu-latest
      timeout-minutes: 30
      steps:
        - uses: actions/checkout@v4
        - name: Install CUDA Toolkit
          uses: Jimver/cuda-toolkit@v0.2.11
          with:
            cuda: '11.4.4'
        - name: Install build dependencies
          run: |
            sudo apt-get update
            sudo apt-get install -y default-jre ninja-build
        - name: Set up env
          run: . ./env.sh
        - name: Configure
          run: cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
        - name: Build
          run: cmake --build build -j$(nproc)
        - name: Test
          run: cd build && ctest --output-on-failure -E "Disabled" -j$(nproc)
  ```
- [ ] 4.4 验证 YAML 语法：`python3 -c "import yaml; yaml.safe_load(open('.github/workflows/build-test.yml'))"`；NOTE：缩进必须 2 空格，YAML 严格要求
- [ ] 4.5 验证 workflow 不破坏现有 `generate-ptxir.yml`：确认两个 workflow 文件并存；`generate-ptxir.yml` 不被修改
- [ ] 4.6 **VERIFICATION**：推送到测试分支触发 CI：
  ```bash
  git push origin feat/phase-1-foundation
  gh pr create --draft --title "[WIP] phase-1-foundation: enable quality gates" \
               --body "测试 CI workflow"
  # 等待 Actions 跑完
  gh pr checks
  ```
- [ ] 4.7 记录首次 CI 跑结果：
  - 哪些测试通过
  - 哪些测试失败（xfail 候选）
  - 哪些测试 Disabled（CI 已排除）
  - 总耗时（应为 10-20 分钟）
- [ ] 4.8 创建 xfail 跟踪 issues：每个失败测试一个 issue，标签 `xfail`，引用 `baseline-2026-06-21.log`
  ```bash
  # 示例（每个失败重复）
  gh issue create --title "xfail: <test_name>" \
                  --body "见 docs/audits/baseline-2026-06-21.log 第 N 行；跟踪 roadmap Phase X T-Y 修复" \
                  --label "xfail"
  ```
- [ ] 4.9 在 baseline-2026-06-21.log 末尾追加"CI xfail 跟踪 issues"章节，列出每个 issue 编号和对应测试名
- [ ] 4.10 标记 PR 为 ready for review：`gh pr ready`
- [ ] 4.11 提交 workflow：
  ```bash
  git add .github/workflows/build-test.yml docs/audits/baseline-2026-06-21.log
  git commit -m "ci: add build-test workflow with xfail policy (T0-2)"
  git push origin feat/phase-1-foundation
  ```

## 5. Phase 1 完成验证

- [ ] 5.1 验证 T0-1 完成：`test -f build/compile_commands.json && python3 -c "import json; print(len(json.load(open('build/compile_commands.json'))))" | grep -q "[1-9][0-9][0-9]"`（>100 TU）
- [ ] 5.2 验证 T0-2 完成：`.github/workflows/build-test.yml` 存在且 YAML 有效；至少一次 PR 触发 CI 跑通
- [ ] 5.3 验证 T0-3 完成：`docs/audits/baseline-2026-06-21.log` 含 sanity 输出 + ctest 三态汇总；`docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` 含 8 项事实错误 + 1 项遗漏
- [ ] 5.4 验证审计快照完整性：`git diff baa8c4e -- docs/audits/HEALTH-AUDIT-2026-06-21.md` 应只有 Errata 引用链接的追加（无内容修改）
- [ ] 5.5 验证 ADR 合规检查（`/AGENTS.md` 声明的 ADR）：CI 启用未涉及架构变更，**无 ADR 需更新**（确认此点）
- [ ] 5.6 合并 worktree 分支到 main：`git checkout main && git merge --no-ff feat/phase-1-foundation -m "phase-1-foundation: enable quality gates (T0-1/T0-2/T0-3)"`
- [ ] 5.7 清理 worktree：`git worktree remove ../ptx-emu-foundation`
- [ ] 5.8 通知用户 Phase 1 完成：
  - compile_commands.json 已生成
  - CI workflow 已启用
  - baseline 已存档
  - Errata 已发布
  - 准备启动 Phase 2 T1-1（替换 Symtable 泄漏）

## 6. ADR 合规检查

- [ ] 6.1 复核 ADR-0001 ~ ADR-0014：本 change 不涉及异常层次/PC 权威/独立线程调度等架构变更，**无 ADR 需新建或更新**
- [ ] 6.2 复核 `/AGENTS.md` 决策日志：本 change 实施 5 条用户决策（2026-06-22），如未来 ADR 涉及应引用本次决策
- [ ] 6.3 复核 `src/ptxsim/AGENTS.md` 约束：本 change 不涉及 barrier/active_mask/simt_stack，**无约束违反**

## 7. 风险缓解与回滚预案

- [ ] 7.1 R1 缓解（首次 CI 大量失败）：xfail 策略已在 Task 4.2/4.8 实施；baseline 已存档 Task 3.1
- [ ] 7.2 R2 缓解（CUDA 拉取慢）：CI 已用 `Jimver/cuda-toolkit@v0.2.11` 缓存 action；timeout 已设 30 min
- [ ] 7.3 R3 缓解（ANTLR Java 缺失）：CI workflow 已 apt-get install `default-jre`
- [ ] 7.4 R8 缓解（xfail ctest 标记）：Task 4.2 已决策 A/C 二选一
- [ ] 7.5 准备回滚预案：
  - CI workflow 回滚：`git revert HEAD~` 或 `git revert <commit-sha>` 然后 push
  - compile_commands.json 回滚：`git checkout HEAD~ -- CMakeLists.txt && rm -rf build && cmake -S . -B build`
  - Errata 回滚：`git revert <errata-commit-sha>`（Errata 文档本身可删可改，不影响审计快照）
- [ ] 7.6 记录回滚命令到 PR 描述（便于紧急回滚）