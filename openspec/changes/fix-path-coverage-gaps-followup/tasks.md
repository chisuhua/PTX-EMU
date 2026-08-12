## 1. Phase 1 — Fix `integration_libptxemu_abi_baseline`

- [ ] 1.1 修改 `run_nm()` 函数使用 `awk '{print $2, $3}'` 剥离地址
- [ ] 1.2 重新生成 `libptxemu_abi_baseline.txt`（不含地址）
- [ ] 1.3 添加 `baselines/README.md` 说明如何重新生成
- [ ] 1.4 验证 `ctest -R integration_libptxemu_abi_baseline --output-on-failure` PASS
- [ ] 1.5 验证 `./scripts/regression.sh --no-build` 全量通过

## 2. 验收

- [ ] 2.1 验证 AC-1: `integration_libptxemu_abi_baseline` ctest PASS（不再 failing）
- [ ] 2.2 验证 AC-2: 完整 ctest 256/256 通过（修复 1 个既有失败）
- [ ] 2.3 验证 AC-3: `libptxemu_abi_baseline.txt` 不含地址（仅 type + name）
- [ ] 2.4 验证 AC-4: `baselines/README.md` regeneration 命令可执行

## 3. 归档

- [ ] 3.1 执行 `openspec validate fix-path-coverage-gaps-followup`
- [ ] 3.2 执行 `openspec archive fix-path-coverage-gaps-followup --yes`
- [ ] 3.3 验证归档目录创建