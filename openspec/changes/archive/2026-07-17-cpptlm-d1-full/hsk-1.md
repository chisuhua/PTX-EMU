# HSK-1: cppTLMBridge 头文件首发 commit hash

> **状态**: ✅ **已发出（待 CppTLM 确认 / rebased / CI 验证 12 端点 static_assert）**
> **回传目标**: CppTLM Team (`#cpptlm-integration` Slack 频道 / PR comment)
> **承诺时间**: D1 开工前
> **形式**: git commit hash + ABI 字段快照
> **PTX-EMU 侧**: 消息草稿 + 手动复制发送已完成（2026-07-15）— 发送侧不可再推进
> **CppTLM 侧 (pending)**: 接收确认 / rebase / 12 端点 `static_assert` — 外部闭环，不阻塞本地交付

---

## 📤 准备发给 CppTLM 团队的完整消息

```
Subject: [HSK-1] PTX-EMU cpptlm_bridge.h ABI source-of-truth ready — please rebase

Cc: CppTLM Team (#cpptlm-integration Slack)

CppTLM Team,

PTX-EMU 仓库已完成 CppTLMBridge ABI 真值源首发。

======================== 关键事实 ========================

- Commit hash:    `8dc000eca9f78e8ee017eafcb305eb4ca62ffd6d`
- ABI path:       include/cudart/cpptlm_bridge.h
- CPPTLMBRIDGE_VERSION: 1
- ABI 真值源定位: PTX-EMU 是提供方（CppTLM 通过 ExternalProject_Add 消费）
- License:        与 PTX-EMU 同源（参考 PTX-EMU 根 LICENSE）

======================== ABI 字段（5 个纯虚方法）========================

1. int version() const
   → 返回 CPPTLMBRIDGE_VERSION (当前 = 1)

2. int submit_kernel(
       uint64_t kernel_id, const char* kernel_name,
       uint32_t grid_x, grid_y, grid_z,
       uint32_t block_x, block_y, block_z,
       const void** kernel_args, size_t args_count,
       size_t shared_mem, uint64_t stream_id)
   → 异步提交，立即返回
   → 0=成功，cudaError_t 错误码=失败
   → **契约**: CppTLM 必须在调用栈内 deep-copy kernel_args（PTX-EMU host 端 args 内存可能在返回后失效）

3. uint64_t poll_kernel(uint64_t kernel_id)
   → 0=已完成，>0=剩余 cycles，UINT64_MAX=未知 kernel_id

4. int synchronize_stream(uint64_t stream_id)
   → 同步 stream 上所有 pending kernels

5. uint64_t global_access(uint64_t device_addr, uint64_t val, uint8_t type)
   → 返回 NoC 路由延迟（cycle 数），UINT64_MAX=地址未映射
   → **Phase 8.B 语义**: timing-only 预计算，数据立即在 PTX-EMU SimpleMemory 完成

======================== ABI 字节级契约 ========================

✅ header includes: <cstddef> + <cstdint> + <cuda_runtime.h>（cudaStream_t 来源）
✅ static_assert: sizeof(cudaStream_t) <= sizeof(uint64_t)
✅ 全局指针: extern CppTLMBridge* g_cpptlm_bridge（默认 nullptr）
✅ 编译期: PTX-EMU `cmake --build build --target cudart` PASS

======================== 后续 bump 流程 ========================

1. 修改 cpptlm_bridge.h 接口签名（添加/删除参数）
2. bump CPPTLMBRIDGE_VERSION（如 1 → 2）
3. 在 PTX-EMU 主分支 commit + 通知（本 HSK 重新发出）
4. CppTLM 通过 ExternalProject_Add 自动拉取新 commit hash → 同步 rebase
5. CppTLM MemoryBridge::version() 返回新版本号

======================== 引用 ========================

- ADR-0021 (PTX-EMU docs/adr/ADR-0021-cpptlm-d1-full-integration.md): D-PTX-1 决策
- openspec/changes/cpptlm-d1-full/ (change artifacts)
- 综合任务书 §2.1 Task #1 (cppTLMBridge 接口定义)
- 协作同步 §5 (cppTLMBridge 接口定义补充)

======================== 请求 ========================

请 CppTLM 团队：

1. 在 ExternalProject_Add 中 git tag <COMMIT_HASH>
2. 验证 `cpptlm_bridge.h` 与 PTX-EMU commit hash 一致
3. CppTLM MemoryBridge::version() 返回 1
4. CppTLM CI 双重 static_assert（12 端点枚举双向一致）

确认收到后回复。

— PTX-EMU Architecture Team
```

---

## 🔧 使用方法（PTX-EMU 内部）

实施 Phase 1（任务 1.5）后：

1. 实施 cpptlm_bridge.h + cpptlm_bridge_impl.h
2. 验证编译通过
3. 运行：
   ```bash
   cd /workspace/project/PTX-EMU
   git add include/cudart/cpptlm_bridge.h include/cudart/cpptlm_bridge_impl.h
   git commit -m "feat(cudart): CppTLMBridge ABI source-of-truth with CPPTLMBRIDGE_VERSION=1 (HSK-1)

   Zero CppTLM dependency. ABI 真值源 — CppTLM 通过 ExternalProject_Add 引用。

   Refs:
   - ADR-0021 (cpptlm-d1-full-integration, D-PTX-1 + D-PTX-6)
   - CppTLM docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #1
   - CppTLM docs/superpowers/specs/PTX-EMU-README.md §10.3
   - openspec/changes/cpptlm-d1-full"
   ```
4. 记录 commit hash：
   ```bash
   PHASE1_HASH=$(git rev-parse HEAD)
   echo "HSK-1 commit hash: $PHASE1_HASH"
   ```
5. 用此 hash 替换上方消息中 `<TO BE FILLED — Phase 1 commit hash>` 占位符
6. 发送给 CppTLM：
   - Slack: `#cpptlm-integration` 频道（如果存在）
   - 或：PR comment on `openspec/changes/cpptlm-d1-full/`

---

## 🔍 验证清单（发出前）

- [x] Phase 1 commit 在本地 main（未 push 到 origin）
- [x] Commit hash 与消息中占位符一致: `8dc000eca9f78e8ee017eafcb305eb4ca62ffd6d`
- [x] CPPTLMBRIDGE_VERSION 仍是 1（未意外 bump）
- [x] `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` 编译通过
- [x] 5 个虚方法签名与消息描述一致
- [x] `g_cpptlm_bridge` extern 声明 + 默认 nullptr
- [x] header 只 include `<cstddef>` + `<cstdint>` + `<cuda_runtime.h>`（无 CppTLM 依赖）

---

## 📋 跟踪

发送后请更新本文件：
- [x] Commit hash 锁定: `8dc000eca9f78e8ee017eafcb305eb4ca62ffd6d`
- [x] 发送日期: 2026-07-15
- [x] 发送渠道: 用户手动复制（无 Slack/邮件）
- [x] Commit `8dc000ec` 在本地 main（未 push 到 origin）
- [ ] CppTLM 确认收到:
- [ ] CppTLM rebased:
- [ ] CppTLM CI 验证 12 端点 static_assert:

---

**最后更新**: 2026-07-16（HSK-1 commit `8dc000ec` 在本地 main，未 push 到 origin；⏳ **待发出**（ADR-0021 Accepted + Phase A commit 完成时启用））
