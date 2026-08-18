# PTX-EMU Action Plan 修正 — HSK-6 索引处理

## 修正要点

**原错误**:我建议在 `2026-07-15-cpptlm-hsk-response.md` 添加 HSK-6 行,但:
1. 该文件实际在 **CppTLM 仓**(不是 PTX-EMU)
2. CppTLM 仓的 `hsk-response.md` 是 HSK 响应文件(consolidated response), 不是 HSK 索引
3. PTX-EMU 仓历史上**没有**集中式 HSK 索引文件

## 修订方案: A + C 组合

### A. PTX-EMU 仓不创建额外索引

PTX-EMU 仓 HSK 文件历史结构(per Oracle 调查):
```
docs/superpowers/hsk-drafts/2026-07-16/
├── HSK-1-cpptlm-bridge-abi.md     ← 单文件,无索引
├── HSK-2-antlr4-version.md       ← 单文件,无索引
└── HSK-3-libcpptlm-cudart-cmake.md ← 单文件,无索引

openspec/changes/archive/{change}/hsk-{N}.md  ← 关联 change 归档
```

**HSK-6 匹配此模式**: `docs/superpowers/specs/2026-08-18-hsk-6-cpptlm-bridge-deprecation.md` (单文件,无索引)。

### C. CppTLM 仓创建对应响应文件(推荐)

per CppTLM 仓 HSK 响应历史:
```
docs/superpowers/specs/
├── 2026-07-14-ptxemu-comprehensive-modification-plan.md
├── 2026-07-15-cpptlm-hsk-response.md
├── 2026-07-17-hsk-1-2-3-responses.md   ← consolidated response
├── 2026-07-17-hsk-4-5-responses.md   ← consolidated response
└── (待创建) 2026-08-18-hsk-6-response.md  ← CppTLM 对 HSK-6 的响应
```

CppTLM owner 创建 `2026-08-18-hsk-6-response.md`:
- ack HSK-6 接收
- 接受 P0-1 门禁(G-D4 17 条静态断言迁至 abi_guards.h)
- 确认 11 项删除清单
- 接受 HSK-5 关闭(`advance()` deferred → CANCELLED)
- Ack 截止: 2026-09-01

## 修订后 PTX-EMU Action Plan

| 步骤 | 操作 | 文件 / 位置 |
|---|---|---|
| 1 | **核对 HSK-6 草案内容** | 读 `docs/superpowers/specs/2026-08-18-hsk-6-cpptlm-bridge-deprecation.md` |
| 2 | **不创建集中式 HSK 索引**(per 修订方案 A) | N/A |
| 3 | **Push 到远程** | `cd /workspace/project/PTX-EMU && git push origin main` |
| 4 | **正式发出 HSK-6 公告**: 通过 PTX-EMU 仓 issue tracker / 邮件通知 CppTLM maintainer + UsrLinuxEmu Architecture Team | (per v2 §D5.1) |
| 5 | **跟踪 ack**(14 天窗口, 至 2026-09-01) | 收到 CppTLM maintainer + UsrLinuxEmu ack comment 后, 标记 HSK-6 = ACCEPTED |
| 6 | **(可选) CppTLM 端建议**: 在 CppTLM 仓创建 `2026-08-18-hsk-6-response.md` (consolidated response per historical pattern) | CppTLM 仓 |

## Reference

- Oracle session `ses_fef78854dffeLfDJh7p8ELuMLy` (确认 PTX-EMU 仓 HSK 文件分散结构)
- v2 §D5.1 HSK-6 协议(本 HSK-6 草案)
- CppTLM 历史响应模式(`2026-07-17-hsk-1-2-3-responses.md` + `2026-07-17-hsk-4-5-responses.md`)