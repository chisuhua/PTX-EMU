# public-device-api Specification

## Purpose
TBD - created by archiving change ptxemu-public-device-api. Update Purpose after archive.
## Requirements
### Requirement: PTX-EMU 端公共头 `ptxemu/device_api.h` 必须存在

PTX-EMU 仓 `include/ptxemu/device_api.h` 必须存在,且 MUST 包含 5 项内容:
1. `namespace ptxemu` 包裹
2. `IPtxEmuDevice` 抽象接口 (纯虚类, 含虚析构函数)
3. DTO 集合: `DeviceConfig` + `WarpStatus` + `LaneStatus` + `ThreadState` (enum)
4. 工厂函数: `std::unique_ptr<IPtxEmuDevice> create_device(const DeviceConfig&)` + `void destroy_device(IPtxEmuDevice*)`
5. `#define PTXEMU_API_VERSION 1` 守卫宏 + frozen 静态自检

#### Scenario: 公共头 header 包含 5 项契约内容
- **WHEN** 读取 `include/ptxemu/device_api.h`
- **THEN** 包含 namespace ptxemu + IPtxEmuDevice + 4 DTO + 2 factory + VERSION 守卫宏

#### Scenario: 公共头不可见 PTX-EMU 内部头
- **WHEN** 编译依赖 `device_api.h` 的消费者 TU
- **THEN** 编译命令中 `-I` 不含 `${PTXEMU_SRC}/include/ptxsim` 或 `${PTXEMU_SRC}/include/ptx_ir`(内部头)

### Requirement: `IPtxEmuDevice` 抽象接口 MUST 至少覆盖 S1 facade 12 callsites

`IPtxEmuDevice` 抽象方法集 MUST 覆盖 CppTLM S1 facade.cc 12 callsites 的所有用法,否则消费方无法 `add_subdirectory(external/PTX-EMU)`。

参考 S1 facade.cc (CppTLM 仓 `e608ba2f` 之前 archive 状态) 关键调用:
- `gpu_context.exe_once()` / `set_scoreboard()` 
- `sm_context.exe_once()` / `set_scoreboard()`
- `warp_context.exe_once()` / `thread->get_state()`
- `thread->set_active_mask()` / `set_next_pc()`

#### Scenario: S1 facade 全部 12 callsites 1:1 覆盖
- **WHEN** 实施 Phase 2 PR 时机械抽取 S1 facade.cc 调用点
- **THEN** `IPtxEmuDevice` 提供 1:1 对应虚方法, 无 facade.cc 调用点未覆盖

### Requirement: HSK-4 vendored 3 接口位置复用

`IPtxEmuDevice::attach_timing()` MUST 接收已 vendored 接口 (复用 HSK-4 位置),不允许重复定义:
- `IScoreboard*`
- `IPipelineLatencyProvider*`
- `ITensorCoreTiming*`

#### Scenario: HSK-4 3 接口零重复定义
- **WHEN** 包含 `ptxemu/device_api.h` 并查找 `IScoreboard` 等符号
- **THEN** 符号已通过 HSK-4 vendored header 提供 (per `cpp_ptlm_abi_guards.h` 不变)

### Requirement: 静态自检锁冻结 `PTXEMU_API_VERSION=1`

impl 层 MUST 提供 `static_assert(PTXEMU_API_VERSION == 1)`,任何公共签名变更需签发 HSK-9 bump,不允许就地 bump VERSION。

#### Scenario: VERSION 改动必须 HSK-9 触发
- **WHEN** 实施者尝试 `#define PTXEMU_API_VERSION 2`
- **THEN** CppTLM 仓 bump PR 必须先签发 HSK-9 (per HSK-7 类似 precedent)

### Requirement: `device_api.h` MUST 仅使用 C++17 子集特性

`device_api.h` 公共头 MUST 仅依赖 C++17 标准库特性,确保 CppTLM 仓 (HSK-8 spec §"CppTLM 端接受条件" #4 锁定 C++17) 可直接 `#include`。PTX-EMU 仓本身用 C++20 (`CMakeLists.txt:34 CMAKE_CXX_STANDARD 20`), 公共头编译到 C++17 兼容模式时不依赖 C++20 特性。

**禁止使用** (C++20 特性):
- `std::format` / `std::format_string`
- `requires` 表达式 / `concept` / `constexpr` 概念约束
- `<=>` spaceship operator
- designated initializers (`{ .field = value }`)
- `consteval` / `constinit`
- `[[likely]]` / `[[unlikely]]` attributes

**允许使用** (C++17 兼容):
- `std::optional` / `std::variant` / `std::string_view` / `std::any`
- `if constexpr` / structured bindings
- `inline` 变量 / `constexpr` lambda
- `std::unique_ptr` / `std::shared_ptr` / `std::make_unique`
- `mutable` / `[[nodiscard]]` / `[[deprecated]]`
- `noexcept` / `override` / `final`

#### Scenario: CppTLM C++17 项目可消费 device_api.h
- **WHEN** CppTLM 仓 (`CMAKE_CXX_STANDARD 17`) `#include <ptxemu/device_api.h>`
- **THEN** 0 编译错误 (无 C++20 特性依赖)

#### Scenario: device_api.h 0 C++20 特性引用
- **WHEN** `grep -E "std::format|requires |concept |<=>|consteval|constinit|likely\]\]" include/ptxemu/device_api.h`
- **THEN** 0 matches

#### Scenario: CI 静态门禁强制 C++17 子集
- **WHEN** `drift_check` workflow 跑 `grep -E "std::format|requires |concept |<=>|consteval|constinit|likely\]\]" include/ptxemu/device_api.h`
- **THEN** 0 matches 验证通过; 否则 workflow fail (防止 C++20 特性误用)

