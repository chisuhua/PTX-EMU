# Spec: libptxemu-abi-freeze

## ADDED Requirements

### Requirement: `libptxemu_device.so` 5 `ptxemu_image_*` ABI 入口签名冻结

The system SHALL keep the 5 `extern "C"` ABI entries of `libptxemu_device.so` byte-identical to Phase 1 ship state:
- `ptxemu_image_load`
- `ptxemu_image_kernel_name`
- `ptxemu_image_execute`
- `ptxemu_image_unload`
- `ptxemu_module_version`

The `CPPTLM_MODULE_VERSION` constant MUST remain `1`. `SONAME` (`libptxemu_device.so.12`) and symlinks MUST NOT change.

#### Scenario: nm 验证 5 ABI 入口符号不变

- **WHEN** `nm -D build/lib/libptxemu_device.so | grep ptxemu_` is run after Phase 13 freeze
- **THEN** output is byte-identical to the Phase 1 ship baseline (same names, types, exported flag)

#### Scenario: SONAME 不 bump

- **WHEN** Phase 13 freezes ABI
- **THEN** `libptxemu_device.so.12` major version is NOT bumped; `libptxemu_device.so → libptxemu_device.so.12 → libptxemu_device.so.12.0` chain preserved

### Requirement: DL-isolated 测试通过

The system SHALL maintain the property that `dlopen libptxemu_device.so` works without `libcudart.so` dependency — i.e., all 5 ABI entries can be called in isolation.

#### Scenario: libptxemu_device.so 在无 libcudart.so 时 dlopen 成功

- **WHEN** test process does NOT load `libcudart.so` and calls `dlopen("libptxemu_device.so", RTLD_NOW)`
- **THEN** dlopen returns non-null handle and all 5 `ptxemu_image_*` symbols can be `dlsym`'d

### Requirement: in-flight unload 返回 busy

The system SHALL return non-zero error code from `ptxemu_image_unload` when the handle is in-flight (kernel still executing).

#### Scenario: in-flight unload 返回 busy

- **WHEN** `ptxemu_image_unload(in_flight_handle)` is called while a kernel from this handle is still executing
- **THEN** system returns non-zero error code (per architecture §10 items 16/24)

### Requirement: PTX-EMU 仓零跨仓污染

The system SHALL ensure `grep -r "UsrLinuxEmu\|TaskRunner" src/ include/ CMakeLists.txt` returns empty output — no include paths, no link dependencies, no header references to external repos.

#### Scenario: archive 前 grep 验证

- **WHEN** archive gate runs `grep -r "UsrLinuxEmu\|TaskRunner" src/ include/ CMakeLists.txt`
- **THEN** output is empty