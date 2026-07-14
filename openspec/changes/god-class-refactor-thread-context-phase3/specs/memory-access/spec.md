## ADDED Requirements

### Requirement: `MemoryAccessor` owns per-thread memory state

The `MemoryAccessor` class SHALL own the per-thread memory state previously held as public data members on `ThreadContext`:

- `void *shared_mem_space_` (private)
- `void *local_mem_space_` (private)
- `std::map<std::string, std::unique_ptr<Symtable>> *name2Sym_` (private, non-owning)
- `std::map<std::string, std::unique_ptr<Symtable>> *name2Share_` (private, non-owning)
- `CTAContext *cta_context_` (private, non-owning)
- `ThreadContext *thread_` (private, non-owning — for `acquire_register` calls)

`ThreadContext` exposes setters that delegate to `MemoryAccessor`:

- `void set_shared_memory_space(void*)` — sets `mem_access_->shared_mem_space_`
- `void *get_shared_memory_space() const` — returns `mem_access_->shared_mem_space_`
- `void set_local_memory_space(void*)` — sets `mem_access_->local_mem_space_`
- `void *get_local_memory_space() const` — returns `mem_access_->local_mem_space_`
- `void set_name2sym(std::map<...>*)` — sets `mem_access_->name2Sym_`
- `void set_name2share(std::map<...>*)` — sets `mem_access_->name2Share_`
- `void set_cta_context(CTAContext*)` — sets `mem_access_->cta_context_`

After Phase 3.0, all external direct assignments to `ThreadContext::shared_mem_space` / `local_mem_space` / `name2Sym` / `name2Share` SHALL go through the setters. Verified direct assignment at `src/ptxsim/core/cta_context.cpp:320` (`thread->shared_mem_space = shared_mem_space`) SHALL be migrated to `thread->set_shared_memory_space(shared_mem_space)` in Phase 3.0.

#### Scenario: Setter delegation keeps state synchronized

- **WHEN** `ThreadContext::set_shared_memory_space(addr)` is called from any caller (handler, `CTAContext`, etc.)
- **THEN** `MemoryAccessor::shared_mem_space_` SHALL equal `addr` after the call
- **AND** no divergence is possible (the field is `private` on `ThreadContext`, so external code cannot bypass the setter)

#### Scenario: External direct assignment is migrated in Phase 3.0

- **WHEN** Phase 3.0 is committed
- **THEN** `src/ptxsim/core/cta_context.cpp:320` SHALL read `thread->set_shared_memory_space(shared_mem_space)` (replacing the direct `thread->shared_mem_space = ...` assignment)
- **AND** all other external direct assignments identified by `grep -rn 'shared_mem_space\s*=' src/` SHALL be similarly migrated

#### Scenario: `init()` stores into private fields

- **WHEN** `ThreadContext::init()` is called
- **THEN** the parameters `name2Sym` and `name2Share` SHALL be stored via `set_name2sym()` and `set_name2share()` setters (which delegate to `MemoryAccessor`)
- **AND** `cta_context_` SHALL be stored via `set_cta_context()` setter

### Requirement: `MemoryAccessor::get_memory_addr` uses virtual method, not `std::function` callback

`MemoryAccessor::get_memory_addr` SHALL be a regular method on `MemoryAccessor`. The register lookup is `thread_->acquire_register()` — a direct virtual call on `ThreadContext`. The signature SHALL match the current `ThreadContext::get_memory_addr` signature:

```cpp
void *get_memory_addr(const AddrOperand &op, const std::vector<Qualifier> &qualifiers);
```

There SHALL be **no** `std::function<void*(const std::string&, std::vector<Qualifier>)> acquire_reg` parameter (this design was used in the cancelled Phase 3.1 and is explicitly rejected for this revision per design.md Decision 5).

`MemoryAccessor` SHALL hold a `ThreadContext *thread_` pointer (set in the constructor) so it can call `thread_->acquire_register(reg, qualifiers)` directly.

#### Scenario: Special register resolution

- **WHEN** `op` references a register name (e.g. `%tid.x`, `%ctaid.x`, `%clock`)
- **THEN** `get_memory_addr` SHALL invoke `thread_->acquire_register(reg, qualifiers)` (where `reg` is a `RegOperand` constructed from the register name)
- **AND** SHALL return the host address from the resulting register value

#### Scenario: Symbol table resolution via `name2Sym`

- **WHEN** `op` references a symbol name present in `name2Sym_`
- **THEN** `get_memory_addr` SHALL look up the symbol and return its absolute address

#### Scenario: Shared memory symbol table resolution via `name2Share`

- **WHEN** `op` references a symbol name present in `name2Share_`
- **THEN** `get_memory_addr` SHALL look up the shared memory symbol and return its absolute address

#### Scenario: Local memory symbol table resolution via `cta_context_->name2Local`

- **WHEN** `op` references a symbol name not in `name2Sym_` or `name2Share_` but present in `cta_context_->name2Local`
- **THEN** `get_memory_addr` SHALL look up the local memory symbol and return its absolute address

#### Scenario: `MemoryAccessor` is testable with a mock `ThreadContext`

- **WHEN** `tests/unit/core/test_memory_accessor.cpp` constructs a `MemoryAccessor` with a `ThreadContext*` argument
- **THEN** the test SHALL be able to invoke `get_memory_addr` with a register name and assert the returned address equals the value provided by `thread_->acquire_register`

### Requirement: `MemoryAccessor` provides data movement

`MemoryAccessor` SHALL provide `mov()` and `mov_data()` methods with the same signatures as the current `ThreadContext::mov` and `ThreadContext::mov_data`:

```cpp
void mov_data(void *src, void *dst, std::vector<Qualifier> &qualifiers);
void mov(void *from, void *to, const std::vector<Qualifier> &q);
```

The semantics SHALL match the current implementation (byte-count derivation via `getBytes(qualifiers)` and `memcpy` from `src` to `dst`). These methods SHALL be `non-static` so that future tests can mock the qualifier resolution path.

#### Scenario: mov_data copies bytes between operand addresses

- **WHEN** `mov_data(src, dst, qualifiers)` is called
- **THEN** `getBytes(qualifiers)` bytes SHALL be copied from `src` to `dst` via `memcpy`
- **AND** the operation SHALL have the same observable effect as the pre-Phase-3.1 implementation

#### Scenario: mov delegates to mov_data

- **WHEN** `mov(from, to, qualifiers)` is called
- **THEN** `mov_data(from, to, qualifiers)` SHALL execute (one-line delegation)

### Requirement: `MemoryAccessor::SHMEMADDR_` is a class static member

The current `static uint64_t SHMEMADDR = 0;` declared at `src/ptxsim/core/thread_context.cpp:25` SHALL migrate to `MemoryAccessor` as a **class static member**:

- Declaration in `include/ptxsim/core/memory_accessor.h`: `static uint64_t SHMEMADDR_;`
- Definition in `src/ptxsim/core/memory_accessor.cpp`: `uint64_t MemoryAccessor::SHMEMADDR_ = 0;`

Class static (not per-instance) preserves current behavior, where all `MemoryAccessor` instances in the program share one SHMEMADDR. This is correct because all threads in a CTA share the same shared memory base address, and the value is set once per kernel launch by `initialize_shared_memory`.

#### Scenario: First initialization sets SHMEMADDR_

- **WHEN** `MemoryAccessor::SHMEMADDR_` is 0 and `initialize_shared_memory(name, address)` is called
- **THEN** `SHMEMADDR_` SHALL be set to `address >> 32` (high 32 bits of the address)
- **AND** the call SHALL return normally

#### Scenario: Duplicate initialization with matching address is allowed

- **WHEN** `SHMEMADDR_` is non-zero and `initialize_shared_memory(name, address)` is called with an address whose high 32 bits match
- **THEN** the call SHALL return normally (no throw, no overwrite)

#### Scenario: Duplicate initialization with mismatched address throws

- **WHEN** `SHMEMADDR_` is non-zero and `initialize_shared_memory(name, address)` is called with an address whose high 32 bits differ
- **THEN** `InvalidMemoryAccessException` SHALL throw
- **AND** the test `tests/unit/core/test_memory_accessor.cpp` case 2 SHALL verify this

### Requirement: `ThreadContext` delegates memory methods transparently

`ThreadContext` SHALL retain all existing public method signatures for:

- `void *get_memory_addr(const AddrOperand&, const std::vector<Qualifier>&)`
- `void set_local_memory_space(void*)`
- `void mov(void*, void*, const std::vector<Qualifier>&)`
- `void mov_data(void*, void*, std::vector<Qualifier>&)`
- `void initialize_shared_memory(const std::string&, uint64_t)`

Each method SHALL forward to the corresponding `MemoryAccessor` method as a one-line inline delegator. No instruction handler or caller outside `ThreadContext` SHALL require modification beyond the Phase 3.0 setter migration.

#### Scenario: handler calls `get_memory_addr` through `ThreadContext` unchanged

- **WHEN** any instruction handler calls `context->get_memory_addr(op, qualifiers)`
- **THEN** `ThreadContext::get_memory_addr` SHALL forward to `mem_access_->get_memory_addr(op, qualifiers)`
- **AND** the returned pointer SHALL match what `MemoryAccessor::get_memory_addr` returns

#### Scenario: external `set_local_memory_space` calls remain unchanged at the API level

- **WHEN** `CTAContext` or other code calls `thread->set_local_memory_space(space)`
- **THEN** `ThreadContext::set_local_memory_space` SHALL store locally (Phase 3.0) and forward to `MemoryAccessor::set_local_memory_space(space)` (Phase 3.1)
- **AND** no divergence is possible (the field is `private` on `ThreadContext` after Phase 3.0)

### Requirement: `MemoryAccessor` unit tests are required

Per `AGENTS.md` §TDD 测试覆盖率检查清单 ("新数据结构 → 类型一单元测试必须"), `MemoryAccessor` SHALL have at least 3 type-1 unit tests in `tests/unit/core/test_memory_accessor.cpp`:

1. **Setter round-trip**: `set_shared_memory_space(addr)` followed by `get_shared_memory_space()` returns the same address
2. **`SHMEMADDR_` duplicate detection**: First `initialize_shared_memory` succeeds; second with matching address succeeds; second with mismatched address throws `InvalidMemoryAccessException` (covers the SHMEMADDR_ ownership decision)
3. **`get_memory_addr` with special register**: Construct a `MemoryAccessor` with a `ThreadContext*` mock; call `get_memory_addr` with `%tid.x`; assert the returned address matches `thread_->acquire_register` output

These tests SHALL be written in TDD Red phase **before** the `MemoryAccessor` implementation is committed (per design.md Phase 3.1 step 1).
