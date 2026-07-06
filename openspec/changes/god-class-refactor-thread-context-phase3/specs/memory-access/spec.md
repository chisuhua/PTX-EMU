## ADDED Requirements

### Requirement: MemoryAccessor owns per-thread memory state

The `MemoryAccessor` class SHALL own the per-thread memory state previously held as public data members on `ThreadContext`:
- `shared_mem_space` (private `shared_mem_space_` in `MemoryAccessor`)
- `local_mem_space` (private `local_mem_space_` in `MemoryAccessor`)
- `name2Sym`, `name2Share` symbol table references (non-owning pointers)
- `cta_context_` (non-owning pointer to shared/local memory symbol tables)

`ThreadContext` exposes setters `set_shared_memory_space(void*)` and `set_local_memory_space(void*)` that delegate to `MemoryAccessor`, ensuring no external code can bypass the delegation and cause state divergence.

#### Scenario: Setter delegation keeps state synchronized
- **WHEN** `ThreadContext::set_shared_memory_space(addr)` is called
- **THEN** `MemoryAccessor::shared_mem_space_` SHALL equal `addr`, no divergence possible

#### Scenario: backward compatibility for external setters
- **WHEN** external code (e.g. `CTAContext`) needs to set memory space pointers
- **THEN** `ThreadContext::set_local_memory_space(space)` SHALL forward to `MemoryAccessor::set_local_memory_space(space)` atomically

### Requirement: MemoryAccessor provides memory address resolution

The `MemoryAccessor` class SHALL provide `get_memory_addr()` that resolves address expressions to host pointers. The method SHALL accept:
- `const AddrOperand& fa` — the address expression
- `const std::vector<Qualifier>& qualifiers` — qualifier list
- `std::function<void*(const std::string&, std::vector<Qualifier>)> acquire_reg` — register lookup callback

This contract matches the prior `ThreadContext::get_memory_addr` signature with one change: the register lookup is now a `std::function` callback (lambda with capture) rather than a member function call, to decouple `MemoryAccessor` from `ThreadContext`'s register access implementation.

#### Scenario: Special register resolution
- **WHEN** `fa` references a register name (`%tid.x`, `%ctaid.x`, etc.)
- **THEN** `get_memory_addr` SHALL invoke the `acquire_reg` callback to resolve the address

#### Scenario: Symbol table resolution
- **WHEN** `fa` references a symbol name in `name2Sym`, `name2Share`, or `cta_context_->name2Local`
- **THEN** `get_memory_addr` SHALL look up the symbol and return its absolute address

### Requirement: MemoryAccessor provides data movement

The `MemoryAccessor` class SHALL provide `mov()` and `mov_data()` static methods for per-thread data movement. These SHALL delegate to existing `mov_data()` / `memcpy()` semantics already in `ThreadContext`.

#### Scenario: mov copies bytes between operand addresses
- **WHEN** `mov_data(src, dst, qualifiers)` is called
- **THEN** `getBytes(qualifiers)` bytes SHALL be copied from `src` to `dst`

#### Scenario: mov delegates to mov_data
- **WHEN** `mov(from, to, qualifiers)` is called
- **THEN** `mov_data(from, to, qualifiers)` SHALL execute

### Requirement: MemoryAccessor provides shared memory initialization

The `MemoryAccessor::initialize_shared_memory(name, address)` SHALL set the high 32 bits of `SHMEMADDR` from `address`, or throw `InvalidMemoryAccessException` if `SHMEMADDR` is already set and doesn't match.

This MUST happen atomically within `MemoryAccessor` so no other code path can race on `SHMEMADDR`.

#### Scenario: First initialization
- **WHEN** `SHMEMADDR` is 0 and `initialize_shared_memory(name, address)` is called
- **THEN** `SHMEMADDR = address >> 32` SHALL execute and return

#### Scenario: Duplicate initialization with mismatch
- **WHEN** `SHMEMADDR` is non-zero and `initialize_shared_memory(name, address)` is called with a different address
- **THEN** `InvalidMemoryAccessException` SHALL throw

### Requirement: ThreadContext delegates memory methods transparently

`ThreadContext` SHALL retain all existing public method signatures for `get_memory_addr()`, `set_local_memory_space()`, `mov()`, `mov_data()`, `initialize_shared_memory()`. Each method SHALL forward to the corresponding `MemoryAccessor` method.

No instruction handler or caller outside `ThreadContext` SHALL require modification.

#### Scenario: handler calls get_memory_addr through ThreadContext unchanged
- **WHEN** any instruction handler calls `context->get_memory_addr(fa, qualifiers)`
- **THEN** `ThreadContext::get_memory_addr` SHALL forward to `mem_access_->get_memory_addr`, passing a lambda that bridges register lookup to `acquire_register()`

#### Scenario: external set_local_memory_space calls remain unchanged
- **WHEN** `CTAContext` or other code calls `thread->set_local_memory_space(space)`
- **THEN** `ThreadContext::set_local_memory_space` SHALL store locally AND forward to `MemoryAccessor::set_local_memory_space`, no divergence
