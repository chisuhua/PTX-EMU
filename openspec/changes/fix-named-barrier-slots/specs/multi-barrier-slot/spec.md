# Capability: multi-barrier-slot

## Requirements

### REQ-1: Sixteen Barrier Slot Initialization
- **GIVEN** a WarpContext is initialized
- **WHEN** any slot is initialized via `WarpBarrier::init()`
- **THEN** up to 16 slots (`wbars[0..15]`) can be independently initialized with their own participation mask and reconvergence PC

### REQ-2: Slot-Routed bar.sync / bar.arrive
- **GIVEN** a `bar.sync` or `bar.arrive` instruction with operand `bar_id`
- **WHEN** the handler processes it
- **THEN** the barrier operation is routed to `wbars[bar_id]` exclusively

### REQ-3: Invalid Bar ID Handling
- **GIVEN** a `bar_id` value >= 16
- **WHEN** the barrier handler executes
- **THEN** an `std::out_of_range` exception is thrown

### REQ-4: Backward Compatibility
- **GIVEN** existing tests that use `bar_id = 0`
- **WHEN** the system runs
- **THEN** all existing barrier tests pass without modification