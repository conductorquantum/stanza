# Stanza Device Configuration Guide

A comprehensive guide to creating and understanding Stanza device configuration files.

## Table of Contents

1. [Introduction](#introduction)
2. [Configuration File Structure](#configuration-file-structure)
3. [Pad Definitions](#pad-definitions)
4. [Conditional Filtering & Inheritance](#conditional-filtering--inheritance)
5. [Groups System](#groups-system)
6. [Routine Configuration](#routine-configuration)
7. [Instrument Configuration](#instrument-configuration)
8. [Complete Examples](#complete-examples)
9. [Validation Rules Reference](#validation-rules-reference)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)

---

## Introduction

Stanza uses YAML configuration files to define quantum devices, including:
- **Pads**: Physical electrodes (gates, contacts, GPIOs)
- **Groups**: Logical subsets of pads for isolated testing
- **Routines**: Automated characterization procedures
- **Instruments**: Hardware connections and settings

The configuration system supports advanced features like:
- Conditional pad inheritance (omitted vs. explicit specification)
- Multi-group device characterization
- Automatic device filtering by group
- Parameter inheritance in nested routines

---

## Configuration File Structure

### Top-Level Schema

```yaml
name: string                          # Required: Device identifier
gates: dict[str, Gate]               # Gate electrode definitions
contacts: dict[str, Contact]         # Contact pad definitions
gpios: dict[str, GPIO]               # GPIO pin definitions
groups: dict[str, DeviceGroup]       # Logical groupings of pads
routines: list[RoutineConfig]        # Test/characterization routines
instruments: list[InstrumentConfig]  # Required: Hardware configuration
```

### Minimal Valid Configuration

```yaml
name: "MyDevice"

gates:
  G1:
    type: PLUNGER
    control_channel: 1
    v_lower_bound: -3.0
    v_upper_bound: 3.0

contacts:
  IN:
    type: SOURCE
    measure_channel: 1
    v_lower_bound: 0.0
    v_upper_bound: 0.0

instruments:
  - name: qdac2
    type: GENERAL
    ip_addr: 192.168.1.100
    port: 5025
    slew_rate: 100.0
    measurement_duration: 1e-3
    sample_time: 10e-6
```

---

## Pad Definitions

### Gate Definition

Gates are the primary control electrodes for quantum dots.

```yaml
gates:
  G1:
    type: PLUNGER | BARRIER | RESERVOIR | SCREEN
    control_channel: int              # Optional: 0-1024
    measure_channel: int              # Optional: 0-1024
    breakout_channel: int             # Optional: 0-1024
    v_lower_bound: float              # Required if control_channel present
    v_upper_bound: float              # Required if control_channel present
```

#### Gate Types

| Type | Purpose | Can Be Shared? |
|------|---------|----------------|
| **PLUNGER** | Controls quantum dot charge occupancy | No |
| **BARRIER** | Controls tunneling barriers between dots | No |
| **RESERVOIR** | Source/drain reservoir gates | **Yes** (only type that can be shared) |
| **SCREEN** | Screening gates for electric field management | No |

#### Field Descriptions

- **`type`** (required): Gate functional type
- **`control_channel`** (optional): Channel for voltage control (0-1024)
- **`measure_channel`** (optional): Channel for voltage measurement (0-1024)
- **`breakout_channel`** (optional): Physical breakout box channel mapping
- **`v_lower_bound`** (required if `control_channel` present): Minimum safe voltage
- **`v_upper_bound`** (required if `control_channel` present): Maximum safe voltage

**Note**: At least one of `control_channel` or `measure_channel` must be specified.

#### Example

```yaml
gates:
  # Plunger gate for dot 1
  G1:
    type: PLUNGER
    control_channel: 1
    measure_channel: 10
    breakout_channel: 1
    v_lower_bound: -3.0
    v_upper_bound: 3.0

  # Barrier between dots
  G2:
    type: BARRIER
    control_channel: 2
    v_lower_bound: -3.0
    v_upper_bound: 3.0

  # Shared reservoir gate
  RES1:
    type: RESERVOIR
    control_channel: 5
    v_lower_bound: -3.0
    v_upper_bound: 3.0
```

### Contact Definition

Contacts are source/drain electrodes for current measurement.

```yaml
contacts:
  IN:
    type: SOURCE | DRAIN
    control_channel: int              # Optional: for biasing
    measure_channel: int              # Optional: for current measurement
    breakout_channel: int             # Optional
    v_lower_bound: float              # Required if control_channel present
    v_upper_bound: float              # Required if control_channel present
```

#### Contact Types

- **`SOURCE`**: Electron/hole injection contact
- **`DRAIN`**: Electron/hole collection contact

#### Example

```yaml
contacts:
  # Source contact (biased)
  IN:
    type: SOURCE
    control_channel: 20
    measure_channel: 20
    v_lower_bound: -0.001
    v_upper_bound: 0.001

  # Drain contact for measurement
  OUT_A:
    type: DRAIN
    measure_channel: 21
    v_lower_bound: -3.0
    v_upper_bound: 3.0

  # Second drain for differential measurement
  OUT_B:
    type: DRAIN
    measure_channel: 22
    v_lower_bound: -3.0
    v_upper_bound: 3.0
```

### GPIO Definition

GPIOs are general-purpose input/output pins for control signals.

```yaml
gpios:
  VDD:
    type: INPUT | OUTPUT
    control_channel: int              # Optional
    measure_channel: int              # Optional
    breakout_channel: int             # Optional
    v_lower_bound: float              # Required if control_channel present
    v_upper_bound: float              # Required if control_channel present
```

#### GPIO Types

- **`INPUT`**: Reading digital/analog signals
- **`OUTPUT`**: Writing digital/analog signals

#### Typical Uses

- Power supply control (VDD, VSS)
- Device multiplexer addressing (A0-A5)
- Substrate bias
- Temperature sensors
- External trigger signals

#### Example

```yaml
gpios:
  # Power supply
  VDD:
    type: INPUT
    control_channel: 30
    v_lower_bound: 0.0
    v_upper_bound: 3.3

  VSS:
    type: INPUT
    control_channel: 31
    v_lower_bound: -3.3
    v_upper_bound: 0.0

  # Multiplexer address lines
  A0:
    type: INPUT
    control_channel: 32
    v_lower_bound: -5.0
    v_upper_bound: 5.0

  A1:
    type: INPUT
    control_channel: 33
    v_lower_bound: -5.0
    v_upper_bound: 5.0
```

### Channel Assignment Rules

**Critical**: All channel numbers must be globally unique across all pads.

```yaml
# INVALID - Duplicate channel
gates:
  G1: {control_channel: 1, ...}
contacts:
  IN: {control_channel: 1, ...}  # ERROR: Channel 1 already used!

# VALID - Unique channels
gates:
  G1: {control_channel: 1, ...}
contacts:
  IN: {control_channel: 2, ...}  # OK: Different channel
```

**Error message if violated**:
```
Duplicate channels found: gate 'G1' control_channel 1, contact 'IN' control_channel 1
```

---

## Conditional Filtering & Inheritance

**This is the most important and nuanced feature of the configuration system.**

When you define groups and omit certain pad types, Stanza uses **conditional filtering** to determine which pads are accessible. The behavior differs between pad types.

### The Core Principle

When filtering a device by group:
- **Gates**: ALWAYS explicitly filtered (only listed gates accessible)
- **Contacts**: Conditionally filtered (ALL if omitted, ONLY specified if explicit)
- **GPIOs**: Conditionally filtered (ALL if omitted, ONLY specified if explicit)

### Implementation Detail

Stanza uses Pydantic's `model_fields_set` to detect whether a field was explicitly provided in the YAML:

```python
# From device.py filter_by_group() method
if "contacts" in group.model_fields_set:
    # User explicitly wrote "contacts: [...]" in YAML
    # Include ONLY the specified contacts
    group_pad_names.update(group.contacts)
else:
    # User didn't write "contacts:" at all in YAML
    # Include ALL device contacts
    all_contacts = [name for name, config in self.channel_configs.items()
                    if config.pad_type == PadType.CONTACT]
    group_pad_names.update(all_contacts)
```

### Filtering Behavior Table

| Pad Type | Omitted from Group | Empty List `[]` | Specified List |
|----------|-------------------|-----------------|----------------|
| **Gates** | Not included | Not included | ONLY specified |
| **Contacts** | ALL included | NOT included | ONLY specified |
| **GPIOs** | ALL included | NOT included | ONLY specified |

### Scenario Examples

#### Scenario 1: Omit Contacts and GPIOs (Common Pattern)

```yaml
groups:
  control:
    gates: [G1, G2]
    # contacts: omitted
    # gpios: omitted
```

**Result when filtering**:
- ✓ Gates: `[G1, G2]` (only specified)
- ✓ Contacts: `[IN, OUT_A, OUT_B]` (ALL device contacts)
- ✓ GPIOs: `[VDD, VSS, A0, A1, A2]` (ALL device GPIOs)

**Use case**: Infrastructure pads (power, measurement, mux control) should be accessible to all groups.

#### Scenario 2: Explicitly Specify Empty List

```yaml
groups:
  control:
    gates: [G1, G2]
    contacts: []     # Explicit empty list
    gpios: []        # Explicit empty list
```

**Result when filtering**:
- ✓ Gates: `[G1, G2]` (only specified)
- ✓ Contacts: `[]` (none)
- ✓ GPIOs: `[]` (none)

**Use case**: Isolated gate testing without any contacts or GPIOs.

#### Scenario 3: Explicitly Specify Subset

```yaml
groups:
  control:
    gates: [G1, G2]
    contacts: [IN, OUT_A]    # Explicit subset
    # gpios: omitted
```

**Result when filtering**:
- ✓ Gates: `[G1, G2]` (only specified)
- ✓ Contacts: `[IN, OUT_A]` (only specified)
- ✓ GPIOs: `[VDD, VSS, A0, A1]` (ALL device GPIOs)

**Use case**: Group needs specific measurement contacts, but all GPIOs for mux control.

#### Scenario 4: Different Filtering Per Group

```yaml
# Device has: G1, G2, G3, IN, OUT_A, OUT_B, VDD, VSS, A0

groups:
  control:
    gates: [G1, G2]
    contacts: [IN, OUT_A]    # Explicit
    # gpios: omitted         # Gets ALL

  sensor:
    gates: [G3]
    # contacts: omitted       # Gets ALL
    gpios: [VDD, VSS]        # Explicit
```

**Control group filtering**:
- Gates: `[G1, G2]`
- Contacts: `[IN, OUT_A]` (only specified)
- GPIOs: `[VDD, VSS, A0]` (ALL - omitted)

**Sensor group filtering**:
- Gates: `[G3]`
- Contacts: `[IN, OUT_A, OUT_B]` (ALL - omitted)
- GPIOs: `[VDD, VSS]` (only specified)

### Why This Design?

**Gates** are always explicit because:
- Incorrect gate voltages can damage quantum dots
- Gate sharing (except RESERVOIR) is physically problematic
- Safety requires explicit specification

**Contacts/GPIOs** are conditional because:
- Infrastructure (power, mux control) naturally shared
- Default "all accessible" is usually desired
- Opt-in restriction available when needed

### Practical Implications

#### Infrastructure Sharing Pattern

When you have shared infrastructure (power supplies, mux control), omit those fields:

```yaml
# 64-device multiplexed system
gpios:
  VDD: {...}
  VSS: {...}
  A0: {...}   # Mux address bit 0
  A1: {...}   # Mux address bit 1
  A2: {...}   # Mux address bit 2
  A3: {...}   # Mux address bit 3
  A4: {...}   # Mux address bit 4
  A5: {...}   # Mux address bit 5

groups:
  side_A:
    gates: [G1, G2, G3, ...]
    contacts: [IN_A_B, OUT_A]
    # gpios omitted - ALL accessible for mux control

  side_B:
    gates: [G7, G8, G9, ...]
    contacts: [IN_A_B, OUT_B]
    # gpios omitted - ALL accessible for mux control
```

Both groups can access mux control pins without duplicating the list.

#### Isolated Measurement Pattern

When groups have separate measurement paths, explicitly specify:

```yaml
groups:
  control:
    gates: [G1, G2]
    contacts: [IN, OUT_A]     # Only control measurement
    gpios: [MUX1]             # Only control mux

  sensor:
    gates: [G3, G4]
    contacts: [IN, OUT_B]     # Only sensor measurement
    gpios: [MUX2]             # Only sensor mux
```

Each group has isolated measurement infrastructure.

### Testing Conditional Filtering

You can verify filtering behavior in Python:

```python
from stanza.utils import load_device_config, device_from_config

# Load config
config = load_device_config("my_device.yaml")
device = device_from_config(config)

# Filter by group
control_device = device.filter_by_group("control")

# Check what's accessible
print(f"Gates: {list(control_device.gates)}")
print(f"Contacts: {list(control_device.contacts)}")
print(f"GPIOs: {list(control_device.gpios)}")

# Verify conditional filtering
assert "VDD" in control_device.gpios  # If omitted from group
```

---

## Groups System

Groups define logical subsets of pads for isolated testing and characterization.

### DeviceGroup Schema

```yaml
groups:
  control:
    name: string               # Optional: Human-readable name
    gates: list[str]          # Required: List of gate names
    contacts: list[str]       # Optional: List of contact names (see conditional filtering)
    gpios: list[str]          # Optional: List of GPIO names (see conditional filtering)
```

### Validation Rules

#### Rule 1: Pad Existence

All referenced pads must exist in the device configuration.

```yaml
# INVALID
gates:
  G1: {...}

groups:
  control:
    gates: [G1, G999]  # ERROR: G999 doesn't exist
```

**Error message**:
```
Group 'control' references unknown gate 'G999'
```

#### Rule 2: Gate Sharing Rules

**Only RESERVOIR gates can be shared between groups.**

```yaml
# VALID - RESERVOIR shared
gates:
  G1: {type: PLUNGER, ...}
  RES1: {type: RESERVOIR, ...}

groups:
  control:
    gates: [G1, RES1]
  sensor:
    gates: [G2, RES1]  # OK: RES1 is RESERVOIR

# INVALID - PLUNGER shared
groups:
  control:
    gates: [G1]
  sensor:
    gates: [G1]  # ERROR: PLUNGER cannot be shared
```

**Error message**:
```
Gate 'G1' referenced by group 'sensor' already assigned to group 'control'.
Only RESERVOIR gates can be shared between groups.
```

**Rationale**: RESERVOIR gates are designed to be common voltage references. Other gate types controlling individual quantum dots should not be shared to prevent interference.

#### Rule 3: Contact Sharing

Contacts CAN be shared between groups (explicitly specified).

```yaml
# VALID - Contact shared
contacts:
  IN: {type: SOURCE, ...}

groups:
  control:
    contacts: [IN, OUT_A]
  sensor:
    contacts: [IN, OUT_B]  # OK: IN shared
```

**Use case**: Common source contact for multiple quantum dots.

#### Rule 4: GPIO Sharing

GPIOs CAN be shared between groups (explicitly specified).

```yaml
# VALID - GPIO shared
gpios:
  VDD: {...}
  VSS: {...}

groups:
  control:
    gpios: [VDD, VSS]
  sensor:
    gpios: [VDD, VSS]  # OK: Power supplies shared
```

**Use case**: Power supplies, common control signals.

### Group Filtering Example

```yaml
name: "DualDot"

gates:
  G1: {type: PLUNGER, control_channel: 1, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G2: {type: BARRIER, control_channel: 2, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G3: {type: PLUNGER, control_channel: 3, v_lower_bound: -3.0, v_upper_bound: 3.0}
  RES1: {type: RESERVOIR, control_channel: 4, v_lower_bound: -3.0, v_upper_bound: 3.0}

contacts:
  IN: {type: SOURCE, measure_channel: 1, v_lower_bound: 0.0, v_upper_bound: 0.0}
  OUT_A: {type: DRAIN, measure_channel: 2, v_lower_bound: -3.0, v_upper_bound: 3.0}
  OUT_B: {type: DRAIN, measure_channel: 3, v_lower_bound: -3.0, v_upper_bound: 3.0}

gpios:
  VDD: {type: INPUT, control_channel: 10, v_lower_bound: 0.0, v_upper_bound: 3.3}

groups:
  dot1:
    name: "Quantum Dot 1"
    gates: [G1, G2, RES1]       # G1, G2, shared RES1
    contacts: [IN, OUT_A]        # Measure through OUT_A
    # gpios omitted - gets VDD

  dot2:
    name: "Quantum Dot 2"
    gates: [G3, RES1]            # G3, shared RES1
    contacts: [IN, OUT_B]        # Measure through OUT_B
    # gpios omitted - gets VDD
```

**Python usage**:
```python
device = device_from_config(config)

# Filter to dot1
dot1_device = device.filter_by_group("dot1")
print(dot1_device.gates)     # ['G1', 'G2', 'RES1']
print(dot1_device.contacts)  # ['IN', 'OUT_A']
print(dot1_device.gpios)     # ['VDD'] - inherited

# Filter to dot2
dot2_device = device.filter_by_group("dot2")
print(dot2_device.gates)     # ['G3', 'RES1']
print(dot2_device.contacts)  # ['IN', 'OUT_B']
print(dot2_device.gpios)     # ['VDD'] - inherited
```

---

## Routine Configuration

Routines define automated characterization procedures.

### RoutineConfig Schema

```yaml
routines:
  - name: string                    # Required: Routine function name
    group: string                   # Optional: Group to filter device by
    parameters: dict[str, Any]      # Optional: Parameters passed to routine
    routines: list[RoutineConfig]   # Optional: Nested sub-routines
```

### Basic Routine

```yaml
routines:
  - name: leakage_test
    parameters:
      leakage_threshold_resistance: 50e6
      num_points: 5
```

This calls the `leakage_test` routine function with the specified parameters.

### Routine with Group

```yaml
routines:
  - name: leakage_test
    group: control             # Device filtered to control group
    parameters:
      leakage_threshold_resistance: 50e6
```

The routine receives a filtered device containing only pads from the `control` group.

### Nested Routines

```yaml
routines:
  - name: health_check
    parameters:
      charge_carrier_type: HOLE
      step_size: 1e-2
    routines:
      - name: noise_floor_measurement
        parameters:
          num_points: 10
          # charge_carrier_type inherited
          # step_size inherited

      - name: leakage_test
        parameters:
          step_size: 5e-3      # Override parent
          # charge_carrier_type still inherited
```

**Parameter inheritance**: Child routines inherit parent parameters but can override them.

### Group-Specific Routines

```yaml
routines:
  - name: health_check
    parameters:
      charge_carrier_type: HOLE
    routines:
      # Run on control group
      - name: noise_floor_measurement
        group: control
        parameters:
          measure_electrode: OUT_A

      - name: leakage_test
        group: control
        parameters:
          measure_electrode: OUT_A

      # Run on sensor group
      - name: noise_floor_measurement
        group: sensor
        parameters:
          measure_electrode: OUT_B

      - name: leakage_test
        group: sensor
        parameters:
          measure_electrode: OUT_B
```

Each sub-routine gets a device filtered to its specific group.

### Parameter Type Conversion

Stanza automatically converts string parameters to appropriate numeric types:

```yaml
parameters:
  count: "100"           # → int(100)
  frequency: "50e6"      # → int(50000000)
  amplitude: "0.5"       # → float(0.5)
  threshold: "1.23"      # → float(1.23)
  name: "test"           # → str("test")
  flag: true             # → bool(True)
```

**Conversion rules**:
1. Try to parse as float
2. If successful and no fractional part, convert to int
3. Otherwise keep as float
4. If parsing fails, keep as string

### Running Routines

**From Python**:
```python
from stanza.routines import RoutineRunner

# Load config and create runner
config = load_device_config("device.yaml")
runner = RoutineRunner(configs=[config])

# Run specific routine
result = runner.run("leakage_test")

# Run specific routine on group
result = runner.run("leakage_test", group="control")

# Run all routines in config
results = runner.run_all()

# Run all sub-routines under a parent
results = runner.run_all(parent_routine="health_check")
```

---

## Instrument Configuration

Instruments define hardware connections and capabilities.

### Required Instruments

Every device configuration MUST have:
- At least one CONTROL or GENERAL instrument (for setting voltages)
- At least one MEASUREMENT or GENERAL instrument (for measuring currents)

### Base Instrument Fields

```yaml
instruments:
  - name: string                    # Required: Instrument identifier
    type: CONTROL | MEASUREMENT | GENERAL | BREAKOUT_BOX
    driver: string                  # Optional: Driver module name
    ip_addr: string                 # Required (OR serial_addr)
    serial_addr: string             # Required (OR ip_addr)
    port: int                       # Optional: Network port
```

**Validation**: Must provide EITHER `ip_addr` OR `serial_addr` (not both, not neither).

### CONTROL Instrument

Controls gate voltages.

```yaml
- name: qdac_control
  type: CONTROL
  ip_addr: 192.168.1.100
  port: 5025
  driver: qdac2
  slew_rate: 100.0                  # Required: V/s (must be > 0)
```

**Required fields**:
- `slew_rate`: Voltage slew rate in V/s (must be positive)

**Configuration**: `extra="forbid"` (no additional fields allowed)

### MEASUREMENT Instrument

Measures currents.

```yaml
- name: qdac_measure
  type: MEASUREMENT
  ip_addr: 192.168.1.101
  port: 5025
  driver: qdac2
  measurement_duration: 1e-3        # Required: seconds (> 0)
  sample_time: 10e-6                # Required: seconds (> 0)
  conversion_factor: 1.0            # Optional: ADC to amperes
```

**Required fields**:
- `measurement_duration`: Total measurement time per point (seconds, > 0)
- `sample_time`: Individual sample time (seconds, > 0)

**Validation**: `sample_time` ≤ `measurement_duration`

**Optional fields**:
- `conversion_factor`: Convert ADC counts to amperes (default: 1.0)

**Configuration**: `extra="forbid"`

### GENERAL Instrument

Combined control and measurement capabilities.

```yaml
- name: qdac2
  type: GENERAL
  ip_addr: 192.168.88.250
  port: 5025
  driver: qdac2
  # Control fields
  slew_rate: 100.0
  # Measurement fields
  measurement_duration: 1e-3
  sample_time: 10e-6
  conversion_factor: 1.0
  # Additional fields allowed
  measurement_aperature_s: 1e-3
  breakout_line: 9
```

**Required fields**: Combination of CONTROL + MEASUREMENT
- `slew_rate`
- `measurement_duration`
- `sample_time`

**Configuration**: `extra="allow"` (additional fields permitted for device-specific settings)

**Use case**: Single instrument handling both control and measurement (e.g., QDevil QDAC-II).

### BREAKOUT_BOX Instrument

Manages channel routing.

```yaml
- name: qswitch
  type: BREAKOUT_BOX
  ip_addr: 192.168.88.252
  port: 5025
  driver: qswitch
```

**Configuration**: `extra="allow"` (device-specific fields allowed)

**Use case**: Multiplexers, switch matrices.

### Multiple Instruments Example

```yaml
instruments:
  # Control only
  - name: qdac_gates
    type: CONTROL
    ip_addr: 192.168.1.100
    port: 5025
    driver: qdac2
    slew_rate: 100.0

  # Measurement only
  - name: lockin_amplifier
    type: MEASUREMENT
    ip_addr: 192.168.1.101
    port: 5025
    driver: lockin
    measurement_duration: 100e-3
    sample_time: 1e-3

  # Breakout box
  - name: switching_matrix
    type: BREAKOUT_BOX
    ip_addr: 192.168.1.102
    port: 5025
    driver: qswitch
```

### OPX-Specific Fields

For OPX measurement instruments:

```yaml
- name: opx
  type: MEASUREMENT
  ip_addr: 192.168.1.100
  measurement_duration: 1e-3
  sample_time: 10e-6
  # OPX-specific
  machine_type: opx1
  cluster_name: my_cluster
  measurement_channels: [1, 2, 3]
  connection_headers: {"port1": "header1"}
  octave:
    name: octave1
    port: 50
```

---

## Complete Examples

### Example 1: Simple Single-Dot Device

```yaml
name: "SingleDot"

gates:
  # Barrier gates
  G1:
    type: BARRIER
    control_channel: 1
    measure_channel: 10
    v_lower_bound: -3.0
    v_upper_bound: 3.0

  # Plunger gate
  G2:
    type: PLUNGER
    control_channel: 2
    measure_channel: 11
    v_lower_bound: -3.0
    v_upper_bound: 3.0

  G3:
    type: BARRIER
    control_channel: 3
    measure_channel: 12
    v_lower_bound: -3.0
    v_upper_bound: 3.0

contacts:
  # Source
  IN:
    type: SOURCE
    control_channel: 20
    measure_channel: 20
    v_lower_bound: -0.001
    v_upper_bound: 0.001

  # Drain
  OUT:
    type: DRAIN
    control_channel: 21
    measure_channel: 21
    v_lower_bound: -3.0
    v_upper_bound: 3.0

gpios:
  VDD:
    type: INPUT
    control_channel: 30
    v_lower_bound: 0.0
    v_upper_bound: 3.3

  VSS:
    type: INPUT
    control_channel: 31
    v_lower_bound: -3.3
    v_upper_bound: 0.0

# No groups - full device accessible
routines:
  - name: health_check
    routines:
      - name: leakage_test
        parameters:
          leakage_threshold_resistance: 50e6
          num_points: 5
          measure_electrode: OUT

      - name: pinchoff_characterization
        parameters:
          step_size: 1e-2
          measure_electrode: OUT

instruments:
  - name: qdac2
    type: GENERAL
    ip_addr: 192.168.1.100
    port: 5025
    driver: qdac2
    slew_rate: 100.0
    measurement_duration: 1e-3
    sample_time: 10e-6
```

### Example 2: Dual-Dot Device with Groups

```yaml
name: "DualDot_Grouped"

gates:
  # Dot 1 gates
  G1: {type: BARRIER, control_channel: 1, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G2: {type: PLUNGER, control_channel: 2, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G3: {type: BARRIER, control_channel: 3, v_lower_bound: -3.0, v_upper_bound: 3.0}

  # Dot 2 gates
  G4: {type: BARRIER, control_channel: 4, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G5: {type: PLUNGER, control_channel: 5, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G6: {type: BARRIER, control_channel: 6, v_lower_bound: -3.0, v_upper_bound: 3.0}

  # Shared reservoirs
  RES_LEFT: {type: RESERVOIR, control_channel: 10, v_lower_bound: -3.0, v_upper_bound: 3.0}
  RES_RIGHT: {type: RESERVOIR, control_channel: 11, v_lower_bound: -3.0, v_upper_bound: 3.0}

contacts:
  IN: {type: SOURCE, measure_channel: 1, v_lower_bound: 0.0, v_upper_bound: 0.0}
  OUT_1: {type: DRAIN, measure_channel: 2, v_lower_bound: -3.0, v_upper_bound: 3.0}
  OUT_2: {type: DRAIN, measure_channel: 3, v_lower_bound: -3.0, v_upper_bound: 3.0}

gpios:
  VDD: {type: INPUT, control_channel: 20, v_lower_bound: 0.0, v_upper_bound: 3.3}
  VSS: {type: INPUT, control_channel: 21, v_lower_bound: -3.3, v_upper_bound: 0.0}
  SUB: {type: INPUT, control_channel: 22, v_lower_bound: -5.0, v_upper_bound: 5.0}

groups:
  dot1:
    name: "Quantum Dot 1"
    gates: [G1, G2, G3, RES_LEFT, RES_RIGHT]
    contacts: [IN, OUT_1]
    # gpios omitted - gets ALL (VDD, VSS, SUB)

  dot2:
    name: "Quantum Dot 2"
    gates: [G4, G5, G6, RES_LEFT, RES_RIGHT]
    contacts: [IN, OUT_2]
    # gpios omitted - gets ALL (VDD, VSS, SUB)

routines:
  - name: health_check
    parameters:
      charge_carrier_type: ELECTRON
    routines:
      # Dot 1 characterization
      - name: noise_floor_measurement
        group: dot1
        parameters:
          num_points: 10
          measure_electrode: OUT_1

      - name: leakage_test
        group: dot1
        parameters:
          leakage_threshold_resistance: 50e6
          measure_electrode: OUT_1

      - name: reservoir_characterization
        group: dot1
        parameters:
          bias_gate: IN
          step_size: 1e-2
          measure_electrode: OUT_1

      # Dot 2 characterization
      - name: noise_floor_measurement
        group: dot2
        parameters:
          num_points: 10
          measure_electrode: OUT_2

      - name: leakage_test
        group: dot2
        parameters:
          leakage_threshold_resistance: 50e6
          measure_electrode: OUT_2

      - name: reservoir_characterization
        group: dot2
        parameters:
          bias_gate: IN
          step_size: 1e-2
          measure_electrode: OUT_2

instruments:
  - name: qdac2
    type: GENERAL
    ip_addr: 192.168.1.100
    port: 5025
    driver: qdac2
    slew_rate: 100.0
    measurement_duration: 1e-3
    sample_time: 10e-6
```

**Key features**:
- Shared RESERVOIR gates (RES_LEFT, RES_RIGHT) between groups
- Shared source contact (IN) between groups
- GPIOs omitted from groups → ALL inherited

### Example 3: Multiplexed Multi-Device System

```yaml
name: "SiMOS_Multiplexed"

contacts:
  IN_A_B: {type: SOURCE, control_channel: 21, measure_channel: 21, v_lower_bound: -0.001, v_upper_bound: 0.001}
  OUT_A: {type: DRAIN, control_channel: 12, measure_channel: 12, v_lower_bound: -3.0, v_upper_bound: 3.0}
  OUT_B: {type: DRAIN, control_channel: 11, measure_channel: 11, v_lower_bound: -3.0, v_upper_bound: 3.0}

gpios:
  # Multiplexer address (6 bits = 64 devices)
  A0: {type: INPUT, control_channel: 24, v_lower_bound: -5.0, v_upper_bound: 5.0}
  A1: {type: INPUT, control_channel: 2, v_lower_bound: -5.0, v_upper_bound: 5.0}
  A2: {type: INPUT, control_channel: 1, v_lower_bound: -5.0, v_upper_bound: 5.0}
  A3: {type: INPUT, control_channel: 4, v_lower_bound: -5.0, v_upper_bound: 5.0}
  A4: {type: INPUT, control_channel: 5, v_lower_bound: -5.0, v_upper_bound: 5.0}
  A5: {type: INPUT, control_channel: 8, v_lower_bound: -5.0, v_upper_bound: 5.0}

  # Power
  VDD: {type: INPUT, control_channel: 22, v_lower_bound: -3.0, v_upper_bound: 3.0}
  VSS: {type: INPUT, control_channel: 9, v_lower_bound: -3.0, v_upper_bound: 3.0}
  SUB: {type: INPUT, control_channel: 7, v_lower_bound: -3.0, v_upper_bound: 3.0}

gates:
  # Side A gates (7 gates)
  G1: {type: BARRIER, control_channel: 20, measure_channel: 20, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G2: {type: PLUNGER, control_channel: 18, measure_channel: 18, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G3: {type: BARRIER, control_channel: 19, measure_channel: 19, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G4: {type: PLUNGER, control_channel: 23, measure_channel: 23, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G5: {type: BARRIER, control_channel: 16, measure_channel: 16, v_lower_bound: -3.0, v_upper_bound: 3.0}

  # Side B gates (5 gates)
  G7: {type: BARRIER, control_channel: 15, measure_channel: 15, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G8: {type: PLUNGER, control_channel: 14, measure_channel: 14, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G9: {type: BARRIER, control_channel: 13, measure_channel: 13, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G10: {type: PLUNGER, control_channel: 3, measure_channel: 3, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G11: {type: BARRIER, control_channel: 6, measure_channel: 6, v_lower_bound: -3.0, v_upper_bound: 3.0}

  # Shared reservoirs
  G6: {type: RESERVOIR, control_channel: 17, measure_channel: 17, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G12: {type: RESERVOIR, control_channel: 10, measure_channel: 10, v_lower_bound: -3.0, v_upper_bound: 3.0}

groups:
  side_A:
    name: "Side A Quantum Dots"
    gates: [G1, G2, G3, G4, G5, G6, G12]
    contacts: [IN_A_B, OUT_A]
    # gpios omitted - ALL accessible (mux address + power)

  side_B:
    name: "Side B Quantum Dots"
    gates: [G7, G8, G9, G10, G11, G6, G12]
    contacts: [IN_A_B, OUT_B]
    # gpios omitted - ALL accessible (mux address + power)

routines:
  - name: health_check
    parameters:
      charge_carrier_type: HOLE
    routines:
      # Side A routines
      - name: noise_floor_measurement
        group: side_A
        parameters:
          num_points: 10
          measure_electrode: OUT_A

      - name: leakage_test
        group: side_A
        parameters:
          leakage_threshold_resistance: 50e6
          leakage_threshold_count: 0
          num_points: 5
          measure_electrode: OUT_A

      - name: global_accumulation
        group: side_A
        parameters:
          bias_gate: IN_A_B
          bias_voltage: 5e-4
          step_size: 1e-2
          measure_electrode: OUT_A

      - name: reservoir_characterization
        group: side_A
        parameters:
          bias_gate: IN_A_B
          bias_voltage: 5e-4
          step_size: 1e-2
          measure_electrode: OUT_A

      - name: finger_gate_characterization
        group: side_A
        parameters:
          bias_gate: IN_A_B
          bias_voltage: 5e-4
          step_size: 1e-2
          measure_electrode: OUT_A

      # Side B routines (mirror of side A)
      - name: noise_floor_measurement
        group: side_B
        parameters:
          num_points: 10
          measure_electrode: OUT_B

      - name: leakage_test
        group: side_B
        parameters:
          leakage_threshold_resistance: 50e6
          leakage_threshold_count: 0
          num_points: 5
          measure_electrode: OUT_B

      - name: global_accumulation
        group: side_B
        parameters:
          bias_gate: IN_A_B
          bias_voltage: 5e-4
          step_size: 1e-2
          measure_electrode: OUT_B

      - name: reservoir_characterization
        group: side_B
        parameters:
          bias_gate: IN_A_B
          bias_voltage: 5e-4
          step_size: 1e-2
          measure_electrode: OUT_B

      - name: finger_gate_characterization
        group: side_B
        parameters:
          bias_gate: IN_A_B
          bias_voltage: 5e-4
          step_size: 1e-2
          measure_electrode: OUT_B

instruments:
  - name: qdac2
    type: GENERAL
    ip_addr: 192.168.88.250
    port: 5025
    driver: qdac2
    measurement_duration: 1e-3
    sample_time: 10e-6
    slew_rate: 100.0
    measurement_aperature_s: 1e-3
    breakout_line: 9

  - name: qswitch
    type: BREAKOUT_BOX
    ip_addr: 192.168.88.252
    port: 5025
    driver: qswitch
```

**Python usage for device 0**:
```python
# Load config
config = load_device_config("SiMOS_Multiplexed.yaml")
config.name = "SiMOS_device_0"

# Create runner
runner = RoutineRunner(configs=[config])

# Set mux to device 0 (binary 000000)
device = runner.resources.device
device.jump({
    "A0": -1.5,  # 0
    "A1": -1.5,  # 0
    "A2": -1.5,  # 0
    "A3": -1.5,  # 0
    "A4": -1.5,  # 0
    "A5": -1.5,  # 0
    "VDD": 1.5,
    "VSS": -1.5,
})

# Run full health check (both sides)
try:
    results = runner.run_all(parent_routine="health_check")
finally:
    # ALWAYS zero gates before switching devices
    device.jump({gate: 0.0 for gate in device.control_gates})
```

---

## Validation Rules Reference

### Channel Uniqueness

All `control_channel`, `measure_channel`, and `breakout_channel` values must be unique across **all pads** (gates, contacts, GPIOs).

```yaml
# INVALID
gates:
  G1: {control_channel: 1, ...}
contacts:
  IN: {control_channel: 1, ...}  # ERROR: Duplicate

# VALID
gates:
  G1: {control_channel: 1, ...}
contacts:
  IN: {control_channel: 2, ...}  # OK: Unique
```

**Error format**:
```
Duplicate channels found: gate 'G1' control_channel 1, contact 'IN' control_channel 1
```

### Voltage Bounds Requirement

If `control_channel` is specified, both `v_lower_bound` and `v_upper_bound` are required.

```yaml
# INVALID
gates:
  G1:
    type: PLUNGER
    control_channel: 1
    # Missing v_lower_bound and v_upper_bound

# VALID
gates:
  G1:
    type: PLUNGER
    control_channel: 1
    v_lower_bound: -3.0
    v_upper_bound: 3.0
```

### Gate Sharing Validation

**Rule**: Only RESERVOIR gates can be shared between groups.

```yaml
# VALID
gates:
  RES1: {type: RESERVOIR, ...}
groups:
  g1: {gates: [RES1]}
  g2: {gates: [RES1]}  # OK: RESERVOIR

# INVALID
gates:
  G1: {type: PLUNGER, ...}
groups:
  g1: {gates: [G1]}
  g2: {gates: [G1]}  # ERROR: PLUNGER
```

**Error**:
```
Gate 'G1' referenced by group 'g2' already assigned to group 'g1'.
Only RESERVOIR gates can be shared between groups.
```

### Contact/GPIO Sharing Validation

Contacts and GPIOs CAN be shared (no error).

```yaml
# VALID
contacts:
  IN: {...}
gpios:
  VDD: {...}

groups:
  g1:
    contacts: [IN]
    gpios: [VDD]
  g2:
    contacts: [IN]     # OK: Contact sharing allowed
    gpios: [VDD]       # OK: GPIO sharing allowed
```

### Routine Group Validation

If a routine specifies a `group`, that group must exist.

```yaml
# INVALID
routines:
  - name: test
    group: nonexistent  # ERROR: Group doesn't exist

groups:
  actual: {gates: [G1]}

# VALID
routines:
  - name: test
    group: actual  # OK: Group exists

groups:
  actual: {gates: [G1]}
```

**Error**:
```
Routine 'test' references unknown group 'nonexistent'.
Available groups: actual
```

### Instrument Validation

**Required instruments**:
- At least one CONTROL or GENERAL instrument
- At least one MEASUREMENT or GENERAL instrument

```yaml
# INVALID - No measurement capability
instruments:
  - name: control_only
    type: CONTROL
    slew_rate: 100.0

# INVALID - No control capability
instruments:
  - name: measure_only
    type: MEASUREMENT
    measurement_duration: 1e-3
    sample_time: 10e-6

# VALID - GENERAL provides both
instruments:
  - name: combined
    type: GENERAL
    slew_rate: 100.0
    measurement_duration: 1e-3
    sample_time: 10e-6

# VALID - Separate instruments
instruments:
  - name: control
    type: CONTROL
    slew_rate: 100.0
  - name: measure
    type: MEASUREMENT
    measurement_duration: 1e-3
    sample_time: 10e-6
```

### Measurement Timing Validation

For measurement instruments: `sample_time` ≤ `measurement_duration`

```yaml
# INVALID
instruments:
  - name: measure
    type: MEASUREMENT
    measurement_duration: 1e-3
    sample_time: 10e-3        # ERROR: Longer than duration

# VALID
instruments:
  - name: measure
    type: MEASUREMENT
    measurement_duration: 1e-3
    sample_time: 10e-6        # OK: Shorter than duration
```

---

## Best Practices

### 1. Channel Organization

Use a consistent numbering scheme:

```yaml
# Good: Organized by function
gates:
  G1: {control_channel: 1, measure_channel: 51, ...}
  G2: {control_channel: 2, measure_channel: 52, ...}
  G3: {control_channel: 3, measure_channel: 53, ...}

contacts:
  IN: {control_channel: 20, measure_channel: 70, ...}
  OUT: {control_channel: 21, measure_channel: 71, ...}

gpios:
  VDD: {control_channel: 40, ...}
  VSS: {control_channel: 41, ...}

# Control: 1-39
# Measure: 51-99
# GPIO control: 40-50
```

### 2. Voltage Bounds

Set conservative initial bounds:

```yaml
# Conservative (safer)
gates:
  G1:
    v_lower_bound: -3.0
    v_upper_bound: 3.0

# Can tighten after initial testing
gates:
  G1:
    v_lower_bound: -2.5
    v_upper_bound: 2.0
```

### 3. Group Design

**For shared infrastructure** (power, mux control), omit from groups:

```yaml
gpios:
  VDD: {...}
  VSS: {...}
  A0: {...}  # Mux control

groups:
  control:
    gates: [G1, G2]
    # gpios omitted - shared infrastructure

  sensor:
    gates: [G3, G4]
    # gpios omitted - shared infrastructure
```

**For isolated resources**, explicitly specify:

```yaml
groups:
  control:
    gates: [G1, G2]
    contacts: [IN, OUT_A]  # Control-specific measurement

  sensor:
    gates: [G3, G4]
    contacts: [IN, OUT_B]  # Sensor-specific measurement
```

### 4. Routine Organization

Use hierarchical organization:

```yaml
routines:
  - name: full_characterization
    parameters:
      charge_carrier_type: ELECTRON
    routines:
      - name: quick_tests
        routines:
          - name: noise_floor_measurement
          - name: leakage_test

      - name: detailed_characterization
        routines:
          - name: reservoir_characterization
          - name: finger_gate_characterization
          - name: coulomb_peak_fitting
```

### 5. Parameter Management

Put common parameters in parent routines:

```yaml
routines:
  - name: health_check
    parameters:
      charge_carrier_type: HOLE
      step_size: 1e-2
      bias_voltage: 5e-4
    routines:
      - name: routine_a
        # Inherits all parent parameters

      - name: routine_b
        parameters:
          step_size: 5e-3  # Override specific parameter
```

### 6. Documentation

Add comments for clarity:

```yaml
gates:
  G1: {type: PLUNGER, control_channel: 1, v_lower_bound: -3.0, v_upper_bound: 3.0}  # Left dot plunger
  G2: {type: BARRIER, control_channel: 2, v_lower_bound: -3.0, v_upper_bound: 3.0}  # Interdot barrier
  G3: {type: PLUNGER, control_channel: 3, v_lower_bound: -3.0, v_upper_bound: 3.0}  # Right dot plunger

groups:
  left_dot:
    gates: [G1, G2]  # Left dot + barrier

  right_dot:
    gates: [G3, G2]  # Right dot + barrier (shared barrier)
```

---

## Common Pitfalls

### Pitfall 1: Forgetting Voltage Bounds

```yaml
# WRONG
gates:
  G1:
    type: PLUNGER
    control_channel: 1
    # Missing bounds!

# CORRECT
gates:
  G1:
    type: PLUNGER
    control_channel: 1
    v_lower_bound: -3.0
    v_upper_bound: 3.0
```

**Fix**: Always specify `v_lower_bound` and `v_upper_bound` when using `control_channel`.

### Pitfall 2: Duplicate Channels

```yaml
# WRONG
gates:
  G1: {control_channel: 1, ...}
contacts:
  IN: {control_channel: 1, ...}  # Duplicate!

# CORRECT
gates:
  G1: {control_channel: 1, ...}
contacts:
  IN: {control_channel: 2, ...}  # Unique
```

**Fix**: Maintain a channel allocation table during config creation.

### Pitfall 3: Sharing Non-Reservoir Gates

```yaml
# WRONG
gates:
  G1: {type: PLUNGER, ...}

groups:
  group1: {gates: [G1]}
  group2: {gates: [G1]}  # Error!

# CORRECT - Use RESERVOIR for shared gates
gates:
  G1: {type: PLUNGER, ...}
  RES1: {type: RESERVOIR, ...}

groups:
  group1: {gates: [G1, RES1]}
  group2: {gates: [G2, RES1]}  # OK: RESERVOIR shared
```

**Fix**: Only share RESERVOIR gates. Use separate gates for each quantum dot.

### Pitfall 4: Misunderstanding Conditional Filtering

```yaml
# WRONG EXPECTATION
groups:
  control:
    gates: [G1, G2]
    # User expects: No GPIOs accessible

# ACTUAL BEHAVIOR
# All device GPIOs are accessible (omitted = all)

# CORRECT FOR INTENT
groups:
  control:
    gates: [G1, G2]
    gpios: []  # Explicit empty list = no GPIOs
```

**Fix**:
- Want all: Omit the field
- Want none: Use empty list `[]`
- Want specific: List them explicitly

### Pitfall 5: Missing Required Instruments

```yaml
# WRONG - Only control
instruments:
  - name: control
    type: CONTROL
    slew_rate: 100.0
    # Missing measurement capability!

# CORRECT
instruments:
  - name: qdac
    type: GENERAL
    slew_rate: 100.0
    measurement_duration: 1e-3
    sample_time: 10e-6
```

**Fix**: Ensure at least one MEASUREMENT or GENERAL instrument.

### Pitfall 6: Parameter Type Issues

```yaml
# WRONG - Will fail type checks
parameters:
  count: 100.5  # Float where int expected

# CORRECT - Use string with scientific notation
parameters:
  count: "100"        # → int(100)
  frequency: "50e6"   # → int(50000000)
```

**Fix**: Use string format with scientific notation. Stanza converts automatically.

### Pitfall 7: Group Doesn't Exist

```yaml
# WRONG
routines:
  - name: test
    group: sensor  # Error: No sensor group

groups:
  control: {gates: [G1]}

# CORRECT
routines:
  - name: test
    group: control  # OK: Group exists

groups:
  control: {gates: [G1]}
```

**Fix**: Ensure all routine groups are defined in the `groups` section.

### Pitfall 8: Instrument Address Issues

```yaml
# WRONG - Both ip_addr and serial_addr
instruments:
  - name: test
    type: CONTROL
    ip_addr: 192.168.1.100
    serial_addr: /dev/ttyUSB0  # Error!

# WRONG - Neither provided
instruments:
  - name: test
    type: CONTROL
    # Missing both!

# CORRECT - One or the other
instruments:
  - name: network_device
    type: CONTROL
    ip_addr: 192.168.1.100

  - name: usb_device
    type: CONTROL
    serial_addr: /dev/ttyUSB0
```

**Fix**: Provide exactly one of `ip_addr` or `serial_addr`.

### Pitfall 9: Not Zeroing Gates Between Devices

```python
# WRONG - Don't leave voltages on
device_0 = get_device(0)
run_characterization(device_0)
device_1 = get_device(1)  # Voltages still on device_0!

# CORRECT - Always zero before switching
device_0 = get_device(0)
try:
    run_characterization(device_0)
finally:
    device_0.jump({gate: 0.0 for gate in device_0.control_gates})

device_1 = get_device(1)
```

**Fix**: Use try/finally to ensure gates are zeroed even if characterization fails.

### Pitfall 10: Inconsistent Naming

```yaml
# CONFUSING
gates:
  Gate_1: {...}
  g2: {...}
  G_3: {...}

# CLEAR
gates:
  G1: {...}
  G2: {...}
  G3: {...}
```

**Fix**: Use consistent naming conventions (e.g., `G1`, `G2`, `G3`).

---

## Quick Reference Card

### Conditional Filtering

| Field | Omitted | Empty `[]` | Specified |
|-------|---------|------------|-----------|
| gates | NOT included | NOT included | ONLY specified |
| contacts | ALL included | NOT included | ONLY specified |
| gpios | ALL included | NOT included | ONLY specified |

### Sharing Rules

| Pad Type | Can Share? | Notes |
|----------|------------|-------|
| RESERVOIR gates | Yes | Only gate type that can share |
| Other gates | No | PLUNGER, BARRIER, SCREEN |
| Contacts | Yes | Common sources allowed |
| GPIOs | Yes | Power, control signals |

### Required Fields

| Section | Required | Optional |
|---------|----------|----------|
| Device | name, instruments | gates, contacts, gpios, groups, routines |
| Gate | type | control_channel, measure_channel, bounds |
| Contact | type | control_channel, measure_channel, bounds |
| GPIO | type | control_channel, measure_channel, bounds |
| Group | gates | name, contacts, gpios |
| Routine | name | group, parameters, routines |
| Instrument | name, type, addr | driver, port, device-specific |

### Validation Checklist

- [ ] All channels unique
- [ ] Bounds provided with control_channel
- [ ] At least one control instrument
- [ ] At least one measurement instrument
- [ ] sample_time ≤ measurement_duration
- [ ] Only RESERVOIR gates shared
- [ ] Routine groups exist in groups section
- [ ] Either ip_addr OR serial_addr (not both)

---

## Conclusion

The Stanza configuration system provides a powerful and flexible way to define quantum devices with:

- **Hierarchical organization** through nested routines
- **Flexible resource sharing** via conditional filtering
- **Safety guarantees** through validation and type checking
- **Hardware abstraction** through instrument configurations

The **conditional filtering** feature is particularly powerful, allowing you to:
- Share infrastructure (power, mux control) across groups by omitting fields
- Isolate measurement resources by explicitly specifying them
- Balance flexibility and safety

When creating configurations:
1. Start with a minimal valid config
2. Add pads incrementally, checking channel uniqueness
3. Define groups based on physical device structure
4. Use conditional filtering for infrastructure sharing
5. Organize routines hierarchically
6. Test with small parameter sweeps before full characterization

For questions or issues, refer to:
- Stanza source code: `stanza/models.py`, `stanza/device.py`
- Example configs: `stanza/configs/devices/`
- Test files: `tests/test_device_group_filtering.py`

---

*Document version: 1.0*
*Compatible with: Stanza 0.1.0+*
*Last updated: 2025*
