# SpinQICK Integration with Stanza

This document describes the integration between SpinQICK and Stanza's driver system.

## Overview

The SpinQICK driver (`stanza/drivers/spinqick.py`) provides integration between Stanza's instrument framework and SpinQICK's DCSource voltage control system. This allows you to use SpinQICK's advanced features within Stanza experiments:

- Gate-level voltage control with cross-coupling compensation
- Hardware voltage ramps and sweeps
- Voltage state saving/loading
- Integration with SpinQICK's tune-up routines and experiments

## Architecture

### Integration Approach

SpinQICK uses a **gate-centric** approach where voltages are controlled by gate names (P1, P2, M1, B1, etc.) defined in `hardware_config.json`. The Stanza driver wraps SpinQICK's `DCSource` class and maps Stanza channels to SpinQICK gate names.

### Key Components

1. **SpinQickControlChannel**: Control channel that interfaces with SpinQICK's DCSource
2. **SpinQick**: Main instrument class that manages multiple gates
3. **StanzaVoltageSourceAdapter**: Adapter to use Stanza instruments as SpinQICK voltage sources (future use)

### Class Hierarchy

```
BaseInstrument (Stanza)
    └── SpinQick
            └── SpinQickControlChannel (per gate)
                    └── DCSource (SpinQICK)
                            └── VoltageSource (QDAC, Basel LNHR, etc.)
```

## Installation

### Requirements

1. Install Stanza with SpinQICK support:
   ```bash
   pip install cq-stanza[spinqick]
   ```

2. Ensure you have:
   - SpinQICK v2.0.0 or later
   - A valid `hardware_config.json` file configured for SpinQICK
   - An initialized voltage source (QDAC, Basel LNHR, etc.)

### Configuration

SpinQICK requires a `hardware_config.json` file that defines gate-to-channel mappings. See [SpinQICK DC_INSTRUCTIONS.md](../spinqick/DC_INSTRUCTIONS.md) for details.

Example structure:
```json
{
  "voltage_source": "slow_dac",
  "channels": {
    "P1": {
      "slow_dac_address": "192.168.1.100",
      "slow_dac_channel": 1,
      "dc_conversion_factor": 1.0,
      "max_v": 0.5,
      "crosscoupling": {
        "M1": 0.05,
        "P2": 0.02
      }
    },
    "M1": {
      "slow_dac_address": "192.168.1.100",
      "slow_dac_channel": 2,
      "dc_conversion_factor": 1.0,
      "max_v": 1.0
    }
  }
}
```

## Usage

### Basic Setup

```python
from stanza.drivers.spinqick import SpinQick
from stanza.models import BaseInstrumentConfig
from stanza.base.channels import ChannelConfig
from stanza.models import PadType, GateType

# Create your SpinQICK-compatible voltage source
# (This could be a QDAC implementation from SpinQICK)
from my_voltage_sources import QDevil_QDAC
voltage_source = QDevil_QDAC()
voltage_source.open("192.168.1.100")

# Configure instrument
instrument_config = BaseInstrumentConfig(
    name="spinqick_control",
    ip_addr="192.168.1.100",
    port=5025
)

# Configure channels (map to SpinQICK gate names)
channel_configs = {
    "P1": ChannelConfig(
        name="P1",  # Must match gate name in hardware_config.json
        voltage_range=(-0.5, 0.5),
        pad_type=PadType.GATE,
        electrode_type=GateType.PLUNGER,
        control_channel=1,  # Physical channel on voltage source
    ),
    "M1": ChannelConfig(
        name="M1",
        voltage_range=(-1.0, 1.0),
        pad_type=PadType.GATE,
        electrode_type=GateType.SENSOR,
        control_channel=2,
    ),
}

# Create SpinQICK instrument
spinqick = SpinQick(
    instrument_config=instrument_config,
    channel_configs=channel_configs,
    voltage_source=voltage_source
)
```

### Basic Voltage Control

```python
# Set voltage on a gate
spinqick.set_voltage('P1', 0.5)

# Read voltage
voltage = spinqick.get_voltage('P1')

# Get all voltages
all_voltages = spinqick.get_all_voltages()
print(all_voltages)  # {'P1': 0.5, 'M1': 0.234, ...}
```

### Cross-Coupling Compensation

One of SpinQICK's key features is automatic cross-coupling compensation:

```python
# Set P1 voltage while keeping M1 (charge sensor) constant
spinqick.set_voltage_compensated(
    channel_name='P1',
    voltage=0.6,
    iso_gates=['M1']  # Keep M1 at constant potential
)

# Or via the channel directly
channel = spinqick.get_channel('control_P1')
channel.set_voltage_compensated(0.6, iso_gates=['M1'])
```

This automatically adjusts M1 voltage based on the cross-coupling matrix to keep the chemical potential constant at M1.

### Voltage Ramps and Sweeps

```python
# Program a voltage ramp
spinqick.program_ramp(
    channel_name='P2',
    vstart=0.0,
    vstop=0.1,
    tstep=10e-6,  # 10 microseconds per step
    nsteps=100
)

# Arm the sweep
spinqick.arm_sweep('P2')

# Trigger execution
spinqick.trigger_sweep('P2')
```

### State Management

```python
# Save current voltage state
spinqick.save_voltage_state()  # Saves to timestamped YAML file

# Or save to specific path
spinqick.save_voltage_state('/path/to/voltage_state.yaml')
```

## Integration with SpinQICK Experiments

The Stanza driver is designed to work alongside SpinQICK's high-level experiment classes:

```python
from spinqick.experiments import tune_electrostatics

# Create SpinQICK experiment using the same voltage source
charge = tune_electrostatics.TuneElectrostatics(
    soccfg,
    soc,
    voltage_source,  # Same voltage source as Stanza driver
    save_data=True,
    plot=True
)

# Run SpinQICK experiments
retune_data = charge.retune_dcs(
    m_dot='M1',
    m_range=(-0.02, 0.02, 150),
    measure_buffer=50.0,
    set_v=True
)

# Meanwhile, Stanza can control other aspects
spinqick.set_voltage('B1', 0.3)  # Set barrier gate
```

## Advanced Features

### Accessing DCSource Directly

For advanced SpinQICK features, you can access the underlying DCSource:

```python
# Access DCSource
dc_source = spinqick.dc_source

# Calculate compensated voltages without applying
gate_list, v_apply, g_array = dc_source.calculate_compensated_voltage(
    delta_v=[0.1],
    gates=['P1'],
    iso_gates=['M1']
)

# Program compensated ramp
dc_source.program_ramp_compensate(
    vstart=0.0,
    vstop=0.1,
    tstep=10e-6,
    nsteps=100,
    gates='P1',
    iso_gates='M1'
)
```

### Channel-Level Control

Each channel exposes SpinQICK-specific functionality:

```python
# Get channel
p1_channel = spinqick.get_channel('control_P1')

# Access gate name
print(p1_channel.gate_name)  # 'P1'

# Set compensated voltage
p1_channel.set_voltage_compensated(0.6, iso_gates=['M1'])

# Access DCSource
print(p1_channel.dc_source.all_voltages)
```

## Use Cases

### 1. Electrostatic Tune-Up with Stanza Control

```python
from stanza.routines import ChargeStabilityRoutine

# Use Stanza's high-level routines with SpinQICK compensation
routine = ChargeStabilityRoutine(
    instrument=spinqick,
    x_gate='P1',
    y_gate='P2',
    compensate_gate='M1'
)

# Run 2D sweep with automatic compensation
data = routine.run(
    x_range=(-0.1, 0.1, 100),
    y_range=(-0.1, 0.1, 100)
)
```

### 2. Hybrid Stanza-SpinQICK Workflow

```python
# 1. Use Stanza for initial setup
spinqick.set_voltage('P1', 0.0)
spinqick.set_voltage('P2', 0.0)
spinqick.save_voltage_state('initial_state.yaml')

# 2. Use SpinQICK experiments for tuning
charge.retune_dcs('M1', (-0.01, 0.01, 100), 50.0, set_v=True)

# 3. Use SpinQICK for charge stability
stability_data = charge.gvg_dc(
    g_gates=(['P1'], ['P2']),
    g_range=((-0.1, 0.1, 100), (-0.1, 0.1, 100)),
    measure_buffer=50.0,
    compensate='M1'
)

# 4. Use Stanza for fine adjustments
spinqick.set_voltage_compensated('P1', 0.05, iso_gates=['M1'])
```

## Architecture Details

### Gate Name Mapping

The driver uses a **convention-over-configuration** approach:

- Stanza channel names **must match** SpinQICK gate names
- The `hardware_config.json` defines available gates
- Channel configs in Stanza map directly to gates in SpinQICK

Example:
```python
# In Stanza
ChannelConfig(name="P1", ...)  # Maps to SpinQICK gate 'P1'

# In hardware_config.json
{
  "channels": {
    "P1": { ... }  # Same name
  }
}
```

### VoltageSource Protocol

SpinQICK defines a `VoltageSource` protocol that hardware must implement:

```python
class VoltageSource(Protocol):
    def open(self, address: str): ...
    def close(self): ...
    def get_voltage(self, ch: int) -> float: ...
    def set_voltage(self, ch: int, volts: float): ...
    def set_sweep(self, ch: int, start: float, stop: float,
                  length: float, num_steps: int): ...
    def trigger(self, ch: int): ...
    def arm_sweep(self, ch: int): ...
```

The `StanzaVoltageSourceAdapter` class (included but not currently used) can adapt Stanza instruments to this protocol for future bidirectional integration.

## Troubleshooting

### Import Errors

```python
ImportError: SpinQICK is not installed
```
**Solution**: Install SpinQICK with `pip install spinqick>=2.0.0`

### Configuration Errors

```python
KeyError: 'P1' not found in hardware_config
```
**Solution**: Ensure your `hardware_config.json` includes all gates referenced in channel_configs

### Voltage Out of Range

```python
Exception: requested voltage would exceed max_v on gate P1
```
**Solution**: Check `max_v` setting in `hardware_config.json` and adjust voltage ranges

### Cross-Coupling Errors

```python
RuntimeError: Can't set compensated voltage without a defined cross-coupling matrix
```
**Solution**: Add `crosscoupling` entries to gates in `hardware_config.json`

## Reference

### Key Classes

- **SpinQick**: Main instrument class
- **SpinQickControlChannel**: Control channel for individual gates
- **StanzaVoltageSourceAdapter**: Adapter for using Stanza instruments as SpinQICK voltage sources

### Key Methods

#### SpinQick Methods

- `set_voltage(channel_name, voltage)`: Set gate voltage
- `get_voltage(channel_name)`: Read gate voltage
- `set_voltage_compensated(channel_name, voltage, iso_gates)`: Set with compensation
- `program_ramp(channel_name, vstart, vstop, tstep, nsteps)`: Program sweep
- `arm_sweep(channel_name)`: Arm programmed sweep
- `trigger_sweep(channel_name)`: Trigger sweep execution
- `get_all_voltages()`: Get all gate voltages
- `save_voltage_state(file_path)`: Save voltage state to file

#### SpinQickControlChannel Methods

- `set_voltage_compensated(voltage, iso_gates)`: Set with compensation

### Configuration Parameters

See [SpinQICK DC_INSTRUCTIONS.md](../spinqick/DC_INSTRUCTIONS.md) for complete details on:

- DCS configuration parameters
- Gate channel mappings
- Cross-coupling matrix setup
- Voltage conversion factors

## Future Enhancements

1. **Bidirectional Integration**: Use `StanzaVoltageSourceAdapter` to allow SpinQICK to control Stanza instruments directly
2. **Sweep Support**: Full implementation of ramp/sweep functionality in the adapter
3. **Measurement Integration**: Support for SpinQICK's DCS (DC Charge Sensor) readout through Stanza
4. **Health Checks**: Integration with Stanza's health check routines

## Contributing

When contributing to the SpinQICK driver:

1. Maintain compatibility with SpinQICK v2.0+
2. Follow Stanza's driver patterns (see `qdac2.py` for reference)
3. Keep gate name mapping convention consistent
4. Document any changes to the VoltageSource protocol

## Related Documentation

- [SpinQICK DC_INSTRUCTIONS.md](../spinqick/DC_INSTRUCTIONS.md) - Complete DC measurement guide
- [SpinQICK README.md](../spinqick/README.md) - SpinQICK overview
- [Stanza Driver Development](./docs/driver_development.md) - General driver guide
- [QDAC2 Driver](./stanza/drivers/qdac2.py) - Reference implementation

## License

This integration is part of the Stanza project and follows the same license terms.
