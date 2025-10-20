# Stanza

<p align="center">
<a href="https://pypi.org/project/cq-stanza" target="_blank">
    <img src="https://img.shields.io/pypi/v/cq-stanza?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/cq-stanza" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/cq-stanza.svg?color=%2334D058" alt="Supported Python versions">
</a>
<a href="https://github.com/conductorquantum/stanza/actions?query=workflow%3A%22CI%2FCD+Tests%22+event%3Apush+branch%3Amain" target="_blank">
    <img src="https://github.com/conductorquantum/stanza/actions/workflows/test.yml/badge.svg?event=push&branch=main" alt="Test">
</a>
<a href="https://codecov.io/gh/conductorquantum/stanza" >
 <img src="https://codecov.io/gh/conductorquantum/stanza/graph/badge.svg?token=7J2Z8TRRVG"/>
 </a>
</p>


**Documentation**: <a href="https://docs.conductorquantum.com/stanza" target="_blank">https://docs.conductorquantum.com/stanza</a>



Stanza is a Python framework for building tune-up sequences for quantum devices. Configure devices with YAML, write routines as decorated functions, and execute them with automatic logging.

## Quick Start

```bash
pip install cq-stanza
```

Or with Quantum Machines drivers:

```bash
pip install "cq-stanza[qm]"
```

### Define Your Device and Routines

Configure your device topology and routine parameters in YAML. Parameters defined in the config are automatically passed to decorated routines:

```yaml
# device.yaml
name: "My Quantum Device"

gates:
  G1: {type: BARRIER, control_channel: 1, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G2: {type: PLUNGER, control_channel: 2, v_lower_bound: -3.0, v_upper_bound: 3.0}

contacts:
  SOURCE: {type: SOURCE, control_channel: 3, measure_channel: 1, v_lower_bound: -3.0, v_upper_bound: 3.0}
  DRAIN: {type: DRAIN, control_channel: 4, measure_channel: 2, v_lower_bound: -3.0, v_upper_bound: 3.0}

gpios:
  VSS: {type: INPUT, control_channel: 5, v_lower_bound: 0, v_upper_bound: 5}
  VDD: {type: INPUT, control_channel: 6, v_lower_bound: -5, v_upper_bound: 0}

routines:
  - name: sweep_barrier
    parameters:
      gate: G1
      v_start: -2.0
      v_stop: 0.0
      n_points: 100
      contact: DRAIN

instruments:
  - name: qdac2-control
    type: CONTROL
    driver: qdac2
    ip_addr: 127.0.0.1
    slew_rate: 1.0
```

### Write a Routine

Routine parameters from YAML are passed as kwargs. You can override them at runtime:

```python
import numpy as np
from stanza.routines import routine

@routine
def sweep_barrier(ctx, gate, v_start, v_stop, n_points, contact):
    """Sweep a barrier gate and measure current."""
    device = ctx.resources.device
    voltages = np.linspace(v_start, v_stop, n_points)
    v_data, i_data = device.sweep_1d(gate, voltages.tolist(), contact)
    return {"voltages": v_data, "currents": i_data}
```

### Run It

First, initialize a session directory for your experiment data:

```bash
# Initialize a new timestamped session directory
stanza init

# Or with a custom name
stanza init --name my_experiment

# Check current session
stanza status
```

Then run your routines - all data will be logged to the active session directory:

```python
from stanza.routines import RoutineRunner
from stanza.models import DeviceConfig

# Load configuration
config = DeviceConfig.from_yaml("device.yaml")

# Create runner - automatically loads routine parameters from config
runner = RoutineRunner(configs=[config])

# Execute with config parameters
result = runner.run("sweep_barrier")

# Or override specific parameters at runtime
result = runner.run("sweep_barrier", gate="G2", n_points=50)
print(result["currents"])
```

## Core Features

**CLI Session Management**: Initialize timestamped experiment directories with `stanza init` to organize your data.

**Device Abstraction**: Define quantum devices with gates, contacts, and instruments in YAML. Access them uniformly in code.

**Decorator-Based Routines**: Write tune-up sequences as simple Python functions with the `@routine` decorator.

**Resource Management**: Access devices, loggers, and other resources through a unified context object.

**Result Tracking**: Store and retrieve results from previous routines to build complex workflows.

**Automatic Logging**: Sessions and data are logged automatically with support for HDF5 and JSONL formats.

**Type Safety**: Built on Pydantic for configuration validation and type checking.

## Architecture

Stanza separates concerns into three layers:

1. **Configuration** (YAML) - Define device topology and routine parameters
2. **Routines** (Python functions) - Implement tune-up logic
3. **Execution** (Runner) - Orchestrate resources and logging

```python
@routine
def my_routine(ctx, **params):
    # Access resources
    device = ctx.resources.device
    logger = ctx.resources.logger

    # Use previous results
    prev_result = ctx.results.get("previous_routine")

    # Your logic here
    return result
```

## Device Operations

Stanza devices support common operations:

```python
# Jump to voltages
device.jump({"G1": -1.5, "G2": -0.8})

# Set VSS and VDD for GPIO pins
device.jump({"VSS": 1.5, "VDD": -1.5})

# Zero specific pads or all pads
device.zero(["G1", "G2"])  # Zero specific pads
device.zero()              # Zero all pads

# Measure current
current = device.measure("DRAIN")

# Check current voltage
voltage = device.check("G1")

# Sweep operations
v_data, i_data = device.sweep_1d("G1", voltages, "DRAIN")
v_data, i_data = device.sweep_2d("G1", v1, "G2", v2, "DRAIN")
v_data, i_data = device.sweep_nd(["G1", "G2"], voltages, "DRAIN")
```

## Built-in Routines

Stanza includes health check routines for device characterization:

```python
from stanza.routines.builtins import (
    noise_floor_measurement,
    leakage_test,
    global_accumulation,
)

# Run health checks
runner.run("noise_floor_measurement", measure_electrode="DRAIN", num_points=10)
runner.run("leakage_test", leakage_threshold_resistance=50e6, num_points=10)
runner.run("global_accumulation", measure_electrode="DRAIN", step_size=0.01, bias_gate="SOURCE", bias_voltage=0.005)
```

These routines include automatic analysis and fitting for device diagnostics.

## Examples

See the [cookbooks](cookbooks/) directory for:
- Basic device configuration
- Writing custom routines
- Running built-in routines
- Jupyter notebook workflows

## Development

```bash
git clone https://github.com/conductorquantum/stanza.git
cd stanza
pip install -e ".[dev]"
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.