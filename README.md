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

This creates a directory structure like:

```
your-project/
├── .stanza/
│   └── active_session.json                       # Tracks the active session
├── 20251020143022_untitled/                      # Timestamped session directory
│   ├── 20251020143022_untitled_notebook.ipynb    # Jupyter notebook for the session
│   └── .stanza/
│       └── config.json                           # Session metadata
└── device.yaml                                   # Your device config
```

The Jupyter notebook is pre-configured with Stanza imports and ready for running your routines. When you provide a custom name with `--name`, the notebook will be named `{timestamp}_{name}.ipynb` instead.

Then run your routines - all data will be logged to the active session directory:

```python
from stanza.routines import RoutineRunner
from stanza.utils import load_device_config

# Load configuration
config = load_device_config("device.yaml")

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

## Live Plotting

Visualize your data in real-time as routines execute. Stanza supports two plotting backends:

![Stanza Live Plotting Demo](docs/images/stanza_live_plotting.gif)

### Quick Start with CLI

Enable live plotting in your notebook or script:

```bash
# Plots in browser (default) - opens at http://localhost:5006
stanza live-plot enable

# Plots in notebook cells (requires: pip install jupyter_bokeh)
stanza live-plot enable --backend inline
```

Once enabled, the `DataLogger` automatically detects the configuration and streams data to plots as your routines run.

### Python API

You can also enable live plotting programmatically:

**Server Backend** - Opens plots in a browser window (works in any environment):

```python
from stanza.plotter import enable_live_plotting

# Enable live plotting with browser backend
backend = enable_live_plotting(runner.context.resources.logger, backend="server", port=5006)

# Run your routine - plots update live at http://localhost:5006
result = runner.run("sweep_barrier")
```

**Inline Backend** - Displays plots directly in notebook cells (requires `jupyter_bokeh`):

```python
from stanza.plotter import enable_live_plotting

# Enable inline plotting
backend = enable_live_plotting(runner.context.resources.logger, backend="inline")

# Run your routine - plots appear and update in the notebook cell
result = runner.run("sweep_barrier")
```

Plots automatically update as data is logged, supporting both 1D line plots and 2D heatmaps for real-time visualization of sweeps and measurements.

## Jupyter Integration

Stanza provides CLI commands to manage a Jupyter notebook server for interactive development and analysis.

### Server Management

Start a Jupyter server in the background that survives terminal closure:

```bash
# Start server in current directory on default port 8888
stanza jupyter start

# Start in specific directory with custom port
stanza jupyter start /path/to/notebooks --port 8889

# Check server status (shows PID, URL, uptime)
stanza jupyter status

# Open Jupyter in browser with authentication token
stanza jupyter open

# Stop the server gracefully
stanza jupyter stop
```

### Notebook Monitoring

Monitor and interact with running notebooks:

```bash
# List all active notebook sessions
stanza jupyter list

# View log files for active notebooks
stanza jupyter logs

# Tail a specific notebook's output (Ctrl+C to detach)
stanza jupyter logs my_notebook.ipynb

# Attach to a notebook with kernel control (Ctrl+C kills kernel, ESC exits)
stanza jupyter attach my_notebook.ipynb
```

![Attaching to a Jupyter Notebook Kernel](docs/images/stanza_jupyter_notebook.gif)

The `logs` command is useful for monitoring long-running experiments, while `attach` provides active control for debugging and development. Both commands support showing initial context with the `-n/--lines` option (default: 10 lines).

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

### Breakout Box Operations

Stanza supports digital breakout boxes for routing control and measurement signals. Configure channels with `breakout_channel` and instruments with `breakout_line`:

```yaml
# device.yaml with QSwitch breakout box
gates:
  G1: {type: BARRIER, control_channel: 1, breakout_channel: 1, v_lower_bound: -3.0, v_upper_bound: 3.0}
  G2: {type: PLUNGER, control_channel: 2, breakout_channel: 2, v_lower_bound: -3.0, v_upper_bound: 3.0}

contacts:
  DRAIN: {type: DRAIN, control_channel: 3, measure_channel: 1, breakout_channel: 3, v_lower_bound: -3.0, v_upper_bound: 3.0}

instruments:
  - name: qdac2
    type: GENERAL
    driver: qdac2
    ip_addr: 192.168.1.100
    port: 5025
    slew_rate: 100.0
    breakout_line: 9  # QSwitch relay that connects to this instrument

  - name: qswitch
    type: BREAKOUT_BOX
    driver: qswitch
    ip_addr: 192.168.1.102
    port: 5025
```

Control the breakout box in your routines:

```python
@routine
def configure_breakout(ctx):
    device = ctx.resources.device

    # Ground all breakout lines for safety
    device.ground_breakout_lines()

    # Unground specific lines to enable control
    device.unground_breakout_lines()

    # Connect all lines to their configured instruments
    device.connect_breakout_lines()

    # Disconnect breakout lines when not in use
    device.disconnect_breakout_lines()
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
