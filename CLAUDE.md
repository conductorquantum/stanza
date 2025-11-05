# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Stanza is a Python framework for building tune-up sequences for quantum devices. It uses a three-layer architecture:
1. **Configuration** (YAML) - Device topology and routine parameters
2. **Routines** (Python functions) - Tune-up logic with `@routine` decorator
3. **Execution** (Runner) - Resource orchestration and automatic logging

## Development Commands

### Environment Setup
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install with specific optional dependencies
pip install -e ".[dev,routines,qm,notebooks]"
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_routines.py

# Run with coverage
uv run pytest --cov=stanza --cov-report=html

# Run specific test by name
uv run pytest tests/test_routines.py::test_routine_decorator
```

### Code Quality
```bash
# Format code with ruff
ruff format .

# Run ruff linter
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type checking with mypy
mypy stanza/

# Run pre-commit hooks
pre-commit run --all-files
```

### Building and Publishing
```bash
# Build package
python -m build

# Install local build
pip install -e .
```

## Architecture

### Protocol-Based Design
The codebase uses Python protocols (from `typing.Protocol`) to define instrument interfaces. This allows for flexible, duck-typed instrument implementations without inheritance:

- `ControlInstrument` - Sets/gets voltages on channels
- `MeasurementInstrument` - Measures current on channels
- `BreakoutBoxInstrument` - Controls digital breakout box routing
- `NamedResource` - Any resource with a `name` attribute

All drivers must implement the appropriate protocols. Validation happens via `validate_driver_protocols()` in `stanza/base/registry.py`.

### Key Components

**Device (`stanza/device.py`)**
- Central abstraction for quantum device control
- Coordinates between control and measurement instruments
- Maps logical pad names (e.g., "G1", "DRAIN") to physical channels
- Provides high-level operations: `jump()`, `sweep_1d()`, `sweep_2d()`, `measure()`, `zero()`
- Manages breakout box routing when configured

**RoutineRunner (`stanza/routines/core.py`)**
- Executes decorated `@routine` functions
- Manages resource and results registries
- Can instantiate resources from configs or accept pre-initialized resources
- Automatically creates DataLogger when initialized with configs

**Context System (`stanza/context.py`, `stanza/registry.py`)**
- `RoutineContext` - Passed to all routines, contains `resources` and `results`
- `ResourceRegistry` - Provides attribute access to resources: `ctx.resources.device`, `ctx.resources.logger`
- `ResultsRegistry` - Stores/retrieves results from previous routines: `ctx.results.get("previous_routine")`

**DataLogger (`stanza/logger/data_logger.py`)**
- Handles automatic data logging to HDF5, JSONL, and live plotting
- Uses `LoggerSession` for buffered per-routine data collection
- Supports multiple writer backends via registry pattern
- Auto-enables live plotting when configured via CLI

**Session Management (`stanza/context.py`)**
- `StanzaSession` - Manages timestamped experiment directories (e.g., `20251020100010_untitled/`)
- Tracks active session in `.stanza/active_session.json`
- Creates Jupyter notebooks pre-configured with Stanza imports
- Separate from `LoggerSession` which handles per-routine logging

### Configuration System

Device configs use Pydantic models (`stanza/models.py`) for validation:
- `DeviceConfig` - Top-level device definition
- `Gate`, `Contact`, `GPIO` - Electrode types with channels and voltage bounds
- `InstrumentConfig` - Instrument definitions with driver specifications

YAML configs are loaded via `load_device_config()` and converted to `Device` instances via `device_from_config()` in `stanza/utils.py`.

### Driver System

Drivers in `stanza/drivers/` implement protocol interfaces:
- **qdac2** - QDevil QDAC-II DAC via PyVISA
- **qswitch** - QDevil QSwitch digital breakout box
- **opx** - Quantum Machines OPX+ control system

Drivers are dynamically loaded via `load_driver_class()` in `stanza/base/registry.py`. New drivers must:
1. Implement required protocols (ControlInstrument, MeasurementInstrument, or both)
2. Accept `(instrument_config, channel_configs, **kwargs)` constructor signature
3. Map logical channel names to physical instrument channels

## Testing Infrastructure

### Test Organization
- Tests use pytest fixtures defined in `tests/conftest.py`
- PyVISA-sim used for hardware simulation (QDAC2, QSwitch)
- QM cloud simulator for OPX+ testing
- Separate directories for integration tests: `test_logging_integration/`, `test_opx_cloud_sim/`

### Key Test Patterns
```python
# Routine testing - access registry directly
from stanza.routines.core import get_registered_routines

routines = get_registered_routines()
assert "my_routine" in routines

# Mock instruments in tests
class MockControlInstrument:
    def set_voltage(self, channel_name: str, voltage: float) -> None:
        pass
    def get_voltage(self, channel_name: str) -> float:
        return 0.0
    def get_slew_rate(self, channel_name: str) -> float:
        return 1.0
```

### Coverage Exclusions
See `pyproject.toml` for coverage exclusions - optional imports (QM, PyVISA) are excluded since they may not be available in all environments.

## Built-in Routines

Located in `stanza/routines/builtins/`:
- **health_check.py** - Device health checks (noise floor, leakage test, global accumulation, pinch-off)
- **charge_sensor_compensation.py** - Charge sensor compensation routines

Built-in routines include analysis and fitting functionality in `stanza/analysis/fitting/`.

## Live Plotting

Two backends supported via `stanza/plotter/`:
- **Server backend** - Bokeh server on configurable port (default: 5006)
- **Inline backend** - Direct notebook cell plotting (requires `jupyter_bokeh`)

Configuration stored in `.stanza/live_plot_config.json`. DataLogger auto-detects and enables if configured.

## Important Notes

### Type Safety
- Project enforces strict mypy checking (see `pyproject.toml` settings)
- Use `from __future__ import annotations` for forward references
- All functions should have type hints for parameters and return values

### Pydantic Compatibility
- Code supports both Pydantic v1 and v2 via `BaseModelWithConfig`
- Use `model_validator` decorators for cross-field validation
- Access version via `PYDANTIC_VERSION_MINOR_TUPLE`

### Routine Parameter Handling
Parameters can be defined in YAML config under `routines:` section. These are automatically passed to decorated functions. Runtime overrides via `runner.run("routine_name", param=value)` take precedence.

### Channel Mapping
The `ChannelConfig` class in `stanza/base/channels.py` maps logical electrode names to:
- `control_channel` - Physical DAC channel number
- `measure_channel` - Physical measurement channel number
- `breakout_channel` - Breakout box routing channel

This mapping is generated by `generate_channel_configs()` in `stanza/utils.py`.
