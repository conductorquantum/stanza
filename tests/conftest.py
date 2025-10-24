from unittest.mock import patch

import pytest

from stanza.device import Device
from stanza.models import (
    Contact,
    ContactType,
    ControlInstrumentConfig,
    DeviceConfig,
    Gate,
    GateType,
    InstrumentType,
    MeasurementInstrumentConfig,
    RoutineConfig,
)
from stanza.utils import generate_channel_configs


class MockInstrumentConfig:
    """Mock instrument config with breakout_line."""

    def __init__(self, breakout_line: int | None = 1):
        self.breakout_line = breakout_line


class MockControlInstrument:
    """Mock implementation of ControlInstrument protocol."""

    def __init__(self, breakout_line: int | None = 1):
        self.voltages = {}
        self.slew_rates = {}
        self.should_fail = False
        self.instrument_config = MockInstrumentConfig(breakout_line)

    def set_voltage(self, channel_name: str, voltage: float) -> None:
        if self.should_fail:
            raise RuntimeError("Mock voltage set failure")
        self.voltages[channel_name] = voltage

    def get_voltage(self, channel_name: str) -> float:
        return self.voltages.get(channel_name, 0.0)

    def get_slew_rate(self, channel_name: str) -> float:
        return self.slew_rates.get(channel_name, 1.0)


class MockMeasurementInstrument:
    """Mock implementation of MeasurementInstrument protocol."""

    def __init__(self, breakout_line: int | None = 2):
        self.measurements = {}
        self.instrument_config = MockInstrumentConfig(breakout_line)

    def measure(self, channel_name: str) -> float:
        return self.measurements.get(channel_name, 0.0)


class MockBreakoutBoxInstrument:
    """Mock implementation of BreakoutBoxInstrument protocol."""

    def __init__(self):
        self.grounded_lines = set()
        self.connected_calls = []
        self.disconnected_calls = []
        self.ungrounded_lines = []

    def get_grounded(self, channel_name: str | list[str]) -> bool | list[bool]:
        if isinstance(channel_name, str):
            return channel_name in self.grounded_lines
        return [name in self.grounded_lines for name in channel_name]

    def set_grounded(self, channel_name: str | list[str]) -> None:
        if isinstance(channel_name, str):
            self.grounded_lines.add(channel_name)
        else:
            self.grounded_lines.update(channel_name)

    def get_ungrounded(self, channel_name: str | list[str]) -> bool | list[bool]:
        if isinstance(channel_name, str):
            return channel_name not in self.grounded_lines
        return [name not in self.grounded_lines for name in channel_name]

    def set_ungrounded(self, channel_name: str | list[str]) -> None:
        if isinstance(channel_name, str):
            self.grounded_lines.discard(channel_name)
            self.ungrounded_lines.append(channel_name)
        else:
            for name in channel_name:
                self.grounded_lines.discard(name)
                self.ungrounded_lines.append(name)

    def get_connected(
        self, channel_name: str | list[str], line_number: int
    ) -> bool | list[bool]:
        if isinstance(channel_name, str):
            return (channel_name, line_number) in self.connected_calls
        return [(name, line_number) in self.connected_calls for name in channel_name]

    def set_connected(self, channel_name: str | list[str], line_number: int) -> None:
        if isinstance(channel_name, str):
            self.connected_calls.append((channel_name, line_number))
        else:
            for name in channel_name:
                self.connected_calls.append((name, line_number))

    def get_disconnected(
        self, channel_name: str | list[str], line_number: int
    ) -> bool | list[bool]:
        if isinstance(channel_name, str):
            return (channel_name, line_number) not in self.connected_calls
        return [
            (name, line_number) not in self.connected_calls for name in channel_name
        ]

    def set_disconnected(self, channel_name: str | list[str], line_number: int) -> None:
        if isinstance(channel_name, str):
            self.disconnected_calls.append((channel_name, line_number))
        else:
            for name in channel_name:
                self.disconnected_calls.append((name, line_number))


@pytest.fixture(autouse=True)
def mock_time_sleep():
    """Automatically patch time.sleep for all tests to avoid actual delays."""
    with patch("time.sleep"):
        yield


@pytest.fixture
def control_instrument():
    """Fixture providing a mock control instrument."""
    return MockControlInstrument()


@pytest.fixture
def measurement_instrument():
    """Fixture providing a mock measurement instrument."""
    return MockMeasurementInstrument()


@pytest.fixture
def device_config():
    """Fixture providing a basic device configuration."""
    return DeviceConfig(
        name="test_device",
        gates={
            "gate1": Gate(
                name="gate1",
                type=GateType.PLUNGER,
                v_lower_bound=-2.0,
                v_upper_bound=2.0,
                control_channel=1,
                measure_channel=1,
            )
        },
        contacts={
            "contact1": Contact(
                name="contact1",
                type=ContactType.SOURCE,
                v_lower_bound=-1.0,
                v_upper_bound=1.0,
                measure_channel=2,
            )
        },
        routines=[],
        instruments=[
            ControlInstrumentConfig(
                name="mock_control",
                type=InstrumentType.CONTROL,
                ip_addr="127.0.0.1",
                slew_rate=1.0,
            ),
            MeasurementInstrumentConfig(
                name="mock_measurement",
                type=InstrumentType.MEASUREMENT,
                ip_addr="127.0.0.1",
                measurement_duration=1.0,
                sample_time=0.1,
            ),
        ],
    )


@pytest.fixture
def device_config_no_instruments():
    """Fixture providing a device configuration without instruments."""
    return DeviceConfig(
        name="test_device",
        gates={
            "gate1": Gate(
                name="gate1",
                type=GateType.PLUNGER,
                v_lower_bound=-2.0,
                v_upper_bound=2.0,
                control_channel=1,
                measure_channel=1,
            )
        },
        contacts={},
        routines=[],
        instruments=[
            ControlInstrumentConfig(
                name="mock_control",
                type=InstrumentType.CONTROL,
                ip_addr="127.0.0.1",
                slew_rate=1.0,
            ),
            MeasurementInstrumentConfig(
                name="mock_measurement",
                type=InstrumentType.MEASUREMENT,
                ip_addr="127.0.0.1",
                measurement_duration=1.0,
                sample_time=0.1,
            ),
        ],
    )


@pytest.fixture
def device(device_config, control_instrument, measurement_instrument):
    """Fixture providing a configured Device instance."""
    channel_configs = generate_channel_configs(device_config)
    return Device(
        device_config.name,
        device_config,
        channel_configs,
        control_instrument,
        measurement_instrument,
    )


@pytest.fixture
def device_no_instruments(device_config_no_instruments):
    """Fixture providing a Device instance without instruments."""
    channel_configs = generate_channel_configs(device_config_no_instruments)
    return Device(
        device_config_no_instruments.name,
        device_config_no_instruments,
        channel_configs,
        None,
        None,
    )


@pytest.fixture
def routine_configs():
    """Fixture providing sample routine configurations."""
    return [
        RoutineConfig(
            name="test_routine", parameters={"param1": "value1", "param2": 42}
        ),
        RoutineConfig(
            name="configured_routine",
            parameters={"threshold": 1e-12, "multiplier": 2.5},
        ),
        RoutineConfig(name="no_params_routine"),
    ]


@pytest.fixture
def valid_device_yaml():
    """Fixture providing valid device YAML configuration."""
    return """
name: test_device
gates:
  G1:
    type: PLUNGER
    control_channel: 1
    measure_channel: null
    v_lower_bound: -1.0
    v_upper_bound: 1.0
contacts:
  C1:
    type: SOURCE
    control_channel: 2
    measure_channel: 3
    v_lower_bound: 0.0
    v_upper_bound: 0.5
gpios:
  GPIO1:
    type: OUTPUT
    control_channel: 4
    measure_channel: null
    v_lower_bound: 0.0
    v_upper_bound: 3.3
routines: []
instruments:
  - name: test_control
    type: CONTROL
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: test_measurement
    type: MEASUREMENT
    ip_addr: "127.0.0.1"
    measurement_duration: 1.0
    sample_time: 0.1
"""


@pytest.fixture
def nested_routines_yaml():
    """Fixture providing device YAML configuration with nested routines."""
    return """
name: "Test Device with Nested Routines"
contacts:
  IN:
    type: SOURCE
    control_channel: 1
    measure_channel: 1
    v_lower_bound: -3.0
    v_upper_bound: 3.0
gates:
  G1:
    type: PLUNGER
    control_channel: 2
    measure_channel: 2
    v_lower_bound: -3.0
    v_upper_bound: 3.0
routines:
  - name: health_check
    parameters:
      parent_param: value
      charge_carrier_type: holes
    routines:
      - name: leakage_test
        parameters:
          leakage_threshold_resistance: 50000000.0
          leakage_threshold_count: 0
      - name: global_accumulation
        parameters:
          step_size: 0.01
      - name: reservoir_characterization
        parameters:
          step_size: 0.01
instruments:
  - name: test_control
    type: CONTROL
    driver: qdac2
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: test_measurement
    type: MEASUREMENT
    driver: qdac2
    ip_addr: "127.0.0.1"
    measurement_duration: 0.001
    sample_time: 0.00001
"""


@pytest.fixture
def device_yaml_with_qswitch():
    """Fixture providing device YAML configuration with qswitch breakout box."""
    return """
name: test_device
gates:
  G1: {type: PLUNGER, control_channel: 1, measure_channel: 1, breakout_channel: 1, v_lower_bound: -1.0, v_upper_bound: 1.0}
contacts:
  C1: {type: SOURCE, control_channel: 2, measure_channel: 2, breakout_channel: 2, v_lower_bound: -1.0, v_upper_bound: 1.0}
routines: []
instruments:
  - name: ctrl
    type: CONTROL
    driver: qdac2
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: meas
    type: MEASUREMENT
    driver: qdac2
    ip_addr: "127.0.0.1"
    measurement_duration: 0.001
    sample_time: 0.0001
  - name: qswitch
    type: BREAKOUT_BOX
    driver: qswitch
    ip_addr: "192.168.1.100"
    port: 5025
"""


@pytest.fixture
def deeply_nested_routines_yaml():
    """Fixture providing device YAML configuration with deeply nested routines (3+ levels)."""
    return """
name: "Test Device with Deeply Nested Routines"
contacts:
  IN:
    type: SOURCE
    control_channel: 1
    measure_channel: 1
    v_lower_bound: -3.0
    v_upper_bound: 3.0
gates:
  G1:
    type: PLUNGER
    control_channel: 2
    measure_channel: 2
    v_lower_bound: -3.0
    v_upper_bound: 3.0
routines:
  - name: level1
    parameters:
      level: 1
    routines:
      - name: level2
        parameters:
          level: 2
        routines:
          - name: level3
            parameters:
              level: 3
instruments:
  - name: test_control
    type: CONTROL
    driver: qdac2
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: test_measurement
    type: MEASUREMENT
    driver: qdac2
    ip_addr: "127.0.0.1"
    measurement_duration: 0.001
    sample_time: 0.00001
"""


@pytest.fixture
def simple_routines_yaml():
    """Fixture providing device YAML configuration with simple top-level routines."""
    return """
name: test
gates:
  G1:
    type: PLUNGER
    control_channel: 1
    measure_channel: 1
    v_lower_bound: -1.0
    v_upper_bound: 1.0
contacts:
  C1:
    type: SOURCE
    control_channel: 2
    measure_channel: 2
    v_lower_bound: -1.0
    v_upper_bound: 1.0
routines:
  - name: routine1
    parameters:
      param1: value1
  - name: routine2
    parameters:
      param2: value2
instruments:
  - name: ctrl
    type: CONTROL
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: meas
    type: MEASUREMENT
    ip_addr: "127.0.0.1"
    measurement_duration: 0.001
    sample_time: 0.0001
"""


@pytest.fixture
def param_override_yaml():
    """Fixture for testing parent parameter override behavior."""
    return """
name: test_device
gates:
  G1:
    type: PLUNGER
    control_channel: 1
    measure_channel: 1
    v_lower_bound: -1.0
    v_upper_bound: 1.0
contacts:
  C1:
    type: SOURCE
    control_channel: 2
    measure_channel: 2
    v_lower_bound: -1.0
    v_upper_bound: 1.0
routines:
  - name: parent_routine
    parameters:
      shared_param: parent_value
      parent_only: parent_data
    routines:
      - name: child_routine
        parameters:
          shared_param: child_value
          child_only: child_data
instruments:
  - name: ctrl
    type: CONTROL
    ip_addr: "127.0.0.1"
    slew_rate: 1.0
  - name: meas
    type: MEASUREMENT
    ip_addr: "127.0.0.1"
    measurement_duration: 0.001
    sample_time: 0.0001
"""
