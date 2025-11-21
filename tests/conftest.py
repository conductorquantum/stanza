from unittest.mock import patch

import numpy as np
import pytest

from stanza.device import Device
from stanza.models import (
    GPIO,
    Contact,
    ContactType,
    ControlInstrumentConfig,
    DeviceConfig,
    Gate,
    GateType,
    GPIOType,
    InstrumentType,
    MeasurementInstrumentConfig,
    RoutineConfig,
)
from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
    StabilityMeasurement,
)
from stanza.routines.builtins.utils.peak_fitting import FittedPeak, ModelFitResult
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


# Helper functions for test setup
def make_gate(
    gate_type: GateType = GateType.PLUNGER,
    control_channel: int | None = None,
    measure_channel: int | None = None,
    v_lower_bound: float = 0.0,
    v_upper_bound: float = 1.0,
) -> Gate:
    """Helper function to create Gate instances with common defaults."""
    return Gate(
        type=gate_type,
        control_channel=control_channel,
        measure_channel=measure_channel,
        v_lower_bound=v_lower_bound,
        v_upper_bound=v_upper_bound,
    )


def make_contact(
    contact_type: ContactType = ContactType.SOURCE,
    control_channel: int | None = None,
    measure_channel: int | None = None,
    v_lower_bound: float = 0.0,
    v_upper_bound: float = 1.0,
) -> Contact:
    """Helper function to create Contact instances with common defaults."""
    return Contact(
        type=contact_type,
        control_channel=control_channel,
        measure_channel=measure_channel,
        v_lower_bound=v_lower_bound,
        v_upper_bound=v_upper_bound,
    )


def standard_instrument_configs() -> list[
    ControlInstrumentConfig | MeasurementInstrumentConfig
]:
    """Helper function to create standard instrument configurations for testing."""
    return [
        ControlInstrumentConfig(
            name="control",
            type=InstrumentType.CONTROL,
            ip_addr="192.168.1.1",
            slew_rate=1.0,
        ),
        MeasurementInstrumentConfig(
            name="measurement",
            type=InstrumentType.MEASUREMENT,
            ip_addr="192.168.1.2",
            measurement_duration=1.0,
            sample_time=0.5,
        ),
    ]


@pytest.fixture
def model_fit_result():
    """Baseline ModelFitResult for charge sensor unit tests."""
    return ModelFitResult(
        model_name="Lorentzian",
        amplitude=1.0,
        center_idx=50.0,
        width=5.0,
        offset=0.0,
        r_squared=0.95,
        rmse=0.01,
        aicc=10.0,
        fwhm=10.0,
        area=100.0,
        skew_resid=0.0,
        eta=0.5,
    )


@pytest.fixture
def fitted_peak_factory(model_fit_result):
    """Factory fixture for constructing FittedPeak instances with sensible defaults."""

    def _factory(
        *,
        best_model: str = "Lorentzian",
        sensitivity: float = 1e-6,
        sensitivity_voltage: float = 0.0,
        peak_idx: int = 0,
        peak_voltage: float = 0.0,
        quality_score: float | None = None,
        sensitivity_score: float | None = None,
        window_currents: np.ndarray | None = None,
        window_voltages: np.ndarray | None = None,
        lorentzian_fit: ModelFitResult | None = None,
        sech2_fit: ModelFitResult | None = None,
        voigt_fit: ModelFitResult | None = None,
    ) -> FittedPeak:
        return FittedPeak(
            best_model=best_model,
            lorentzian_fit=lorentzian_fit or model_fit_result,
            sech2_fit=sech2_fit or model_fit_result,
            voigt_fit=voigt_fit or model_fit_result,
            sensitivity=sensitivity,
            sensitivity_voltage=sensitivity_voltage,
            peak_idx=peak_idx,
            peak_voltage=peak_voltage,
            window_currents=window_currents
            if window_currents is not None
            else np.ones(10),
            window_voltages=window_voltages
            if window_voltages is not None
            else np.linspace(0, 0.1, 10),
            quality_score=quality_score,
            sensitivity_score=sensitivity_score,
        )

    return _factory


@pytest.fixture
def stability_measurement_factory():
    """Factory fixture for constructing StabilityMeasurement instances."""

    def _factory(
        *,
        peak_index: int = 0,
        peak_voltage: float = -0.7,
        max_gradient_voltage: float = -0.68,
        time_array: np.ndarray | None = None,
        current_array: np.ndarray | None = None,
        current_mean: float = 1e-9,
        current_std: float = 1e-11,
        local_slope: float = 1e-6,
        voltage_noise: float = 1e-5,
    ) -> StabilityMeasurement:
        time_array_np = (
            np.array(time_array, dtype=np.float64)
            if time_array is not None
            else np.array([0.0, 1.0], dtype=np.float64)
        )
        current_array_np = (
            np.array(current_array, dtype=np.float64)
            if current_array is not None
            else np.array([1e-9, 1e-9], dtype=np.float64)
        )

        return StabilityMeasurement(
            peak_index=peak_index,
            peak_voltage=peak_voltage,
            max_gradient_voltage=max_gradient_voltage,
            time_array=time_array_np,
            current_array=current_array_np,
            current_mean=current_mean,
            current_std=current_std,
            local_slope=local_slope,
            voltage_noise=voltage_noise,
        )

    return _factory


@pytest.fixture
def control_instrument_config():
    """Fixture providing a standard ControlInstrumentConfig for testing."""
    return ControlInstrumentConfig(
        name="control",
        type=InstrumentType.CONTROL,
        ip_addr="192.168.1.1",
        slew_rate=1.0,
    )


@pytest.fixture
def measurement_instrument_config():
    """Fixture providing a standard MeasurementInstrumentConfig for testing."""
    return MeasurementInstrumentConfig(
        name="measurement",
        type=InstrumentType.MEASUREMENT,
        ip_addr="192.168.1.2",
        measurement_duration=1.0,
        sample_time=0.5,
    )


@pytest.fixture
def sample_gate():
    """Fixture providing a standard Gate for testing."""
    return Gate(
        type=GateType.PLUNGER,
        control_channel=1,
        v_lower_bound=0.0,
        v_upper_bound=1.0,
    )


@pytest.fixture
def sample_contact():
    """Fixture providing a standard Contact for testing."""
    return Contact(
        type=ContactType.SOURCE,
        measure_channel=2,
        v_lower_bound=0.0,
        v_upper_bound=1.0,
    )


@pytest.fixture
def sample_gpio():
    """Fixture providing a standard GPIO for testing."""
    return GPIO(
        type=GPIOType.OUTPUT,
        control_channel=3,
        v_lower_bound=0.0,
        v_upper_bound=3.3,
    )


@pytest.fixture
def create_device():
    """Fixture that returns a function to create a Device from a DeviceConfig."""

    def _create_device(
        device_config: DeviceConfig,
        control_instrument: MockControlInstrument | None = None,
        measurement_instrument: MockMeasurementInstrument | None = None,
    ) -> Device:
        """Create a Device instance from a DeviceConfig."""
        channel_configs = generate_channel_configs(device_config)
        return Device(
            name=device_config.name,
            device_config=device_config,
            channel_configs=channel_configs,
            control_instrument=control_instrument or MockControlInstrument(),
            measurement_instrument=measurement_instrument
            or MockMeasurementInstrument(),
        )

    return _create_device


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
