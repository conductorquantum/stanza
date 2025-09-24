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
)
from stanza.utils import generate_channel_configs


class MockControlInstrument:
    """Mock implementation of ControlInstrument protocol."""

    def __init__(self):
        self.voltages = {}
        self.slew_rates = {}
        self.should_fail = False

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

    def __init__(self):
        self.measurements = {}

    def measure(self, channel_name: str) -> float:
        return self.measurements.get(channel_name, 0.0)


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
                readout=False,
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
                readout=True,
                v_lower_bound=-1.0,
                v_upper_bound=1.0,
                measure_channel=2,
            )
        },
        experiments=[],
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
                readout=False,
                v_lower_bound=-2.0,
                v_upper_bound=2.0,
                control_channel=1,
                measure_channel=1,
            )
        },
        contacts={},
        experiments=[],
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
