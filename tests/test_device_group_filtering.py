import pytest

from stanza.device import Device
from stanza.exceptions import DeviceError
from stanza.models import (
    Contact,
    ContactType,
    ControlInstrumentConfig,
    DeviceConfig,
    DeviceGroup,
    Gate,
    GateType,
    InstrumentType,
    MeasurementInstrumentConfig,
)
from stanza.base.channels import ChannelConfig
from stanza.models import PadType

class MockControlInstrument:
    """Mock control instrument for testing."""

    def __init__(self, *args, **kwargs):
        pass

    def set_voltage(self, channel_name: str, voltage: float) -> None:
        pass

    def get_voltage(self, channel_name: str) -> float:
        return 0.0

    def get_slew_rate(self, channel_name: str) -> float:
        return 1.0


class MockMeasurementInstrument:
    """Mock measurement instrument for testing."""

    def __init__(self, *args, **kwargs):
        pass

    def measure(self, channel_name: str) -> float:
        return 0.0


def test_device_filter_by_group_basic():
    """Test basic device filtering by group."""
    device_config = DeviceConfig(
        name="test_device",
        gates={
            "G1": Gate(
                type=GateType.PLUNGER,
                control_channel=1,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "G2": Gate(
                type=GateType.BARRIER,
                control_channel=2,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "G3": Gate(
                type=GateType.PLUNGER,
                control_channel=3,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
        },
        contacts={
            "IN": Contact(
                type=ContactType.SOURCE,
                measure_channel=1,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "OUT": Contact(
                type=ContactType.DRAIN,
                measure_channel=2,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
        },
        groups={
            "control": DeviceGroup(gates=["G1", "G2"], contacts=["IN"]),
            "sensor": DeviceGroup(gates=["G3"], contacts=["OUT"]),
        },
        routines=[],
        instruments=[
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
        ],
    )

    channel_configs = {
        "G1": ChannelConfig(
            name="G1",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        ),
        "G2": ChannelConfig(
            name="G2",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.BARRIER,
            control_channel=2,
        ),
        "G3": ChannelConfig(
            name="G3",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=3,
        ),
        "IN": ChannelConfig(
            name="IN",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.CONTACT,
            electrode_type=ContactType.SOURCE,
            measure_channel=1,
        ),
        "OUT": ChannelConfig(
            name="OUT",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.CONTACT,
            electrode_type=ContactType.DRAIN,
            measure_channel=2,
        ),
    }

    device = Device(
        name="test_device",
        device_config=device_config,
        channel_configs=channel_configs,
        control_instrument=MockControlInstrument(),
        measurement_instrument=MockMeasurementInstrument(),
    )

    # Filter by control group
    control_device = device.filter_by_group("control")

    # Check that only control group gates and contacts are present
    assert set(control_device.gates) == {"G1", "G2"}
    assert set(control_device.contacts) == {"IN"}
    assert control_device.name == "test_device_control"

    # Filter by sensor group
    sensor_device = device.filter_by_group("sensor")

    # Check that only sensor group gates and contacts are present
    assert set(sensor_device.gates) == {"G3"}
    assert set(sensor_device.contacts) == {"OUT"}
    assert sensor_device.name == "test_device_sensor"


def test_device_filter_by_group_unknown_group():
    """Test that filtering by unknown group raises error."""
    device_config = DeviceConfig(
        name="test_device",
        gates={
            "G1": Gate(
                type=GateType.PLUNGER,
                control_channel=1,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
        },
        contacts={},
        groups={"control": DeviceGroup(gates=["G1"])},
        routines=[],
        instruments=[
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
        ],
    )



    channel_configs = {
        "G1": ChannelConfig(
            name="G1",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        ),
    }

    device = Device(
        name="test_device",
        device_config=device_config,
        channel_configs=channel_configs,
        control_instrument=MockControlInstrument(),
        measurement_instrument=MockMeasurementInstrument(),
    )

    # Try to filter by unknown group
    with pytest.raises(DeviceError, match="Group 'unknown' not found"):
        device.filter_by_group("unknown")


def test_device_filter_by_group_shares_instruments():
    """Test that filtered devices share the same instrument instances."""
    device_config = DeviceConfig(
        name="test_device",
        gates={
            "G1": Gate(
                type=GateType.PLUNGER,
                control_channel=1,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "G2": Gate(
                type=GateType.BARRIER,
                control_channel=2,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
        },
        contacts={},
        groups={
            "control": DeviceGroup(gates=["G1"]),
            "sensor": DeviceGroup(gates=["G2"]),
        },
        routines=[],
        instruments=[
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
        ],
    )

    channel_configs = {
        "G1": ChannelConfig(
            name="G1",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        ),
        "G2": ChannelConfig(
            name="G2",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.BARRIER,
            control_channel=2,
        ),
    }

    control_inst = MockControlInstrument()
    measure_inst = MockMeasurementInstrument()

    device = Device(
        name="test_device",
        device_config=device_config,
        channel_configs=channel_configs,
        control_instrument=control_inst,
        measurement_instrument=measure_inst,
    )

    # Filter by groups
    control_device = device.filter_by_group("control")
    sensor_device = device.filter_by_group("sensor")

    # Check that instruments are shared (same instance)
    assert control_device.control_instrument is control_inst
    assert control_device.measurement_instrument is measure_inst
    assert sensor_device.control_instrument is control_inst
    assert sensor_device.measurement_instrument is measure_inst


def test_device_get_shared_gates():
    """Test getting list of shared gates."""
    device_config = DeviceConfig(
        name="test_device",
        gates={
            "G1": Gate(
                type=GateType.PLUNGER,
                control_channel=1,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "G2": Gate(
                type=GateType.BARRIER,
                control_channel=2,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "RES1": Gate(
                type=GateType.RESERVOIR,
                control_channel=3,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "RES2": Gate(
                type=GateType.RESERVOIR,
                control_channel=4,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
        },
        contacts={},
        groups={
            "control": DeviceGroup(gates=["G1", "RES1", "RES2"]),
            "sensor": DeviceGroup(gates=["G2", "RES1", "RES2"]),
        },
        routines=[],
        instruments=[
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
        ],
    )

    channel_configs = {
        "G1": ChannelConfig(
            name="G1",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        ),
        "G2": ChannelConfig(
            name="G2",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.BARRIER,
            control_channel=2,
        ),
        "RES1": ChannelConfig(
            name="RES1",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.RESERVOIR,
            control_channel=3,
        ),
        "RES2": ChannelConfig(
            name="RES2",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.RESERVOIR,
            control_channel=4,
        ),
    }

    device = Device(
        name="test_device",
        device_config=device_config,
        channel_configs=channel_configs,
        control_instrument=MockControlInstrument(),
        measurement_instrument=MockMeasurementInstrument(),
    )

    # Check shared gates
    shared_gates = device.get_shared_gates()
    assert set(shared_gates) == {"RES1", "RES2"}


def test_device_get_other_group_gates():
    """Test getting gates from other groups (excluding shared)."""
    device_config = DeviceConfig(
        name="test_device",
        gates={
            "G1": Gate(
                type=GateType.PLUNGER,
                control_channel=1,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "G2": Gate(
                type=GateType.BARRIER,
                control_channel=2,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "G3": Gate(
                type=GateType.PLUNGER,
                control_channel=3,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
            "RES1": Gate(
                type=GateType.RESERVOIR,
                control_channel=4,
                v_lower_bound=0.0,
                v_upper_bound=1.0,
            ),
        },
        contacts={},
        groups={
            "control": DeviceGroup(gates=["G1", "G2", "RES1"]),
            "sensor": DeviceGroup(gates=["G3", "RES1"]),
        },
        routines=[],
        instruments=[
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
        ],
    )


    channel_configs = {
        "G1": ChannelConfig(
            name="G1",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        ),
        "G2": ChannelConfig(
            name="G2",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.BARRIER,
            control_channel=2,
        ),
        "G3": ChannelConfig(
            name="G3",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=3,
        ),
        "RES1": ChannelConfig(
            name="RES1",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.RESERVOIR,
            control_channel=4,
        ),
    }

    device = Device(
        name="test_device",
        device_config=device_config,
        channel_configs=channel_configs,
        control_instrument=MockControlInstrument(),
        measurement_instrument=MockMeasurementInstrument(),
    )

    # Get gates from other groups (not in control, not shared)
    other_gates = device.get_other_group_gates("control")
    # Should get G3 (from sensor group), but NOT RES1 (shared)
    assert set(other_gates) == {"G3"}

    # Get gates from other groups (not in sensor, not shared)
    other_gates = device.get_other_group_gates("sensor")
    # Should get G1, G2 (from control group), but NOT RES1 (shared)
    assert set(other_gates) == {"G1", "G2"}
