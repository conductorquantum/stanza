import pytest

from stanza.device import Device
from stanza.exceptions import DeviceError
from stanza.models import ContactType, DeviceConfig, DeviceGroup, GateType
from stanza.utils import generate_channel_configs
from tests.conftest import (
    MockControlInstrument,
    MockMeasurementInstrument,
    make_contact,
    make_gate,
    standard_instrument_configs,
)


def test_device_filter_by_group_basic():
    """Test basic device filtering by group."""
    device_config = DeviceConfig(
        name="test_device",
        gates={
            "G1": make_gate(GateType.PLUNGER, control_channel=1),
            "G2": make_gate(GateType.BARRIER, control_channel=2),
            "G3": make_gate(GateType.PLUNGER, control_channel=3),
        },
        contacts={
            "IN": make_contact(ContactType.SOURCE, measure_channel=1),
            "OUT": make_contact(ContactType.DRAIN, measure_channel=2),
        },
        groups={
            "control": DeviceGroup(gates=["G1", "G2"], contacts=["IN"]),
            "sensor": DeviceGroup(gates=["G3"], contacts=["OUT"]),
        },
        routines=[],
        instruments=standard_instrument_configs(),
    )

    channel_configs = generate_channel_configs(device_config)

    device = Device(
        name="test_device",
        device_config=device_config,
        channel_configs=channel_configs,
        control_instrument=MockControlInstrument(),
        measurement_instrument=MockMeasurementInstrument(),
    )

    # Filter by control group
    control_device = device.filter_by_group("control")
    assert set(control_device.gates) == {"G1", "G2"}
    assert set(control_device.contacts) == {"IN"}
    assert control_device.name == "test_device_control"

    # Filter by sensor group
    sensor_device = device.filter_by_group("sensor")
    assert set(sensor_device.gates) == {"G3"}
    assert set(sensor_device.contacts) == {"OUT"}
    assert sensor_device.name == "test_device_sensor"


def test_device_filter_by_group_unknown_group():
    """Test that filtering by unknown group raises error."""
    device_config = DeviceConfig(
        name="test_device",
        gates={"G1": make_gate(GateType.PLUNGER, control_channel=1)},
        contacts={},
        groups={"control": DeviceGroup(gates=["G1"])},
        routines=[],
        instruments=standard_instrument_configs(),
    )

    channel_configs = generate_channel_configs(device_config)

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
            "G1": make_gate(GateType.PLUNGER, control_channel=1),
            "G2": make_gate(GateType.BARRIER, control_channel=2),
        },
        contacts={},
        groups={
            "control": DeviceGroup(gates=["G1"]),
            "sensor": DeviceGroup(gates=["G2"]),
        },
        routines=[],
        instruments=standard_instrument_configs(),
    )

    channel_configs = generate_channel_configs(device_config)

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
            "G1": make_gate(GateType.PLUNGER, control_channel=1),
            "G2": make_gate(GateType.BARRIER, control_channel=2),
            "RES1": make_gate(GateType.RESERVOIR, control_channel=3),
            "RES2": make_gate(GateType.RESERVOIR, control_channel=4),
        },
        contacts={},
        groups={
            "control": DeviceGroup(gates=["G1", "RES1", "RES2"]),
            "sensor": DeviceGroup(gates=["G2", "RES1", "RES2"]),
        },
        routines=[],
        instruments=standard_instrument_configs(),
    )

    channel_configs = generate_channel_configs(device_config)

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
            "G1": make_gate(GateType.PLUNGER, control_channel=1),
            "G2": make_gate(GateType.BARRIER, control_channel=2),
            "G3": make_gate(GateType.PLUNGER, control_channel=3),
            "RES1": make_gate(GateType.RESERVOIR, control_channel=4),
        },
        contacts={},
        groups={
            "control": DeviceGroup(gates=["G1", "G2", "RES1"]),
            "sensor": DeviceGroup(gates=["G3", "RES1"]),
        },
        routines=[],
        instruments=standard_instrument_configs(),
    )

    channel_configs = generate_channel_configs(device_config)

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
