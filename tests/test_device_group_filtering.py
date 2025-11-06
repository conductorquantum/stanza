import pytest

from stanza.device import Device
from stanza.exceptions import DeviceError
from stanza.models import (
    ContactType,
    DeviceConfig,
    DeviceGroup,
    GPIO,
    GPIOType,
    GateType,
)
from stanza.utils import generate_channel_configs
from tests.conftest import (
    MockControlInstrument,
    MockMeasurementInstrument,
    make_contact,
    make_gate,
    standard_instrument_configs,
)


def make_gpio(
    gpio_type: GPIOType = GPIOType.INPUT,
    control_channel: int | None = None,
    measure_channel: int | None = None,
    v_lower_bound: float = -5.0,
    v_upper_bound: float = 5.0,
) -> GPIO:
    """Helper function to create GPIO instances with common defaults."""
    return GPIO(
        type=gpio_type,
        control_channel=control_channel,
        measure_channel=measure_channel,
        v_lower_bound=v_lower_bound,
        v_upper_bound=v_upper_bound,
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


class TestConditionalFiltering:
    """Tests for conditional filtering of GPIOs and contacts."""

    def test_group_with_omitted_gpios_includes_all_device_gpios(self):
        """Test that when gpios are omitted from group, ALL device GPIOs are included."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
            },
            contacts={
                "IN": make_contact(ContactType.SOURCE, measure_channel=1),
                "OUT": make_contact(ContactType.DRAIN, measure_channel=2),
            },
            gpios={
                "A0": make_gpio(GPIOType.INPUT, control_channel=10),
                "A1": make_gpio(GPIOType.INPUT, control_channel=11),
                "A2": make_gpio(GPIOType.INPUT, control_channel=12),
                "VDD": make_gpio(GPIOType.INPUT, control_channel=13),
            },
            groups={
                # Group doesn't specify gpios - should get ALL gpios
                "control": DeviceGroup(gates=["G1"], contacts=["IN"]),
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
        filtered_device = device.filter_by_group("control")

        # Should include only specified gates and contacts
        assert set(filtered_device.gates) == {"G1"}
        assert set(filtered_device.contacts) == {"IN"}

        # Should include ALL device GPIOs (not specified, so all included)
        assert set(filtered_device.gpios) == {"A0", "A1", "A2", "VDD"}

    def test_group_with_omitted_contacts_includes_all_device_contacts(self):
        """Test that when contacts are omitted from group, ALL device contacts are included."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
            },
            contacts={
                "IN_A": make_contact(ContactType.SOURCE, measure_channel=1),
                "OUT_A": make_contact(ContactType.DRAIN, measure_channel=2),
                "OUT_B": make_contact(ContactType.DRAIN, measure_channel=3),
            },
            gpios={
                "VDD": make_gpio(GPIOType.INPUT, control_channel=10),
            },
            groups={
                # Group doesn't specify contacts - should get ALL contacts
                "control": DeviceGroup(gates=["G1"], gpios=["VDD"]),
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
        filtered_device = device.filter_by_group("control")

        # Should include only specified gates and gpios
        assert set(filtered_device.gates) == {"G1"}
        assert set(filtered_device.gpios) == {"VDD"}

        # Should include ALL device contacts (not specified, so all included)
        assert set(filtered_device.contacts) == {"IN_A", "OUT_A", "OUT_B"}

    def test_group_with_explicit_gpios_includes_only_specified(self):
        """Test that when gpios are explicitly specified, ONLY those are included."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
            },
            contacts={
                "IN": make_contact(ContactType.SOURCE, measure_channel=1),
            },
            gpios={
                "A0": make_gpio(GPIOType.INPUT, control_channel=10),
                "A1": make_gpio(GPIOType.INPUT, control_channel=11),
                "A2": make_gpio(GPIOType.INPUT, control_channel=12),
                "VDD": make_gpio(GPIOType.INPUT, control_channel=13),
                "VSS": make_gpio(GPIOType.INPUT, control_channel=14),
            },
            groups={
                # Group explicitly specifies only A0 and VDD
                "control": DeviceGroup(gates=["G1"], contacts=["IN"], gpios=["A0", "VDD"]),
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
        filtered_device = device.filter_by_group("control")

        # Should include only specified elements
        assert set(filtered_device.gates) == {"G1"}
        assert set(filtered_device.contacts) == {"IN"}
        assert set(filtered_device.gpios) == {"A0", "VDD"}  # Only specified ones

        # Should NOT include A1, A2, VSS
        assert "A1" not in filtered_device.gpios
        assert "A2" not in filtered_device.gpios
        assert "VSS" not in filtered_device.gpios

    def test_group_with_explicit_contacts_includes_only_specified(self):
        """Test that when contacts are explicitly specified, ONLY those are included."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
            },
            contacts={
                "IN_A": make_contact(ContactType.SOURCE, measure_channel=1),
                "OUT_A": make_contact(ContactType.DRAIN, measure_channel=2),
                "OUT_B": make_contact(ContactType.DRAIN, measure_channel=3),
            },
            gpios={
                "VDD": make_gpio(GPIOType.INPUT, control_channel=10),
            },
            groups={
                # Group explicitly specifies only IN_A and OUT_A
                "control": DeviceGroup(
                    gates=["G1"], contacts=["IN_A", "OUT_A"], gpios=["VDD"]
                ),
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
        filtered_device = device.filter_by_group("control")

        # Should include only specified elements
        assert set(filtered_device.gates) == {"G1"}
        assert set(filtered_device.contacts) == {"IN_A", "OUT_A"}  # Only specified ones
        assert set(filtered_device.gpios) == {"VDD"}

        # Should NOT include OUT_B
        assert "OUT_B" not in filtered_device.contacts

    def test_group_with_empty_contacts_and_gpios_includes_none(self):
        """Test that when contacts and gpios are explicitly empty lists, NONE are included."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
            },
            contacts={
                "IN": make_contact(ContactType.SOURCE, measure_channel=1),
                "OUT_A": make_contact(ContactType.DRAIN, measure_channel=2),
                "OUT_B": make_contact(ContactType.DRAIN, measure_channel=3),
            },
            gpios={
                "VDD": make_gpio(GPIOType.INPUT, control_channel=10),
                "VSS": make_gpio(GPIOType.INPUT, control_channel=11),
                "A0": make_gpio(GPIOType.INPUT, control_channel=12),
            },
            groups={
                # Group explicitly specifies empty lists for contacts and gpios
                "control": DeviceGroup(gates=["G1", "G2"], contacts=[], gpios=[]),
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
        filtered_device = device.filter_by_group("control")

        # Should include specified gates
        assert set(filtered_device.gates) == {"G1", "G2"}

        # Should NOT include any contacts (empty list specified)
        assert len(filtered_device.contacts) == 0
        assert "IN" not in filtered_device.contacts
        assert "OUT_A" not in filtered_device.contacts
        assert "OUT_B" not in filtered_device.contacts

        # Should NOT include any gpios (empty list specified)
        assert len(filtered_device.gpios) == 0
        assert "VDD" not in filtered_device.gpios
        assert "VSS" not in filtered_device.gpios
        assert "A0" not in filtered_device.gpios

    def test_mixed_groups_different_filtering_behavior(self):
        """Test mixed scenario: one group with explicit gpios, one without."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
            },
            contacts={
                "IN": make_contact(ContactType.SOURCE, measure_channel=1),
                "OUT_A": make_contact(ContactType.DRAIN, measure_channel=2),
                "OUT_B": make_contact(ContactType.DRAIN, measure_channel=3),
            },
            gpios={
                "A0": make_gpio(GPIOType.INPUT, control_channel=10),
                "A1": make_gpio(GPIOType.INPUT, control_channel=11),
                "VDD": make_gpio(GPIOType.INPUT, control_channel=12),
                "VSS": make_gpio(GPIOType.INPUT, control_channel=13),
            },
            groups={
                # control: explicit gpios (only VDD), omits contacts (gets all)
                "control": DeviceGroup(gates=["G1"], gpios=["VDD"]),
                # sensor: omits gpios (gets all), explicit contacts (only OUT_B)
                "sensor": DeviceGroup(gates=["G2"], contacts=["OUT_B"]),
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
        assert set(control_device.gates) == {"G1"}
        assert set(control_device.gpios) == {"VDD"}  # Only specified
        assert set(control_device.contacts) == {"IN", "OUT_A", "OUT_B"}  # All (omitted)

        # Filter by sensor group
        sensor_device = device.filter_by_group("sensor")
        assert set(sensor_device.gates) == {"G2"}
        assert set(sensor_device.gpios) == {"A0", "A1", "VDD", "VSS"}  # All (omitted)
        assert set(sensor_device.contacts) == {"OUT_B"}  # Only specified

    def test_gates_always_filter_explicitly(self):
        """Test that gates ALWAYS filter explicitly regardless of omission."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
                "G3": make_gate(GateType.PLUNGER, control_channel=3),
            },
            contacts={
                "IN": make_contact(ContactType.SOURCE, measure_channel=1),
            },
            gpios={
                "VDD": make_gpio(GPIOType.INPUT, control_channel=10),
            },
            groups={
                # Group specifies only G1 and G2 - should NOT get G3
                "control": DeviceGroup(gates=["G1", "G2"]),
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
        filtered_device = device.filter_by_group("control")

        # Should include ONLY specified gates
        assert set(filtered_device.gates) == {"G1", "G2"}
        assert "G3" not in filtered_device.gates

        # Contacts and GPIOs should include all (omitted from group)
        assert set(filtered_device.contacts) == {"IN"}
        assert set(filtered_device.gpios) == {"VDD"}

    def test_gpios_can_be_explicitly_shared_between_groups(self):
        """Test that GPIOs can be explicitly listed in multiple groups (like contacts)."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
            },
            contacts={
                "IN": make_contact(ContactType.SOURCE, measure_channel=1),
            },
            gpios={
                "VDD": make_gpio(GPIOType.INPUT, control_channel=10),
                "VSS": make_gpio(GPIOType.INPUT, control_channel=11),
                "A0": make_gpio(GPIOType.INPUT, control_channel=12),
            },
            groups={
                # Both groups explicitly list VDD and VSS (shared infrastructure)
                "control": DeviceGroup(gates=["G1"], gpios=["VDD", "VSS"]),
                "sensor": DeviceGroup(gates=["G2"], gpios=["VDD", "VSS", "A0"]),
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
        assert set(control_device.gates) == {"G1"}
        assert set(control_device.gpios) == {"VDD", "VSS"}  # Shared GPIOs
        assert "A0" not in control_device.gpios

        # Filter by sensor group
        sensor_device = device.filter_by_group("sensor")
        assert set(sensor_device.gates) == {"G2"}
        assert set(sensor_device.gpios) == {"VDD", "VSS", "A0"}

    def test_non_reservoir_gates_cannot_be_shared(self):
        """Test that non-RESERVOIR gates cannot be explicitly shared between groups."""
        with pytest.raises(
            ValueError,
            match="Gate 'G1' referenced by group 'sensor' already assigned to group 'control'",
        ):
            DeviceConfig(
                name="test_device",
                gates={
                    "G1": make_gate(GateType.PLUNGER, control_channel=1),
                    "G2": make_gate(GateType.BARRIER, control_channel=2),
                },
                contacts={},
                groups={
                    "control": DeviceGroup(gates=["G1"]),
                    "sensor": DeviceGroup(gates=["G1", "G2"]),  # G1 is shared - not allowed
                },
                routines=[],
                instruments=standard_instrument_configs(),
            )

    def test_reservoir_gates_can_be_shared(self):
        """Test that RESERVOIR gates can be explicitly shared between groups."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
                "RES1": make_gate(GateType.RESERVOIR, control_channel=3),
            },
            contacts={},
            groups={
                "control": DeviceGroup(gates=["G1", "RES1"]),
                "sensor": DeviceGroup(gates=["G2", "RES1"]),  # RES1 is shared - allowed
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
        assert set(control_device.gates) == {"G1", "RES1"}

        # Filter by sensor group
        sensor_device = device.filter_by_group("sensor")
        assert set(sensor_device.gates) == {"G2", "RES1"}

    def test_contacts_can_be_explicitly_shared_between_groups(self):
        """Test that contacts can be explicitly listed in multiple groups."""
        device_config = DeviceConfig(
            name="test_device",
            gates={
                "G1": make_gate(GateType.PLUNGER, control_channel=1),
                "G2": make_gate(GateType.BARRIER, control_channel=2),
            },
            contacts={
                "IN": make_contact(ContactType.SOURCE, measure_channel=1),
                "OUT_A": make_contact(ContactType.DRAIN, measure_channel=2),
                "OUT_B": make_contact(ContactType.DRAIN, measure_channel=3),
            },
            gpios={},
            groups={
                # Both groups explicitly list IN (shared source contact)
                "control": DeviceGroup(gates=["G1"], contacts=["IN", "OUT_A"]),
                "sensor": DeviceGroup(gates=["G2"], contacts=["IN", "OUT_B"]),
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
        assert set(control_device.gates) == {"G1"}
        assert set(control_device.contacts) == {"IN", "OUT_A"}

        # Filter by sensor group
        sensor_device = device.filter_by_group("sensor")
        assert set(sensor_device.gates) == {"G2"}
        assert set(sensor_device.contacts) == {"IN", "OUT_B"}
