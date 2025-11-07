import pytest

from stanza.device import Device
from stanza.exceptions import DeviceError
from stanza.models import (
    GPIO,
    ContactType,
    DeviceConfig,
    DeviceGroup,
    GateType,
    GPIOType,
    PadType,
)
from stanza.utils import generate_channel_configs
from tests.conftest import (
    MockControlInstrument,
    MockMeasurementInstrument,
    make_contact,
    make_gate,
    standard_instrument_configs,
)


def get_gates(configs):
    """Extract gate names from channel configs."""
    return [name for name, config in configs.items() if config.pad_type == PadType.GATE]


def get_contacts(configs):
    """Extract contact names from channel configs."""
    return [
        name for name, config in configs.items() if config.pad_type == PadType.CONTACT
    ]


def get_gpios(configs):
    """Extract GPIO names from channel configs."""
    return [name for name, config in configs.items() if config.pad_type == PadType.GPIO]


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


def test_device_filter_by_group_basic(create_device):
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

    device = create_device(device_config)

    # Filter by control group
    control_configs = device.filter_by_group("control")
    assert set(get_gates(control_configs)) == {"G1", "G2"}
    assert set(get_contacts(control_configs)) == {"IN"}

    # Filter by sensor group
    sensor_configs = device.filter_by_group("sensor")
    assert set(get_gates(sensor_configs)) == {"G3"}
    assert set(get_contacts(sensor_configs)) == {"OUT"}


def test_device_filter_by_group_unknown_group(create_device):
    """Test that filtering by unknown group raises error."""
    device_config = DeviceConfig(
        name="test_device",
        gates={"G1": make_gate(GateType.PLUNGER, control_channel=1)},
        contacts={},
        groups={"control": DeviceGroup(gates=["G1"])},
        routines=[],
        instruments=standard_instrument_configs(),
    )

    device = create_device(device_config)

    # Try to filter by unknown group
    with pytest.raises(DeviceError, match="Group 'unknown' not found"):
        device.filter_by_group("unknown")


def test_device_get_shared_gates(create_device):
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

    device = create_device(device_config)

    # Check shared gates
    shared_gates = device.get_shared_gates()
    assert set(shared_gates) == {"RES1", "RES2"}


def test_device_get_other_group_gates(create_device):
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

    device = create_device(device_config)

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

    def test_group_with_omitted_gpios_includes_all_device_gpios(self, create_device):
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

        device = create_device(device_config)

        # Filter by control group
        filtered_configs = device.filter_by_group("control")

        # Should include only specified gates and contacts
        assert set(get_gates(filtered_configs)) == {"G1"}
        assert set(get_contacts(filtered_configs)) == {"IN"}

        # Should include ALL device GPIOs (not specified, so all included)
        assert set(get_gpios(filtered_configs)) == {"A0", "A1", "A2", "VDD"}

    def test_group_with_omitted_contacts_includes_all_device_contacts(
        self, create_device
    ):
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

        device = create_device(device_config)

        # Filter by control group
        filtered_configs = device.filter_by_group("control")

        # Should include only specified gates and gpios
        assert set(get_gates(filtered_configs)) == {"G1"}
        assert set(get_gpios(filtered_configs)) == {"VDD"}

        # Should include ALL device contacts (not specified, so all included)
        assert set(get_contacts(filtered_configs)) == {"IN_A", "OUT_A", "OUT_B"}

    def test_group_with_explicit_gpios_includes_only_specified(self, create_device):
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
                "control": DeviceGroup(
                    gates=["G1"], contacts=["IN"], gpios=["A0", "VDD"]
                ),
            },
            routines=[],
            instruments=standard_instrument_configs(),
        )

        device = create_device(device_config)

        # Filter by control group
        filtered_configs = device.filter_by_group("control")

        # Should include only specified elements
        assert set(get_gates(filtered_configs)) == {"G1"}
        assert set(get_contacts(filtered_configs)) == {"IN"}
        assert set(get_gpios(filtered_configs)) == {"A0", "VDD"}  # Only specified ones

        # Should NOT include A1, A2, VSS
        gpios = get_gpios(filtered_configs)
        assert "A1" not in gpios
        assert "A2" not in gpios
        assert "VSS" not in gpios

    def test_group_with_explicit_contacts_includes_only_specified(self, create_device):
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

        device = create_device(device_config)

        # Filter by control group
        filtered_configs = device.filter_by_group("control")

        # Should include only specified elements
        assert set(get_gates(filtered_configs)) == {"G1"}
        assert set(get_contacts(filtered_configs)) == {
            "IN_A",
            "OUT_A",
        }  # Only specified ones
        assert set(get_gpios(filtered_configs)) == {"VDD"}

        # Should NOT include OUT_B
        contacts = get_contacts(filtered_configs)
        assert "OUT_B" not in contacts

    def test_group_with_empty_contacts_and_gpios_includes_none(self, create_device):
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

        device = create_device(device_config)

        # Filter by control group
        filtered_configs = device.filter_by_group("control")

        # Should include specified gates
        assert set(get_gates(filtered_configs)) == {"G1", "G2"}

        # Should NOT include any contacts (empty list specified)
        contacts = get_contacts(filtered_configs)
        assert len(contacts) == 0
        assert "IN" not in contacts
        assert "OUT_A" not in contacts
        assert "OUT_B" not in contacts

        # Should NOT include any gpios (empty list specified)
        gpios = get_gpios(filtered_configs)
        assert len(gpios) == 0
        assert "VDD" not in gpios
        assert "VSS" not in gpios
        assert "A0" not in gpios

    def test_mixed_groups_different_filtering_behavior(self, create_device):
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

        device = create_device(device_config)

        # Filter by control group
        control_configs = device.filter_by_group("control")
        assert set(get_gates(control_configs)) == {"G1"}
        assert set(get_gpios(control_configs)) == {"VDD"}  # Only specified
        assert set(get_contacts(control_configs)) == {
            "IN",
            "OUT_A",
            "OUT_B",
        }  # All (omitted)

        # Filter by sensor group
        sensor_configs = device.filter_by_group("sensor")
        assert set(get_gates(sensor_configs)) == {"G2"}
        assert set(get_gpios(sensor_configs)) == {
            "A0",
            "A1",
            "VDD",
            "VSS",
        }  # All (omitted)
        assert set(get_contacts(sensor_configs)) == {"OUT_B"}  # Only specified

    def test_gates_always_filter_explicitly(self, create_device):
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

        device = create_device(device_config)

        # Filter by control group
        filtered_configs = device.filter_by_group("control")

        # Should include ONLY specified gates
        gates = get_gates(filtered_configs)
        assert set(gates) == {"G1", "G2"}
        assert "G3" not in gates

        # Contacts and GPIOs should include all (omitted from group)
        assert set(get_contacts(filtered_configs)) == {"IN"}
        assert set(get_gpios(filtered_configs)) == {"VDD"}

    def test_gpios_can_be_explicitly_shared_between_groups(self, create_device):
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

        device = create_device(device_config)

        # Filter by control group
        control_configs = device.filter_by_group("control")
        assert set(get_gates(control_configs)) == {"G1"}
        control_gpios = get_gpios(control_configs)
        assert set(control_gpios) == {"VDD", "VSS"}  # Shared GPIOs
        assert "A0" not in control_gpios

        # Filter by sensor group
        sensor_configs = device.filter_by_group("sensor")
        assert set(get_gates(sensor_configs)) == {"G2"}
        assert set(get_gpios(sensor_configs)) == {"VDD", "VSS", "A0"}

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
                    "sensor": DeviceGroup(
                        gates=["G1", "G2"]
                    ),  # G1 is shared - not allowed
                },
                routines=[],
                instruments=standard_instrument_configs(),
            )

    def test_reservoir_gates_can_be_shared(self, create_device):
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

        device = create_device(device_config)

        # Filter by control group
        control_configs = device.filter_by_group("control")
        assert set(get_gates(control_configs)) == {"G1", "RES1"}

        # Filter by sensor group
        sensor_configs = device.filter_by_group("sensor")
        assert set(get_gates(sensor_configs)) == {"G2", "RES1"}

    def test_contacts_can_be_explicitly_shared_between_groups(self, create_device):
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

        device = create_device(device_config)

        # Filter by control group
        control_configs = device.filter_by_group("control")
        assert set(get_gates(control_configs)) == {"G1"}
        assert set(get_contacts(control_configs)) == {"IN", "OUT_A"}

        # Filter by sensor group
        sensor_configs = device.filter_by_group("sensor")
        assert set(get_gates(sensor_configs)) == {"G2"}
        assert set(get_contacts(sensor_configs)) == {"IN", "OUT_B"}
