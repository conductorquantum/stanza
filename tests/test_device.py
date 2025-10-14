import pytest

from stanza.device import Device
from stanza.exceptions import DeviceError
from stanza.models import Contact, ContactType, Gate, GateType, PadType
from stanza.utils import generate_channel_configs


class TestDevice:
    def test_initialization(self, device):
        assert device.name == "test_device"
        assert device.device_config.name == "test_device"
        assert device.control_instrument is not None
        assert device.measurement_instrument is not None

    def test_is_configured(self, device):
        assert device.is_configured() is True

    def test_is_configured_no_instruments(self, device_no_instruments):
        assert device_no_instruments.is_configured() is False

    def test_gates_property(self, device):
        gates = device.gates
        assert isinstance(gates, list)
        assert "gate1" in gates

    def test_jump_single_voltage(self, device):
        device.jump({"gate1": 1.5})
        assert device.control_instrument.get_voltage("gate1") == 1.5

    def test_jump_no_control_instrument(self, device_no_instruments):
        with pytest.raises(DeviceError, match="Control instrument not configured"):
            device_no_instruments.jump({"gate1": 1.5})

    def test_measure_single_pad(self, device):
        device.measurement_instrument.measurements["gate1"] = 1e-6
        current = device.measure("gate1")
        assert current == 1e-6

    def test_measure_no_measurement_instrument(self, device_no_instruments):
        with pytest.raises(DeviceError, match="Measurement instrument not configured"):
            device_no_instruments.measure("gate1")

    def test_check_voltage(self, device):
        device.control_instrument.set_voltage("gate1", 2.0)
        voltage = device.check("gate1")
        assert voltage == 2.0

    def test_check_no_control_instrument(self, device_no_instruments):
        with pytest.raises(DeviceError, match="Control instrument not configured"):
            device_no_instruments.check("gate1")

    def test_invalid_control_instrument(self, device_config):
        with pytest.raises(DeviceError, match="Control instrument must implement"):
            Device("test", device_config, None, "invalid", None)

    def test_invalid_measurement_instrument(self, device_config):
        with pytest.raises(DeviceError, match="Measurement instrument must implement"):
            Device("test", device_config, None, None, "invalid")

    def test_contacts_property(self, device):
        contacts = device.contacts
        assert isinstance(contacts, list)

    def test_control_gates_property(self, device):
        control_gates = device.control_gates
        assert isinstance(control_gates, list)
        assert "gate1" in control_gates

    def test_control_contacts_property(self, device):
        control_contacts = device.control_contacts
        assert isinstance(control_contacts, list)

    def test_measurement_gates_property(self, device):
        """Test the measurement_gates property (renamed from measure_gates)."""
        measurement_gates = device.measurement_gates
        assert isinstance(measurement_gates, list)
        assert "gate1" in measurement_gates

    def test_measurement_contacts_property(self, device):
        """Test the measurement_contacts property (renamed from measure_contacts)."""
        measurement_contacts = device.measurement_contacts
        assert isinstance(measurement_contacts, list)

    def test_get_gates_by_type(self, device):
        gates = device.get_gates_by_type("PLUNGER")
        assert "gate1" in gates

    def test_get_contacts_by_type(self, device):
        contacts = device.get_contacts_by_type("SOURCE")
        assert isinstance(contacts, list)

    def test_measure_list_of_pads(self, device):
        device.measurement_instrument.measurements["gate1"] = 1e-6
        currents = device.measure(["gate1"])
        assert currents == [1e-6]

    def test_check_list_of_pads(self, device):
        device.control_instrument.set_voltage("gate1", 2.0)
        voltages = device.check(["gate1"])
        assert voltages == [2.0]

    def test_measure_pad_not_found(self, device):
        with pytest.raises(DeviceError, match="Pad nonexistent not found"):
            device.measure("nonexistent")

    def test_check_pad_not_found(self, device):
        with pytest.raises(DeviceError, match="Pad nonexistent not found"):
            device.check("nonexistent")

    def test_jump_with_settling(self, device):
        device.jump({"gate1": 1.5}, wait_for_settling=True)
        assert device.control_instrument.get_voltage("gate1") == 1.5

    def test_jump_voltage_set_failure(self, device):
        device.control_instrument.should_fail = True
        with pytest.raises(DeviceError, match="Failed to set voltage"):
            device.jump({"gate1": 1.5})

    def test_measure_pad_no_measure_channel(
        self, device_config, control_instrument, measurement_instrument
    ):
        device_config.gates["gate2"] = Gate(
            name="gate2",
            type=GateType.PLUNGER,
            v_lower_bound=-2.0,
            v_upper_bound=2.0,
            control_channel=3,
        )
        channel_configs = generate_channel_configs(device_config)
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
        )

        with pytest.raises(DeviceError, match="Pad gate2 has no measure channel"):
            device.measure("gate2")

    def test_check_pad_no_control_channel(
        self, device_config, control_instrument, measurement_instrument
    ):
        device_config.contacts["contact2"] = Contact(
            name="contact2",
            type=ContactType.SOURCE,
            v_lower_bound=-1.0,
            v_upper_bound=1.0,
            measure_channel=4,
        )
        channel_configs = generate_channel_configs(device_config)
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
        )

        with pytest.raises(DeviceError, match="Pad contact2 has no control channel"):
            device.check("contact2")

    def test_sweep_1d(self, device):
        device.measurement_instrument.measurements["contact1"] = 1e-6
        voltages, currents = device.sweep_1d("gate1", [0.0, 1.0], "contact1")
        assert len(voltages) == 2
        assert len(currents) == 2

    def test_sweep_2d(self, device):
        device.measurement_instrument.measurements["contact1"] = 1e-6
        voltages, currents = device.sweep_2d("gate1", [0.0], "gate1", [1.0], "contact1")
        assert len(voltages) == 2
        assert len(currents) == 1

    def test_sweep_all(self, device):
        device.measurement_instrument.measurements["contact1"] = 1e-6
        voltages, currents = device.sweep_all([0.0, 1.0], "contact1")
        assert len(voltages) == 2
        assert len(currents) == 2

    def test_sweep_nd(self, device):
        device.measurement_instrument.measurements["contact1"] = 1e-6
        voltages, currents = device.sweep_nd(["gate1"], [[0.0], [1.0]], "contact1")
        assert len(voltages) == 2
        assert len(currents) == 2

    def test_zero_default(self, device):
        device.jump({"gate1": 1.5})
        device.zero()
        assert device.check("gate1") == 0.0

    def test_zero_string_types(
        self, device_config, control_instrument, measurement_instrument
    ):
        device_config.contacts["contact2"] = Contact(
            name="contact2",
            type=ContactType.DRAIN,
            v_lower_bound=-1.0,
            v_upper_bound=1.0,
            control_channel=4,
        )
        channel_configs = generate_channel_configs(device_config)
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
        )

        device.jump({"gate1": 1.5, "contact2": 0.5})
        device.zero("gate")
        assert device.check("gate1") == 0.0
        assert device.check("contact2") == 0.5

        device.zero("contact")
        assert device.check("contact2") == 0.0

    def test_zero_enum_types(
        self, device_config, control_instrument, measurement_instrument
    ):
        device_config.gates["gate2"] = Gate(
            name="gate2",
            type=GateType.BARRIER,
            v_lower_bound=-2.0,
            v_upper_bound=2.0,
            control_channel=3,
        )
        device_config.contacts["contact2"] = Contact(
            name="contact2",
            type=ContactType.DRAIN,
            v_lower_bound=-1.0,
            v_upper_bound=1.0,
            control_channel=4,
        )
        channel_configs = generate_channel_configs(device_config)
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
        )

        device.jump({"gate1": 1.0, "gate2": 2.0, "contact2": 0.5})
        device.zero(PadType.ALL)
        assert device.check(["gate1", "gate2", "contact2"]) == [0.0, 0.0, 0.0]

    def test_zero_invalid_type(self, device):
        with pytest.raises(DeviceError, match="Invalid pad type"):
            device.zero("invalid")

    def test_zero_no_control_instrument(self, device_no_instruments):
        with pytest.raises(DeviceError, match="Control instrument not configured"):
            device_no_instruments.zero()

    def test_zero_verification_failure(self, device):
        device.control_instrument.get_voltage = lambda _: 0.5
        with pytest.raises(DeviceError, match="Failed to set all controllable"):
            device.zero()
