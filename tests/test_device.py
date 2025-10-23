import pytest

from stanza.base.channels import ChannelConfig
from stanza.base.instruments import BaseControlInstrument, BaseMeasurementInstrument
from stanza.device import Device
from stanza.exceptions import DeviceError
from stanza.models import (
    GPIO,
    Contact,
    ContactType,
    ControlInstrumentConfig,
    Gate,
    GateType,
    GPIOType,
    InstrumentType,
    MeasurementInstrumentConfig,
    PadType,
)
from stanza.utils import generate_channel_configs
from tests.conftest import MockBreakoutBoxInstrument


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
        assert len(voltages) == 1
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

    def test_gpios_property(
        self, device_config, control_instrument, measurement_instrument
    ):
        """Test the gpios property returns list of all GPIO pad names."""
        device_config.gpios = {
            "gpio1": GPIO(
                name="gpio1",
                type=GPIOType.OUTPUT,
                v_lower_bound=0.0,
                v_upper_bound=3.3,
                control_channel=5,
            )
        }
        channel_configs = generate_channel_configs(device_config)
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
        )
        gpios = device.gpios
        assert isinstance(gpios, list)
        assert "gpio1" in gpios

    def test_control_gpios_property(
        self, device_config, control_instrument, measurement_instrument
    ):
        """Test the control_gpios property returns list of GPIO pads with control channels."""
        device_config.gpios = {
            "gpio1": GPIO(
                name="gpio1",
                type=GPIOType.OUTPUT,
                v_lower_bound=0.0,
                v_upper_bound=3.3,
                control_channel=5,
            )
        }
        channel_configs = generate_channel_configs(device_config)
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
        )
        control_gpios = device.control_gpios
        assert isinstance(control_gpios, list)
        assert "gpio1" in control_gpios

    def test_zero_gpio_type(
        self, device_config, control_instrument, measurement_instrument
    ):
        """Test zeroing GPIO pads specifically using PadType.GPIO."""
        device_config.gpios = {
            "gpio1": GPIO(
                name="gpio1",
                type=GPIOType.OUTPUT,
                v_lower_bound=0.0,
                v_upper_bound=3.3,
                control_channel=5,
            )
        }
        channel_configs = generate_channel_configs(device_config)
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
        )

        device.jump({"gpio1": 3.3})
        assert device.check("gpio1") == 3.3

        device.zero("gpio")
        assert device.check("gpio1") == 0.0

    def test_zero_all_includes_gpios(
        self, device_config, control_instrument, measurement_instrument
    ):
        """Test zeroing all pads includes GPIO pads when using PadType.ALL."""
        device_config.gpios = {
            "gpio1": GPIO(
                name="gpio1",
                type=GPIOType.OUTPUT,
                v_lower_bound=0.0,
                v_upper_bound=3.3,
                control_channel=5,
            )
        }
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

        device.jump({"gate1": 1.5, "contact2": 0.5, "gpio1": 3.3})
        device.zero(PadType.ALL)
        assert device.check(["gate1", "contact2", "gpio1"]) == [0.0, 0.0, 0.0]


class TestDeviceBreakoutBox:
    """Test breakout box functionality in Device."""

    def test_invalid_breakout_box_instrument(self, device_config):
        """Test that invalid breakout box instrument raises error."""
        channel_configs = generate_channel_configs(device_config)
        with pytest.raises(DeviceError, match="Breakout Box instrument must implement"):
            Device("test", device_config, channel_configs, None, None, "invalid")

    def test_breakout_lines_property(
        self, device_config, control_instrument, measurement_instrument
    ):
        """Test breakout_lines property returns channels with breakout_channel set."""
        device_config.gates["gate1"].breakout_channel = 1
        device_config.gates["gate2"] = Gate(
            name="gate2",
            type=GateType.BARRIER,
            v_lower_bound=-2.0,
            v_upper_bound=2.0,
            control_channel=3,
            breakout_channel=2,
        )
        channel_configs = generate_channel_configs(device_config)
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
        )
        breakout_lines = device.breakout_lines
        assert isinstance(breakout_lines, list)
        assert "gate1" in breakout_lines
        assert "gate2" in breakout_lines

    def test_ground_no_breakout_box(self, device):
        """Test ground_breakout_lines() raises error when breakout box not configured."""
        with pytest.raises(DeviceError, match="Breakout box instrument not configured"):
            device.ground_breakout_lines()

    def test_unground_no_breakout_box(self, device):
        """Test unground_breakout_lines() raises error when breakout box not configured."""
        with pytest.raises(DeviceError, match="Breakout box instrument not configured"):
            device.unground_breakout_lines()

    def test_connect_no_breakout_box(self, device):
        """Test connect_breakout_lines() raises error when breakout box not configured."""
        with pytest.raises(DeviceError, match="Breakout box instrument not configured"):
            device.connect_breakout_lines()

    def test_disconnect_no_breakout_box(self, device):
        """Test disconnect_breakout_lines() raises error when breakout box not configured."""
        with pytest.raises(DeviceError, match="Breakout box instrument not configured"):
            device.disconnect_breakout_lines()

    def test_ground_with_breakout_box(
        self, device_config, control_instrument, measurement_instrument
    ):
        """Test ground() calls set_grounded for all breakout lines."""
        device_config.gates["gate1"].breakout_channel = 1
        device_config.gates["gate2"] = Gate(
            name="gate2",
            type=GateType.BARRIER,
            v_lower_bound=-2.0,
            v_upper_bound=2.0,
            control_channel=3,
            breakout_channel=2,
        )
        channel_configs = generate_channel_configs(device_config)
        breakout_box = MockBreakoutBoxInstrument()
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
            breakout_box,
        )

        device.ground_breakout_lines()
        assert "gate1" in breakout_box.grounded_lines
        assert "gate2" in breakout_box.grounded_lines

    def test_unground_with_breakout_box(
        self, device_config, control_instrument, measurement_instrument
    ):
        """Test unground() calls set_ungrounded for all breakout lines."""

        device_config.gates["gate1"].breakout_channel = 1
        channel_configs = generate_channel_configs(device_config)
        breakout_box = MockBreakoutBoxInstrument()
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
            breakout_box,
        )

        device.unground_breakout_lines()
        assert "gate1" in breakout_box.ungrounded_lines

    def test_connect_with_measure_channel(self, device_config, control_instrument):
        class MockMeasurementInst(BaseMeasurementInstrument):
            def __init__(self, instrument_config):
                self.instrument_config = instrument_config
                self.measurements = {}

            def measure(self, channel_name: str) -> float:
                return self.measurements.get(channel_name, 0.0)

        device_config.gates["gate1"].breakout_channel = 1
        channel_configs = generate_channel_configs(device_config)
        breakout_box = MockBreakoutBoxInstrument()

        meas_config = MeasurementInstrumentConfig(
            name="mock_meas",
            type=InstrumentType.MEASUREMENT,
            ip_addr="127.0.0.1",
            measurement_duration=1.0,
            sample_time=0.1,
            breakout_line=5,
        )
        mock_meas = MockMeasurementInst(meas_config)

        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            mock_meas,
            breakout_box,
        )

        device.connect_breakout_lines()
        assert ("gate1", 5) in breakout_box.connected_calls

    def test_connect_with_control_channel(self, device_config, measurement_instrument):
        class MockControlInst(BaseControlInstrument):
            def __init__(self, instrument_config):
                self.instrument_config = instrument_config
                self.voltages = {}

            def set_voltage(self, channel_name: str, voltage: float) -> None:
                self.voltages[channel_name] = voltage

            def get_voltage(self, channel_name: str) -> float:
                return self.voltages.get(channel_name, 0.0)

            def get_slew_rate(self, channel_name: str) -> float:
                return 1.0

        device_config.contacts["contact2"] = Contact(
            name="contact2",
            type=ContactType.DRAIN,
            v_lower_bound=-1.0,
            v_upper_bound=1.0,
            control_channel=4,
            breakout_channel=3,
        )

        channel_configs = generate_channel_configs(device_config)
        breakout_box = MockBreakoutBoxInstrument()

        ctrl_config = ControlInstrumentConfig(
            name="mock_ctrl",
            type=InstrumentType.CONTROL,
            ip_addr="127.0.0.1",
            slew_rate=1.0,
            breakout_line=7,
        )
        mock_ctrl = MockControlInst(ctrl_config)

        device = Device(
            "test",
            device_config,
            channel_configs,
            mock_ctrl,
            measurement_instrument,
            breakout_box,
        )

        device.connect_breakout_lines()
        assert ("contact2", 7) in breakout_box.connected_calls

    def test_disconnect_with_measure_channel(self, device_config, control_instrument):
        class MockMeasurementInst(BaseMeasurementInstrument):
            def __init__(self, instrument_config):
                self.instrument_config = instrument_config
                self.measurements = {}

            def measure(self, channel_name: str) -> float:
                return self.measurements.get(channel_name, 0.0)

        device_config.gates["gate1"].breakout_channel = 1
        channel_configs = generate_channel_configs(device_config)
        breakout_box = MockBreakoutBoxInstrument()

        meas_config = MeasurementInstrumentConfig(
            name="mock_meas",
            type=InstrumentType.MEASUREMENT,
            ip_addr="127.0.0.1",
            measurement_duration=1.0,
            sample_time=0.1,
            breakout_line=5,
        )
        mock_meas = MockMeasurementInst(meas_config)

        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            mock_meas,
            breakout_box,
        )

        device.disconnect_breakout_lines()
        assert ("gate1", 5) in breakout_box.disconnected_calls

    def test_connection_no_instrument_channel(
        self, device_config, control_instrument, measurement_instrument
    ):
        channel_configs = generate_channel_configs(device_config)
        channel_configs["orphan"] = ChannelConfig(
            name="orphan",
            voltage_range=(0.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            breakout_channel=1,
        )

        breakout_box = MockBreakoutBoxInstrument()
        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            measurement_instrument,
            breakout_box,
        )

        with pytest.raises(DeviceError, match="has no associated instrument channel"):
            device.connect_breakout_lines()

    def test_connection_no_measurement_instrument(
        self, device_config, control_instrument
    ):
        device_config.gates["gate1"].breakout_channel = 1
        channel_configs = generate_channel_configs(device_config)
        breakout_box = MockBreakoutBoxInstrument()

        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            None,
            breakout_box,
        )

        with pytest.raises(DeviceError, match="Measurement instrument not configured"):
            device.connect_breakout_lines()

    def test_connection_no_control_instrument(
        self, device_config, measurement_instrument
    ):
        device_config.contacts["contact2"] = Contact(
            name="contact2",
            type=ContactType.DRAIN,
            v_lower_bound=-1.0,
            v_upper_bound=1.0,
            control_channel=4,
            breakout_channel=3,
        )

        channel_configs = generate_channel_configs(device_config)
        breakout_box = MockBreakoutBoxInstrument()

        device = Device(
            "test",
            device_config,
            channel_configs,
            None,
            measurement_instrument,
            breakout_box,
        )

        with pytest.raises(DeviceError, match="Control instrument not configured"):
            device.connect_breakout_lines()

    def test_connection_no_breakout_line_configured(
        self, device_config, control_instrument
    ):
        class MockMeasurementInst(BaseMeasurementInstrument):
            def __init__(self, instrument_config):
                self.instrument_config = instrument_config
                self.measurements = {}

            def measure(self, channel_name: str) -> float:
                return self.measurements.get(channel_name, 0.0)

        device_config.gates["gate1"].breakout_channel = 1
        channel_configs = generate_channel_configs(device_config)
        breakout_box = MockBreakoutBoxInstrument()

        meas_config = MeasurementInstrumentConfig(
            name="mock_meas",
            type=InstrumentType.MEASUREMENT,
            ip_addr="127.0.0.1",
            measurement_duration=1.0,
            sample_time=0.1,
        )
        mock_meas = MockMeasurementInst(meas_config)

        device = Device(
            "test",
            device_config,
            channel_configs,
            control_instrument,
            mock_meas,
            breakout_box,
        )

        with pytest.raises(
            DeviceError, match="instrument breakout line not configured"
        ):
            device.connect_breakout_lines()
