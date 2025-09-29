import pytest

from stanza.instruments.channels import (
    ChannelConfig,
    ControlChannel,
    MeasurementChannel,
    Parameter,
    Validators,
)
from stanza.models import ContactType, GateType, PadType


class TestChannelConfig:
    def test_basic_initialization(self):
        config = ChannelConfig(
            name="gate1",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
        )
        assert config.name == "gate1"
        assert config.voltage_range == (-1.0, 1.0)
        assert config.output_mode == "dc"
        assert config.enabled is True
        assert config.metadata == {}

    def test_custom_initialization(self):
        config = ChannelConfig(
            name="gate2",
            voltage_range=(-5.0, 5.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.BARRIER,
            control_channel=1,
            measure_channel=2,
            output_mode="ac",
            enabled=False,
            unit="mV",
        )
        assert config.control_channel == 1
        assert config.measure_channel == 2
        assert config.output_mode == "ac"
        assert config.enabled is False
        assert config.unit == "mV"


class TestParameter:
    def test_basic_parameter(self):
        param = Parameter(name="voltage", value=1.0, unit="V")
        assert param.name == "voltage"
        assert param.value == 1.0
        assert param.unit == "V"
        assert param.metadata == {}

    def test_parameter_validation_success(self):
        param = Parameter(
            name="voltage",
            value=1.0,
            unit="V",
            validator=Validators.range_validator(-2.0, 2.0),
        )
        param.set(1.5)
        assert param.value == 1.5

    def test_parameter_validation_failure(self):
        param = Parameter(
            name="voltage",
            value=1.0,
            unit="V",
            validator=Validators.range_validator(-2.0, 2.0),
        )
        with pytest.raises(ValueError, match="Invalid value 3.0 for parameter voltage"):
            param.set(3.0)

    def test_parameter_getter_setter(self):
        stored_value = [0.0]

        def setter(value):
            stored_value[0] = value

        def getter():
            return stored_value[0]

        param = Parameter(name="test", getter=getter, setter=setter)
        param.set(2.5)
        assert param.get() == 2.5
        assert stored_value[0] == 2.5


class TestValidators:
    def test_range_validator(self):
        validator = Validators.range_validator(-1.0, 1.0)
        assert validator(0.5) is True
        assert validator(-1.0) is True
        assert validator(1.0) is True
        assert validator(-1.1) is False
        assert validator(1.1) is False

    def test_positive_validator(self):
        assert Validators.positive_validator(1.0) is True
        assert Validators.positive_validator(0.1) is True
        assert Validators.positive_validator(0.0) is False
        assert Validators.positive_validator(-1.0) is False

    def test_negative_validator(self):
        assert Validators.negative_validator(-1.0) is True
        assert Validators.negative_validator(-0.1) is True
        assert Validators.negative_validator(0.0) is False
        assert Validators.negative_validator(1.0) is False

    def test_non_zero_validator(self):
        assert Validators.non_zero_validator(1.0) is True
        assert Validators.non_zero_validator(-1.0) is True
        assert Validators.non_zero_validator(0.0) is False


class TestControlChannel:
    def test_initialization(self):
        config = ChannelConfig(
            name="gate1",
            voltage_range=(-2.0, 2.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        )
        channel = ControlChannel(config=config)

        assert channel.channel_id == 1
        assert channel.config == config
        assert "voltage" in channel.parameters
        assert "slew_rate" in channel.parameters

    def test_voltage_parameter_validation(self):
        config = ChannelConfig(
            name="gate1",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        )
        channel = ControlChannel(config=config)

        channel.set_parameter("voltage", 0.5)
        assert channel.get_parameter_value("voltage") == 0.5

        with pytest.raises(ValueError):
            channel.set_parameter("voltage", 2.0)

    def test_slew_rate_parameter(self):
        config = ChannelConfig(
            name="gate1",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        )
        channel = ControlChannel(config=config)

        channel.set_parameter("slew_rate", 10.0)
        assert channel.get_parameter_value("slew_rate") == 10.0

        with pytest.raises(ValueError):
            channel.set_parameter("slew_rate", 0.0)


class TestMeasurementChannel:
    def test_initialization(self):
        config = ChannelConfig(
            name="sense1",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.CONTACT,
            electrode_type=ContactType.SOURCE,
            measure_channel=1,
        )
        channel = MeasurementChannel(config=config)

        assert channel.channel_id == 1
        assert "current" in channel.parameters
        assert "conversion_factor" in channel.parameters

    def test_conversion_factor_validation(self):
        config = ChannelConfig(
            name="sense1",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.CONTACT,
            electrode_type=ContactType.SOURCE,
            measure_channel=1,
        )
        channel = MeasurementChannel(config=config)

        channel.set_parameter("conversion_factor", 1e-6)
        assert channel.get_parameter_value("conversion_factor") == 1e-6

        with pytest.raises(ValueError):
            channel.set_parameter("conversion_factor", 0.0)


class TestInstrumentChannelBase:
    def test_parameter_management(self):
        config = ChannelConfig(
            name="test",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        )
        channel = ControlChannel(config=config)

        # Test getting existing parameter
        voltage_param = channel.get_parameter("voltage")
        assert voltage_param.name == "voltage"

        # Test getting non-existent parameter
        with pytest.raises(KeyError, match="Parameter 'nonexistent' not found"):
            channel.get_parameter("nonexistent")

        # Test adding duplicate parameter
        duplicate_param = Parameter(name="voltage", value=0.0)
        with pytest.raises(ValueError, match="Parameter 'voltage' already exists"):
            channel.add_parameter(duplicate_param)

    def test_parameter_value_access(self):
        config = ChannelConfig(
            name="test",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        )
        channel = ControlChannel(config=config)

        channel.set_parameter("voltage", 0.8)
        assert channel.get_parameter_value("voltage") == 0.8

        with pytest.raises(KeyError):
            channel.get_parameter_value("nonexistent")

    def test_str_representation(self):
        config = ChannelConfig(
            name="test_channel",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=5,
        )
        channel = ControlChannel(config=config)

        str_repr = str(channel)
        assert "Channel(name=test_channel" in str_repr
        assert "channel_id=5" in str_repr
        assert "config=" in str_repr
        assert "parameters=" in str_repr

    def test_channel_info(self):
        config = ChannelConfig(
            name="info_channel",
            voltage_range=(-2.0, 2.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=3,
        )
        channel = ControlChannel(config=config)

        info = channel.channel_info
        assert info["name"] == "info_channel"
        assert info["channel_id"] == 3
        assert info["config"] == config
        assert "voltage" in info["parameters"]
        assert "slew_rate" in info["parameters"]

    def test_set_parameter_non_validation_error(self):
        config = ChannelConfig(
            name="test",
            voltage_range=(-1.0, 1.0),
            pad_type=PadType.GATE,
            electrode_type=GateType.PLUNGER,
            control_channel=1,
        )
        channel = ControlChannel(config=config)

        with pytest.raises(Exception, match="Set parameter.*failed"):
            channel.set_parameter("nonexistent_param", 1.0)
