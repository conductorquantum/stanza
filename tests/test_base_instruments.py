import pytest

from stanza.base.channels import (
    ChannelConfig,
    ControlChannel,
    MeasurementChannel,
)
from stanza.base.instruments import (
    BaseControlInstrument,
    BaseInstrument,
    BaseMeasurementInstrument,
)
from stanza.models import (
    BaseInstrumentConfig,
    ContactType,
    ControlInstrumentConfig,
    GateType,
    InstrumentType,
    MeasurementInstrumentConfig,
    PadType,
)


@pytest.fixture
def measurement_config():
    return MeasurementInstrumentConfig(
        name="test_measurement",
        type=InstrumentType.MEASUREMENT,
        ip_addr="192.168.1.1",
        measurement_duration=1.0,
        sample_time=0.5,
    )


@pytest.fixture
def control_config():
    return ControlInstrumentConfig(
        name="test_control",
        type=InstrumentType.CONTROL,
        ip_addr="192.168.1.2",
        slew_rate=5.0,
    )


@pytest.fixture
def base_config():
    return BaseInstrumentConfig(
        name="test_base",
        type=InstrumentType.GENERAL,
        ip_addr="192.168.1.3",
    )


@pytest.fixture
def measurement_channel():
    config = ChannelConfig(
        name="sense1",
        voltage_range=(-1.0, 1.0),
        pad_type=PadType.CONTACT,
        electrode_type=ContactType.SOURCE,
        measure_channel=1,
    )
    return MeasurementChannel(config)


@pytest.fixture
def control_channel():
    config = ChannelConfig(
        name="gate1",
        voltage_range=(-2.0, 2.0),
        pad_type=PadType.GATE,
        electrode_type=GateType.PLUNGER,
        control_channel=1,
    )
    return ControlChannel(config)


class TestBaseMeasurementInstrument:
    def test_initialization(self, measurement_config):
        instrument = BaseMeasurementInstrument(measurement_config)

        assert instrument.name == "test_measurement"
        assert instrument.measurement_duration == 1.0
        assert instrument.sample_time == 0.5
        assert instrument.instrument_config == measurement_config

    def test_instrument_info_with_channels(
        self, measurement_config, measurement_channel
    ):
        instrument = BaseMeasurementInstrument(measurement_config)
        instrument.add_channel(measurement_channel)
        info = instrument.instrument_info

        assert info["name"] == "test_measurement"
        assert info["timing"]["measurement_duration"] == 1.0
        assert info["timing"]["sample_time"] == 0.5
        assert info["channels"]["sense1"] == 1
        assert "instrument_config" in info

    def test_instrument_info_cached_property(self, measurement_config):
        instrument = BaseMeasurementInstrument(measurement_config)
        info1 = instrument.instrument_info
        info2 = instrument.instrument_info

        assert info1 is info2

    def test_teardown_measurement(self, measurement_config):
        instrument = BaseMeasurementInstrument(measurement_config)
        instrument.teardown_measurement()


class TestBaseControlInstrument:
    def test_initialization(self, control_config):
        instrument = BaseControlInstrument(control_config)

        assert instrument.name == "test_control"
        assert instrument.instrument_config == control_config

    def test_instrument_info_with_channels(self, control_config, control_channel):
        instrument = BaseControlInstrument(control_config)
        instrument.add_channel(control_channel)
        info = instrument.instrument_info

        assert info["name"] == "test_control"
        assert info["slew_rate"] == 5.0
        assert info["channels"]["gate1"] == 1
        assert "instrument_config" in info

    def test_instrument_info_cached_property(self, control_config):
        instrument = BaseControlInstrument(control_config)
        info1 = instrument.instrument_info
        info2 = instrument.instrument_info

        assert info1 is info2


class TestBaseInstrument:
    def test_initialization(self, base_config):
        instrument = BaseInstrument(base_config)

        assert instrument.name == "test_base"
        assert instrument.instrument_config == base_config

    def test_instrument_info_with_channels(
        self, base_config, measurement_channel, control_channel
    ):
        instrument = BaseInstrument(base_config)
        instrument.add_channel(measurement_channel)
        instrument.add_channel(control_channel)
        info = instrument.instrument_info

        assert info["name"] == "test_base"
        assert info["channels"]["sense1"] == 1
        assert info["channels"]["gate1"] == 1
        assert "instrument_config" in info

    def test_instrument_info_cached_property(self, base_config):
        instrument = BaseInstrument(base_config)
        info1 = instrument.instrument_info
        info2 = instrument.instrument_info

        assert info1 is info2
