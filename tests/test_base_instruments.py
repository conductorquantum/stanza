import pytest

from stanza.instruments.base import (
    BaseControlInstrument,
    BaseInstrument,
    BaseMeasurementInstrument,
)
from stanza.instruments.channels import (
    ChannelConfig,
    ControlChannel,
    MeasurementChannel,
)
from stanza.models import (
    BaseInstrumentConfig,
    ControlInstrumentConfig,
    InstrumentType,
    MeasurementInstrumentConfig,
)


class TestBaseMeasurementInstrument:
    def test_initialization(self):
        config = MeasurementInstrumentConfig(
            name="test_measurement",
            type=InstrumentType.MEASUREMENT,
            ip_addr="192.168.1.1",
            measurement_duration=1.0,
            sample_time=0.5,
        )

        instrument = BaseMeasurementInstrument(config)

        assert instrument.name == "test_measurement"
        assert instrument.measurement_duration == 1.0
        assert instrument.sample_time == 0.5
        assert instrument.instrument_config == config

    def test_prepare_measurement_not_implemented(self):
        config = MeasurementInstrumentConfig(
            name="test_measurement",
            type=InstrumentType.MEASUREMENT,
            ip_addr="192.168.1.1",
            measurement_duration=1.0,
            sample_time=0.5,
        )

        instrument = BaseMeasurementInstrument(config)

        with pytest.raises(
            NotImplementedError,
            match="Specific instrument must implement prepare_measurement",
        ):
            instrument.prepare_measurement()

    def test_instrument_info_with_channels(self):
        config = MeasurementInstrumentConfig(
            name="test_measurement",
            type=InstrumentType.MEASUREMENT,
            ip_addr="192.168.1.1",
            measurement_duration=2.0,
            sample_time=1.0,
        )

        instrument = BaseMeasurementInstrument(config)

        # Add a measurement channel
        channel_config = ChannelConfig(
            name="sense1", voltage_range=(-1.0, 1.0), measure_channel=1
        )
        channel = MeasurementChannel(channel_config)
        instrument.add_channel(channel)

        info = instrument.instrument_info

        assert info["name"] == "test_measurement"
        assert info["timing"]["measurement_duration"] == 2.0
        assert info["timing"]["sample_time"] == 1.0
        assert info["channels"]["sense1"] == 1
        assert "instrument_config" in info

    def test_instrument_info_cached_property(self):
        config = MeasurementInstrumentConfig(
            name="test_measurement",
            type=InstrumentType.MEASUREMENT,
            ip_addr="192.168.1.1",
            measurement_duration=1.0,
            sample_time=0.5,
        )

        instrument = BaseMeasurementInstrument(config)

        # Access instrument_info twice
        info1 = instrument.instrument_info
        info2 = instrument.instrument_info

        # Should be the same object reference (cached)
        assert info1 is info2


class TestBaseControlInstrument:
    def test_initialization(self):
        config = ControlInstrumentConfig(
            name="test_control",
            type=InstrumentType.CONTROL,
            ip_addr="192.168.1.2",
            slew_rate=5.0,
        )

        instrument = BaseControlInstrument(config)

        assert instrument.name == "test_control"
        assert instrument.instrument_config == config

    def test_instrument_info_with_channels(self):
        config = ControlInstrumentConfig(
            name="test_control",
            type=InstrumentType.CONTROL,
            ip_addr="192.168.1.2",
            slew_rate=10.0,
        )

        instrument = BaseControlInstrument(config)

        # Add a control channel
        channel_config = ChannelConfig(
            name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
        )
        channel = ControlChannel(channel_config)
        instrument.add_channel(channel)

        info = instrument.instrument_info

        assert info["name"] == "test_control"
        assert info["slew_rate"] == 10.0
        assert info["channels"]["gate1"] == 1
        assert "instrument_config" in info

    def test_instrument_info_cached_property(self):
        config = ControlInstrumentConfig(
            name="test_control",
            type=InstrumentType.CONTROL,
            ip_addr="192.168.1.2",
            slew_rate=5.0,
        )

        instrument = BaseControlInstrument(config)

        # Access instrument_info twice
        info1 = instrument.instrument_info
        info2 = instrument.instrument_info

        # Should be the same object reference (cached)
        assert info1 is info2


class TestBaseInstrument:
    def test_initialization(self):
        config = BaseInstrumentConfig(
            name="test_base",
            type=InstrumentType.GENERAL,
            ip_addr="192.168.1.3",
        )

        instrument = BaseInstrument(config)

        assert instrument.name == "test_base"
        assert instrument.instrument_config == config

    def test_instrument_info_with_channels(self):
        config = BaseInstrumentConfig(
            name="test_base",
            type=InstrumentType.GENERAL,
            ip_addr="192.168.1.3",
        )

        instrument = BaseInstrument(config)

        # Add both measurement and control channels
        measure_channel_config = ChannelConfig(
            name="sense1", voltage_range=(-1.0, 1.0), measure_channel=1
        )
        control_channel_config = ChannelConfig(
            name="gate1", voltage_range=(-2.0, 2.0), control_channel=2
        )

        measure_channel = MeasurementChannel(measure_channel_config)
        control_channel = ControlChannel(control_channel_config)

        instrument.add_channel(measure_channel)
        instrument.add_channel(control_channel)

        info = instrument.instrument_info

        assert info["name"] == "test_base"
        assert info["channels"]["sense1"] == 1
        assert info["channels"]["gate1"] == 2
        assert "instrument_config" in info

    def test_instrument_info_cached_property(self):
        config = BaseInstrumentConfig(
            name="test_base",
            type=InstrumentType.GENERAL,
            ip_addr="192.168.1.3",
        )

        instrument = BaseInstrument(config)

        # Access instrument_info twice
        info1 = instrument.instrument_info
        info2 = instrument.instrument_info

        # Should be the same object reference (cached)
        assert info1 is info2
