from contextlib import contextmanager

import pytest

from stanza.instruments.channels import (
    ChannelConfig,
    ControlChannel,
    MeasurementChannel,
)
from stanza.instruments.mixins import (
    ControlInstrumentMixin,
    InstrumentChannelMixin,
    MeasurementInstrumentMixin,
)


class TestInstrumentChannelMixin:
    def test_add_and_get_channel(self):
        mixin = InstrumentChannelMixin()

        config = ChannelConfig(
            name="test_channel", voltage_range=(-1.0, 1.0), control_channel=1
        )
        channel = ControlChannel(config=config)

        mixin.add_channel(channel)
        retrieved_channel = mixin.get_channel("test_channel")

        assert retrieved_channel == channel
        assert "test_channel" in mixin.channels

    def test_add_channel_with_custom_name(self):
        mixin = InstrumentChannelMixin()

        config = ChannelConfig(
            name="test_channel", voltage_range=(-1.0, 1.0), control_channel=1
        )
        channel = ControlChannel(config=config)

        mixin.add_channel("custom_name", channel)
        retrieved_channel = mixin.get_channel("custom_name")

        assert retrieved_channel == channel
        assert "custom_name" in mixin.channels
        assert "test_channel" not in mixin.channels

    def test_add_channel_invalid_args(self):
        mixin = InstrumentChannelMixin()

        with pytest.raises(
            ValueError, match="Must provide either channel or channel_name and channel"
        ):
            mixin.add_channel(None)

    def test_remove_channel(self):
        mixin = InstrumentChannelMixin()

        config = ChannelConfig(
            name="test_channel", voltage_range=(-1.0, 1.0), control_channel=1
        )
        channel = ControlChannel(config=config)

        mixin.add_channel(channel)
        mixin.remove_channel("test_channel")

        assert "test_channel" not in mixin.channels


class TestControlInstrumentMixin:
    def test_set_and_get_voltage(self):
        mixin = ControlInstrumentMixin()

        config = ChannelConfig(
            name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
        )
        channel = ControlChannel(config=config)
        mixin.add_channel(channel)

        mixin.set_voltage("gate1", 1.5)
        voltage = mixin.get_voltage("gate1")

        assert voltage == 1.5

    def test_get_slew_rate(self):
        mixin = ControlInstrumentMixin()

        config = ChannelConfig(
            name="gate1", voltage_range=(-2.0, 2.0), control_channel=1
        )
        channel = ControlChannel(config=config)
        mixin.add_channel(channel)

        channel.set_parameter("slew_rate", 5.0)
        slew_rate = mixin.get_slew_rate("gate1")

        assert slew_rate == 5.0


class TestMeasurementInstrumentMixin:
    def test_prepare_measurement_not_implemented(self):
        mixin = MeasurementInstrumentMixin()

        with pytest.raises(
            NotImplementedError,
            match="Specific instrument must implement prepare_measurement",
        ):
            mixin.prepare_measurement()

    def test_measure_basic_functionality(self):
        class TestMeasurementInstrument(MeasurementInstrumentMixin):
            @contextmanager
            def prepare_measurement(self):
                yield

        mixin = TestMeasurementInstrument()

        config = ChannelConfig(
            name="sense1", voltage_range=(-1.0, 1.0), measure_channel=1
        )
        channel = MeasurementChannel(config=config)
        channel.set_parameter("current", 1e-6)
        mixin.add_channel(channel)

        current = mixin.measure("sense1")

        assert current == 1e-6

    def test_prepare_measurement_context_avoids_nested_calls(self):
        class TestMeasurementInstrument(MeasurementInstrumentMixin):
            def __init__(self):
                super().__init__()
                self.prepare_count = 0
                self.teardown_count = 0

            def prepare_measurement(self):
                self.prepare_count += 1

            def teardown_measurement(self):
                self.teardown_count += 1

        mixin = TestMeasurementInstrument()

        with mixin.prepare_measurement_context():
            with mixin.prepare_measurement_context():
                pass

        assert mixin.prepare_count == 1
        assert mixin.teardown_count == 1
