from contextlib import contextmanager

import pytest

from stanza.base.channels import (
    ChannelConfig,
    ControlChannel,
    MeasurementChannel,
)
from stanza.base.mixins import (
    ControlInstrumentMixin,
    InstrumentChannelMixin,
    MeasurementInstrumentMixin,
)
from stanza.models import ContactType, GateType, PadType


@pytest.fixture
def channel_mixin():
    return InstrumentChannelMixin()


@pytest.fixture
def control_config():
    return ChannelConfig(
        "test_channel",
        (-2.0, 2.0),
        PadType.GATE,
        GateType.PLUNGER,
        control_channel=1,
    )


@pytest.fixture
def measurement_config():
    return ChannelConfig(
        "sense1",
        (-1.0, 1.0),
        PadType.CONTACT,
        ContactType.SOURCE,
        measure_channel=1,
    )


class TestInstrumentChannelMixin:
    def test_add_and_get_channel(self, channel_mixin, control_config):
        channel = ControlChannel(config=control_config)
        channel_mixin.add_channel(channel)
        retrieved_channel = channel_mixin.get_channel("test_channel")

        assert retrieved_channel == channel
        assert "test_channel" in channel_mixin.channels

    def test_add_channel_with_custom_name(self, channel_mixin, control_config):
        channel = ControlChannel(config=control_config)
        channel_mixin.add_channel("custom_name", channel)
        retrieved_channel = channel_mixin.get_channel("custom_name")

        assert retrieved_channel == channel
        assert "custom_name" in channel_mixin.channels
        assert "test_channel" not in channel_mixin.channels

    def test_add_channel_invalid_args(self, channel_mixin):
        with pytest.raises(
            ValueError, match="Must provide either channel or channel_name and channel"
        ):
            channel_mixin.add_channel(None)

    def test_remove_channel(self, channel_mixin, control_config):
        channel = ControlChannel(config=control_config)
        channel_mixin.add_channel(channel)
        channel_mixin.remove_channel("test_channel")

        assert "test_channel" not in channel_mixin.channels


class TestControlInstrumentMixin:
    @pytest.fixture
    def control_mixin(self, control_config):
        mixin = ControlInstrumentMixin()
        channel = ControlChannel(config=control_config)
        mixin.add_channel(channel)
        return mixin

    def test_set_and_get_voltage(self, control_mixin):
        control_mixin.set_voltage("test_channel", 1.5)
        assert control_mixin.get_voltage("test_channel") == 1.5

    def test_get_slew_rate(self, control_mixin):
        channel = control_mixin.get_channel("test_channel")
        channel.set_parameter("slew_rate", 5.0)
        assert control_mixin.get_slew_rate("test_channel") == 5.0

    def test_set_slew_rate(self, control_mixin):
        control_mixin.set_slew_rate("test_channel", 3.5)
        channel = control_mixin.get_channel("test_channel")
        assert channel.get_parameter_value("slew_rate") == 3.5


class TestMeasurementInstrumentMixin:
    def test_measure_basic_functionality(self, measurement_config):
        class TestMeasurementInstrument(MeasurementInstrumentMixin):
            @contextmanager
            def prepare_measurement(self):
                yield

        mixin = TestMeasurementInstrument()
        channel = MeasurementChannel(config=measurement_config)
        channel.set_parameter("current", 1e-6)
        mixin.add_channel(channel)

        assert mixin.measure("sense1") == 1e-6

    def test_prepare_measurement_context_avoids_nested_calls(self):
        class TestMeasurementInstrument(MeasurementInstrumentMixin):
            def __init__(self):
                super().__init__()
                self.prepare_count = self.teardown_count = 0

            def prepare_measurement(self):
                self.prepare_count += 1

            def teardown_measurement(self):
                self.teardown_count += 1

        mixin = TestMeasurementInstrument()
        with mixin.prepare_measurement_context():
            with mixin.prepare_measurement_context():
                pass
        assert (mixin.prepare_count, mixin.teardown_count) == (1, 1)
