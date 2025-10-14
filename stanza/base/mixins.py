from collections.abc import Generator
from contextlib import contextmanager

from stanza.base.channels import InstrumentChannel


class InstrumentChannelMixin:
    """Mixin class that provides the default InstrumentChannel implementation."""

    def __init__(self) -> None:
        self.channels: dict[str, InstrumentChannel] = {}

    def add_channel(
        self,
        channel_name_or_channel: str | InstrumentChannel | None,
        channel: InstrumentChannel | None = None,
    ) -> None:
        """Add a channel to the instrument.

        Can be called as:
        - add_channel(channel) - uses channel.name as key
        - add_channel(channel_name, channel) - uses provided name as key
        """
        if isinstance(channel_name_or_channel, InstrumentChannel):
            actual_channel = channel_name_or_channel
            actual_name = actual_channel.name
        elif channel is not None:
            actual_channel = channel
            actual_name = (
                channel_name_or_channel
                if channel_name_or_channel is not None
                else channel.name
            )
        else:
            raise ValueError("Must provide either channel or channel_name and channel")

        self.channels[actual_name] = actual_channel

    def get_channel(self, channel_name: str) -> InstrumentChannel:
        """Get a channel by name."""
        return self.channels[channel_name]

    def remove_channel(self, channel_name: str) -> None:
        """Remove a channel from the instrument."""
        del self.channels[channel_name]


class ControlInstrumentMixin(InstrumentChannelMixin):
    """Mixin class that provides the default ControlInstrument implementation using channel parameters."""

    def set_voltage(self, channel_name: str, voltage: float) -> None:
        """Set voltage on a specific channel using parameter system."""
        channel_obj = self.get_channel(channel_name)
        channel_obj.set_parameter("voltage", voltage)

    def get_voltage(self, channel_name: str) -> float:
        """Get current voltage on a specific channel using parameter system."""
        channel_obj = self.get_channel(channel_name)
        return float(channel_obj.get_parameter_value("voltage"))

    def get_slew_rate(self, channel_name: str) -> float:
        """Get current slew rate on a specific channel using parameter system."""
        channel_obj = self.get_channel(channel_name)
        return float(channel_obj.get_parameter_value("slew_rate"))

    def set_slew_rate(self, channel_name: str, slew_rate: float) -> None:
        """Set slew rate on a specific channel using parameter system."""
        channel_obj = self.get_channel(channel_name)
        channel_obj.set_parameter("slew_rate", slew_rate)


class MeasurementInstrumentMixin(InstrumentChannelMixin):
    """Mixin class that provides default MeasurementInstrument implementations."""

    @contextmanager
    def prepare_measurement_context(self) -> Generator[None, None, None]:
        """Context manager that handles preparation, avoiding nested calls."""
        if getattr(self, "_in_prepare_measurement", False):
            yield
        else:
            # Not in context, do actual preparation
            self._in_prepare_measurement = True
            try:
                self.prepare_measurement()
                yield
            finally:
                self._in_prepare_measurement = False
                self.teardown_measurement()

    def prepare_measurement(self) -> None:
        """Optional override in subclasses for actual preparation logic."""
        ...

    def teardown_measurement(self) -> None:
        """Optional override in subclasses for actual teardown logic."""
        ...

    def measure(self, channel_name: str) -> float:
        """Default measurement that reads the 'current' parameter from the first channel.

        Subclasses are expected to expose channels via get_channel(int) and a
        'current' Parameter on each channel whose getter performs the measurement.
        """
        with self.prepare_measurement_context():
            channel_obj = self.get_channel(channel_name)
            return float(channel_obj.get_parameter_value("current"))
