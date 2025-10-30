from typing import Protocol, overload, runtime_checkable

from stanza.models import TriggerEdge


@runtime_checkable
class BreakoutBoxInstrument(Protocol):
    """Protocol for a digital breakout box."""

    @overload
    def get_grounded(self, channel_name: list[str]) -> list[bool]:
        """Check if breakout box channels are grounded."""
        ...

    @overload
    def get_grounded(self, channel_name: str) -> bool:
        """Check if breakout box channels are grounded."""
        ...

    def get_grounded(self, channel_name: str | list[str]) -> bool | list[bool]:
        """Check if breakout box channel(s) are grounded."""
        ...

    @overload
    def set_grounded(self, channel_name: list[str]) -> None:
        """Set breakout box channels to grounded."""
        ...

    @overload
    def set_grounded(self, channel_name: str) -> None:
        """Set breakout box channel to grounded."""
        ...

    def set_grounded(self, channel_name: str | list[str]) -> None:
        """Set breakout box channel(s) to grounded."""
        ...

    @overload
    def get_ungrounded(self, channel_name: list[str]) -> list[bool]:
        """Check if breakout box channels are ungrounded."""
        ...

    @overload
    def get_ungrounded(self, channel_name: str) -> bool:
        """Check if breakout box channel is ungrounded."""
        ...

    def get_ungrounded(self, channel_name: str | list[str]) -> bool | list[bool]:
        """Check if breakout box channel(s) are ungrounded."""
        ...

    @overload
    def set_ungrounded(self, channel_name: list[str]) -> None:
        """Set breakout box channels to ungrounded."""
        ...

    @overload
    def set_ungrounded(self, channel_name: str) -> None:
        """Set breakout box channel to ungrounded."""
        ...

    def set_ungrounded(self, channel_name: str | list[str]) -> None:
        """Set breakout box channel(s) to ungrounded."""
        ...

    @overload
    def get_connected(self, channel_name: list[str], line_number: int) -> list[bool]:
        """Check if breakout box channels are connected to the line number."""
        ...

    @overload
    def get_connected(self, channel_name: str, line_number: int) -> bool:
        """Check if breakout box channel is connected to the line number."""
        ...

    def get_connected(
        self, channel_name: str | list[str], line_number: int
    ) -> bool | list[bool]:
        """Get if breakout box channel(s) are connected to the line number."""
        ...

    @overload
    def set_connected(self, channel_name: list[str], line_number: int) -> None:
        """Set breakout box channels to connected to the line number."""
        ...

    @overload
    def set_connected(self, channel_name: str, line_number: int) -> None:
        """Set breakout box channel to connected to the line number."""
        ...

    def set_connected(self, channel_name: str | list[str], line_number: int) -> None:
        """Set breakout box channel(s) to connected to the line number."""
        ...

    @overload
    def get_disconnected(self, channel_name: list[str], line_number: int) -> list[bool]:
        """Check if breakout box channels are disconnected from the line number."""
        ...

    @overload
    def get_disconnected(self, channel_name: str, line_number: int) -> bool:
        """Check if breakout box channel is disconnected from the line number."""
        ...

    def get_disconnected(
        self, channel_name: str | list[str], line_number: int
    ) -> bool | list[bool]:
        """Get if breakout box channel(s) are disconnected from the line number."""
        ...

    @overload
    def set_disconnected(self, channel_name: list[str], line_number: int) -> None:
        """Set breakout box channels to disconnected from the line number."""
        ...

    @overload
    def set_disconnected(self, channel_name: str, line_number: int) -> None:
        """Set breakout box channel to disconnected from the line number."""
        ...

    def set_disconnected(self, channel_name: str | list[str], line_number: int) -> None:
        """Set breakout box channel(s) to disconnected from the line number."""
        ...


@runtime_checkable
class ControlInstrument(Protocol):
    """Protocol for control instruments."""

    def set_voltage(self, channel_name: str, voltage: float) -> None:
        """Set the voltage on a specific channel."""
        ...

    def get_voltage(self, channel_name: str) -> float:
        """Get the voltage on a specific channel."""
        ...

    def get_slew_rate(self, channel_name: str) -> float:
        """Get the slew rate on a specific channel."""
        ...


@runtime_checkable
class MeasurementInstrument(Protocol):
    """Protocol for measurement instruments."""

    @overload
    def measure(self, channel_name: str) -> float:
        """Measure the current on a specific channel."""
        ...

    @overload
    def measure(self, channel_name: list[str]) -> list[float]:
        """Optional overload for measuring the current on multiple channels."""
        ...

    def measure(self, channel_name: str | list[str]) -> float | list[float]:
        """Measure the current on a specific channel(s)."""
        ...


@runtime_checkable
class TriggerSequencer(Protocol):
    """Instrument that can load sequences and respond to hardware triggers."""

    name: str

    def load_sequence(self, pad: str, values: list[float], trigger_port: str) -> None:
        """Load a sequence of values to output on trigger events.

        Args:
            pad: Channel/pad to sequence
            values: List of values to step through
            trigger_port: Physical trigger input (e.g., "ext1", "trig_in_2")
        """
        ...

    def arm_triggers(self) -> None:
        """Enable trigger listening mode."""
        ...


@runtime_checkable
class TriggerCoordinator(Protocol):
    """Instrument that can generate trigger pulses and coordinate timing."""

    name: str

    def configure_trigger_output(
        self, channel: str, edge: TriggerEdge = TriggerEdge.RISING
    ) -> None:
        """Configure a channel to output trigger pulses.

        Args:
            channel: Physical trigger output (e.g., "sync_out", "marker1")
            edge: Which edge to use for triggering
        """
        ...

    def triggered_acquisition(
        self, pad: str, num_points: int, trigger_output: str, trigger_period_us: float
    ) -> list[float]:
        """Run acquisition while sending trigger pulses.

        Args:
            pad: Channel/pad to measure from
            num_points: Number of measurements (and triggers) to perform
            trigger_output: Which output channel sends the triggers
            trigger_period_us: Time between trigger pulses in microseconds

        Returns:
            List of measured values, one per trigger
        """
        ...


class NamedResource(Protocol):
    """Protocol for resources that have a name attribute."""

    name: str
