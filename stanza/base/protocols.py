from typing import Protocol, overload, runtime_checkable


@runtime_checkable
class SetupInstrument(Protocol):
    """Protocol for setup instruments."""

    def setup(self) -> None:
        """Setup the setup instrument."""
        ...

    def teardown(self) -> None:
        """Teardown the setup instrument."""
        ...


class BreakoutBox(SetupInstrument):
    """Protocol for a digital breakout box."""

    @overload
    def ground(self, channel_name: str) -> None:
        """Ground a specific channel."""
        ...

    @overload
    def ground(self, channel_name: list[str]) -> None:
        """Ground a list of channels."""
        ...

    def ground(self, channel_name: str | list[str]) -> None:
        """Ground specific channel(s)."""
        ...

    @overload
    def unground(self, channel_name: str) -> None:
        """Unground a specific channel."""
        ...

    @overload
    def unground(self, channel_name: list[str]) -> None:
        """Unground a list of channels."""
        ...

    def unground(self, channel_name: str | list[str]) -> None:
        """Unground specific channel(s)."""
        ...

    @overload
    def connect(self, channel_name: str, line_number: int) -> None:
        """COnnect a specific channel to a specific breakout boxline."""
        ...

    @overload
    def connect(self, channel_name: list[str], line_number: int) -> None:
        """Connect a specific channels to a specific breakout boxline."""
        ...

    def connect(self, channel_name: str | list[str], line_number: int) -> None:
        """Connect specific channel(s) to a specific breakout boxline."""
        ...

    @overload
    def disconnect(self, channel_name: str) -> None:
        """Disconnect a specific channel from all breakout boxlines."""
        ...

    @overload
    def disconnect(self, channel_name: list[str]) -> None:
        """Disconnect a specific channels from all breakout boxlines."""
        ...

    def disconnect(self, channel_name: str | list[str]) -> None:
        """Disconnect specific channel(s) from all breakout boxlines."""
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


class NamedResource(Protocol):
    """Protocol for resources that have a name attribute."""

    name: str
