from typing import Protocol, overload, runtime_checkable


@runtime_checkable
class BreakoutBoxInstrument(Protocol):
    """Protocol for a digital breakout box."""

    @overload
    def get_grounded(self, channel_name: list[str]) -> list[bool]: ...

    @overload
    def get_grounded(self, channel_name: str) -> bool: ...

    def get_grounded(self, channel_name: str | list[str]) -> bool | list[bool]:
        """Get if the channel is grounded."""
        ...

    @overload
    def set_grounded(self, channel_name: list[str]) -> None: ...

    @overload
    def set_grounded(self, channel_name: str) -> None: ...

    def set_grounded(self, channel_name: str | list[str]) -> None:
        """Set the channel to grounded."""
        ...

    @overload
    def get_ungrounded(self, channel_name: list[str]) -> list[bool]: ...

    @overload
    def get_ungrounded(self, channel_name: str) -> bool: ...

    def get_ungrounded(self, channel_name: str | list[str]) -> bool | list[bool]:
        """Get if the channel is ungrounded."""
        ...

    @overload
    def set_ungrounded(self, channel_name: list[str]) -> None: ...

    @overload
    def set_ungrounded(self, channel_name: str) -> None: ...

    def set_ungrounded(self, channel_name: str | list[str]) -> None:
        """Set the channel to ungrounded."""
        ...

    @overload
    def get_connected(
        self, channel_name: list[str], line_number: int
    ) -> list[bool]: ...

    @overload
    def get_connected(self, channel_name: str, line_number: int) -> bool: ...

    def get_connected(
        self, channel_name: str | list[str], line_number: int
    ) -> bool | list[bool]:
        """Get if the channel is connected to the line number."""
        ...

    @overload
    def set_connected(self, channel_name: list[str], line_number: int) -> None: ...

    @overload
    def set_connected(self, channel_name: str, line_number: int) -> None: ...

    def set_connected(self, channel_name: str | list[str], line_number: int) -> None:
        """Set the channel to connected to the line number."""
        ...

    @overload
    def get_disconnected(
        self, channel_name: list[str], line_number: int
    ) -> list[bool]: ...

    @overload
    def get_disconnected(self, channel_name: str, line_number: int) -> bool: ...

    def get_disconnected(
        self, channel_name: str | list[str], line_number: int
    ) -> bool | list[bool]:
        """Get if the channel is disconnected from the line number."""
        ...

    @overload
    def set_disconnected(self, channel_name: list[str], line_number: int) -> None: ...

    @overload
    def set_disconnected(self, channel_name: str, line_number: int) -> None: ...

    def set_disconnected(self, channel_name: str | list[str], line_number: int) -> None:
        """Set the channel to disconnected from the line number."""
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
