from typing import Protocol, runtime_checkable


@runtime_checkable
class Controlinstrument(Protocol):
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

    def measure(self, channel_name: str) -> float:
        """Measure the current on a specific channel."""
        ...
