from typing import Any, Protocol, overload, runtime_checkable

import numpy as np


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


class NamedResource(Protocol):
    """Protocol for resources that have a name attribute."""

    name: str


@runtime_checkable
class ListSweepInstrument(Protocol):
    """Instrument that supports pre-loaded voltage list sweeps."""

    def load_voltage_list(
        self,
        channel_name: str,
        voltages: list[float] | np.ndarray,
        trigger_port: str,
        **kwargs: Any,
    ) -> None: ...

    def reset_voltage_list(self, channel_name: str) -> None: ...


@runtime_checkable
class HardwareSweepController(Protocol):
    """Controller that executes hardware-triggered sweep programs."""

    def execute_sweep_1d(
        self,
        trigger_link_name: str,
        n_points: int,
        measure_electrode: str,
        n_avg: int = 1,
        settling_wait_ns: int = 250_000,
    ) -> np.ndarray: ...

    def execute_sweep_2d(
        self,
        outer_trigger_name: str,
        inner_trigger_name: str,
        n_outer: int,
        n_inner: int,
        measure_electrode: str,
        n_avg: int = 1,
        settling_wait_ns: int = 250_000,
    ) -> np.ndarray: ...
