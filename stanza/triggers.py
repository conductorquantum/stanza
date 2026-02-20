"""Trigger configuration for instrument synchronization.

Driver-agnostic. Describes when things should happen, not how a specific AWG implements it.
All durations stored in QUA clock cycles (1 cycle = 4 ns).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stanza.timing import ns_to_cycles


class TriggerMode(Enum):
    """Trigger mode for instrument synchronization."""

    SOFTWARE = "software"
    HARDWARE_OUT = "hardware_out"
    HARDWARE_IN = "hardware_in"
    TIMED = "timed"


@dataclass(frozen=True)
class TriggerConfig:
    """Configuration for a trigger mode.

    All durations are in QUA clock cycles (1 cycle = 4 ns).
    Use convenience constructors (software_trigger, timed_trigger, etc.)
    for ergonomic creation with nanosecond inputs.
    """

    mode: TriggerMode

    # For HARDWARE_OUT / HARDWARE_IN
    digital_port: tuple[str, int, int] | None = None

    # For HARDWARE_OUT: marker duration
    trigger_duration_cycles: int = 25  # 100 ns default

    # For HARDWARE_IN: timeout and debounce
    input_timeout_cycles: int | None = None  # None = wait forever
    input_debounce_cycles: int = 0

    # For TIMED: interval and repetition count
    interval_cycles: int | None = None
    repetitions: int | None = None

    def __post_init__(self) -> None:
        if self.mode == TriggerMode.HARDWARE_OUT and self.digital_port is None:
            raise ValueError("HARDWARE_OUT requires digital_port")
        if self.mode == TriggerMode.HARDWARE_IN and self.digital_port is None:
            raise ValueError("HARDWARE_IN requires digital_port")
        if self.mode == TriggerMode.TIMED and self.interval_cycles is None:
            raise ValueError("TIMED requires interval_cycles")
        if self.interval_cycles is not None and self.interval_cycles <= 0:
            raise ValueError("interval_cycles must be positive")
        if self.trigger_duration_cycles <= 0:
            raise ValueError("trigger_duration_cycles must be positive")


def software_trigger() -> TriggerConfig:
    """Create a software trigger (host-controlled pause/resume)."""
    return TriggerConfig(mode=TriggerMode.SOFTWARE)


def timed_trigger(interval_ns: int, repetitions: int | None = None) -> TriggerConfig:
    """Create a timed trigger with fixed interval between iterations."""
    return TriggerConfig(
        mode=TriggerMode.TIMED,
        interval_cycles=ns_to_cycles(interval_ns),
        repetitions=repetitions,
    )


def hardware_out_trigger(
    digital_port: tuple[str, int, int],
    duration_ns: int = 100,
) -> TriggerConfig:
    """Create a hardware output trigger (instrument emits digital marker)."""
    return TriggerConfig(
        mode=TriggerMode.HARDWARE_OUT,
        digital_port=digital_port,
        trigger_duration_cycles=ns_to_cycles(duration_ns),
    )


def hardware_in_trigger(
    digital_port: tuple[str, int, int],
    timeout_ns: int | None = None,
    debounce_ns: int = 0,
) -> TriggerConfig:
    """Create a hardware input trigger (instrument waits for external digital edge)."""
    return TriggerConfig(
        mode=TriggerMode.HARDWARE_IN,
        digital_port=digital_port,
        input_timeout_cycles=ns_to_cycles(timeout_ns)
        if timeout_ns is not None
        else None,
        input_debounce_cycles=ns_to_cycles(debounce_ns) if debounce_ns > 0 else 0,
    )
