"""Trigger wiring description — driver-agnostic.

TriggerLink describes a physical digital trigger wire between two instruments.
All timing is in nanoseconds.  Driver-specific trigger *configuration*
(modes, clock-cycle timing) lives in the respective driver module
(e.g. stanza.drivers.opx_triggers).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerLink:
    """Describes a digital trigger wire between two instruments.

    The source instrument emits a digital pulse; the sink instrument
    acts on it (e.g., steps to next voltage in a list).

    Attributes:
        name: Human-readable identifier (e.g., "qdac_ch1_trigger").
        source_port: Port identifier on the source instrument (format is driver-specific).
        sink_instrument: Reference name of the receiving instrument (e.g., "qdac").
        sink_channel: Channel name on the sink instrument.
        sink_trigger_port: External trigger port on sink (e.g., "ext1").
        trigger_duration_ns: Width of the digital pulse in nanoseconds.
    """

    name: str
    source_port: tuple[str | int, ...]
    sink_instrument: str
    sink_channel: str
    sink_trigger_port: str
    trigger_duration_ns: int = 100

    def __post_init__(self) -> None:
        if self.trigger_duration_ns <= 0:
            raise ValueError("trigger_duration_ns must be positive")
