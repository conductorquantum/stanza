"""Pulse definitions and registry for arbitrary waveform generators.

Driver-agnostic. Pairs waveform samples with metadata (operation type, digital markers).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from stanza.exceptions import PulseError
from stanza.timing import cycles_to_ns, ns_to_cycles, ns_to_samples
from stanza.waveforms import (
    OPX_PLUS_CONSTRAINTS,
    WaveformConstraints,
    cosine_waveform,
    drag_waveform,
    gaussian_waveform,
    square_waveform,
)

_QUA_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_pulse_name(name: str) -> None:
    if not _QUA_NAME_RE.match(name):
        raise PulseError(
            f"Invalid pulse name '{name}': must be a valid identifier "
            f"(alphanumeric + underscores, no leading digit)"
        )


@dataclass(frozen=True)
class PulseDefinition:
    """A named pulse with waveform data and metadata."""

    name: str
    operation: Literal["control", "measurement"]
    length_cycles: int
    waveform_i: np.ndarray
    waveform_q: np.ndarray | None = None
    digital_marker: str | None = None

    @property
    def length_ns(self) -> int:
        return cycles_to_ns(self.length_cycles)

    @property
    def is_iq(self) -> bool:
        return self.waveform_q is not None

    def __post_init__(self) -> None:
        _validate_pulse_name(self.name)
        expected_samples = ns_to_samples(self.length_ns)
        if len(self.waveform_i) != expected_samples:
            raise PulseError(
                f"Pulse '{self.name}': waveform_i has {len(self.waveform_i)} samples, "
                f"expected {expected_samples} for {self.length_cycles} cycles"
            )
        if self.waveform_q is not None and len(self.waveform_q) != expected_samples:
            raise PulseError(
                f"Pulse '{self.name}': waveform_q has {len(self.waveform_q)} samples, "
                f"expected {expected_samples}"
            )


@dataclass(frozen=True)
class DigitalMarker:
    """Digital marker definition matching QUA schema."""

    name: str
    samples: list[tuple[int, int]]  # [(value, duration_cycles), ...]

    def __post_init__(self) -> None:
        _validate_pulse_name(self.name)
        for value, duration in self.samples:
            if value not in (0, 1):
                raise PulseError(f"Digital marker value must be 0 or 1, got {value}")
            if duration <= 0:
                raise PulseError(
                    f"Digital marker duration must be positive, got {duration}"
                )

    @property
    def total_cycles(self) -> int:
        return sum(d for _, d in self.samples)


def marker_high(name: str, duration_cycles: int) -> DigitalMarker:
    """Digital marker held HIGH for the full duration."""
    return DigitalMarker(name=name, samples=[(1, duration_cycles)])


def marker_trigger(name: str, width_cycles: int, total_cycles: int) -> DigitalMarker:
    """Short trigger pulse at the start, LOW for the rest."""
    return DigitalMarker(
        name=name,
        samples=[
            (1, width_cycles),
            (0, total_cycles - width_cycles),
        ],
    )


class PulseRegistry:
    """Collects named pulse and marker definitions. Deduplicates waveforms by content hash."""

    def __init__(self) -> None:
        self._pulses: dict[str, PulseDefinition] = {}
        self._markers: dict[str, DigitalMarker] = {}
        self._waveform_hashes: dict[bytes, str] = {}

    def add_pulse(self, pulse: PulseDefinition) -> None:
        """Register a pulse. Validates uniqueness and marker/pulse length parity."""
        if pulse.name in self._pulses:
            raise PulseError(f"Duplicate pulse name: '{pulse.name}'")
        if pulse.digital_marker is not None:
            if pulse.digital_marker not in self._markers:
                raise PulseError(
                    f"Pulse '{pulse.name}' references marker '{pulse.digital_marker}' "
                    f"which has not been registered. Call add_marker() first."
                )
            marker = self._markers[pulse.digital_marker]
            if marker.total_cycles != pulse.length_cycles:
                raise PulseError(
                    f"Pulse '{pulse.name}' has length {pulse.length_cycles} cycles but "
                    f"marker '{pulse.digital_marker}' has total duration {marker.total_cycles} cycles. "
                    f"These must match to avoid silent desync."
                )
        self._pulses[pulse.name] = pulse

    def add_marker(self, marker: DigitalMarker) -> None:
        if marker.name in self._markers:
            raise PulseError(f"Duplicate marker name: '{marker.name}'")
        self._markers[marker.name] = marker

    def get_pulse(self, name: str) -> PulseDefinition:
        if name not in self._pulses:
            raise KeyError(f"Pulse '{name}' not found")
        return self._pulses[name]

    def get_marker(self, name: str) -> DigitalMarker:
        if name not in self._markers:
            raise KeyError(f"Marker '{name}' not found")
        return self._markers[name]

    @property
    def pulse_names(self) -> list[str]:
        return list(self._pulses)

    @property
    def marker_names(self) -> list[str]:
        return list(self._markers)

    def waveform_key(self, samples: np.ndarray) -> str:
        """Return a deduplicated waveform name based on content hash."""
        h = hashlib.sha256(samples.tobytes()).digest()
        if h in self._waveform_hashes:
            return self._waveform_hashes[h]
        name = f"wf_{len(self._waveform_hashes)}"
        self._waveform_hashes[h] = name
        return name


# --- Convenience factories ---


def make_square_pulse(
    name: str,
    amplitude: float,
    duration_ns: int,
    digital_marker: str | None = None,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> PulseDefinition:
    wf = square_waveform(amplitude, duration_ns, constraints)
    return PulseDefinition(
        name=name,
        operation="control",
        length_cycles=ns_to_cycles(duration_ns),
        waveform_i=wf,
        digital_marker=digital_marker,
    )


def make_gaussian_pulse(
    name: str,
    amplitude: float,
    duration_ns: int,
    sigma_ns: float,
    truncation_sigma: float = 4.0,
    normalize: bool = False,
    digital_marker: str | None = None,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> PulseDefinition:
    wf = gaussian_waveform(
        amplitude, duration_ns, sigma_ns, truncation_sigma, constraints
    )
    if normalize and np.max(np.abs(wf)) > 0:
        wf = wf * (amplitude / np.max(np.abs(wf)))
        wf = wf.astype(np.float32)
    return PulseDefinition(
        name=name,
        operation="control",
        length_cycles=ns_to_cycles(duration_ns),
        waveform_i=wf,
        digital_marker=digital_marker,
    )


def make_drag_pulse(
    name: str,
    amplitude: float,
    duration_ns: int,
    sigma_ns: float,
    alpha: float,
    delta_hz: float,
    truncation_sigma: float = 4.0,
    normalize: bool = False,
    digital_marker: str | None = None,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> PulseDefinition:
    wf_i, wf_q = drag_waveform(
        amplitude, duration_ns, sigma_ns, alpha, delta_hz, truncation_sigma, constraints
    )
    if normalize and np.max(np.abs(wf_i)) > 0:
        scale = amplitude / np.max(np.abs(wf_i))
        wf_i = (wf_i * scale).astype(np.float32)
        wf_q = (wf_q * scale).astype(np.float32)
    return PulseDefinition(
        name=name,
        operation="control",
        length_cycles=ns_to_cycles(duration_ns),
        waveform_i=wf_i,
        waveform_q=wf_q,
        digital_marker=digital_marker,
    )


def make_cosine_pulse(
    name: str,
    amplitude: float,
    frequency_hz: float,
    duration_ns: int,
    dc_offset: float = 0.0,
    phase_rad: float = 0.0,
    digital_marker: str | None = None,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> PulseDefinition:
    wf = cosine_waveform(
        amplitude, frequency_hz, duration_ns, dc_offset, phase_rad, constraints
    )
    return PulseDefinition(
        name=name,
        operation="control",
        length_cycles=ns_to_cycles(duration_ns),
        waveform_i=wf,
        digital_marker=digital_marker,
    )
