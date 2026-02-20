"""Waveform generation primitives for arbitrary waveform generators.

Pure functions that produce sample arrays. No hardware dependency.
All functions generate in float64 for precision, return float32 to match DAC resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stanza.exceptions import WaveformError
from stanza.timing import QUA_CYCLE_NS, ns_to_samples


@dataclass(frozen=True)
class WaveformConstraints:
    """Hardware constraints for waveform generation."""

    min_amplitude: float = -0.5
    max_amplitude: float = 0.5
    min_length_cycles: int = 4
    sample_rate_hz: float = 1e9


OPX_PLUS_CONSTRAINTS = WaveformConstraints()


def _samples_per_cycle(constraints: WaveformConstraints) -> int:
    return int(constraints.sample_rate_hz / 1e9 * QUA_CYCLE_NS)


def validate_waveform(
    samples: np.ndarray,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> np.ndarray:
    """Validate waveform samples against hardware constraints.

    Checks: NaN/Inf, cycle alignment, minimum length, amplitude bounds.
    Returns the array unchanged on success.
    """
    if not np.isfinite(samples).all():
        bad_indices = np.where(~np.isfinite(samples))[0]
        raise WaveformError(
            f"Waveform contains non-finite values at index {bad_indices[0]}: {samples[bad_indices[0]]}"
        )

    spc = _samples_per_cycle(constraints)
    if len(samples) % spc != 0:
        raise WaveformError(
            f"Waveform length {len(samples)} is not aligned to {spc}-sample cycle boundary"
        )

    min_samples = constraints.min_length_cycles * spc
    if len(samples) < min_samples:
        raise WaveformError(
            f"Waveform length {len(samples)} is below minimum {min_samples} samples "
            f"({constraints.min_length_cycles} cycles)"
        )

    out_of_bounds = np.where(
        (samples < constraints.min_amplitude) | (samples > constraints.max_amplitude)
    )[0]
    if len(out_of_bounds) > 0:
        idx = out_of_bounds[0]
        raise WaveformError(
            f"Waveform sample at index {idx} has value {samples[idx]}, "
            f"outside [{constraints.min_amplitude}, {constraints.max_amplitude}]"
        )

    return samples


def _check_amplitude(amplitude: float, constraints: WaveformConstraints) -> None:
    """Pre-allocation amplitude check."""
    if abs(amplitude) > constraints.max_amplitude:
        raise WaveformError(
            f"Amplitude {amplitude} exceeds maximum {constraints.max_amplitude}"
        )


def _check_duration(duration_ns: int, constraints: WaveformConstraints) -> None:
    """Pre-allocation duration check."""
    min_ns = constraints.min_length_cycles * QUA_CYCLE_NS
    if duration_ns < min_ns:
        raise WaveformError(
            f"Duration {duration_ns} ns is below minimum {min_ns} ns "
            f"({constraints.min_length_cycles} cycles)"
        )
    spc = _samples_per_cycle(constraints)
    n_samples = ns_to_samples(duration_ns, constraints.sample_rate_hz)
    if n_samples % spc != 0:
        raise WaveformError(
            f"Duration {duration_ns} ns produces {n_samples} samples, "
            f"not aligned to {spc}-sample cycle boundary"
        )


def _apply_taper(samples: np.ndarray, taper_samples: int = 2) -> np.ndarray:
    """Apply raised-cosine taper to first and last taper_samples."""
    if len(samples) < 2 * taper_samples:
        return samples
    taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_samples)))
    samples[:taper_samples] *= taper
    samples[-taper_samples:] *= taper[::-1]
    return samples


def square_waveform(
    amplitude: float,
    duration_ns: int,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> np.ndarray:
    """Generate a square (constant amplitude) waveform."""
    _check_amplitude(amplitude, constraints)
    _check_duration(duration_ns, constraints)
    n_samples = ns_to_samples(duration_ns, constraints.sample_rate_hz)
    samples = np.full(n_samples, amplitude, dtype=np.float64)
    validate_waveform(samples, constraints)
    return samples.astype(np.float32)


def gaussian_waveform(
    amplitude: float,
    duration_ns: int,
    sigma_ns: float,
    truncation_sigma: float = 4.0,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> np.ndarray:
    """Generate a Gaussian envelope waveform with raised-cosine taper at edges."""
    _check_amplitude(amplitude, constraints)
    _check_duration(duration_ns, constraints)
    if sigma_ns <= 0:
        raise WaveformError(f"sigma_ns must be positive, got {sigma_ns}")

    n_samples = ns_to_samples(duration_ns, constraints.sample_rate_hz)
    t = np.arange(n_samples, dtype=np.float64)
    center = (n_samples - 1) / 2.0
    sigma_samples = sigma_ns * constraints.sample_rate_hz / 1e9

    # Base Gaussian
    samples = amplitude * np.exp(-0.5 * ((t - center) / sigma_samples) ** 2)

    # Explicit truncation outside ±truncation_sigma * sigma
    if truncation_sigma is not None and truncation_sigma > 0:
        limit = truncation_sigma * sigma_samples
        mask = np.abs(t - center) <= limit
        samples = samples * mask.astype(np.float64)

    _apply_taper(samples)
    validate_waveform(samples, constraints)
    return samples.astype(np.float32)


def drag_waveform(
    amplitude: float,
    duration_ns: int,
    sigma_ns: float,
    alpha: float,
    delta_hz: float,
    truncation_sigma: float = 4.0,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate DRAG (Derivative Removal by Adiabatic Gate) I/Q waveforms.

    Returns (i_samples, q_samples) both as float32.
    """
    if delta_hz == 0:
        raise WaveformError("delta_hz cannot be zero (anharmonicity)")

    _check_amplitude(amplitude, constraints)
    _check_duration(duration_ns, constraints)
    if sigma_ns <= 0:
        raise WaveformError(f"sigma_ns must be positive, got {sigma_ns}")

    n_samples = ns_to_samples(duration_ns, constraints.sample_rate_hz)
    t = np.arange(n_samples, dtype=np.float64)
    center = (n_samples - 1) / 2.0
    sigma_samples = sigma_ns * constraints.sample_rate_hz / 1e9

    # I channel: Gaussian
    i_samples = amplitude * np.exp(-0.5 * ((t - center) / sigma_samples) ** 2)
    if truncation_sigma is not None and truncation_sigma > 0:
        limit = truncation_sigma * sigma_samples
        mask = np.abs(t - center) <= limit
        i_samples = i_samples * mask.astype(np.float64)
    _apply_taper(i_samples)

    # Q channel: DRAG correction = alpha * d(gaussian)/dt / delta_hz
    dt_ns = 1e9 / constraints.sample_rate_hz  # time step in ns
    dgaussian = (
        amplitude
        * (-(t - center) / (sigma_samples**2))
        * np.exp(-0.5 * ((t - center) / sigma_samples) ** 2)
    )
    q_samples = alpha * dgaussian / (delta_hz * dt_ns)
    if truncation_sigma is not None and truncation_sigma > 0:
        limit = truncation_sigma * sigma_samples
        mask = np.abs(t - center) <= limit
        q_samples = q_samples * mask.astype(np.float64)
    _apply_taper(q_samples)

    # Validate I channel
    validate_waveform(i_samples, constraints)

    # Validate Q channel with actionable error
    q_max = np.max(np.abs(q_samples))
    if q_max > constraints.max_amplitude:
        suggested_alpha = alpha * constraints.max_amplitude / q_max
        raise WaveformError(
            f"DRAG Q channel exceeds bounds (peak={q_max:.4f}). "
            f"Reduce alpha from {alpha} to ~{suggested_alpha:.4f}"
        )
    validate_waveform(q_samples, constraints)

    return i_samples.astype(np.float32), q_samples.astype(np.float32)


def cosine_waveform(
    amplitude: float,
    frequency_hz: float,
    duration_ns: int,
    dc_offset: float = 0.0,
    phase_rad: float = 0.0,
    constraints: WaveformConstraints = OPX_PLUS_CONSTRAINTS,
) -> np.ndarray:
    """Generate a cosine waveform with optional DC offset."""
    peak = abs(dc_offset) + abs(amplitude)
    if peak > constraints.max_amplitude:
        raise WaveformError(
            f"|dc_offset| + |amplitude| = {peak} exceeds maximum {constraints.max_amplitude}"
        )
    _check_duration(duration_ns, constraints)

    n_samples = ns_to_samples(duration_ns, constraints.sample_rate_hz)
    t = np.arange(n_samples, dtype=np.float64) / constraints.sample_rate_hz
    samples = dc_offset + amplitude * np.cos(2 * np.pi * frequency_hz * t + phase_rad)
    validate_waveform(samples, constraints)
    return samples.astype(np.float32)
