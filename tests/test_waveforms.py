"""Tests for stanza.waveforms module."""

from __future__ import annotations

import time

import numpy as np
import pytest

from stanza.exceptions import WaveformError
from stanza.timing import ns_to_samples
from stanza.waveforms import (
    WaveformConstraints,
    cosine_waveform,
    drag_waveform,
    gaussian_waveform,
    square_waveform,
    validate_waveform,
)

# ---------------------------------------------------------------------------
# validate_waveform
# ---------------------------------------------------------------------------


def test_validate_waveform():
    """Valid array passes through unchanged."""
    samples = np.full(16, 0.1, dtype=np.float64)
    result = validate_waveform(samples)
    assert result is samples


def test_validate_waveform_raises_on_nan():
    samples = np.full(16, 0.1, dtype=np.float64)
    samples[5] = np.nan
    with pytest.raises(WaveformError, match="non-finite"):
        validate_waveform(samples)


def test_validate_waveform_raises_on_inf():
    samples = np.full(16, 0.1, dtype=np.float64)
    samples[3] = np.inf
    with pytest.raises(WaveformError, match="non-finite"):
        validate_waveform(samples)


def test_validate_waveform_raises_on_out_of_bounds():
    samples = np.full(16, 0.1, dtype=np.float64)
    samples[7] = 0.6
    with pytest.raises(WaveformError, match="outside"):
        validate_waveform(samples)


def test_validate_waveform_raises_on_too_short():
    # min_length_cycles=4, samples_per_cycle=4 => min 16 samples
    samples = np.full(12, 0.1, dtype=np.float64)
    with pytest.raises(WaveformError, match="below minimum"):
        validate_waveform(samples)


def test_validate_waveform_raises_on_misaligned_length():
    # 17 samples is not aligned to 4-sample cycle boundary
    samples = np.full(17, 0.1, dtype=np.float64)
    with pytest.raises(WaveformError, match="not aligned"):
        validate_waveform(samples)


# ---------------------------------------------------------------------------
# square_waveform
# ---------------------------------------------------------------------------


def test_square_waveform():
    """Correct amplitude, length, and dtype."""
    duration_ns = 20
    amplitude = 0.3
    wf = square_waveform(amplitude, duration_ns)

    expected_len = ns_to_samples(duration_ns, sample_rate_hz=1e9)
    assert len(wf) == expected_len
    assert wf.dtype == np.float32
    np.testing.assert_allclose(wf, amplitude, atol=1e-7)


def test_square_waveform_raises_on_amplitude():
    with pytest.raises(WaveformError, match="exceeds maximum"):
        square_waveform(0.6, 20)


# ---------------------------------------------------------------------------
# gaussian_waveform
# ---------------------------------------------------------------------------


def test_gaussian_waveform():
    """Peak at center and symmetric."""
    duration_ns = 40
    amplitude = 0.4
    sigma_ns = 10.0
    wf = gaussian_waveform(amplitude, duration_ns, sigma_ns)

    expected_len = ns_to_samples(duration_ns, sample_rate_hz=1e9)
    assert len(wf) == expected_len

    # Peak should be near the center
    peak_idx = np.argmax(wf)
    center = len(wf) // 2
    assert abs(peak_idx - center) <= 1

    # Symmetric: compare first half with reversed second half
    n = len(wf)
    first_half = wf[: n // 2]
    second_half = wf[n // 2 :][::-1]
    # After taper, still approximately symmetric
    np.testing.assert_allclose(first_half, second_half, atol=1e-6)


def test_gaussian_waveform_tapered_edges():
    """First and last samples should be near zero due to taper."""
    wf = gaussian_waveform(0.4, 40, 10.0)
    assert abs(float(wf[0])) < 1e-4
    assert abs(float(wf[-1])) < 1e-4


def test_gaussian_waveform_raises_on_invalid_sigma():
    with pytest.raises(WaveformError, match="sigma_ns must be positive"):
        gaussian_waveform(0.3, 20, -5.0)


# ---------------------------------------------------------------------------
# drag_waveform
# ---------------------------------------------------------------------------


def test_drag_waveform():
    """I channel matches Gaussian shape; Q channel is antisymmetric."""
    duration_ns = 40
    amplitude = 0.3
    sigma_ns = 10.0
    alpha = 0.5
    delta_hz = -200e6

    i_wf, q_wf = drag_waveform(amplitude, duration_ns, sigma_ns, alpha, delta_hz)

    expected_len = ns_to_samples(duration_ns, sample_rate_hz=1e9)
    assert len(i_wf) == expected_len
    assert len(q_wf) == expected_len

    # I channel peak near center
    peak_idx = np.argmax(i_wf)
    center = len(i_wf) // 2
    assert abs(peak_idx - center) <= 1

    # Q channel antisymmetric: q(center - k) ~ -q(center + k)
    n = len(q_wf)
    mid = n // 2
    # Check a few interior points (avoid tapered edges)
    for offset in range(3, mid - 2):
        left = float(q_wf[mid - offset])
        right = float(q_wf[mid + offset])
        if abs(left) > 1e-6:
            assert np.sign(left) != np.sign(right) or abs(right) < 1e-5


def test_drag_waveform_raises_on_zero_delta():
    with pytest.raises(WaveformError, match="delta_hz cannot be zero"):
        drag_waveform(0.3, 40, 10.0, 0.5, 0.0)


def test_drag_waveform_raises_on_q_bounds():
    """Q channel overflow should suggest a reduced alpha."""
    with pytest.raises(WaveformError, match="Reduce alpha"):
        drag_waveform(0.4, 100, 20.0, 100.0, -1.0)


# ---------------------------------------------------------------------------
# cosine_waveform
# ---------------------------------------------------------------------------


def test_cosine_waveform():
    """Correct frequency verified via FFT."""
    frequency_hz = 50e6  # 50 MHz
    duration_ns = 200
    amplitude = 0.3

    wf = cosine_waveform(amplitude, frequency_hz, duration_ns)
    expected_len = ns_to_samples(duration_ns, sample_rate_hz=1e9)
    assert len(wf) == expected_len

    # FFT peak should be at the expected frequency
    fft_vals = np.abs(np.fft.rfft(wf))
    freqs = np.fft.rfftfreq(len(wf), d=1e-9)
    peak_freq = freqs[np.argmax(fft_vals[1:]) + 1]  # skip DC bin
    assert abs(peak_freq - frequency_hz) < 1e6


def test_cosine_waveform_dc_only():
    """Zero amplitude produces a pure DC waveform."""
    dc = 0.2
    wf = cosine_waveform(0.0, 1e6, 40, dc_offset=dc)
    np.testing.assert_allclose(wf, dc, atol=1e-7)


def test_cosine_waveform_raises_on_peak_bounds():
    with pytest.raises(WaveformError, match="exceeds maximum"):
        cosine_waveform(0.4, 1e6, 40, dc_offset=0.2)


# ---------------------------------------------------------------------------
# Custom constraints & dtype
# ---------------------------------------------------------------------------


def test_custom_constraints():
    """Custom constraints are respected."""
    custom = WaveformConstraints(
        min_amplitude=-1.0,
        max_amplitude=1.0,
        min_length_cycles=2,
        sample_rate_hz=1e9,
    )
    wf = square_waveform(0.9, 16, constraints=custom)
    np.testing.assert_allclose(wf, 0.9, atol=1e-7)


def test_output_dtype_is_float32():
    """All generator functions return float32."""
    assert square_waveform(0.1, 20).dtype == np.float32
    assert gaussian_waveform(0.1, 20, 5.0).dtype == np.float32
    i, q = drag_waveform(0.1, 40, 10.0, 0.1, -200e6)
    assert i.dtype == np.float32
    assert q.dtype == np.float32
    assert cosine_waveform(0.1, 1e6, 20).dtype == np.float32


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_large_waveform_no_quadratic_copy():
    """1M sample waveform generates in under 1 second."""
    # 1_000_000 samples = 1_000_000 ns at 1 GHz
    duration_ns = 1_000_000
    start = time.perf_counter()
    wf = square_waveform(0.1, duration_ns)
    elapsed = time.perf_counter() - start

    assert len(wf) == ns_to_samples(duration_ns, sample_rate_hz=1e9)
    assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1s"
