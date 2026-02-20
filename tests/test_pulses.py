"""Tests for stanza.pulses module."""

from __future__ import annotations

import numpy as np
import pytest

from stanza.exceptions import PulseError
from stanza.pulses import (
    DigitalMarker,
    PulseDefinition,
    PulseRegistry,
    make_cosine_pulse,
    make_drag_pulse,
    make_gaussian_pulse,
    make_square_pulse,
    marker_high,
    marker_trigger,
)
from stanza.timing import ns_to_cycles, ns_to_samples

# ---------------------------------------------------------------------------
# PulseDefinition
# ---------------------------------------------------------------------------


class TestPulseDefinition:
    def test_pulse_definition(self):
        """Valid creation, length_ns derived correctly."""
        duration_ns = 100
        cycles = ns_to_cycles(duration_ns)
        n_samples = ns_to_samples(duration_ns)
        wf = np.zeros(n_samples, dtype=np.float32)

        pulse = PulseDefinition(
            name="test_pulse",
            operation="control",
            length_cycles=cycles,
            waveform_i=wf,
        )
        assert pulse.length_ns == duration_ns
        assert pulse.length_cycles == cycles
        assert pulse.name == "test_pulse"
        assert pulse.operation == "control"

    def test_pulse_definition_raises_on_length_mismatch(self):
        """waveform samples != expected for cycles."""
        cycles = ns_to_cycles(100)
        wrong_wf = np.zeros(50, dtype=np.float32)  # 100 ns -> 100 samples, not 50

        with pytest.raises(PulseError, match="waveform_i has 50 samples"):
            PulseDefinition(
                name="bad_length",
                operation="control",
                length_cycles=cycles,
                waveform_i=wrong_wf,
            )

    def test_pulse_definition_raises_on_iq_mismatch(self):
        """Q length != I length."""
        duration_ns = 100
        cycles = ns_to_cycles(duration_ns)
        n_samples = ns_to_samples(duration_ns)
        wf_i = np.zeros(n_samples, dtype=np.float32)
        wf_q = np.zeros(n_samples // 2, dtype=np.float32)

        with pytest.raises(PulseError, match="waveform_q"):
            PulseDefinition(
                name="iq_mismatch",
                operation="control",
                length_cycles=cycles,
                waveform_i=wf_i,
                waveform_q=wf_q,
            )

    def test_pulse_definition_is_iq(self):
        """is_iq property True/False."""
        duration_ns = 100
        cycles = ns_to_cycles(duration_ns)
        n_samples = ns_to_samples(duration_ns)
        wf = np.zeros(n_samples, dtype=np.float32)

        pulse_no_q = PulseDefinition(
            name="no_q",
            operation="control",
            length_cycles=cycles,
            waveform_i=wf,
        )
        assert pulse_no_q.is_iq is False

        pulse_with_q = PulseDefinition(
            name="with_q",
            operation="control",
            length_cycles=cycles,
            waveform_i=wf,
            waveform_q=wf.copy(),
        )
        assert pulse_with_q.is_iq is True

    def test_pulse_definition_raises_on_invalid_name(self):
        """'123bad' -> PulseError."""
        duration_ns = 100
        cycles = ns_to_cycles(duration_ns)
        n_samples = ns_to_samples(duration_ns)
        wf = np.zeros(n_samples, dtype=np.float32)

        with pytest.raises(PulseError, match="Invalid pulse name"):
            PulseDefinition(
                name="123bad",
                operation="control",
                length_cycles=cycles,
                waveform_i=wf,
            )

    def test_pulse_definition_raises_on_name_with_spaces(self):
        """'my pulse' -> PulseError."""
        duration_ns = 100
        cycles = ns_to_cycles(duration_ns)
        n_samples = ns_to_samples(duration_ns)
        wf = np.zeros(n_samples, dtype=np.float32)

        with pytest.raises(PulseError, match="Invalid pulse name"):
            PulseDefinition(
                name="my pulse",
                operation="control",
                length_cycles=cycles,
                waveform_i=wf,
            )


# ---------------------------------------------------------------------------
# DigitalMarker
# ---------------------------------------------------------------------------


class TestDigitalMarker:
    def test_digital_marker(self):
        """Valid creation."""
        marker = DigitalMarker(name="my_marker", samples=[(1, 10), (0, 15)])
        assert marker.name == "my_marker"
        assert len(marker.samples) == 2

    def test_digital_marker_total_cycles(self):
        """Sum of durations."""
        marker = DigitalMarker(name="marker_a", samples=[(1, 10), (0, 15)])
        assert marker.total_cycles == 25

    def test_digital_marker_raises_on_invalid_value(self):
        """value=2 should raise PulseError."""
        with pytest.raises(PulseError, match="must be 0 or 1"):
            DigitalMarker(name="bad_val", samples=[(2, 10)])

    def test_digital_marker_raises_on_negative_duration(self):
        """duration=-1 should raise PulseError."""
        with pytest.raises(PulseError, match="must be positive"):
            DigitalMarker(name="bad_dur", samples=[(1, -1)])

    def test_marker_high(self):
        """Factory function."""
        marker = marker_high("high_marker", duration_cycles=25)
        assert marker.name == "high_marker"
        assert marker.samples == [(1, 25)]
        assert marker.total_cycles == 25

    def test_marker_trigger(self):
        """Factory function."""
        marker = marker_trigger("trig", width_cycles=2, total_cycles=25)
        assert marker.name == "trig"
        assert marker.samples == [(1, 2), (0, 23)]
        assert marker.total_cycles == 25


# ---------------------------------------------------------------------------
# PulseRegistry
# ---------------------------------------------------------------------------


class TestPulseRegistry:
    def _make_pulse(
        self,
        name: str = "p1",
        duration_ns: int = 100,
        digital_marker: str | None = None,
    ):
        cycles = ns_to_cycles(duration_ns)
        n_samples = ns_to_samples(duration_ns)
        wf = np.zeros(n_samples, dtype=np.float32)
        return PulseDefinition(
            name=name,
            operation="control",
            length_cycles=cycles,
            waveform_i=wf,
            digital_marker=digital_marker,
        )

    def test_registry_add_and_get(self):
        """Round-trip add then get."""
        reg = PulseRegistry()
        pulse = self._make_pulse("my_pulse")
        reg.add_pulse(pulse)
        assert reg.get_pulse("my_pulse") is pulse

    def test_registry_raises_on_duplicate_pulse(self):
        """Same name twice."""
        reg = PulseRegistry()
        reg.add_pulse(self._make_pulse("dup"))
        with pytest.raises(PulseError, match="Duplicate pulse name"):
            reg.add_pulse(self._make_pulse("dup"))

    def test_registry_raises_on_duplicate_marker(self):
        """Same marker name twice."""
        reg = PulseRegistry()
        m = marker_high("m1", 25)
        reg.add_marker(m)
        with pytest.raises(PulseError, match="Duplicate marker name"):
            reg.add_marker(m)

    def test_registry_raises_on_missing_pulse(self):
        """KeyError on get."""
        reg = PulseRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get_pulse("nonexistent")

    def test_registry_pulse_names(self):
        """Listing names."""
        reg = PulseRegistry()
        reg.add_pulse(self._make_pulse("alpha"))
        reg.add_pulse(self._make_pulse("beta"))
        assert reg.pulse_names == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Waveform dedup
# ---------------------------------------------------------------------------


class TestWaveformDedup:
    def test_waveform_dedup(self):
        """Same samples -> same key."""
        reg = PulseRegistry()
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        key1 = reg.waveform_key(samples)
        key2 = reg.waveform_key(samples.copy())
        assert key1 == key2

    def test_waveform_dedup_different(self):
        """Different samples -> different key."""
        reg = PulseRegistry()
        s1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        s2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        key1 = reg.waveform_key(s1)
        key2 = reg.waveform_key(s2)
        assert key1 != key2


# ---------------------------------------------------------------------------
# Pulse + Marker integration
# ---------------------------------------------------------------------------


class TestPulseMarkerIntegration:
    def test_add_pulse_with_marker(self):
        """Marker registered first, lengths match, succeeds."""
        reg = PulseRegistry()
        duration_ns = 100
        cycles = ns_to_cycles(duration_ns)
        marker = marker_high("mk", duration_cycles=cycles)
        reg.add_marker(marker)

        n_samples = ns_to_samples(duration_ns)
        wf = np.zeros(n_samples, dtype=np.float32)
        pulse = PulseDefinition(
            name="with_mk",
            operation="control",
            length_cycles=cycles,
            waveform_i=wf,
            digital_marker="mk",
        )
        reg.add_pulse(pulse)
        assert reg.get_pulse("with_mk") is pulse

    def test_add_pulse_raises_on_unregistered_marker(self):
        """Marker not in registry -> PulseError."""
        reg = PulseRegistry()
        duration_ns = 100
        cycles = ns_to_cycles(duration_ns)
        n_samples = ns_to_samples(duration_ns)
        wf = np.zeros(n_samples, dtype=np.float32)
        pulse = PulseDefinition(
            name="orphan",
            operation="control",
            length_cycles=cycles,
            waveform_i=wf,
            digital_marker="ghost_marker",
        )
        with pytest.raises(PulseError, match="has not been registered"):
            reg.add_pulse(pulse)

    def test_add_pulse_raises_on_marker_length_mismatch(self):
        """Marker cycles != pulse cycles -> PulseError."""
        reg = PulseRegistry()
        duration_ns = 100
        pulse_cycles = ns_to_cycles(duration_ns)
        marker_cycles = pulse_cycles + 5  # different

        marker = marker_high("mk_wrong", duration_cycles=marker_cycles)
        reg.add_marker(marker)

        n_samples = ns_to_samples(duration_ns)
        wf = np.zeros(n_samples, dtype=np.float32)
        pulse = PulseDefinition(
            name="bad_mk_len",
            operation="control",
            length_cycles=pulse_cycles,
            waveform_i=wf,
            digital_marker="mk_wrong",
        )
        with pytest.raises(PulseError, match="must match"):
            reg.add_pulse(pulse)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


class TestConvenienceFactories:
    def test_make_square_pulse(self):
        """Returns correct PulseDefinition."""
        pulse = make_square_pulse("sq", amplitude=0.3, duration_ns=100)
        assert pulse.name == "sq"
        assert pulse.operation == "control"
        assert pulse.length_ns == 100
        assert len(pulse.waveform_i) == 100
        np.testing.assert_allclose(pulse.waveform_i, 0.3, atol=1e-6)

    def test_make_gaussian_pulse(self):
        """Waveform shape correct: peak near center, decays at edges."""
        pulse = make_gaussian_pulse(
            "gauss", amplitude=0.4, duration_ns=200, sigma_ns=30.0
        )
        assert pulse.name == "gauss"
        assert pulse.length_ns == 200
        wf = pulse.waveform_i
        center = len(wf) // 2
        # Peak should be near the center
        assert np.argmax(np.abs(wf)) == pytest.approx(center, abs=5)
        # Edges should be smaller than center
        assert abs(wf[0]) < abs(wf[center])

    def test_make_gaussian_pulse_normalize(self):
        """normalize=True -> peak == amplitude."""
        amp = 0.4
        pulse = make_gaussian_pulse(
            "gauss_norm",
            amplitude=amp,
            duration_ns=200,
            sigma_ns=30.0,
            normalize=True,
        )
        np.testing.assert_allclose(np.max(np.abs(pulse.waveform_i)), amp, atol=1e-5)

    def test_make_drag_pulse(self):
        """Returns IQ pulse (is_iq=True)."""
        pulse = make_drag_pulse(
            "drag",
            amplitude=0.3,
            duration_ns=200,
            sigma_ns=30.0,
            alpha=0.01,
            delta_hz=200e6,
        )
        assert pulse.is_iq is True
        assert pulse.waveform_q is not None
        assert len(pulse.waveform_i) == len(pulse.waveform_q)
        assert pulse.length_ns == 200

    def test_make_cosine_pulse(self):
        """Correct waveform: starts at amplitude + dc_offset when phase=0."""
        pulse = make_cosine_pulse(
            "cos",
            amplitude=0.2,
            frequency_hz=10e6,
            duration_ns=200,
        )
        assert pulse.name == "cos"
        assert pulse.length_ns == 200
        # At t=0, cos(0)=1, so first sample should be ~amplitude
        np.testing.assert_allclose(pulse.waveform_i[0], 0.2, atol=1e-5)
