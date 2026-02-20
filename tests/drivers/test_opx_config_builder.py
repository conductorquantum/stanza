"""Tests for OPXConfigBuilder."""

from __future__ import annotations

import json

import numpy as np
import pytest

from stanza.drivers.opx_config_builder import OPXConfigBuilder
from stanza.pulses import (
    PulseDefinition,
    PulseRegistry,
    marker_high,
)
from stanza.timing import ns_to_cycles, ns_to_samples

# --- Helpers ---


def _make_waveform(duration_ns: int, amplitude: float = 0.1) -> np.ndarray:
    """Create a valid float32 waveform for testing."""
    n = ns_to_samples(duration_ns, sample_rate_hz=1e9)
    return np.full(n, amplitude, dtype=np.float32)


def _make_pulse(
    name: str,
    duration_ns: int = 20,
    amplitude: float = 0.1,
    iq: bool = False,
    digital_marker: str | None = None,
) -> PulseDefinition:
    wf_i = _make_waveform(duration_ns, amplitude)
    wf_q = _make_waveform(duration_ns, amplitude * 0.5) if iq else None
    return PulseDefinition(
        name=name,
        operation="control",
        length_cycles=ns_to_cycles(duration_ns),
        waveform_i=wf_i,
        waveform_q=wf_q,
        digital_marker=digital_marker,
    )


def _minimal_builder() -> OPXConfigBuilder:
    """Builder with one FEM and basic analog I/O."""
    b = OPXConfigBuilder(machine_type="OPX1000")
    b.add_fem(slot=2, fem_type="LF")
    b.add_analog_output(fem_slot=2, port=1)
    b.add_analog_input(fem_slot=2, port=1, offset=-0.007)
    return b


# --- Structure Tests ---


class TestBuildMinimalConfig:
    def test_build_minimal_config(self):
        b = _minimal_builder()
        b.add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=100)
        config = b.build()

        assert config["version"] == "1"
        assert "con1" in config["controllers"]
        assert "ch1" in config["elements"]
        assert "pulses" in config
        assert "waveforms" in config
        assert "integration_weights" in config

    def test_controller_has_fem(self):
        b = _minimal_builder()
        b.add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=100)
        config = b.build()
        controller = config["controllers"]["con1"]

        assert controller["type"] == "OPX1000"
        assert "2" in controller["fems"]
        assert controller["fems"]["2"]["type"] == "LF"

    def test_analog_ports_in_config(self):
        b = _minimal_builder()
        b.add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=100)
        config = b.build()
        fem = config["controllers"]["con1"]["fems"]["2"]

        assert "1" in fem["analog_outputs"]
        assert "1" in fem["analog_inputs"]
        assert fem["analog_inputs"]["1"]["offset"] == -0.007


class TestAddMeasurementElement:
    def test_auto_creates_readout_infrastructure(self):
        b = _minimal_builder()
        b.add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=100)
        config = b.build()

        assert "readout_pulse" in config["pulses"]
        assert "zero_wf" in config["waveforms"]
        assert "const_weights" in config["integration_weights"]

    def test_element_has_readout_operation(self):
        b = _minimal_builder()
        b.add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=100)
        config = b.build()
        elem = config["elements"]["ch1"]

        assert elem["operations"]["readout"] == "readout_pulse"
        assert elem["singleInput"]["port"] == ("con1", 2, 1)
        assert elem["outputs"]["out1"] == ("con1", 2, 1)

    def test_readout_pulse_length_matches_read_len(self):
        b = _minimal_builder()
        b.add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=200)
        config = b.build()

        assert config["pulses"]["readout_pulse"]["length"] == 200

    def test_const_weights_length_matches_read_len(self):
        b = _minimal_builder()
        b.add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=200)
        config = b.build()
        weights = config["integration_weights"]["const_weights"]

        assert weights["cosine"] == [[1.0, 200]]
        assert weights["sine"] == [[0.0, 200]]


class TestAddControlElement:
    def test_basic_control_element(self):
        b = _minimal_builder()
        b.add_control_element(
            name="gate1",
            port=("con1", 2, 1),
            operations={"square": "square_pulse"},
        )
        # Add the pulse so build() validates
        b._pulses["square_pulse"] = {
            "operation": "control",
            "length": 20,
            "waveforms": {"single": "some_wf"},
        }
        b._waveforms["some_wf"] = {"type": "constant", "sample": 0.1}
        config = b.build()
        elem = config["elements"]["gate1"]

        assert "singleInput" in elem
        assert elem["operations"]["square"] == "square_pulse"

    def test_control_element_with_digital_input(self):
        b = _minimal_builder()
        b.add_digital_input(fem_slot=2, port=3)
        b.add_control_element(
            name="gate1",
            port=("con1", 2, 1),
            digital_input_port=("con1", 2, 3),
        )
        config = b.build()
        elem = config["elements"]["gate1"]

        assert "digitalInputs" in elem
        assert "trigger" in elem["digitalInputs"]
        assert elem["digitalInputs"]["trigger"]["port"] == ("con1", 2, 3)

    def test_control_element_iq(self):
        b = OPXConfigBuilder(machine_type="OPX1000")
        b.add_analog_output(fem_slot=2, port=1)
        b.add_analog_output(fem_slot=2, port=2)
        b.add_control_element(
            name="qubit1",
            port=("con1", 2, 1),  # ignored when iq_ports provided
            iq_ports=(("con1", 2, 1), ("con1", 2, 2)),
        )
        config = b.build()
        elem = config["elements"]["qubit1"]

        assert "mixInputs" in elem
        assert elem["mixInputs"]["I"] == ("con1", 2, 1)
        assert elem["mixInputs"]["Q"] == ("con1", 2, 2)


# --- Pulse Registry Integration ---


class TestAddPulseRegistry:
    def test_single_channel_pulse(self):
        b = _minimal_builder()
        reg = PulseRegistry()
        pulse = _make_pulse("sq_20", duration_ns=20)
        reg.add_pulse(pulse)

        b.add_pulse_registry(reg)
        b.add_control_element(
            name="gate1",
            port=("con1", 2, 1),
            operations={"play": "sq_20"},
        )
        config = b.build()

        assert "sq_20" in config["pulses"]
        pulse_cfg = config["pulses"]["sq_20"]
        assert "single" in pulse_cfg["waveforms"]
        assert pulse_cfg["operation"] == "control"
        assert pulse_cfg["length"] == 20

    def test_iq_pulse(self):
        b = OPXConfigBuilder(machine_type="OPX1000")
        b.add_analog_output(fem_slot=2, port=1)
        b.add_analog_output(fem_slot=2, port=2)

        reg = PulseRegistry()
        pulse = _make_pulse("drag_20", duration_ns=20, iq=True)
        reg.add_pulse(pulse)

        b.add_pulse_registry(reg)
        b.add_control_element(
            name="qubit1",
            port=("con1", 2, 1),
            iq_ports=(("con1", 2, 1), ("con1", 2, 2)),
            operations={"play": "drag_20"},
        )
        config = b.build()
        pulse_cfg = config["pulses"]["drag_20"]

        assert "I" in pulse_cfg["waveforms"]
        assert "Q" in pulse_cfg["waveforms"]

    def test_dedup_identical_waveforms(self):
        b = _minimal_builder()
        reg = PulseRegistry()

        # Two pulses with identical waveform data
        wf = _make_waveform(20, 0.1)
        p1 = PulseDefinition(
            name="pulse_a",
            operation="control",
            length_cycles=ns_to_cycles(20),
            waveform_i=wf.copy(),
        )
        p2 = PulseDefinition(
            name="pulse_b",
            operation="control",
            length_cycles=ns_to_cycles(20),
            waveform_i=wf.copy(),
        )
        reg.add_pulse(p1)
        reg.add_pulse(p2)

        b.add_pulse_registry(reg)
        config = b.build()

        # Both pulses should reference the same waveform key
        wf_key_a = config["pulses"]["pulse_a"]["waveforms"]["single"]
        wf_key_b = config["pulses"]["pulse_b"]["waveforms"]["single"]
        assert wf_key_a == wf_key_b

    def test_digital_marker_in_config(self):
        b = _minimal_builder()
        reg = PulseRegistry()

        marker = marker_high("trig_on", duration_cycles=ns_to_cycles(20))
        reg.add_marker(marker)
        pulse = _make_pulse("sq_20", duration_ns=20, digital_marker="trig_on")
        reg.add_pulse(pulse)

        b.add_pulse_registry(reg)
        b.add_control_element(
            name="gate1",
            port=("con1", 2, 1),
            operations={"play": "sq_20"},
        )
        config = b.build()

        assert "digital_waveforms" in config
        assert "trig_on" in config["digital_waveforms"]
        assert config["pulses"]["sq_20"]["digital_marker"] == "trig_on"

    def test_dedup_bounds_config_size(self):
        """100 pulses sharing 3 distinct waveforms should produce only 3 waveform entries."""
        b = _minimal_builder()
        reg = PulseRegistry()

        wf_a = _make_waveform(20, 0.1)
        wf_b = _make_waveform(20, 0.2)
        wf_c = _make_waveform(20, 0.3)
        waveforms = [wf_a, wf_b, wf_c]

        for i in range(100):
            wf = waveforms[i % 3].copy()
            p = PulseDefinition(
                name=f"p_{i}",
                operation="control",
                length_cycles=ns_to_cycles(20),
                waveform_i=wf,
            )
            reg.add_pulse(p)

        b.add_pulse_registry(reg)
        config = b.build()

        # Should have exactly 3 arbitrary waveforms (not 100)
        arbitrary_wfs = [
            k for k, v in config["waveforms"].items() if v.get("type") == "arbitrary"
        ]
        assert len(arbitrary_wfs) == 3


# --- Validation Tests ---


class TestValidation:
    def test_raises_on_missing_pulse(self):
        b = _minimal_builder()
        b.add_control_element(
            name="gate1",
            port=("con1", 2, 1),
            operations={"play": "nonexistent_pulse"},
        )
        with pytest.raises(ValueError, match="undefined pulse 'nonexistent_pulse'"):
            b.build()

    def test_raises_on_missing_waveform(self):
        b = _minimal_builder()
        b._pulses["bad_pulse"] = {
            "operation": "control",
            "length": 20,
            "waveforms": {"single": "nonexistent_wf"},
        }
        b.add_control_element(
            name="gate1",
            port=("con1", 2, 1),
            operations={"play": "bad_pulse"},
        )
        with pytest.raises(ValueError, match="undefined waveform 'nonexistent_wf'"):
            b.build()

    def test_raises_on_missing_fem_port(self):
        b = OPXConfigBuilder(machine_type="OPX1000")
        b.add_fem(slot=2)
        # Don't add analog output port 5
        b.add_element(
            name="gate1",
            input_ports={"single": ("con1", 2, 5)},
        )
        with pytest.raises(ValueError, match="analog output port"):
            b.build()

    def test_raises_on_iq_pulse_with_single_input_element(self):
        b = _minimal_builder()
        reg = PulseRegistry()
        pulse = _make_pulse("iq_pulse", duration_ns=20, iq=True)
        reg.add_pulse(pulse)
        b.add_pulse_registry(reg)

        # Single-input element with IQ pulse
        b.add_control_element(
            name="gate1",
            port=("con1", 2, 1),
            operations={"play": "iq_pulse"},
        )
        with pytest.raises(ValueError, match="IQ but element.*singleInput"):
            b.build()

    def test_raises_on_single_pulse_with_mix_input_element(self):
        b = OPXConfigBuilder(machine_type="OPX1000")
        b.add_analog_output(fem_slot=2, port=1)
        b.add_analog_output(fem_slot=2, port=2)

        reg = PulseRegistry()
        pulse = _make_pulse("single_pulse", duration_ns=20, iq=False)
        reg.add_pulse(pulse)
        b.add_pulse_registry(reg)

        # mixInputs element with single-channel pulse
        b.add_control_element(
            name="qubit1",
            port=("con1", 2, 1),
            iq_ports=(("con1", 2, 1), ("con1", 2, 2)),
            operations={"play": "single_pulse"},
        )
        with pytest.raises(ValueError, match="single-channel but element.*mixInputs"):
            b.build()

    def test_raises_on_missing_digital_marker(self):
        b = _minimal_builder()
        b._pulses["marked_pulse"] = {
            "operation": "control",
            "length": 20,
            "waveforms": {"single": "some_wf"},
            "digital_marker": "missing_marker",
        }
        b._waveforms["some_wf"] = {"type": "constant", "sample": 0.1}
        b.add_control_element(
            name="gate1",
            port=("con1", 2, 1),
            operations={"play": "marked_pulse"},
        )
        with pytest.raises(ValueError, match="digital marker 'missing_marker'"):
            b.build()

    def test_integration_weights_length_mismatch(self):
        b = _minimal_builder()
        b.add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=100)
        # Override with wrong-length weights
        b._integration_weights["const_weights"] = {
            "cosine": [[1.0, 200]],  # 200 ns != 100 ns read_len
            "sine": [[0.0, 200]],
        }
        with pytest.raises(ValueError, match="does not match read_len"):
            b.build()


class TestConstIntegrationWeightsFactory:
    def test_returns_correct_structure(self):
        name, cosine, sine = OPXConfigBuilder.const_integration_weights(25)
        assert name == "const_weights"
        assert cosine == [(1.0, 25)]
        assert sine == [(0.0, 25)]


# --- OPX1 Compatibility ---


class TestOPX1:
    def test_flat_controller_structure(self):
        b = OPXConfigBuilder(machine_type="OPX", opx_version="opx1")
        b.add_analog_output(fem_slot=1, port=1)
        b.add_analog_input(fem_slot=1, port=1)
        b.add_measurement_element(name="ch1", port=("con1", 1, 1), read_len_ns=100)
        config = b.build()
        controller = config["controllers"]["con1"]

        assert "fems" not in controller
        assert "1" in controller["analog_outputs"]
        assert "1" in controller["analog_inputs"]

    def test_add_fem_is_noop(self):
        b = OPXConfigBuilder(machine_type="OPX", opx_version="opx1")
        b.add_fem(slot=2)  # Should be a no-op
        assert len(b._fems) == 0


# --- Golden Test ---


class TestGoldenConfig:
    def test_golden_measurement_config(self):
        """Full measurement config matches known-good dict."""
        b = OPXConfigBuilder(machine_type="OPX1000")
        b.add_fem(slot=2, fem_type="LF")
        b.add_analog_output(fem_slot=2, port=1, offset=0.0, sampling_rate=1e9)
        b.add_analog_input(fem_slot=2, port=1, offset=-0.007, sampling_rate=1e9)
        b.add_measurement_element(
            name="measure_ch1", port=("con1", 2, 1), read_len_ns=100
        )
        config = b.build()

        expected = {
            "version": "1",
            "controllers": {
                "con1": {
                    "type": "OPX1000",
                    "fems": {
                        "2": {
                            "type": "LF",
                            "analog_outputs": {
                                "1": {"offset": 0.0, "sampling_rate": 1e9},
                            },
                            "analog_inputs": {
                                "1": {"offset": -0.007, "sampling_rate": 1e9},
                            },
                        }
                    },
                }
            },
            "elements": {
                "measure_ch1": {
                    "singleInput": {"port": ("con1", 2, 1)},
                    "outputs": {"out1": ("con1", 2, 1)},
                    "operations": {"readout": "readout_pulse"},
                    "intermediate_frequency": 0,
                    "time_of_flight": 28,
                    "smearing": 0,
                }
            },
            "pulses": {
                "readout_pulse": {
                    "operation": "measurement",
                    "length": 100,
                    "waveforms": {"single": "zero_wf"},
                    "integration_weights": {"const": "const_weights"},
                }
            },
            "waveforms": {
                "zero_wf": {"type": "constant", "sample": 0.0},
            },
            "integration_weights": {
                "const_weights": {
                    "cosine": [[1.0, 100]],
                    "sine": [[0.0, 100]],
                }
            },
        }

        # Use JSON canonical form for clear diff on failure
        assert json.loads(
            json.dumps(config, sort_keys=True, default=str)
        ) == json.loads(json.dumps(expected, sort_keys=True, default=str))


# --- Chaining ---


class TestChaining:
    def test_fluent_api(self):
        config = (
            OPXConfigBuilder(machine_type="OPX1000")
            .add_fem(slot=2)
            .add_analog_output(fem_slot=2, port=1)
            .add_analog_output(fem_slot=2, port=2)
            .add_analog_input(fem_slot=2, port=1)
            .add_analog_input(fem_slot=2, port=2)
            .add_digital_output(fem_slot=2, port=1)
            .add_measurement_element(name="ch1", port=("con1", 2, 1), read_len_ns=100)
            .build()
        )
        assert "ch1" in config["elements"]
        assert "1" in config["controllers"]["con1"]["fems"]["2"]["digital_outputs"]
