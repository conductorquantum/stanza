"""Tests for device health check routines."""

from unittest.mock import patch

import numpy as np
import pytest

from stanza.analysis.fitting import pinchoff_curve
from stanza.exceptions import RoutineError
from stanza.models import GateType
from stanza.registry import ResourceRegistry, ResultsRegistry
from stanza.routines import RoutineContext
from stanza.routines.builtins.health_check import (
    _calculate_leakage_matrix,
    _check_leakage_threshold,
    analyze_single_gate_heuristic,
    finger_gate_characterization,
    global_accumulation,
    leakage_test,
    noise_floor_measurement,
    reservoir_characterization,
)


class MockDevice:
    def __init__(self):
        self.name = "device"
        self.control_gates = ["G1", "G2", "G3"]
        self.voltages = dict.fromkeys(self.control_gates, 0.0)
        self.currents = dict.fromkeys(self.control_gates, 1e-11)
        self.channel_configs = {
            gate: type("Config", (), {"voltage_range": (-10.0, 10.0)})()
            for gate in self.control_gates
        }
        self.gate_types = {
            "R1": GateType.RESERVOIR,
            "R2": GateType.RESERVOIR,
            "P1": GateType.PLUNGER,
            "B1": GateType.BARRIER,
        }

    def measure(self, electrodes):
        if isinstance(electrodes, str):
            return self.currents.get(electrodes, 1e-11)
        return np.array([self.currents.get(e, 1e-11) for e in electrodes])

    def check(self, electrodes):
        if isinstance(electrodes, str):
            return self.voltages.get(electrodes, 0.0)
        return [self.voltages.get(e, 0.0) for e in electrodes]

    def jump(self, voltage_dict, wait_for_settling=False):
        self.voltages.update(voltage_dict)

    def sweep_all(self, voltages, measure_electrode, session=None):
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        return voltages, currents

    def sweep_1d(self, gate, voltages, measure_electrode, session=None):
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        return voltages, currents

    def get_gates_by_type(self, gate_type):
        return [name for name, gtype in self.gate_types.items() if gtype == gate_type]


@pytest.fixture
def mock_device():
    return MockDevice()


@pytest.fixture
def routine_context(mock_device):
    resources = ResourceRegistry(mock_device)
    results = ResultsRegistry()
    return RoutineContext(resources, results)


@pytest.fixture(autouse=True)
def mock_sleep():
    with patch("time.sleep"):
        yield


class MockLoggerSession:
    def __init__(self):
        self.measurements = []
        self.analyses = []

    def log_measurement(self, name, data):
        self.measurements.append((name, data))

    def log_analysis(self, name, data):
        self.analyses.append((name, data))


class TestNoiseFloorMeasurement:
    def test_basic_measurement(self, routine_context):
        result = noise_floor_measurement(
            routine_context, measure_electrode="G1", num_points=10
        )

        assert "current_mean" in result
        assert "current_std" in result
        assert isinstance(result["current_mean"], float)
        assert isinstance(result["current_std"], float)

    def test_measurement_statistics(self, routine_context):
        routine_context.resources.device.currents["G1"] = 1e-10

        result = noise_floor_measurement(
            routine_context, measure_electrode="G1", num_points=100
        )

        assert abs(result["current_mean"] - 1e-10) < 1e-12
        assert result["current_std"] >= 0

    def test_different_num_points(self, routine_context):
        result_10 = noise_floor_measurement(
            routine_context, measure_electrode="G1", num_points=10
        )
        result_100 = noise_floor_measurement(
            routine_context, measure_electrode="G1", num_points=100
        )

        assert "current_mean" in result_10
        assert "current_mean" in result_100

    def test_with_logger_session(self, routine_context):
        session = MockLoggerSession()
        result = noise_floor_measurement(
            routine_context, measure_electrode="G1", num_points=10, session=session
        )

        assert len(session.analyses) == 1
        assert session.analyses[0][0] == "noise_floor_measurement"
        assert "currents" in session.analyses[0][1]
        assert result["current_mean"] == session.analyses[0][1]["current_mean"]


class TestLeakageTest:
    def test_basic_leakage_test(self, routine_context):
        routine_context.results.store("current_std", 1e-11)

        result = leakage_test(
            routine_context,
            leakage_threshold_resistance=1e6,
            leakage_threshold_count=0,
            num_points=3,
        )

        assert "max_safe_voltage_bound" in result
        assert "min_safe_voltage_bound" in result
        assert isinstance(result["max_safe_voltage_bound"], (int, float))
        assert isinstance(result["min_safe_voltage_bound"], (int, float))

    def test_leakage_uses_current_std_from_results(self, routine_context):
        routine_context.results.store("current_std", 5e-11)

        result = leakage_test(
            routine_context,
            leakage_threshold_resistance=1e6,
            leakage_threshold_count=0,
            num_points=2,
        )

        assert result is not None

    def test_leakage_default_current_threshold(self, routine_context):
        result = leakage_test(
            routine_context,
            leakage_threshold_resistance=1e6,
            leakage_threshold_count=0,
            num_points=2,
        )

        assert result is not None

    def test_leakage_restores_initial_voltages(self, routine_context):
        device = routine_context.resources.device
        initial_voltages = dict.fromkeys(device.control_gates, 0.5)
        device.jump(initial_voltages)
        routine_context.results.store("current_std", 1e-11)

        leakage_test(
            routine_context,
            leakage_threshold_resistance=1e6,
            leakage_threshold_count=0,
            num_points=2,
        )

        for gate in device.control_gates:
            assert device.voltages[gate] == initial_voltages[gate]

    def test_leakage_with_session_logging(self, routine_context):
        session = MockLoggerSession()
        routine_context.results.store("current_std", 1e-11)

        result = leakage_test(
            routine_context,
            leakage_threshold_resistance=1e6,
            leakage_threshold_count=0,
            num_points=2,
            session=session,
        )

        assert result is not None
        assert any("leakage_test_success" in a[0] for a in session.analyses)


class TestGlobalAccumulation:
    def test_invalid_step_size(self, routine_context):
        routine_context.results.store("max_safe_voltage_bound", 10.0)
        routine_context.results.store("min_safe_voltage_bound", -10.0)

        with pytest.raises(RoutineError, match="Step size must be greater than 0"):
            global_accumulation(routine_context, measure_electrode="G1", step_size=0)

    def test_calls_sweep_all(self, routine_context):
        class TrackedDevice(MockDevice):
            def __init__(self):
                super().__init__()
                self.sweep_all_called = False
                self.sweep_params = {}

            def sweep_all(self, voltages, measure_electrode, session=None):
                self.sweep_all_called = True
                self.sweep_params = {
                    "voltages": voltages,
                    "measure_electrode": measure_electrode,
                    "session": session,
                }
                return voltages, np.ones_like(voltages) * 1e-10

        tracked_device = TrackedDevice()
        routine_context.resources._resources["device"] = tracked_device
        routine_context.results.store("max_safe_voltage_bound", 10.0)
        routine_context.results.store("min_safe_voltage_bound", -10.0)

        with pytest.raises((RoutineError, ValueError)):
            global_accumulation(routine_context, measure_electrode="G1", step_size=2.0)

        assert tracked_device.sweep_all_called
        assert tracked_device.sweep_params["measure_electrode"] == "G1"
        assert len(tracked_device.sweep_params["voltages"]) >= 2


class TestReservoirCharacterization:
    @pytest.fixture
    def reservoir_context(self, routine_context):
        routine_context.results.store("max_safe_voltage_bound", 10.0)
        routine_context.results.store("min_safe_voltage_bound", -10.0)
        routine_context.results.store("global_turn_on_voltage", 0.5)
        return routine_context

    def test_invalid_step_size(self, reservoir_context):
        with pytest.raises(RoutineError, match="Step size must be greater than 0"):
            reservoir_characterization(
                reservoir_context, measure_electrode="G1", step_size=-0.1
            )

    def test_sweeps_each_reservoir(self, reservoir_context):
        class TrackedDevice(MockDevice):
            def __init__(self):
                super().__init__()
                self.swept_gates = []

            def sweep_1d(self, gate, voltages, measure_electrode, session=None):
                self.swept_gates.append(gate)
                return voltages, np.ones_like(voltages) * 1e-10

        tracked_device = TrackedDevice()
        reservoir_context.resources._resources["device"] = tracked_device

        with pytest.raises((RoutineError, ValueError)):
            reservoir_characterization(
                reservoir_context, measure_electrode="G1", step_size=2.0
            )

        assert len(tracked_device.swept_gates) >= 1
        assert any(g in ["R1", "R2"] for g in tracked_device.swept_gates)


class TestFingerGateCharacterization:
    @pytest.fixture
    def finger_context(self, routine_context):
        routine_context.results.store("max_safe_voltage_bound", 10.0)
        routine_context.results.store("min_safe_voltage_bound", -10.0)
        routine_context.results.store("global_turn_on_voltage", 0.5)
        return routine_context

    def test_invalid_step_size(self, finger_context):
        with pytest.raises(RoutineError, match="Step size must be greater than 0"):
            finger_gate_characterization(
                finger_context, measure_electrode="G1", step_size=0
            )

    def test_sweeps_plunger_and_barrier_gates(self, finger_context):
        class TrackedDevice(MockDevice):
            def __init__(self):
                super().__init__()
                self.swept_gates = []

            def sweep_1d(self, gate, voltages, measure_electrode, session=None):
                self.swept_gates.append(gate)
                return voltages, np.ones_like(voltages) * 1e-10

        tracked_device = TrackedDevice()
        finger_context.resources._resources["device"] = tracked_device

        with pytest.raises((RoutineError, ValueError)):
            finger_gate_characterization(
                finger_context, measure_electrode="G1", step_size=2.0
            )

        assert len(tracked_device.swept_gates) >= 1
        assert any(g in ["P1", "B1"] for g in tracked_device.swept_gates)


class TestAnalyzeSingleGateHeuristic:
    def test_poor_fit_raises_error(self):
        voltages = np.linspace(-1, 1, 10)
        currents = np.random.random(10) * 1e-15

        with pytest.raises(ValueError, match="poor fit"):
            analyze_single_gate_heuristic(voltages, currents)

    def test_linear_data_fails(self):
        voltages = np.linspace(-1, 1, 20)
        currents = voltages * 1e-11

        with pytest.raises(ValueError):
            analyze_single_gate_heuristic(voltages, currents)

    def test_returns_all_expected_keys(self):
        voltages = np.linspace(-2, 2, 100)
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        np.random.seed(42)
        noise = np.random.normal(0, 0.01 * np.max(currents), size=currents.shape)
        currents_noisy = currents + noise

        result = analyze_single_gate_heuristic(voltages, currents_noisy)
        assert "vp" in result
        assert "vt" in result
        assert "vc" in result
        assert "popt" in result
        assert "pcov" in result
        assert isinstance(result["vp"], float)
        assert isinstance(result["vt"], float)
        assert isinstance(result["vc"], float)

    def test_negative_amplitude_curve(self):
        voltages = np.linspace(-2, 2, 200)
        currents = pinchoff_curve(voltages, -0.5, 2.0, -1.0)
        np.random.seed(42)
        noise = np.random.normal(
            0, 0.01 * np.abs(np.max(currents) - np.min(currents)), size=currents.shape
        )
        currents_noisy = currents + noise

        result = analyze_single_gate_heuristic(voltages, currents_noisy)
        assert "vp" in result
        assert "vt" in result
        assert "vc" in result
        assert result["vc"] < result["vt"] < result["vp"]


class TestLeakageHelperFunctions:
    def test_calculate_leakage_matrix(self):
        delta_V = 0.1
        current_diff = np.array([1e-9, 2e-9, 0.0, 5e-9])

        leakage_matrix = _calculate_leakage_matrix(delta_V, current_diff)

        assert leakage_matrix.shape == (4,)
        assert leakage_matrix[0] == abs(delta_V / 1e-9)
        assert leakage_matrix[2] == np.inf
        assert np.all(np.isfinite(leakage_matrix[:2]))

    def test_calculate_leakage_matrix_handles_negatives(self):
        delta_V = -0.1
        current_diff = np.array([-1e-9, 1e-9])

        leakage_matrix = _calculate_leakage_matrix(delta_V, current_diff)

        assert np.all(leakage_matrix > 0)
        assert np.all(np.isfinite(leakage_matrix))

    def test_check_leakage_threshold_no_leakage(self):
        leakage_matrix = np.array([[np.inf, 1e8], [1e8, np.inf]])
        control_gates = ["G1", "G2"]

        leaked = _check_leakage_threshold(
            leakage_matrix,
            leakage_threshold_resistance=1e6,
            leakage_threshold_count=0,
            control_gates=control_gates,
            delta_V=0.1,
            session=None,
        )

        assert not leaked

    def test_check_leakage_threshold_with_leakage(self):
        leakage_matrix = np.array([[np.inf, 1e5], [1e5, np.inf]])
        control_gates = ["G1", "G2"]
        session = MockLoggerSession()

        leaked = _check_leakage_threshold(
            leakage_matrix,
            leakage_threshold_resistance=1e6,
            leakage_threshold_count=0,
            control_gates=control_gates,
            delta_V=0.1,
            session=session,
        )

        assert leaked
        assert len(session.analyses) == 1
        assert session.analyses[0][0] == "leaky_gate_pairs"
