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
        self.control_contacts = []
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
        return voltages, pinchoff_curve(voltages, 1.0, 1.0, 1.0)

    def sweep_1d(self, gate, voltages, measure_electrode, session=None):
        return voltages, pinchoff_curve(voltages, 1.0, 1.0, 1.0)

    def get_gates_by_type(self, gate_type):
        return [name for name, gtype in self.gate_types.items() if gtype == gate_type]

    def zero(self, pad_type=None):
        self.voltages = dict.fromkeys(self.control_gates, 0.0)


class MockLoggerSession:
    def __init__(self):
        self.measurements = []
        self.analyses = []

    def log_measurement(self, name, data):
        self.measurements.append((name, data))

    def log_analysis(self, name, data):
        self.analyses.append((name, data))


class HighQualityMockDevice(MockDevice):
    """Mock device that returns high-quality pinchoff curves."""

    def sweep_all(self, voltages, measure_electrode, session=None):
        voltages_array = np.array(voltages)
        return voltages_array, pinchoff_curve(voltages_array, 1.5, 3.0, -1.5)

    def sweep_1d(self, gate, voltages, measure_electrode, session=None):
        voltages_array = np.array(voltages)
        return voltages_array, pinchoff_curve(voltages_array, 1.5, 3.0, -1.5)


@pytest.fixture
def mock_device():
    return MockDevice()


@pytest.fixture
def routine_context(mock_device):
    return RoutineContext(ResourceRegistry(mock_device), ResultsRegistry())


@pytest.fixture
def characterization_context(routine_context):
    """Context with leakage and global accumulation results."""
    routine_context.results.store(
        "leakage_test", {"max_safe_voltage_bound": 1.0, "min_safe_voltage_bound": -10.0}
    )
    routine_context.results.store(
        "global_accumulation", {"global_turn_on_voltage": 0.5}
    )
    return routine_context


@pytest.fixture(autouse=True)
def mock_sleep():
    with patch("time.sleep"):
        yield


class TestNoiseFloorMeasurement:
    def test_basic_measurement(self, routine_context):
        result = noise_floor_measurement(
            routine_context, measure_electrode="G1", num_points=10
        )
        assert "current_mean" in result and "current_std" in result
        assert isinstance(result["current_mean"], float)

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
        assert "current_mean" in result_10 and "current_mean" in result_100

    def test_with_logger_session(self, routine_context):
        session = MockLoggerSession()
        result = noise_floor_measurement(
            routine_context, measure_electrode="G1", num_points=10, session=session
        )
        assert len(session.analyses) == 1
        assert session.analyses[0][0] == "noise_floor_measurement"
        assert result["current_mean"] == session.analyses[0][1]["current_mean"]


class TestLeakageTest:
    @pytest.fixture
    def leakage_params(self):
        return {
            "leakage_threshold_resistance": 50e6,
            "leakage_threshold_count": 0,
            "num_points": 2,
        }

    def test_basic_leakage_test(self, routine_context, leakage_params):
        routine_context.results.store("current_std", 1e-11)
        result = leakage_test(routine_context, measure_electrode="G1", **leakage_params)
        assert "max_safe_voltage_bound" in result and "min_safe_voltage_bound" in result

    def test_leakage_uses_current_std_from_results(
        self, routine_context, leakage_params
    ):
        routine_context.results.store("current_std", 5e-11)
        assert (
            leakage_test(routine_context, measure_electrode="G1", **leakage_params)
            is not None
        )

    def test_leakage_default_current_threshold(self, routine_context, leakage_params):
        assert (
            leakage_test(routine_context, measure_electrode="G1", **leakage_params)
            is not None
        )

    def test_leakage_restores_initial_voltages(self, routine_context, leakage_params):
        device = routine_context.resources.device
        initial_voltages = dict.fromkeys(device.control_gates, 0.5)
        device.jump(initial_voltages)
        routine_context.results.store("current_std", 1e-11)
        leakage_test(routine_context, measure_electrode="G1", **leakage_params)
        for gate in device.control_gates:
            assert device.voltages[gate] == initial_voltages[gate]

    def test_leakage_with_session_logging(self, routine_context, leakage_params):
        session = MockLoggerSession()
        routine_context.results.store("current_std", 1e-11)
        result = leakage_test(
            routine_context, measure_electrode="G1", session=session, **leakage_params
        )
        assert result is not None
        assert any("leakage_test_success" in a[0] for a in session.analyses)

    def test_exception_handling(self, routine_context, leakage_params):
        routine_context.results.store("current_std", 1e-11)
        call_count = [0]

        def failing_measure(electrodes):
            call_count[0] += 1
            if call_count[0] > 2:
                raise RuntimeError("Measurement failed")
            return (
                np.array([1e-11] * len(electrodes))
                if isinstance(electrodes, list)
                else 1e-11
            )

        routine_context.resources.device.measure = failing_measure
        with pytest.raises(RuntimeError, match="Measurement failed"):
            leakage_test(routine_context, measure_electrode="G1", **leakage_params)

    def test_continues_when_voltage_too_close(self, routine_context, leakage_params):
        routine_context.results.store("current_std", 1e-11)

        class NoVoltageChangeDevice(MockDevice):
            def measure(self, electrodes):
                return (
                    np.array([1e-8] * len(electrodes))
                    if isinstance(electrodes, list)
                    else 1e-8
                )

            def jump(self, voltage_dict, wait_for_settling=False):
                pass

        routine_context.resources._resources["device"] = NoVoltageChangeDevice()
        assert (
            leakage_test(routine_context, measure_electrode="G1", **leakage_params)
            is not None
        )


class TestGlobalAccumulation:
    def test_invalid_step_size(self, routine_context):
        routine_context.results.store(
            "leakage_test",
            {"max_safe_voltage_bound": 10.0, "min_safe_voltage_bound": -10.0},
        )
        with pytest.raises(RoutineError, match="Step size must be greater than 0"):
            global_accumulation(
                routine_context,
                measure_electrode="G1",
                step_size=0,
                bias_gate="G1",
                bias_voltage=0.0,
                charge_carrier_type="electron",
            )

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
                }
                return voltages, np.ones_like(voltages) * 1e-10

        routine_context.resources._resources["device"] = TrackedDevice()
        routine_context.results.store(
            "leakage_test",
            {"max_safe_voltage_bound": 10.0, "min_safe_voltage_bound": -10.0},
        )
        with pytest.raises((RoutineError, ValueError)):
            global_accumulation(
                routine_context,
                measure_electrode="G1",
                step_size=2.0,
                bias_gate="G1",
                bias_voltage=0.0,
                charge_carrier_type="electron",
            )
        assert routine_context.resources.device.sweep_all_called

    def test_with_session_logging(self, routine_context):
        session = MockLoggerSession()
        routine_context.results.store(
            "leakage_test",
            {"max_safe_voltage_bound": 1.0, "min_safe_voltage_bound": -10.0},
        )
        routine_context.resources._resources["device"] = HighQualityMockDevice()
        result = global_accumulation(
            routine_context,
            measure_electrode="G1",
            step_size=0.1,
            bias_gate="G1",
            bias_voltage=0.0,
            charge_carrier_type="electron",
            session=session,
        )
        assert "global_turn_on_voltage" in result
        assert len(session.analyses) == 1
        assert session.analyses[0][0] == "global_turn_on_voltage"
        cutoff = result["global_turn_on_voltage"]
        for gate in routine_context.resources.device.control_gates:
            assert routine_context.resources.device.voltages[gate] == cutoff


class TestReservoirCharacterization:
    def test_invalid_step_size(self, characterization_context):
        with pytest.raises(RoutineError, match="Step size must be greater than 0"):
            reservoir_characterization(
                characterization_context,
                measure_electrode="G1",
                step_size=-0.1,
                bias_gate="G1",
                bias_voltage=0.0,
            )

    def test_sweeps_each_reservoir(self, characterization_context):
        class TrackedDevice(MockDevice):
            def __init__(self):
                super().__init__()
                self.swept_gates = []

            def sweep_1d(self, gate, voltages, measure_electrode, session=None):
                self.swept_gates.append(gate)
                return voltages, np.ones_like(voltages) * 1e-10

        characterization_context.resources._resources["device"] = TrackedDevice()
        with pytest.raises((RoutineError, ValueError)):
            reservoir_characterization(
                characterization_context,
                measure_electrode="G1",
                step_size=2.0,
                bias_gate="G1",
                bias_voltage=0.0,
            )
        assert any(
            g in ["R1", "R2"]
            for g in characterization_context.resources.device.swept_gates
        )

    def test_with_session_logging(self, characterization_context):
        session = MockLoggerSession()
        characterization_context.resources._resources["device"] = (
            HighQualityMockDevice()
        )
        result = reservoir_characterization(
            characterization_context,
            measure_electrode="G1",
            step_size=0.1,
            bias_gate="G1",
            bias_voltage=0.0,
            session=session,
        )
        assert "reservoir_characterization" in result
        assert len(session.analyses) >= 1
        assert all("reservoir_characterization" in name for name, _ in session.analyses)


class TestFingerGateCharacterization:
    def test_invalid_step_size(self, characterization_context):
        with pytest.raises(RoutineError, match="Step size must be greater than 0"):
            finger_gate_characterization(
                characterization_context,
                measure_electrode="G1",
                step_size=0,
                bias_gate="G1",
                bias_voltage=0.0,
            )

    def test_sweeps_plunger_and_barrier_gates(self, characterization_context):
        class TrackedDevice(MockDevice):
            def __init__(self):
                super().__init__()
                self.swept_gates = []

            def sweep_1d(self, gate, voltages, measure_electrode, session=None):
                self.swept_gates.append(gate)
                return voltages, np.ones_like(voltages) * 1e-10

        characterization_context.resources._resources["device"] = TrackedDevice()
        with pytest.raises((RoutineError, ValueError)):
            finger_gate_characterization(
                characterization_context,
                measure_electrode="G1",
                step_size=2.0,
                bias_gate="G1",
                bias_voltage=0.0,
            )
        assert any(
            g in ["P1", "B1"]
            for g in characterization_context.resources.device.swept_gates
        )

    def test_with_session_logging(self, characterization_context):
        session = MockLoggerSession()
        characterization_context.resources._resources["device"] = (
            HighQualityMockDevice()
        )
        result = finger_gate_characterization(
            characterization_context,
            measure_electrode="G1",
            step_size=0.1,
            bias_gate="G1",
            bias_voltage=0.0,
            session=session,
        )
        assert "finger_gate_characterization" in result
        assert len(session.analyses) >= 1
        assert all(
            "finger_gate_characterization" in name for name, _ in session.analyses
        )


class TestAnalyzeSingleGateHeuristic:
    def test_poor_fit_raises_error(self):
        np.random.seed(42)
        voltages = np.linspace(-1, 1, 10)
        currents = np.random.random(10) * 1e-15
        with pytest.raises(ValueError, match="Curve fit quality is poor"):
            analyze_single_gate_heuristic(voltages, currents)

    def test_returns_all_expected_keys(self):
        voltages = np.linspace(-2, 2, 100)
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        np.random.seed(42)
        noise = np.random.normal(0, 0.01 * np.max(currents), size=currents.shape)
        result = analyze_single_gate_heuristic(voltages, currents + noise)
        for key in [
            "cutoff_voltage",
            "transition_voltage",
            "saturation_voltage",
            "popt",
            "pcov",
        ]:
            assert key in result
            if key.endswith("_voltage"):
                assert isinstance(result[key], float)

    def test_negative_amplitude_curve(self):
        voltages = np.linspace(-2, 2, 200)
        currents = pinchoff_curve(voltages, -0.5, 2.0, -1.0)
        np.random.seed(42)
        noise = np.random.normal(0, 0.01 * np.ptp(currents), size=currents.shape)
        result = analyze_single_gate_heuristic(voltages, currents + noise)
        assert (
            result["saturation_voltage"]
            < result["transition_voltage"]
            < result["cutoff_voltage"]
        )


class TestLeakageHelperFunctions:
    def test_calculate_leakage_matrix(self):
        leakage_matrix = _calculate_leakage_matrix(
            0.1, np.array([1e-9, 2e-9, 0.0, 5e-9])
        )
        assert leakage_matrix.shape == (4,)
        assert leakage_matrix[0] == abs(0.1 / 1e-9)
        assert leakage_matrix[2] == np.inf
        assert np.all(np.isfinite(leakage_matrix[:2]))

    def test_calculate_leakage_matrix_handles_negatives(self):
        leakage_matrix = _calculate_leakage_matrix(-0.1, np.array([-1e-9, 1e-9]))
        assert np.all(leakage_matrix > 0)
        assert np.all(np.isfinite(leakage_matrix))

    def test_check_leakage_threshold_no_leakage(self):
        leaked = _check_leakage_threshold(
            np.array([[np.inf, 1e8], [1e8, np.inf]]),
            leakage_threshold_resistance=50e6,
            leakage_threshold_count=0,
            control_gates=["G1", "G2"],
            delta_V=0.1,
            session=None,
        )
        assert not leaked

    def test_check_leakage_threshold_with_leakage(self):
        session = MockLoggerSession()
        leaked = _check_leakage_threshold(
            np.array([[np.inf, 1e5], [1e5, np.inf]]),
            leakage_threshold_resistance=50e6,
            leakage_threshold_count=0,
            control_gates=["G1", "G2"],
            delta_V=0.1,
            session=session,
        )
        assert leaked
        assert len(session.analyses) == 1
        assert session.analyses[0][0] == "leaky_gate_pairs"
