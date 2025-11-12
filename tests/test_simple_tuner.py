"""Tests for simple tuner routines."""

from unittest.mock import patch

import numpy as np
import pytest

from stanza.models import GateType
from stanza.registry import ResourceRegistry, ResultsRegistry
from stanza.routines import RoutineContext
from stanza.routines.builtins.simple_tuner import (
    compute_peak_spacing,
    run_dqd_search,
    run_dqd_search_fixed_barriers,
)


class MockResult:
    def __init__(self, output):
        self.output = output


class MockModels:
    def __init__(self, responses):
        self.responses = responses

    def execute(self, model, data):
        return MockResult(
            self.responses.get(model, {"classification": False, "score": 0.0})
        )


class MockModelsClient:
    def __init__(self):
        self.name = "models_client"
        self.responses = {}
        self.models = MockModels(self.responses)

    def set_response(self, model_name, response):
        self.responses[model_name] = response


class MockDevice:
    def __init__(self):
        self.name = "device"
        self.control_gates = ["P1", "P2", "R1", "R2", "B0", "B1", "B2"]
        self.voltages = dict.fromkeys(self.control_gates, 0.0)
        self.gate_types = {
            "P1": GateType.PLUNGER,
            "P2": GateType.PLUNGER,
            "R1": GateType.RESERVOIR,
            "R2": GateType.RESERVOIR,
            "B0": GateType.BARRIER,
            "B1": GateType.BARRIER,
            "B2": GateType.BARRIER,
        }

    def measure(self, electrodes):
        return (
            1e-11
            if isinstance(electrodes, str)
            else np.array([1e-11] * len(electrodes))
        )

    def check(self, electrodes):
        return (
            self.voltages.get(electrodes, 0.0)
            if isinstance(electrodes, str)
            else [self.voltages.get(e, 0.0) for e in electrodes]
        )

    def jump(self, voltage_dict, wait_for_settling=False):
        self.voltages.update(voltage_dict)

    def sweep_nd(self, gate_electrodes, voltages, measure_electrode, session=None):
        return voltages, np.ones(len(voltages)) * 1e-10

    def get_gates_by_type(self, gate_type):
        return [n for n, t in self.gate_types.items() if t == gate_type]


class MockLoggerSession:
    def __init__(self):
        self.measurements = []
        self.analyses = []

    def log_measurement(self, name, data, metadata=None, routine_name=None):
        self.measurements.append((name, data, metadata, routine_name))

    def log_analysis(self, name, data, metadata=None, routine_name=None):
        self.analyses.append((name, data, metadata, routine_name))


@pytest.fixture
def mock_models_client():
    client = MockModelsClient()
    client.set_response(
        "coulomb-blockade-classifier-v3", {"classification": True, "score": 1.0}
    )
    client.set_response(
        "coulomb-blockade-peak-detector-v2", {"peak_indices": [5, 10, 15]}
    )
    client.set_response(
        "charge-stability-diagram-binary-classifier-v2-16x16",
        {"classification": True, "score": 1.0},
    )
    client.set_response(
        "charge-stability-diagram-binary-classifier-v1-48x48",
        {"classification": True, "score": 1.0},
    )
    return client


@pytest.fixture
def characterization_context(mock_models_client):
    ctx = RoutineContext(
        ResourceRegistry(MockDevice(), mock_models_client), ResultsRegistry()
    )

    # Setup gate characterization data
    gate_data = {
        "saturation_voltage": 2.0,
        "cutoff_voltage": -1.0,
        "transition_voltage": 1.0,
    }

    ctx.results.store(
        "leakage_test",
        {"max_safe_voltage_bound": 10.0, "min_safe_voltage_bound": -10.0},
    )
    ctx.results.store("global_accumulation", {"global_turn_on_voltage": 0.5})
    ctx.results.store(
        "reservoir_characterization",
        {"reservoir_characterization": dict.fromkeys(["R1", "R2"], gate_data)},
    )
    ctx.results.store(
        "finger_gate_characterization",
        {
            "finger_gate_characterization": dict.fromkeys(
                ["P1", "P2", "B0", "B1", "B2"], gate_data
            )
        },
    )
    ctx.results.store("compute_peak_spacing", {"peak_spacing": 0.1})

    return ctx


@pytest.fixture(autouse=True)
def mock_deps():
    with (
        patch("time.sleep"),
        patch("numpy.random.seed"),
        patch("numpy.random.uniform", return_value=0.5),
        patch(
            "numpy.random.choice",
            side_effect=lambda x, p=None: x[0] if isinstance(x, list) else 0,
        ),
    ):
        yield


GATES = ["P1", "P2", "R1", "R2", "B0", "B1", "B2"]


class TestComputePeakSpacing:
    def test_returns_peak_spacing(self, characterization_context):
        result = compute_peak_spacing(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_peak_spacing=0.05,
            max_peak_spacing=0.2,
            current_trace_points=32,
            max_number_of_samples=5,
            number_of_samples_for_scale_computation=3,
            seed=42,
        )
        assert "peak_spacing" in result and isinstance(result["peak_spacing"], float)

    def test_logs_to_session(self, characterization_context):
        session = MockLoggerSession()
        compute_peak_spacing(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_peak_spacing=0.05,
            max_peak_spacing=0.2,
            current_trace_points=32,
            max_number_of_samples=10,
            number_of_samples_for_scale_computation=1,
            seed=42,
            session=session,
        )
        assert len(session.measurements) > 0 and len(session.analyses) > 0

    def test_raises_on_insufficient_peaks(
        self, characterization_context, mock_models_client
    ):
        mock_models_client.set_response(
            "coulomb-blockade-peak-detector-v2", {"peak_indices": [1, 2]}
        )
        with pytest.raises(ValueError, match="No peak spacings found"):
            compute_peak_spacing(
                characterization_context,
                gates=GATES,
                measure_electrode="P1",
                min_peak_spacing=0.05,
                max_peak_spacing=0.2,
                current_trace_points=32,
                max_number_of_samples=2,
                number_of_samples_for_scale_computation=1,
                seed=42,
            )

    def test_raises_on_no_coulomb_blockade(
        self, characterization_context, mock_models_client
    ):
        mock_models_client.set_response(
            "coulomb-blockade-classifier-v3", {"classification": False, "score": 0.0}
        )
        with pytest.raises(ValueError, match="No peak spacings found"):
            compute_peak_spacing(
                characterization_context,
                gates=GATES,
                measure_electrode="P1",
                min_peak_spacing=0.05,
                max_peak_spacing=0.2,
                current_trace_points=32,
                max_number_of_samples=2,
                number_of_samples_for_scale_computation=1,
                seed=42,
            )


class TestRunDqdSearchFixedBarriers:
    def test_returns_dqd_squares(self, characterization_context):
        result = run_dqd_search_fixed_barriers(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            current_trace_points=16,
            low_res_csd_points=8,
            high_res_csd_points=16,
            max_samples=3,
            num_dqds_for_exit=1,
            seed=42,
        )
        assert "dqd_squares" in result and isinstance(result["dqd_squares"], list)

    def test_logs_to_session(self, characterization_context):
        session = MockLoggerSession()
        run_dqd_search_fixed_barriers(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            current_trace_points=16,
            low_res_csd_points=8,
            high_res_csd_points=16,
            max_samples=2,
            num_dqds_for_exit=1,
            seed=42,
            session=session,
        )
        assert len(session.measurements) > 0 and len(session.analyses) > 0

    def test_handles_no_dqds(self, characterization_context, mock_models_client):
        mock_models_client.set_response(
            "charge-stability-diagram-binary-classifier-v1-48x48",
            {"classification": False, "score": 0.0},
        )
        result = run_dqd_search_fixed_barriers(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            current_trace_points=16,
            low_res_csd_points=8,
            high_res_csd_points=16,
            max_samples=2,
            num_dqds_for_exit=1,
            seed=42,
        )
        assert len(result["dqd_squares"]) == 0

    def test_supports_diagonals(self, characterization_context):
        result = run_dqd_search_fixed_barriers(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            current_trace_points=16,
            low_res_csd_points=8,
            high_res_csd_points=16,
            max_samples=3,
            num_dqds_for_exit=2,
            include_diagonals=True,
            seed=42,
        )
        assert "dqd_squares" in result

    def test_supports_hole_carriers(self, characterization_context):
        result = run_dqd_search_fixed_barriers(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            current_trace_points=16,
            low_res_csd_points=8,
            high_res_csd_points=16,
            max_samples=2,
            num_dqds_for_exit=1,
            charge_carrier_type="hole",
            seed=42,
        )
        assert "dqd_squares" in result


class TestRunDqdSearch:
    def test_returns_barrier_sweep_results(self, characterization_context):
        result = run_dqd_search(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_peak_spacing=0.05,
            max_peak_spacing=0.2,
            current_trace_points=16,
            outer_barrier_points=2,
            inner_barrier_points=2,
            num_dqds_for_exit=1,
        )
        assert "barrier_sweep_results" in result
        assert isinstance(result["barrier_sweep_results"], list)
        assert len(result["barrier_sweep_results"]) > 0

    def test_logs_to_session(self, characterization_context):
        session = MockLoggerSession()
        run_dqd_search(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_peak_spacing=0.05,
            max_peak_spacing=0.2,
            current_trace_points=16,
            outer_barrier_points=2,
            inner_barrier_points=2,
            num_dqds_for_exit=2,
            session=session,
        )
        assert len(session.measurements) > 0 and len(session.analyses) > 0

    def test_exits_early_on_dqd(self, characterization_context):
        result = run_dqd_search(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_peak_spacing=0.05,
            max_peak_spacing=0.2,
            current_trace_points=16,
            outer_barrier_points=3,
            inner_barrier_points=3,
            num_dqds_for_exit=1,
        )
        assert len(result["barrier_sweep_results"]) >= 1

    def test_result_structure(self, characterization_context, mock_models_client):
        mock_models_client.set_response(
            "charge-stability-diagram-binary-classifier-v1-48x48",
            {"classification": False, "score": 0.0},
        )
        result = run_dqd_search(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_peak_spacing=0.05,
            max_peak_spacing=0.2,
            current_trace_points=16,
            outer_barrier_points=2,
            inner_barrier_points=2,
            num_dqds_for_exit=10,
        )
        assert len(result["barrier_sweep_results"]) == 4
        for point in result["barrier_sweep_results"]:
            assert all(
                k in point
                for k in [
                    "outer_barrier_voltage",
                    "inner_barrier_voltage",
                    "peak_spacing",
                    "dqd_squares",
                ]
            )
            assert isinstance(point["outer_barrier_voltage"], float)
            assert isinstance(point["inner_barrier_voltage"], float)
