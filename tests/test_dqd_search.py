"""Tests for DQD search routines."""

from unittest.mock import patch

import numpy as np
import pytest

from stanza.models import GateType
from stanza.registry import ResourceRegistry, ResultsRegistry
from stanza.routines import RoutineContext
from stanza.routines.builtins.dqd_search import (
    compute_peak_spacing,
    run_dqd_search,
    run_dqd_search_fixed_barriers,
)
from stanza.routines.builtins.dqd_search.grid_search import (
    GRID_SQUARE_MULTIPLIER,
    SearchSquare,
    generate_2d_sweep,
    generate_diagonal_sweep,
    generate_grid_corners,
    select_next_square,
)
from stanza.routines.builtins.dqd_search.utils import (
    get_gate_safe_bounds,
    get_global_turn_on_voltage,
    get_voltages,
)


class MockResult:
    """Mock result object for model execution."""

    def __init__(self, output):
        self.output = output


class MockModels:
    """Mock models client for executing model requests."""

    def __init__(self, responses):
        self.responses = responses

    def execute(self, model, data):
        return MockResult(
            self.responses.get(model, {"classification": False, "score": 0.0})
        )


class MockModelsClient:
    """Mock client for model execution with configurable responses."""

    def __init__(self):
        self.name = "models_client"
        self.responses = {}
        self.models = MockModels(self.responses)

    def set_response(self, model_name, response):
        self.responses[model_name] = response


class MockDevice:
    """Mock device with plunger, reservoir, and barrier gates."""

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
    """Mock logger session for capturing measurements and analyses."""

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
    """Tests for compute_peak_spacing routine."""

    def test_returns_peak_spacing(self, characterization_context):
        """Test that compute_peak_spacing returns a valid peak spacing value."""
        result = compute_peak_spacing(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_search_scale=0.05,
            max_search_scale=0.2,
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
            min_search_scale=0.05,
            max_search_scale=0.2,
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
        """Test that compute_peak_spacing raises error when insufficient peaks are detected."""
        mock_models_client.set_response(
            "coulomb-blockade-peak-detector-v2", {"peak_indices": [1, 2]}
        )
        with pytest.raises(ValueError, match="No peak spacings found"):
            compute_peak_spacing(
                characterization_context,
                gates=GATES,
                measure_electrode="P1",
                min_search_scale=0.05,
                max_search_scale=0.2,
                current_trace_points=32,
                max_number_of_samples=2,
                number_of_samples_for_scale_computation=1,
                seed=42,
            )

    def test_raises_on_no_coulomb_blockade(
        self, characterization_context, mock_models_client
    ):
        """Test that compute_peak_spacing raises error when no Coulomb blockade is detected."""
        mock_models_client.set_response(
            "coulomb-blockade-classifier-v3", {"classification": False, "score": 0.0}
        )
        with pytest.raises(ValueError, match="No peak spacings found"):
            compute_peak_spacing(
                characterization_context,
                gates=GATES,
                measure_electrode="P1",
                min_search_scale=0.05,
                max_search_scale=0.2,
                current_trace_points=32,
                max_number_of_samples=2,
                number_of_samples_for_scale_computation=1,
                seed=42,
            )


class TestRunDQDSearchFixedBarriers:
    """Tests for run_dqd_search_fixed_barriers routine."""

    def test_returns_dqd_squares(self, characterization_context):
        """Test that run_dqd_search_fixed_barriers returns a list of DQD squares."""
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
        """Test that run_dqd_search_fixed_barriers logs measurements and analyses to session."""
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
        """Test that run_dqd_search_fixed_barriers handles case when no DQDs are found."""
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
        """Test that run_dqd_search_fixed_barriers supports diagonal sweeps."""
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
        """Test that run_dqd_search_fixed_barriers supports hole charge carriers."""
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


class TestRunDQDSearchFixedBarriersEdgeCases:
    """Tests for edge cases in run_dqd_search_fixed_barriers routine."""

    def test_handles_all_squares_visited(self, characterization_context):
        """Test that routine handles case when all grid squares are visited."""
        result = run_dqd_search_fixed_barriers(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            current_trace_points=16,
            low_res_csd_points=8,
            high_res_csd_points=16,
            max_samples=1000,
            num_dqds_for_exit=100,
            seed=42,
        )
        assert "dqd_squares" in result

    def test_skips_out_of_bounds_squares(self, characterization_context):
        """Test that routine skips squares that are out of safe voltage bounds."""
        result = run_dqd_search_fixed_barriers(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            current_trace_points=16,
            low_res_csd_points=8,
            high_res_csd_points=16,
            max_samples=2,
            num_dqds_for_exit=1,
            barrier_voltages={"B0": 15.0, "B1": 15.0, "B2": 15.0},
            seed=42,
        )
        assert "dqd_squares" in result


class TestRunDQDSearch:
    """Tests for run_dqd_search routine."""

    def test_returns_barrier_sweep_results(self, characterization_context):
        """Test that run_dqd_search returns barrier sweep results."""
        result = run_dqd_search(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_search_scale=0.05,
            max_search_scale=0.2,
            current_trace_points=16,
            outer_barrier_points=2,
            inner_barrier_points=2,
            num_dqds_for_exit=1,
        )
        assert "run_dqd_search" in result
        assert isinstance(result["run_dqd_search"], list)
        assert len(result["run_dqd_search"]) > 0

    def test_logs_to_session(self, characterization_context):
        """Test that run_dqd_search logs measurements and analyses to session."""
        session = MockLoggerSession()
        run_dqd_search(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_search_scale=0.05,
            max_search_scale=0.2,
            current_trace_points=16,
            outer_barrier_points=2,
            inner_barrier_points=2,
            num_dqds_for_exit=2,
            session=session,
        )
        assert len(session.measurements) > 0 and len(session.analyses) > 0

    def test_exits_early_on_dqd(self, characterization_context):
        """Test that run_dqd_search exits early when target number of DQDs is found."""
        result = run_dqd_search(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_search_scale=0.05,
            max_search_scale=0.2,
            current_trace_points=16,
            outer_barrier_points=3,
            inner_barrier_points=3,
            num_dqds_for_exit=1,
        )
        assert len(result["run_dqd_search"]) >= 1

    def test_result_structure(self, characterization_context, mock_models_client):
        """Test that run_dqd_search returns results with expected structure."""
        mock_models_client.set_response(
            "charge-stability-diagram-binary-classifier-v1-48x48",
            {"classification": False, "score": 0.0},
        )
        result = run_dqd_search(
            characterization_context,
            gates=GATES,
            measure_electrode="P1",
            min_search_scale=0.05,
            max_search_scale=0.2,
            current_trace_points=16,
            outer_barrier_points=2,
            inner_barrier_points=2,
            num_dqds_for_exit=10,
        )
        assert len(result["run_dqd_search"]) == 4
        for point in result["run_dqd_search"]:
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


class TestUtilsErrorPaths:
    """Tests for error handling in DQD search utility functions."""

    def test_get_global_turn_on_voltage_missing(self):
        """Test that get_global_turn_on_voltage raises error when value is missing."""
        results = ResultsRegistry()
        results.store("global_accumulation", None)
        with pytest.raises(ValueError, match="Global turn on voltage not found"):
            get_global_turn_on_voltage(results)

    def test_get_voltages_missing_characterization(self):
        """Test that get_voltages raises error when characterization results are missing."""
        results = ResultsRegistry()
        results.store("reservoir_characterization", {})
        results.store("finger_gate_characterization", {})
        with pytest.raises(ValueError, match="Gate characterization results not found"):
            get_voltages(GATES, "saturation_voltage", results)

    def test_get_gate_safe_bounds_missing(self):
        """Test that get_gate_safe_bounds raises error when leakage test results are missing."""
        results = ResultsRegistry()
        with pytest.raises(ValueError, match="Leakage test results not found"):
            get_gate_safe_bounds(results)


class TestGridGeometry:
    """Tests for grid generation and geometry functions."""

    def test_grid_fits_within_bounds_and_diagonal_trace_correct(
        self, characterization_context
    ):
        """Test that generated grid corners fit within bounds and diagonal trace is correct."""
        peak_spacing = characterization_context.results.get("compute_peak_spacing")[
            "peak_spacing"
        ]
        square_size = peak_spacing * GRID_SQUARE_MULTIPLIER

        device = characterization_context.resources.device
        plunger_gates = device.get_gates_by_type(GateType.PLUNGER)

        results = characterization_context.results
        finger_gate_char = results.get("finger_gate_characterization")[
            "finger_gate_characterization"
        ]
        x_bounds = (
            finger_gate_char[plunger_gates[0]]["cutoff_voltage"],
            finger_gate_char[plunger_gates[0]]["saturation_voltage"],
        )
        y_bounds = (
            finger_gate_char[plunger_gates[1]]["cutoff_voltage"],
            finger_gate_char[plunger_gates[1]]["saturation_voltage"],
        )

        grid_corners, _, _ = generate_grid_corners(x_bounds, y_bounds, square_size)

        for corner in grid_corners:
            assert x_bounds[0] <= corner[0] <= x_bounds[1]
            assert y_bounds[0] <= corner[1] <= y_bounds[1]
            assert x_bounds[0] <= corner[0] + square_size <= x_bounds[1]
            assert y_bounds[0] <= corner[1] + square_size <= y_bounds[1]

        corner = grid_corners[0]
        diagonal_sweep = generate_diagonal_sweep(corner, square_size, 32)

        assert np.allclose(diagonal_sweep[0], corner)
        assert np.allclose(diagonal_sweep[-1], corner + square_size)
        assert len(diagonal_sweep) == 32

    def test_csd_sweeps_cover_square_area(self, characterization_context):
        """Test that charge stability diagram sweeps correctly cover square area."""
        peak_spacing = characterization_context.results.get("compute_peak_spacing")[
            "peak_spacing"
        ]
        square_size = peak_spacing * GRID_SQUARE_MULTIPLIER

        device = characterization_context.resources.device
        plunger_gates = device.get_gates_by_type(GateType.PLUNGER)

        results = characterization_context.results
        finger_gate_char = results.get("finger_gate_characterization")[
            "finger_gate_characterization"
        ]
        x_bounds = (
            finger_gate_char[plunger_gates[0]]["cutoff_voltage"],
            finger_gate_char[plunger_gates[0]]["saturation_voltage"],
        )
        y_bounds = (
            finger_gate_char[plunger_gates[1]]["cutoff_voltage"],
            finger_gate_char[plunger_gates[1]]["saturation_voltage"],
        )

        grid_corners, _, _ = generate_grid_corners(x_bounds, y_bounds, square_size)
        corner = grid_corners[0]

        low_res_points = 8
        low_res_sweep = generate_2d_sweep(corner, square_size, low_res_points)
        assert low_res_sweep.shape == (low_res_points, low_res_points, 2)
        assert np.allclose(low_res_sweep[0, 0], corner)
        assert np.allclose(low_res_sweep[-1, -1], corner + square_size)
        assert np.allclose(low_res_sweep[0, -1], corner + [square_size, 0])
        assert np.allclose(low_res_sweep[-1, 0], corner + [0, square_size])

        high_res_points = 16
        high_res_sweep = generate_2d_sweep(corner, square_size, high_res_points)
        assert high_res_sweep.shape == (high_res_points, high_res_points, 2)
        assert np.allclose(high_res_sweep[0, 0], corner)
        assert np.allclose(high_res_sweep[-1, -1], corner + square_size)
        assert np.allclose(high_res_sweep[0, -1], corner + [square_size, 0])
        assert np.allclose(high_res_sweep[-1, 0], corner + [0, square_size])


class TestGridSearchWeightedSelection:
    """Tests for weighted selection algorithm in grid search."""

    def test_select_next_square_random_fallback(self):
        """Test that select_next_square falls back to random selection when needed."""
        visited = [
            SearchSquare(
                grid_idx=0,
                current_trace_currents=np.array([1.0]),
                current_trace_voltages=np.array([[0.0, 0.0]]),
                current_trace_score=0.5,
                current_trace_classification=True,
                low_res_csd_currents=None,
                low_res_csd_voltages=None,
                low_res_csd_score=0.0,
                low_res_csd_classification=False,
                high_res_csd_currents=None,
                high_res_csd_voltages=None,
                high_res_csd_score=0.0,
                high_res_csd_classification=False,
            )
        ]
        result = select_next_square(visited, [], 3, 3, False)
        assert result is not None and 0 < result < 9

    def test_sampling_priority_order(self):
        """Test that select_next_square prioritizes neighbors of DQD squares."""
        dqd_square = SearchSquare(
            grid_idx=5,
            current_trace_currents=np.array([1.0]),
            current_trace_voltages=np.array([[0.0, 0.0]]),
            current_trace_score=1.0,
            current_trace_classification=True,
            low_res_csd_currents=np.array([[1.0]]),
            low_res_csd_voltages=np.array([[[0.0, 0.0]]]),
            low_res_csd_score=1.0,
            low_res_csd_classification=True,
            high_res_csd_currents=np.array([[1.0]]),
            high_res_csd_voltages=np.array([[[0.0, 0.0]]]),
            high_res_csd_score=1.0,
            high_res_csd_classification=True,
        )

        high_score_square = SearchSquare(
            grid_idx=14,
            current_trace_currents=np.array([1.0]),
            current_trace_voltages=np.array([[0.0, 0.0]]),
            current_trace_score=1.6,
            current_trace_classification=True,
            low_res_csd_currents=None,
            low_res_csd_voltages=None,
            low_res_csd_score=0.0,
            low_res_csd_classification=False,
            high_res_csd_currents=None,
            high_res_csd_voltages=None,
            high_res_csd_score=0.0,
            high_res_csd_classification=False,
        )

        visited = [dqd_square, high_score_square]
        dqd_squares = [dqd_square]

        results = [
            select_next_square(visited, dqd_squares, 4, 4, False) for _ in range(20)
        ]

        dqd_neighbors = {1, 4, 6, 9}
        assert all(r in dqd_neighbors for r in results)
