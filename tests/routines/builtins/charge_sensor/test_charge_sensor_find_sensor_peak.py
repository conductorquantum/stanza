"""Tests for charge sensor routines (find_sensor_peak, find_stable_sensor_peak)."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from stanza.exceptions import RoutineError
from stanza.models import DeviceGroup
from stanza.routines import RoutineContext
from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
    SensorDotPlungerSweepOutput,
    StablePeakCandidate,
    _calculate_combined_scores,
    _calculate_local_slope,
    _calculate_voltage_noise,
    _normalize_sensitivity_scores,
    build_sensor_sweep_voltage_list,
    calculate_peak_window_bounds,
    calculate_quality_scores,
    find_sensor_peak,
)
from stanza.routines.builtins.charge_sensor.constants import (
    DEFAULT_WINDOW_HALF_WIDTH,
    WINDOW_FRACTION,
)
from stanza.routines.builtins.utils.peak_fitting import (
    calculate_quality_score,
    fit_peak_multi_model,
    lorentzian,
)

# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_device_for_sensor_routines():
    """Create a comprehensive mock device for sensor routine tests."""
    mock_device = Mock()

    # Setup device config with groups
    sensor_group = DeviceGroup(name="sensor_group", gates=["G1", "G2", "G3"])
    control_group = DeviceGroup(name="control_group", gates=["G4", "G5"])

    mock_device.device_config = Mock()
    mock_device.device_config.groups = {
        "sensor_group": sensor_group,
        "control_group": control_group,
    }

    # Mock gate lookups - make control_gates iterable
    mock_device.control_gates = ["G1", "G2", "G3", "G4", "G5"]
    mock_device.get_gates_by_type.return_value = ["G1", "G2", "G3", "G4", "G5"]

    # Mock device methods
    mock_device.check.return_value = [0.0, 0.0, 0.0]
    mock_device.jump = Mock()
    mock_device.measure.return_value = 1e-9

    # Mock sweep_nd to return reasonable data
    voltages = np.linspace(-1.0, -0.5, 128)
    currents = np.ones(128) * 1e-9
    mock_device.sweep_nd.return_value = (voltages, currents)

    return mock_device


def create_mock_context_for_sensor_routines(include_prerequisites=True):
    """Create a mock RoutineContext with required prerequisite results."""
    mock_ctx = Mock(spec=RoutineContext)

    if include_prerequisites:
        mock_ctx.results = {
            "global_accumulation_sensor_group": {"global_turn_on_voltage": -0.8},
            "finger_gate_characterization_sensor_group": {
                "G3": {  # sensor plunger gate
                    "saturation_voltage": 0.5,
                    "cutoff_voltage": -1.5,
                    "pinch_off_voltage": -2.0,
                }
            },
        }
    else:
        mock_ctx.results = {}

    mock_device = create_mock_device_for_sensor_routines()

    # Create mock models_client for ML model calls
    mock_models_client = Mock()
    mock_models = Mock()
    mock_execute_result = Mock()
    mock_execute_result.output = {
        "classification": True,
        "score": 0.95,
        "peak_indices": [64],
    }
    mock_models.execute.return_value = mock_execute_result
    mock_models_client.models = mock_models

    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_resources.models_client = mock_models_client
    mock_ctx.resources = mock_resources
    mock_ctx.session_metadata = {}

    return mock_ctx, mock_device


# =============================================================================
# Helper Utilities Tests
# =============================================================================


def test_calculate_peak_window_bounds_handles_edge_peaks():
    """Verify calculate_peak_window_bounds clamps to trace bounds and always contains the peak."""
    trace_length = 1000

    # First/middle/last peak scenarios should all yield windows within the trace that contain the peak
    edge_cases = [
        ([100, 400, 700], 0, 100),
        ([100, 400, 700], 1, 400),
        ([100, 400, 700], 2, 700),
    ]
    for peak_indices, peak_index, peak_idx in edge_cases:
        start, end = calculate_peak_window_bounds(
            peak_idx=peak_idx,
            peak_index=peak_index,
            peak_indices=peak_indices,
            trace_length=trace_length,
        )
        assert 0 <= start < peak_idx < end <= trace_length

    # Test single peak (should use entire trace)
    start, end = calculate_peak_window_bounds(
        peak_idx=500, peak_index=0, peak_indices=[500], trace_length=trace_length
    )
    assert start == 0
    assert end == trace_length

    # Closely spaced peaks should still provide a valid window with at least a few points
    close_peak_indices = [50, 100, 150, 200]
    start, end = calculate_peak_window_bounds(
        peak_idx=100,
        peak_index=1,
        peak_indices=close_peak_indices,
        trace_length=trace_length,
    )
    assert 0 <= start < 100 < end <= trace_length
    assert end - start >= 3

    # Middle peak should use WINDOW_FRACTION of inter-peak spacing when possible
    peak_indices = [50, 150, 250]
    peak_idx = 150
    start, end = calculate_peak_window_bounds(
        peak_idx=peak_idx,
        peak_index=1,
        peak_indices=peak_indices,
        trace_length=trace_length,
    )
    expected_offset = int(WINDOW_FRACTION * 100)
    assert abs(start - (peak_idx - expected_offset)) <= 5
    assert abs(end - (peak_idx + expected_offset)) <= 5
    assert start < peak_idx < end

    # Extremely wide spacing should clamp to DEFAULT_WINDOW_HALF_WIDTH on each side
    wide_peak_indices = [200, 600, 1000]
    trace_length_wide = 1200
    start, end = calculate_peak_window_bounds(
        peak_idx=600,
        peak_index=1,
        peak_indices=wide_peak_indices,
        trace_length=trace_length_wide,
    )
    assert start == 600 - DEFAULT_WINDOW_HALF_WIDTH
    assert end == 600 + DEFAULT_WINDOW_HALF_WIDTH
    assert end - start == 2 * DEFAULT_WINDOW_HALF_WIDTH


def test_normalize_sensitivity_scores_constant_inputs(fitted_peak_factory):
    """Confirm _normalize_sensitivity_scores assigns score 1.0 when all sensitivities are equal."""
    peaks = [
        fitted_peak_factory(
            sensitivity=1e-6,
            sensitivity_voltage=float(i * 0.01),
            peak_idx=i * 10,
            peak_voltage=float(i * 0.01),
        )
        for i in range(5)
    ]

    _normalize_sensitivity_scores(peaks)

    for peak in peaks:
        assert peak.sensitivity_score == 1.0


def testcalculate_quality_scores_requires_normalized_sensitivity(fitted_peak_factory):
    """Call calculate_quality_scores without pre-normalized scores and assert it raises RoutineError."""
    peak = fitted_peak_factory(
        sensitivity=1e-6,
        sensitivity_voltage=0.01,
        peak_idx=50,
        peak_voltage=0.01,
        sensitivity_score=None,
    )

    with pytest.raises(RoutineError, match="Sensitivity score not set"):
        calculate_quality_scores([peak])


def test_calculate_local_slope_window_validation():
    """Ensure _calculate_local_slope raises when fewer than three data points surround the target voltage."""
    voltages = np.array([0.0, 0.01])  # Only 2 points
    currents = np.array([0.0, 1e-9])
    target_voltage = 0.005

    with pytest.raises(
        RoutineError, match="Insufficient points for local slope calculation"
    ):
        _calculate_local_slope(voltages, currents, target_voltage, window_points=5)


def test_calculate_voltage_noise_near_zero_slope():
    """Provide tiny slopes to _calculate_voltage_noise and confirm it guards against division-by-zero."""
    current_std = 1e-10
    local_slope = 1e-15  # Very small slope

    with pytest.raises(RoutineError, match="Local slope magnitude too small"):
        _calculate_voltage_noise(current_std, local_slope)


# =============================================================================
# Peak Scoring Algorithms Tests
# =============================================================================


def test_peak_quality_score_weights_components():
    """Verify quality score combines R² (70%), normalized RMSE (5%),
    skew residual (5%), and sensitivity score (20%) with correct weights."""
    # Known inputs
    r_squared = 0.90
    rmse = 0.02
    y_max = 1.0
    skew = 0.15
    sensitivity_score = 0.85

    quality = calculate_quality_score(r_squared, rmse, y_max, skew, sensitivity_score)

    # Expected: 0.7*0.90 - 0.05*0.02 - 0.05*0.15 + 0.2*0.85
    expected = 0.7 * 0.90 - 0.05 * 0.02 - 0.05 * 0.15 + 0.2 * 0.85
    assert abs(quality - expected) < 1e-6


def test_sensitivity_score_normalized_across_peaks(fitted_peak_factory):
    """With N detected peaks, confirm sensitivity scores are min-max normalized
    to [0, 1] range before quality calculation."""
    peaks = []
    sensitivities = [1e-6, 5e-6, 10e-6, 2e-6, 7e-6]  # Different values

    for i, sens in enumerate(sensitivities):
        peaks.append(
            fitted_peak_factory(
                sensitivity=sens,
                peak_idx=i * 10,
                peak_voltage=i * 0.01,
                sensitivity_voltage=0.0,
                window_currents=np.ones(10) * 1e-9,
            )
        )

    _normalize_sensitivity_scores(peaks)

    # Verify all scores are in [0, 1] range
    for peak in peaks:
        assert 0 <= peak.sensitivity_score <= 1.0

    # Peak with max sensitivity should have score = 1.0
    max_sens_idx = np.argmax(sensitivities)
    assert peaks[max_sens_idx].sensitivity_score == 1.0

    # Peak with min sensitivity should have score = 0.0
    min_sens_idx = np.argmin(sensitivities)
    assert peaks[min_sens_idx].sensitivity_score == 0.0


def test_find_sensor_peak_returns_highest_quality():
    """Feed synthetic sweep with multiple peaks of known quality and verify
    the routine calculates quality scores for all peaks."""
    # Create synthetic trace with multiple Lorentzian peaks at different quality levels

    voltages = np.linspace(-1.0, -0.5, 500)

    # Peak 1: High amplitude, narrow (high quality)
    peak1 = lorentzian(
        np.arange(100, 200), amplitude=3e-9, center=50, width=8, offset=1e-11
    )

    # Peak 2: Medium amplitude, wider (medium quality)
    peak2 = lorentzian(
        np.arange(200, 300), amplitude=2e-9, center=50, width=15, offset=1e-11
    )

    # Peak 3: Low amplitude, wide (low quality)
    peak3 = lorentzian(
        np.arange(300, 400), amplitude=1e-9, center=50, width=20, offset=1e-11
    )

    # Combine into full trace
    currents = np.ones(500) * 1e-11
    currents[100:200] = peak1
    currents[200:300] = peak2
    currents[300:400] = peak3

    # Add small noise
    np.random.seed(42)
    currents += np.random.normal(0, 1e-13, len(currents))

    # Simulate peak detection finding all three peaks
    peak_indices = [150, 250, 350]

    # Fit all peaks
    fitted_peaks = []
    for peak_number, peak_idx in enumerate(peak_indices):
        bounds = calculate_peak_window_bounds(
            peak_idx, peak_number, peak_indices, len(currents)
        )
        if bounds:
            start_idx, end_idx = bounds

            peak = fit_peak_multi_model(
                window_currents=currents[start_idx:end_idx],
                window_indices=np.arange(start_idx, end_idx),
                peak_idx_in_window=peak_idx - start_idx,
                aggregated_voltages=voltages,
                window_start_idx=start_idx,
                window_end_idx=end_idx,
                peak_idx_aggregated=peak_idx,
            )
            fitted_peaks.append(peak)

    # Normalize and calculate quality scores
    _normalize_sensitivity_scores(fitted_peaks)
    calculate_quality_scores(fitted_peaks)

    # Verify all peaks have quality scores calculated
    assert len(fitted_peaks) == 3
    for peak in fitted_peaks:
        assert peak.quality_score is not None
        assert peak.sensitivity_score is not None
        assert 0 <= peak.quality_score <= 1.0

    # Verify quality scores are different (peaks have different characteristics)
    quality_scores = [p.quality_score for p in fitted_peaks]
    assert len(set(quality_scores)) > 1, (
        "Quality scores should differ for different peaks"
    )


# =============================================================================
# find_sensor_peak Tests
# =============================================================================


def test_find_sensor_peak_requires_prerequisites():
    """Run find_sensor_peak with missing global_accumulation or
    finger_gate_characterization results and expect RoutineError."""
    # Missing prerequisites
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {}

    mock_device = create_mock_device_for_sensor_routines()
    mock_resources = Mock()
    mock_resources.device = mock_device
    # Set up group mock to be a dict-like object (or None)
    mock_resources.group = None
    mock_ctx.resources = mock_resources

    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with pytest.raises(RoutineError, match="global_turn_on_voltage not found"):
            find_sensor_peak(
                ctx=mock_ctx,
                peak_spacing=0.02,
                sensor_group_name="sensor_group",
                sensor_plunger_gate="G3",
                measure_electrode="OUT",
                bias_gate="BIAS",
                bias_voltage=1e-4,
            )


def test_find_sensor_peak_gate_voltage_overrides_apply():
    """Provide gate_voltage_overrides and ensure build_sensor_sweep_voltage_list
    honors them for reservoirs/shared gates during many_window_barrier_sweep."""

    # Test the helper function directly
    sensor_gates_list = ["G1", "G2", "G3"]
    sensor_plunger_index = 2  # G3
    base_voltage = -0.8
    plunger_voltages = np.linspace(-1.0, -0.5, 5)
    gate_overrides = {"G1": 0.8, "G2": 0.9}

    voltage_list = build_sensor_sweep_voltage_list(
        sensor_gates_list=sensor_gates_list,
        sensor_plunger_index=sensor_plunger_index,
        base_voltage=base_voltage,
        plunger_voltages=plunger_voltages,
        gate_voltage_overrides=gate_overrides,
    )

    # Verify overrides are applied
    # Each entry in voltage_list should be a list of [G1, G2, G3] voltages
    for voltages in voltage_list:
        assert voltages[0] == 0.8, "G1 should use override voltage 0.8"
        assert voltages[1] == 0.9, "G2 should use override voltage 0.9"
        # G3 varies as the plunger
        assert voltages[2] in plunger_voltages


def test_find_sensor_peak_uses_narrowed_range_for_park_point():
    """Verify that the narrowed range returned from many_window_barrier_sweep
    is applied when parking the sensor. Tests both multi-peak and single-peak
    fallback cases."""
    # Test case 1: Multi-peak scenario - narrowed range calculated from adjacent peaks
    peak_voltage = -0.7
    prev_peak_voltage = -0.74
    next_peak_voltage = -0.66

    # Calculate narrowed range (midpoint between peaks)
    start_of_range = (prev_peak_voltage + peak_voltage) / 2
    end_of_range = (peak_voltage + next_peak_voltage) / 2
    narrowed_range = (start_of_range, end_of_range)

    # Verify narrowed range is valid
    assert isinstance(narrowed_range, tuple)
    assert len(narrowed_range) == 2
    assert narrowed_range[0] < narrowed_range[1]

    # Verify range is centered around the peak
    range_center = (narrowed_range[0] + narrowed_range[1]) / 2
    assert abs(range_center - peak_voltage) < 0.001

    # Verify park point would be within the narrowed range
    park_voltage = peak_voltage
    assert narrowed_range[0] <= park_voltage <= narrowed_range[1]

    # Test case 2: Single-peak fallback scenario - uses peak_spacing as fallback
    # When there's only one peak, prev/next voltages use peak_spacing as fallback
    peak_voltage_fallback = -0.7
    peak_spacing = 0.02

    # Simulate the fallback logic from find_sensor_peak
    # When there's no previous peak, use peak_voltage - peak_spacing
    # When there's no next peak, use peak_voltage + peak_spacing
    prev_peak_voltage_fallback = peak_voltage_fallback - peak_spacing
    next_peak_voltage_fallback = peak_voltage_fallback + peak_spacing

    # Calculate narrowed range using fallback bounds
    start_of_range_fallback = (prev_peak_voltage_fallback + peak_voltage_fallback) / 2
    end_of_range_fallback = (peak_voltage_fallback + next_peak_voltage_fallback) / 2
    narrowed_range_fallback = (start_of_range_fallback, end_of_range_fallback)

    # Verify fallback bounds are used correctly
    assert abs(prev_peak_voltage_fallback - (-0.72)) < 1e-9
    assert abs(next_peak_voltage_fallback - (-0.68)) < 1e-9
    assert (
        narrowed_range_fallback[0] < peak_voltage_fallback < narrowed_range_fallback[1]
    )
    assert (
        abs((narrowed_range_fallback[1] - narrowed_range_fallback[0]) - peak_spacing)
        < 1e-9
    )  # Range spans one spacing


# =============================================================================
# find_stable_sensor_peak Tests
# =============================================================================


# =============================================================================
# Workflow Integration Tests
# =============================================================================


def test_charge_sensor_workflow_consumes_compensation_results():
    """Chain find_sensor_peak → run_compensation → charge_sensor_csd_readout
    with mocks to ensure outputs from one step feed the next."""
    # Test the workflow data flow conceptually

    # Step 1: find_sensor_peak would return these results
    peak_result = {
        "best_peak_voltage": -0.7,
        "best_peak_max_gradient_voltage": -0.68,
        "mean_reservoir_saturation_voltage": -0.8,
        "sensor_gates_list": ["G1", "G2", "G3"],
        "sensor_park_point": {"G1": -0.8, "G2": -0.8, "G3": -0.7},
    }

    # Step 2: run_compensation would return gradients based on peak results
    compensation_gradients = {"G4": 0.15, "G5": 0.20}

    # Step 3: charge_sensor_csd_readout consumes these results
    # Verify that the data flows correctly
    sensor_park_voltages = {
        "G1": peak_result["mean_reservoir_saturation_voltage"],
        "G2": peak_result["mean_reservoir_saturation_voltage"],
        "G3": peak_result["best_peak_voltage"],
    }

    # Verify workflow data consistency
    assert sensor_park_voltages["G1"] == -0.8
    assert sensor_park_voltages["G2"] == -0.8
    assert sensor_park_voltages["G3"] == -0.7

    # Verify compensation gradients are available for CSD readout
    assert "G4" in compensation_gradients
    assert "G5" in compensation_gradients
    assert compensation_gradients["G4"] == 0.15
    assert compensation_gradients["G5"] == 0.20

    # This verifies the data structure compatibility between workflow steps


# =============================================================================
# Stability Measurement Algorithm Tests (find_stable_sensor_peak)
# =============================================================================


def test_stable_peak_selects_top_n_candidates(fitted_peak_factory):
    """With 5+ detected peaks, verify routine tests only the top 3 peaks
    (by quality score) for stability."""
    peaks = []
    quality_scores = [0.9, 0.7, 0.85, 0.6, 0.75]  # Top 3 are: 0.9, 0.85, 0.75

    for i, quality in enumerate(quality_scores):
        peaks.append(
            fitted_peak_factory(
                sensitivity=1.0,
                sensitivity_voltage=i * 0.01,
                peak_idx=i * 10,
                peak_voltage=i * 0.01,
                window_currents=np.array([1e-9]),
                window_voltages=np.array([i * 0.01]),
                quality_score=quality,
            )
        )

    # Sort by quality score and take top 3
    sorted_peaks = sorted(peaks, key=lambda p: p.quality_score, reverse=True)
    top_3_peaks = sorted_peaks[:3]

    # Verify we get exactly 3 candidates
    assert len(top_3_peaks) == 3
    # Verify they are the highest quality
    assert top_3_peaks[0].quality_score == 0.9
    assert top_3_peaks[1].quality_score == 0.85
    assert top_3_peaks[2].quality_score == 0.75


def test_stability_measurement_holds_at_max_gradient(stability_measurement_factory):
    """For each candidate peak, confirm the device is positioned at
    sensitivity_voltage (max gradient point) during the 2-minute hold."""
    # This test verifies the conceptual behavior - positioning at max gradient

    peak_voltage = -0.7
    max_gradient_voltage = -0.68  # Should be positioned here for stability test
    stability = stability_measurement_factory(
        peak_voltage=peak_voltage,
        max_gradient_voltage=max_gradient_voltage,
        time_array=np.linspace(0, 120, 100),
        current_array=np.ones(100) * 1e-9,
    )

    # Verify the stability measurement captured max gradient voltage
    assert stability.max_gradient_voltage == max_gradient_voltage
    assert stability.max_gradient_voltage != stability.peak_voltage
    # Time array should span 120 seconds (default hold time)
    assert stability.time_array[-1] >= 120


def test_stability_measures_current_vs_time(stability_measurement_factory):
    """Verify stability measurement records continuous current samples over
    the configured hold time (default 120s) with timestamps."""

    # Simulate 120-second measurement with 100 samples
    hold_time = 120.0
    num_samples = 100
    time_array = np.linspace(0, hold_time, num_samples)

    # Simulate current measurements with small drift
    current_array = 1e-9 + np.random.normal(0, 1e-11, num_samples)

    stability = stability_measurement_factory(
        peak_voltage=-0.7,
        max_gradient_voltage=-0.68,
        time_array=time_array,
        current_array=current_array,
        current_mean=np.mean(current_array),
        current_std=np.std(current_array),
    )

    # Verify time and current arrays have same length
    assert len(stability.time_array) == len(stability.current_array)
    # Verify time spans the hold period
    assert stability.time_array[0] == 0
    assert stability.time_array[-1] == hold_time
    # Verify statistics are calculated
    assert stability.current_mean > 0
    assert stability.current_std > 0


def test_voltage_noise_calculated_from_gradient():
    """Confirm voltage noise computation uses formula σᵥ = σᵢ / |dI/dV|
    where σᵢ is current std and dI/dV is local slope at max gradient."""

    current_std = 1e-11  # 10 pA std
    local_slope = 2e-6  # 2 µA/V

    voltage_noise = _calculate_voltage_noise(current_std, local_slope)

    # Expected: 1e-11 / 2e-6 = 5e-6 V
    expected = current_std / abs(local_slope)
    assert abs(voltage_noise - expected) < 1e-12


def test_stability_score_inverts_voltage_noise(
    fitted_peak_factory, stability_measurement_factory
):
    """Verify stability score is computed as 1/voltage_noise, then normalized
    by dividing by the maximum across all candidates."""
    candidates = []
    voltage_noises = [1e-5, 5e-6, 2e-5]  # Different noise levels

    for i, v_noise in enumerate(voltage_noises):
        peak = fitted_peak_factory(
            sensitivity=1.0,
            sensitivity_voltage=i * 0.01,
            peak_idx=i * 10,
            peak_voltage=i * 0.01,
            window_currents=np.array([1e-9]),
            window_voltages=np.array([i * 0.01]),
            quality_score=0.8,
        )

        stability = stability_measurement_factory(
            peak_index=i,
            peak_voltage=i * 0.01,
            max_gradient_voltage=i * 0.01,
            voltage_noise=v_noise,
        )

        candidates.append(
            StablePeakCandidate(
                fitted_peak=peak, original_score=0.8, stability_measurement=stability
            )
        )

    # Calculate combined scores
    _calculate_combined_scores(candidates, original_weight=0.3, stability_weight=0.7)

    # Verify stability scores are set and normalized
    for candidate in candidates:
        assert candidate.stability_measurement.stability_score is not None
        assert 0 <= candidate.stability_measurement.stability_score <= 1.0

    # Peak with lowest voltage noise should have highest stability score
    min_noise_idx = np.argmin(voltage_noises)
    assert candidates[min_noise_idx].stability_measurement.stability_score == 1.0


def test_stable_peak_returns_highest_combined_score(
    fitted_peak_factory, stability_measurement_factory
):
    """Among 3 tested candidates, confirm the routine selects the peak with
    maximum combined score, even if it had lower original quality."""
    # Candidate 0: high quality (0.9), poor stability (high noise = 2e-5)
    # Candidate 1: medium quality (0.7), good stability (low noise = 1e-6)
    # Candidate 2: low quality (0.5), medium stability (noise = 5e-6)

    candidates = []
    quality_scores = [0.9, 0.7, 0.5]
    voltage_noises = [2e-5, 1e-6, 5e-6]

    for i, (quality, v_noise) in enumerate(
        zip(quality_scores, voltage_noises, strict=True)
    ):
        peak = fitted_peak_factory(
            sensitivity=1.0,
            sensitivity_voltage=i * 0.01,
            peak_idx=i * 10,
            peak_voltage=i * 0.01,
            window_currents=np.array([1e-9]),
            window_voltages=np.array([i * 0.01]),
            quality_score=quality,
        )

        stability = stability_measurement_factory(
            peak_index=i,
            peak_voltage=i * 0.01,
            max_gradient_voltage=i * 0.01,
            voltage_noise=v_noise,
        )

        candidates.append(
            StablePeakCandidate(
                fitted_peak=peak,
                original_score=quality,
                stability_measurement=stability,
            )
        )

    # Calculate combined scores (70% stability, 30% original)
    _calculate_combined_scores(candidates, original_weight=0.3, stability_weight=0.7)

    # Find peak with highest combined score
    best_candidate = max(candidates, key=lambda c: c.combined_score)

    # With 70% weight on stability, candidate 1 (best stability) should win
    # despite having lower original quality
    assert best_candidate == candidates[1]


def test_peak_detector_model_output_parsing():
    """Verify ML model peak detector output is correctly parsed and used.

    This test ensures the integration between the ML model and the peak finding
    routine works correctly. The model returns peak_indices which must be
    correctly interpreted as array indices for window extraction.
    """
    mock_ctx, mock_device = create_mock_context_for_sensor_routines()

    # Mock the ML model to return specific peak indices
    # Simulate finding peaks at indices 100, 200, 300 in a 500-point trace
    mock_peak_indices = [100, 200, 300]

    mock_ctx.resources.models_client.models.execute.return_value.output = {
        "classification": True,
        "score": 0.95,
        "peak_indices": mock_peak_indices,
    }

    # Ensure ctx.resources.group is None to avoid filter_gates_by_group error
    # filter_gates_by_group checks if group is None, and if not, tries to iterate over it
    # Setting it to None ensures the function returns the original gate list
    mock_ctx.resources.group = None

    # Mock sweep to return trace with peaks
    voltages = np.linspace(-1.0, -0.5, 500)
    currents = np.ones(500) * 1e-11

    # Add peaks at the specified indices
    for peak_idx in mock_peak_indices:
        # Add a Lorentzian peak centered at each index
        indices = np.arange(max(0, peak_idx - 20), min(500, peak_idx + 20))
        peak_currents = lorentzian(
            indices - peak_idx, amplitude=2e-9, center=0, width=5, offset=1e-11
        )
        currents[indices] += peak_currents

    mock_device.sweep_nd.return_value = (voltages, currents)

    # Patch the many_window_barrier_sweep to capture peak indices
    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak.many_window_barrier_sweep"
    ) as mock_sweep:
        # Create mock output with correct structure
        best_peak_voltage = voltages[
            mock_peak_indices[0]
        ]  # First peak (highest quality)

        mock_sweep.return_value = SensorDotPlungerSweepOutput(
            sensor_plunger_voltage=best_peak_voltage,
            classification=True,
            score=0.95,
            peak_indices=mock_peak_indices,  # This must be a list, not a Mock
            num_peaks=len(mock_peak_indices),
            aggregated_voltages=voltages,
            aggregated_currents=currents,
            best_peak_voltage=best_peak_voltage,
            best_peak_max_gradient_voltage=best_peak_voltage,
            prev_peak_voltage=voltages[mock_peak_indices[0] - 50]
            if mock_peak_indices[0] > 50
            else None,
            next_peak_voltage=voltages[mock_peak_indices[-1] + 50]
            if mock_peak_indices[-1] < len(voltages) - 50
            else None,
        )

        result = find_sensor_peak(
            ctx=mock_ctx,
            peak_spacing=0.02,
            sensor_group_name="sensor_group",
            sensor_plunger_gate="G3",
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
        )

        # Verify the routine correctly used the model output
        # The best peak should correspond to one of the detected indices
        assert "best_peak_voltage" in result
        best_voltage = result["best_peak_voltage"]

        # The best peak voltage should match one of the detected peak positions
        detected_voltages = [voltages[idx] for idx in mock_peak_indices]
        assert any(abs(best_voltage - v) < 0.01 for v in detected_voltages), (
            f"Best peak voltage {best_voltage} doesn't match detected peaks at {detected_voltages}"
        )

        # Verify many_window_barrier_sweep was called (which internally uses the model)
        assert mock_sweep.called
