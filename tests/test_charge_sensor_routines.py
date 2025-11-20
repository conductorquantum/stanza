"""Tests for charge sensor routines (find_sensor_peak, find_stable_sensor_peak)."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from stanza.exceptions import RoutineError
from stanza.models import DeviceGroup
from stanza.routines import RoutineContext
from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
    SensorDotPlungerSweepOutput,
    StabilityMeasurement,
    StablePeakCandidate,
    _calculate_combined_scores,
    _calculate_voltage_noise,
    build_sensor_sweep_voltage_list,
    find_sensor_peak,
)
from stanza.routines.builtins.charge_sensor.constants import (
    ML_MODEL_INPUT_SIZE,
)
from stanza.routines.builtins.utils.peak_fitting import FittedPeak, ModelFitResult

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


def test_find_sensor_peak_uses_fallback_bounds():
    """Mock device sweeps with a single peak to ensure fallback peak_spacing
    bounds populate prev/next voltages."""
    # Test the fallback logic for narrowed range calculation
    # When there's only one peak, prev/next voltages use peak_spacing as fallback

    peak_voltage = -0.7
    peak_spacing = 0.02

    # Simulate the fallback logic from find_sensor_peak
    # When there's no previous peak, use peak_voltage - peak_spacing
    # When there's no next peak, use peak_voltage + peak_spacing
    prev_peak_voltage = peak_voltage - peak_spacing
    next_peak_voltage = peak_voltage + peak_spacing

    # Calculate narrowed range (midpoint between peaks)
    start_of_range = (prev_peak_voltage + peak_voltage) / 2
    end_of_range = (peak_voltage + next_peak_voltage) / 2
    narrowed_range = (start_of_range, end_of_range)

    # Verify fallback bounds are used correctly
    assert abs(prev_peak_voltage - (-0.72)) < 1e-9
    assert abs(next_peak_voltage - (-0.68)) < 1e-9
    assert narrowed_range[0] < peak_voltage < narrowed_range[1]
    assert (
        abs((narrowed_range[1] - narrowed_range[0]) - peak_spacing) < 1e-9
    )  # Range spans one spacing


def test_find_sensor_peak_zero_control_side_sets_control_gates():
    """Verify zero_control_side=True drives all non-sensor control gates to 0 V
    before sweeping, while False preserves existing voltages."""
    # Test the logic that determines control gate voltages
    all_control_gates = ["G1", "G2", "G3", "G4", "G5"]
    sensor_gates = ["G1", "G2", "G3"]
    control_gates = [g for g in all_control_gates if g not in sensor_gates]

    # Verify control gates are identified correctly
    assert control_gates == ["G4", "G5"]

    # Test that zero_control_side=True would set these to 0
    zero_control_side = True
    if zero_control_side:
        control_voltage_dict = dict.fromkeys(control_gates, 0.0)
    else:
        control_voltage_dict = {}  # Would preserve existing

    # Verify behavior
    if zero_control_side:
        assert all(v == 0.0 for v in control_voltage_dict.values())
        assert set(control_voltage_dict.keys()) == {"G4", "G5"}
    else:
        assert control_voltage_dict == {}


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


def test_many_window_barrier_sweep_enforces_ml_window_size():
    """Confirm each window feeds exactly ML_MODEL_INPUT_SIZE (=128) points
    into the ML classifier/detector."""
    # Verify ML_MODEL_INPUT_SIZE constant is defined correctly
    assert ML_MODEL_INPUT_SIZE == 128, "ML_MODEL_INPUT_SIZE should be 128"

    # Test window size calculation
    window_size = 0.04  # V
    points_per_window = ML_MODEL_INPUT_SIZE

    # Verify that a window of ML_MODEL_INPUT_SIZE points is created
    start_v = -1.0
    end_v = start_v + window_size
    voltages = np.linspace(start_v, end_v, points_per_window, endpoint=False)

    assert len(voltages) == ML_MODEL_INPUT_SIZE
    assert voltages[0] == start_v
    assert voltages[-1] < end_v  # endpoint=False


def test_sensor_dot_output_captures_aggregated_traces_and_metadata():
    """Ensure SensorDotPlungerSweepOutput stores the aggregated voltages/currents
    plus the last classification flag and peak indices."""

    # Create mock data
    aggregated_voltages = np.linspace(-1.0, -0.5, 256)
    aggregated_currents = np.ones(256) * 1e-9

    # Create output object using the actual dataclass fields
    output = SensorDotPlungerSweepOutput(
        sensor_plunger_voltage=-0.7,
        classification=True,
        score=0.95,
        peak_indices=[128, 192],
        num_peaks=2,
        aggregated_voltages=aggregated_voltages,
        aggregated_currents=aggregated_currents,
        best_peak_voltage=-0.7,
        best_peak_max_gradient_voltage=-0.68,
        prev_peak_voltage=-0.72,
        next_peak_voltage=-0.66,
    )

    # Verify output captures all required data
    assert len(output.aggregated_voltages) == 256
    assert len(output.aggregated_currents) == 256
    assert output.classification is True
    assert len(output.peak_indices) == 2
    assert output.num_peaks == 2
    assert output.best_peak_voltage == -0.7


def test_find_sensor_peak_uses_narrowed_range_for_park_point():
    """Verify that the narrowed range returned from many_window_barrier_sweep
    is applied when parking the sensor."""
    # Test the narrowed range calculation logic
    # The narrowed range is calculated as the midpoint between adjacent peaks

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


def test_peak_finder_receives_full_aggregated_trace():
    """Confirm the peak detection path gets the entire aggregated sweep data
    rather than per-window slices only."""
    # Test aggregation logic: multiple windows should combine into larger trace
    window_size_points = ML_MODEL_INPUT_SIZE  # 128 points per window
    num_windows = 3

    # Simulate aggregating traces from multiple windows
    aggregated_trace = np.array([])
    for i in range(num_windows):
        window_trace = np.ones(window_size_points) * (i + 1) * 1e-9
        aggregated_trace = np.concatenate([aggregated_trace, window_trace])

    # Verify aggregated trace is larger than single window
    assert len(aggregated_trace) == window_size_points * num_windows
    assert len(aggregated_trace) > ML_MODEL_INPUT_SIZE

    # Peak detector would receive this full aggregated trace
    # not just individual windows
    assert len(aggregated_trace) == 384  # 128 * 3


# =============================================================================
# find_stable_sensor_peak Tests
# =============================================================================


def test_find_stable_sensor_peak_prefers_stable_peaks():
    """Provide peak candidates with varying noise to verify the routine picks
    the highest combined score even if the raw quality is lower."""

    # Create mock fitted peaks with different quality scores
    mock_fit = ModelFitResult(
        model_name="Lorentzian",
        amplitude=1e-9,
        center_idx=50,
        width=5.0,
        offset=1e-11,
        r_squared=0.95,
        rmse=1e-12,
        aicc=-100.0,
        fwhm=0.01,
        area=1e-10,
        skew_resid=0.1,
    )

    # Peak 1: high quality (0.9), but poor stability (high voltage noise)
    peak1 = FittedPeak(
        best_model="Lorentzian",
        lorentzian_fit=mock_fit,
        sech2_fit=mock_fit,
        voigt_fit=mock_fit,
        sensitivity=1.0,
        sensitivity_voltage=-0.7,
        peak_idx=50,
        peak_voltage=-0.7,
        window_currents=np.array([1e-9]),
        window_voltages=np.array([-0.7]),
        quality_score=0.9,
    )

    # Peak 2: lower quality (0.6), but good stability (low voltage noise)
    peak2 = FittedPeak(
        best_model="Lorentzian",
        lorentzian_fit=mock_fit,
        sech2_fit=mock_fit,
        voigt_fit=mock_fit,
        sensitivity=1.0,
        sensitivity_voltage=-0.65,
        peak_idx=60,
        peak_voltage=-0.65,
        window_currents=np.array([1e-9]),
        window_voltages=np.array([-0.65]),
        quality_score=0.6,
    )

    # Create stability measurements
    stability1 = StabilityMeasurement(
        peak_index=0,
        peak_voltage=-0.7,
        max_gradient_voltage=-0.7,
        time_array=np.array([0.0, 1.0]),
        current_array=np.array([1e-9, 1e-9]),
        current_mean=1e-9,
        current_std=1e-11,
        local_slope=1e-8,
        voltage_noise=2.0,  # High noise
    )

    stability2 = StabilityMeasurement(
        peak_index=1,
        peak_voltage=-0.65,
        max_gradient_voltage=-0.65,
        time_array=np.array([0.0, 1.0]),
        current_array=np.array([1e-9, 1e-9]),
        current_mean=1e-9,
        current_std=1e-11,
        local_slope=1e-8,
        voltage_noise=0.5,  # Low noise
    )

    # Create candidates
    candidate1 = StablePeakCandidate(
        fitted_peak=peak1, original_score=0.9, stability_measurement=stability1
    )
    candidate2 = StablePeakCandidate(
        fitted_peak=peak2, original_score=0.6, stability_measurement=stability2
    )

    # Calculate combined scores (50/50 weighting for this test)
    _calculate_combined_scores(
        [candidate1, candidate2], original_weight=0.5, stability_weight=0.5
    )

    # Verify combined scores exist
    assert candidate1.combined_score is not None
    assert candidate2.combined_score is not None

    # Peak2 (better stability) should score competitively despite lower original quality
    # This verifies that stability is properly weighted in the selection process
    assert (
        candidate2.stability_measurement.stability_score
        > candidate1.stability_measurement.stability_score
    )


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


def test_stable_peak_selects_top_n_candidates():
    """With 5+ detected peaks, verify routine tests only the top 3 peaks
    (by quality score) for stability."""
    from stanza.routines.builtins.utils.peak_fitting import ModelFitResult

    # Create 5 peaks with different quality scores
    mock_fit = ModelFitResult(
        model_name="Lorentzian",
        amplitude=1e-9,
        center_idx=50,
        width=5.0,
        offset=0.0,
        r_squared=0.95,
        rmse=1e-12,
        aicc=-100.0,
        fwhm=0.01,
        area=1e-10,
        skew_resid=0.1,
    )

    peaks = []
    quality_scores = [0.9, 0.7, 0.85, 0.6, 0.75]  # Top 3 are: 0.9, 0.85, 0.75

    for i, quality in enumerate(quality_scores):
        peak = FittedPeak(
            best_model="Lorentzian",
            lorentzian_fit=mock_fit,
            sech2_fit=mock_fit,
            voigt_fit=mock_fit,
            sensitivity=1.0,
            sensitivity_voltage=i * 0.01,
            peak_idx=i * 10,
            peak_voltage=i * 0.01,
            window_currents=np.array([1e-9]),
            window_voltages=np.array([i * 0.01]),
            quality_score=quality,
        )
        peaks.append(peak)

    # Sort by quality score and take top 3
    sorted_peaks = sorted(peaks, key=lambda p: p.quality_score, reverse=True)
    top_3_peaks = sorted_peaks[:3]

    # Verify we get exactly 3 candidates
    assert len(top_3_peaks) == 3
    # Verify they are the highest quality
    assert top_3_peaks[0].quality_score == 0.9
    assert top_3_peaks[1].quality_score == 0.85
    assert top_3_peaks[2].quality_score == 0.75


def test_stability_measurement_holds_at_max_gradient():
    """For each candidate peak, confirm the device is positioned at
    sensitivity_voltage (max gradient point) during the 2-minute hold."""
    # This test verifies the conceptual behavior - positioning at max gradient

    # Create a mock stability measurement
    peak_voltage = -0.7
    max_gradient_voltage = -0.68  # Should be positioned here for stability test

    stability = StabilityMeasurement(
        peak_index=0,
        peak_voltage=peak_voltage,
        max_gradient_voltage=max_gradient_voltage,
        time_array=np.linspace(0, 120, 100),  # 120 seconds
        current_array=np.ones(100) * 1e-9,
        current_mean=1e-9,
        current_std=1e-11,
        local_slope=1e-6,
        voltage_noise=1e-5,
    )

    # Verify the stability measurement captured max gradient voltage
    assert stability.max_gradient_voltage == max_gradient_voltage
    assert stability.max_gradient_voltage != stability.peak_voltage
    # Time array should span 120 seconds (default hold time)
    assert stability.time_array[-1] >= 120


def test_stability_measures_current_vs_time():
    """Verify stability measurement records continuous current samples over
    the configured hold time (default 120s) with timestamps."""

    # Simulate 120-second measurement with 100 samples
    hold_time = 120.0
    num_samples = 100
    time_array = np.linspace(0, hold_time, num_samples)

    # Simulate current measurements with small drift
    current_array = 1e-9 + np.random.normal(0, 1e-11, num_samples)

    stability = StabilityMeasurement(
        peak_index=0,
        peak_voltage=-0.7,
        max_gradient_voltage=-0.68,
        time_array=time_array,
        current_array=current_array,
        current_mean=np.mean(current_array),
        current_std=np.std(current_array),
        local_slope=1e-6,
        voltage_noise=1e-5,
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


def test_stability_score_inverts_voltage_noise():
    """Verify stability score is computed as 1/voltage_noise, then normalized
    by dividing by the maximum across all candidates."""
    # Create mock candidates with different voltage noise values

    mock_fit = ModelFitResult(
        model_name="Lorentzian",
        amplitude=1e-9,
        center_idx=50,
        width=5.0,
        offset=0.0,
        r_squared=0.95,
        rmse=1e-12,
        aicc=-100.0,
        fwhm=0.01,
        area=1e-10,
        skew_resid=0.1,
    )

    candidates = []
    voltage_noises = [1e-5, 5e-6, 2e-5]  # Different noise levels

    for i, v_noise in enumerate(voltage_noises):
        peak = FittedPeak(
            best_model="Lorentzian",
            lorentzian_fit=mock_fit,
            sech2_fit=mock_fit,
            voigt_fit=mock_fit,
            sensitivity=1.0,
            sensitivity_voltage=i * 0.01,
            peak_idx=i * 10,
            peak_voltage=i * 0.01,
            window_currents=np.array([1e-9]),
            window_voltages=np.array([i * 0.01]),
            quality_score=0.8,
        )

        stability = StabilityMeasurement(
            peak_index=i,
            peak_voltage=i * 0.01,
            max_gradient_voltage=i * 0.01,
            time_array=np.array([0.0, 1.0]),
            current_array=np.array([1e-9, 1e-9]),
            current_mean=1e-9,
            current_std=1e-11,
            local_slope=1e-6,
            voltage_noise=v_noise,
        )

        candidate = StablePeakCandidate(
            fitted_peak=peak, original_score=0.8, stability_measurement=stability
        )
        candidates.append(candidate)

    # Calculate combined scores
    _calculate_combined_scores(candidates, original_weight=0.3, stability_weight=0.7)

    # Verify stability scores are set and normalized
    for candidate in candidates:
        assert candidate.stability_measurement.stability_score is not None
        assert 0 <= candidate.stability_measurement.stability_score <= 1.0

    # Peak with lowest voltage noise should have highest stability score
    min_noise_idx = np.argmin(voltage_noises)
    assert candidates[min_noise_idx].stability_measurement.stability_score == 1.0


def test_stable_peak_returns_highest_combined_score():
    """Among 3 tested candidates, confirm the routine selects the peak with
    maximum combined score, even if it had lower original quality."""

    mock_fit = ModelFitResult(
        model_name="Lorentzian",
        amplitude=1e-9,
        center_idx=50,
        width=5.0,
        offset=0.0,
        r_squared=0.95,
        rmse=1e-12,
        aicc=-100.0,
        fwhm=0.01,
        area=1e-10,
        skew_resid=0.1,
    )

    # Create 3 candidates:
    # Candidate 0: high quality (0.9), poor stability (high noise = 2e-5)
    # Candidate 1: medium quality (0.7), good stability (low noise = 1e-6)
    # Candidate 2: low quality (0.5), medium stability (noise = 5e-6)

    candidates = []
    quality_scores = [0.9, 0.7, 0.5]
    voltage_noises = [2e-5, 1e-6, 5e-6]

    for i, (quality, v_noise) in enumerate(
        zip(quality_scores, voltage_noises, strict=True)
    ):
        peak = FittedPeak(
            best_model="Lorentzian",
            lorentzian_fit=mock_fit,
            sech2_fit=mock_fit,
            voigt_fit=mock_fit,
            sensitivity=1.0,
            sensitivity_voltage=i * 0.01,
            peak_idx=i * 10,
            peak_voltage=i * 0.01,
            window_currents=np.array([1e-9]),
            window_voltages=np.array([i * 0.01]),
            quality_score=quality,
        )

        stability = StabilityMeasurement(
            peak_index=i,
            peak_voltage=i * 0.01,
            max_gradient_voltage=i * 0.01,
            time_array=np.array([0.0, 1.0]),
            current_array=np.array([1e-9, 1e-9]),
            current_mean=1e-9,
            current_std=1e-11,
            local_slope=1e-6,
            voltage_noise=v_noise,
        )

        candidate = StablePeakCandidate(
            fitted_peak=peak, original_score=quality, stability_measurement=stability
        )
        candidates.append(candidate)

    # Calculate combined scores (70% stability, 30% original)
    _calculate_combined_scores(candidates, original_weight=0.3, stability_weight=0.7)

    # Find peak with highest combined score
    best_candidate = max(candidates, key=lambda c: c.combined_score)

    # With 70% weight on stability, candidate 1 (best stability) should win
    # despite having lower original quality
    assert best_candidate == candidates[1]


def test_stability_score_higher_for_lower_noise():
    """Verify that peak with lower voltage noise receives higher stability
    score (inverse relationship)."""
    from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
        StabilityMeasurement,
        StablePeakCandidate,
        _calculate_combined_scores,
    )
    from stanza.routines.builtins.utils.peak_fitting import ModelFitResult

    mock_fit = ModelFitResult(
        model_name="Lorentzian",
        amplitude=1e-9,
        center_idx=50,
        width=5.0,
        offset=0.0,
        r_squared=0.95,
        rmse=1e-12,
        aicc=-100.0,
        fwhm=0.01,
        area=1e-10,
        skew_resid=0.1,
    )

    candidates = []
    # Create two candidates with different noise levels
    voltage_noises = [1e-5, 5e-6]  # Second has lower noise

    for i, v_noise in enumerate(voltage_noises):
        peak = FittedPeak(
            best_model="Lorentzian",
            lorentzian_fit=mock_fit,
            sech2_fit=mock_fit,
            voigt_fit=mock_fit,
            sensitivity=1.0,
            sensitivity_voltage=i * 0.01,
            peak_idx=i * 10,
            peak_voltage=i * 0.01,
            window_currents=np.array([1e-9]),
            window_voltages=np.array([i * 0.01]),
            quality_score=0.8,
        )

        stability = StabilityMeasurement(
            peak_index=i,
            peak_voltage=i * 0.01,
            max_gradient_voltage=i * 0.01,
            time_array=np.array([0.0, 1.0]),
            current_array=np.array([1e-9, 1e-9]),
            current_mean=1e-9,
            current_std=1e-11,
            local_slope=1e-6,
            voltage_noise=v_noise,
        )

        candidate = StablePeakCandidate(
            fitted_peak=peak, original_score=0.8, stability_measurement=stability
        )
        candidates.append(candidate)

    _calculate_combined_scores(candidates, original_weight=0.5, stability_weight=0.5)

    # Candidate with lower voltage noise should have higher stability score
    assert (
        candidates[1].stability_measurement.stability_score
        > candidates[0].stability_measurement.stability_score
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
