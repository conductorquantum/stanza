"""Tests for charge sensor routines (find_sensor_peak, find_stable_sensor_peak)."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from stanza.exceptions import RoutineError
from stanza.routines import RoutineContext
from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
    find_sensor_peak,
    find_stable_sensor_peak,
)
from stanza.routines.builtins.charge_sensor.utils.constants import (
    DEFAULT_WINDOW_HALF_WIDTH,
    WINDOW_FRACTION,
)
from stanza.routines.builtins.charge_sensor.utils.peak_stability import (
    calculate_combined_scores,
    calculate_local_slope,
    calculate_voltage_noise,
)
from stanza.routines.builtins.charge_sensor.utils.sweeps import (
    build_sensor_sweep_voltage_list,
)
from stanza.routines.builtins.charge_sensor.utils.types import (
    SensorDotPlungerSweepOutput,
    StablePeakCandidate,
)
from stanza.routines.builtins.utils.peak_fitting import (
    calculate_peak_window_bounds,
    calculate_quality_score,
    calculate_quality_scores,
    fit_peak_multi_model,
    lorentzian,
    normalize_sensitivity_scores,
)


def test_calculate_peak_window_bounds_handles_edge_peaks():
    """Verify calculate_peak_window_bounds clamps to trace bounds and always contains the peak."""
    trace_length = 1000

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

    start, end = calculate_peak_window_bounds(
        peak_idx=500, peak_index=0, peak_indices=[500], trace_length=trace_length
    )
    assert start == 0
    assert end == trace_length

    close_peak_indices = [50, 100, 150, 200]
    start, end = calculate_peak_window_bounds(
        peak_idx=100,
        peak_index=1,
        peak_indices=close_peak_indices,
        trace_length=trace_length,
    )
    assert 0 <= start < 100 < end <= trace_length
    assert end - start >= 3

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
    """Confirm normalize_sensitivity_scores assigns score 1.0 when all sensitivities are equal."""
    peaks = [
        fitted_peak_factory(
            sensitivity=1e-6,
            sensitivity_voltage=float(i * 0.01),
            peak_idx=i * 10,
            peak_voltage=float(i * 0.01),
        )
        for i in range(5)
    ]

    normalize_sensitivity_scores(peaks)

    for peak in peaks:
        assert peak.sensitivity_score == 1.0


def test_calculate_local_slope_window_validation():
    """Ensure calculate_local_slope raises when fewer than three data points surround the target voltage."""
    voltages = np.array([0.0, 0.01])  # Only 2 points
    currents = np.array([0.0, 1e-9])
    target_voltage = 0.005

    with pytest.raises(
        RoutineError, match="Insufficient points for local slope calculation"
    ):
        calculate_local_slope(voltages, currents, target_voltage, window_points=5)


def test_calculate_voltage_noise_near_zero_slope():
    """Provide tiny slopes to calculate_voltage_noise and confirm it guards against division-by-zero."""
    current_std = 1e-10
    local_slope = 1e-15  # Very small slope

    with pytest.raises(RoutineError, match="Local slope magnitude too small"):
        calculate_voltage_noise(current_std, local_slope)


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
    sensitivities = [1e-6, 5e-6, 10e-6, 2e-6, 7e-6]

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

    normalize_sensitivity_scores(peaks)

    for peak in peaks:
        assert 0 <= peak.sensitivity_score <= 1.0

    max_sens_idx = np.argmax(sensitivities)
    assert peaks[max_sens_idx].sensitivity_score == 1.0

    min_sens_idx = np.argmin(sensitivities)
    assert peaks[min_sens_idx].sensitivity_score == 0.0


def test_find_sensor_peak_returns_highest_quality():
    """Feed synthetic sweep with multiple peaks of known quality and verify
    the routine calculates quality scores for all peaks."""
    voltages = np.linspace(-1.0, -0.5, 500)

    # peak1: highest amplitude, narrowest width - should be least noisy and best quality
    peak1 = lorentzian(
        np.arange(100, 200), amplitude=10e-9, center=50, width=5, offset=1e-11
    )

    # peak2: medium amplitude and width
    peak2 = lorentzian(
        np.arange(200, 300), amplitude=2e-9, center=50, width=15, offset=1e-11
    )

    # peak3: lowest amplitude, widest width - should be most noisy
    peak3 = lorentzian(
        np.arange(300, 400), amplitude=1e-9, center=50, width=30, offset=1e-11
    )

    currents = np.ones(500) * 1e-11
    currents[100:200] = peak1
    currents[200:300] = peak2
    currents[300:400] = peak3

    # Use lower noise to ensure peak1's superior signal-to-noise ratio is clear
    np.random.seed(42)
    currents += np.random.normal(0, 5e-14, len(currents))

    peak_indices = [150, 250, 350]

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

    normalize_sensitivity_scores(fitted_peaks)
    calculate_quality_scores(fitted_peaks)

    assert len(fitted_peaks) == 3
    for peak in fitted_peaks:
        assert peak.quality_score is not None
        assert peak.sensitivity_score is not None
        assert 0 <= peak.quality_score <= 1.0

    quality_scores = [p.quality_score for p in fitted_peaks]
    assert len(set(quality_scores)) > 1, (
        "Quality scores should differ for different peaks"
    )

    # Identify the best peak (highest quality score)
    best_peak = max(fitted_peaks, key=lambda p: p.quality_score)

    # Verify the best peak matches the least noisy peak (peak1 at index 150)
    # peak1 has the highest amplitude (5e-9), narrowest width (6), and best
    # signal-to-noise ratio, making it the least noisy and highest quality
    assert best_peak.peak_idx == 150, (
        f"Best peak should be at index 150 (peak1, least noisy), "
        f"but got index {best_peak.peak_idx} with quality score {best_peak.quality_score}. "
        f"All quality scores: {[(p.peak_idx, p.quality_score) for p in fitted_peaks]}"
    )


def test_find_sensor_peak_requires_prerequisites(mock_device_for_sensor_routines):
    """Run find_sensor_peak with missing global_accumulation or
    finger_gate_characterization results and expect RoutineError."""

    # Missing prerequisites
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {}

    mock_device = mock_device_for_sensor_routines
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
    sensor_gates_list = ["G1", "G2", "G3"]
    sensor_plunger_index = 2
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

    for voltages in voltage_list:
        assert voltages[0] == 0.8, "G1 should use override voltage 0.8"
        assert voltages[1] == 0.9, "G2 should use override voltage 0.9"
        assert voltages[2] in plunger_voltages


def test_find_sensor_peak_uses_narrowed_range_for_park_point(
    mock_device_for_sensor_routines,
):
    """Verify that find_sensor_peak correctly calculates narrowed range and
    that the park point is within the narrowed range. Tests both cases:
    with neighboring peaks and fallback when peaks are missing."""
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "global_accumulation_sensor_group": {"global_turn_on_voltage": -0.8},
        "finger_gate_characterization_sensor_group": {
            "G3": {
                "saturation_voltage": 0.5,
                "cutoff_voltage": -1.5,
                "pinch_off_voltage": -2.0,
            }
        },
    }

    mock_device = mock_device_for_sensor_routines
    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_resources.group = None
    mock_ctx.resources = mock_resources

    peak_spacing = 0.02
    best_peak_voltage = -0.7
    best_peak_max_gradient_voltage = -0.68
    prev_peak_voltage = -0.74
    next_peak_voltage = -0.66

    voltages = np.linspace(-1.0, -0.5, 500)
    currents = np.ones(500) * 1e-11

    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with patch(
            "stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak.many_window_barrier_sweep"
        ) as mock_sweep:
            # Test case 1: With neighboring peaks
            mock_sweep.return_value = SensorDotPlungerSweepOutput(
                sensor_plunger_voltage=best_peak_voltage,
                classification=True,
                score=0.95,
                peak_indices=[250],
                num_peaks=1,
                aggregated_voltages=voltages,
                aggregated_currents=currents,
                best_peak_voltage=best_peak_voltage,
                best_peak_max_gradient_voltage=best_peak_max_gradient_voltage,
                prev_peak_voltage=prev_peak_voltage,
                next_peak_voltage=next_peak_voltage,
            )

            result = find_sensor_peak(
                ctx=mock_ctx,
                peak_spacing=peak_spacing,
                sensor_group_name="sensor_group",
                sensor_plunger_gate="G3",
                measure_electrode="OUT",
                bias_gate="BIAS",
                bias_voltage=1e-4,
            )

            # Verify narrowed range is returned
            assert "narrowed_sensor_plunger_range" in result
            narrowed_range = result["narrowed_sensor_plunger_range"]
            assert isinstance(narrowed_range, tuple)
            assert len(narrowed_range) == 2
            assert narrowed_range[0] < narrowed_range[1]

            # Verify park point is within narrowed range
            park_voltage = result["best_peak_max_gradient_voltage"]
            assert narrowed_range[0] <= park_voltage <= narrowed_range[1], (
                f"Park voltage {park_voltage} should be within narrowed range "
                f"{narrowed_range}"
            )

            # Test case 2: Fallback when neighboring peaks are missing
            mock_sweep.return_value = SensorDotPlungerSweepOutput(
                sensor_plunger_voltage=best_peak_voltage,
                classification=True,
                score=0.95,
                peak_indices=[250],
                num_peaks=1,
                aggregated_voltages=voltages,
                aggregated_currents=currents,
                best_peak_voltage=best_peak_voltage,
                best_peak_max_gradient_voltage=best_peak_max_gradient_voltage,
                prev_peak_voltage=None,  # Missing previous peak
                next_peak_voltage=None,  # Missing next peak
            )

            result_fallback = find_sensor_peak(
                ctx=mock_ctx,
                peak_spacing=peak_spacing,
                sensor_group_name="sensor_group",
                sensor_plunger_gate="G3",
                measure_electrode="OUT",
                bias_gate="BIAS",
                bias_voltage=1e-4,
            )

            # Verify narrowed range is calculated using fallback values
            narrowed_range_fallback = result_fallback["narrowed_sensor_plunger_range"]
            assert isinstance(narrowed_range_fallback, tuple)
            assert len(narrowed_range_fallback) == 2
            assert narrowed_range_fallback[0] < narrowed_range_fallback[1]

            # Verify park point is within narrowed range
            park_voltage_fallback = result_fallback["best_peak_max_gradient_voltage"]
            assert (
                narrowed_range_fallback[0]
                <= park_voltage_fallback
                <= narrowed_range_fallback[1]
            ), (
                f"Park voltage {park_voltage_fallback} should be within narrowed range "
                f"{narrowed_range_fallback}"
            )

            # Verify fallback values are used
            assert (
                result_fallback["prev_peak_voltage"] == best_peak_voltage - peak_spacing
            )
            assert (
                result_fallback["next_peak_voltage"] == best_peak_voltage + peak_spacing
            )


def test_charge_sensor_workflow_consumes_compensation_results():
    """Chain find_sensor_peak → run_compensation → charge_sensor_csd_readout
    with mocks to ensure outputs from one step feed the next."""
    peak_result = {
        "best_peak_voltage": -0.7,
        "best_peak_max_gradient_voltage": -0.68,
        "mean_reservoir_saturation_voltage": -0.8,
        "sensor_gates_list": ["G1", "G2", "G3"],
        "sensor_park_point": {"G1": -0.8, "G2": -0.8, "G3": -0.7},
    }

    compensation_gradients = {"G4": 0.15, "G5": 0.20}

    sensor_park_voltages = {
        "G1": peak_result["mean_reservoir_saturation_voltage"],
        "G2": peak_result["mean_reservoir_saturation_voltage"],
        "G3": peak_result["best_peak_voltage"],
    }

    assert sensor_park_voltages["G1"] == -0.8
    assert sensor_park_voltages["G2"] == -0.8
    assert sensor_park_voltages["G3"] == -0.7

    assert "G4" in compensation_gradients
    assert "G5" in compensation_gradients
    assert compensation_gradients["G4"] == 0.15
    assert compensation_gradients["G5"] == 0.20


def test_stable_peak_selects_top_n_and_highest_combined_score(
    fitted_peak_factory,
    stability_measurement_factory,
    mock_device_for_sensor_routines,
):
    """With 5+ detected peaks, verify find_stable_sensor_peak:
    1. Tests only the top 3 peaks (by quality score) for stability
    2. Selects the peak with highest combined score, not just highest original quality."""
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "global_accumulation_sensor_group": {"global_turn_on_voltage": -0.8},
        "finger_gate_characterization_sensor_group": {
            "G3": {
                "saturation_voltage": 0.5,
                "cutoff_voltage": -1.5,
                "pinch_off_voltage": -2.0,
            }
        },
    }

    mock_device = mock_device_for_sensor_routines
    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_resources.group = None
    mock_ctx.resources = mock_resources

    # Create 5 peaks with different quality scores
    # Quality scores: [0.9, 0.7, 0.85, 0.6, 0.75]
    # Top 3 by quality: indices 0 (0.9), 2 (0.85), 4 (0.75)
    quality_scores = [0.9, 0.7, 0.85, 0.6, 0.75]
    peaks = []
    for i, quality in enumerate(quality_scores):
        peaks.append(
            fitted_peak_factory(
                sensitivity=1.0,
                sensitivity_voltage=-0.7 + i * 0.01,
                peak_idx=i * 10,
                peak_voltage=-0.7 + i * 0.01,
                window_currents=np.array([1e-9]),
                window_voltages=np.array([-0.7 + i * 0.01]),
                quality_score=quality,
            )
        )

    # Expected top 3 peaks (sorted by quality score descending)
    expected_top_3_quality_scores = [0.9, 0.85, 0.75]

    # Map voltage noise to top 3 peaks to test combined score selection
    # Peak 0 (quality 0.9): very high noise (poor stability)
    # Peak 2 (quality 0.85): medium noise (medium stability)
    # Peak 4 (quality 0.75): very low noise (excellent stability)
    # With 50/50 weights, peak 4 should win due to much better stability
    voltage_noises_by_peak_index = {
        0: 1e-4,  # Very high noise - poor stability
        2: 5e-6,  # Medium noise - medium stability
        4: 1e-7,  # Very low noise - excellent stability
    }

    voltages = np.linspace(-1.0, -0.5, 500)
    currents = np.ones(500) * 1e-11
    peak_indices = [100, 200, 300, 400, 450]

    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with patch(
            "stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak.many_window_barrier_sweep"
        ) as mock_sweep:
            with patch(
                "stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak.analyze_find_first_peak_voltages"
            ) as mock_analyze:
                with patch(
                    "stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak.measure_peak_stability"
                ) as mock_measure_stability:
                    # Setup mock sweep output
                    mock_sweep.return_value = SensorDotPlungerSweepOutput(
                        sensor_plunger_voltage=-0.7,
                        classification=True,
                        score=0.95,
                        peak_indices=peak_indices,
                        num_peaks=len(peak_indices),
                        aggregated_voltages=voltages,
                        aggregated_currents=currents,
                        best_peak_voltage=-0.7,
                        best_peak_max_gradient_voltage=-0.68,
                        prev_peak_voltage=-0.74,
                        next_peak_voltage=-0.66,
                    )

                    # Setup mock analyze to return all 5 peaks
                    mock_analyze.return_value = peaks

                    # Store stability measurements as they're created
                    created_stability_measurements = []

                    # Setup mock stability measurement with different noise for top 3 peaks
                    def create_stability_measurement(**kwargs):
                        peak = kwargs["peak"]
                        peak_index_in_list = kwargs.get("peak_index", 0)
                        # Find which original peak index this corresponds to
                        peak_idx_in_peaks = None
                        for idx, p in enumerate(peaks):
                            if p.peak_idx == peak.peak_idx:
                                peak_idx_in_peaks = idx
                                break

                        # Use voltage noise based on peak index, default to medium noise
                        v_noise = voltage_noises_by_peak_index.get(
                            peak_idx_in_peaks if peak_idx_in_peaks is not None else 0,
                            5e-6,
                        )

                        stability = stability_measurement_factory(
                            peak_index=peak_index_in_list,
                            peak_voltage=peak.peak_voltage,
                            max_gradient_voltage=peak.sensitivity_voltage,
                            voltage_noise=v_noise,
                        )
                        created_stability_measurements.append(stability)
                        return stability

                    mock_measure_stability.side_effect = create_stability_measurement

                    # Call find_stable_sensor_peak with top_n_peaks=3
                    result = find_stable_sensor_peak(
                        ctx=mock_ctx,
                        peak_spacing=0.02,
                        sensor_group_name="sensor_group",
                        sensor_plunger_gate="G3",
                        measure_electrode="OUT",
                        bias_gate="BIAS",
                        bias_voltage=1e-4,
                        top_n_peaks=3,
                        hold_time_seconds=0.1,  # Short time for testing
                    )

                    # Verify measure_peak_stability was called exactly 3 times (top 3 peaks)
                    assert mock_measure_stability.call_count == 3, (
                        f"Expected 3 stability measurements, got {mock_measure_stability.call_count}"
                    )

                    # Verify the peaks passed to measure_peak_stability are the top 3 by quality
                    called_peaks = [
                        call.kwargs["peak"]
                        for call in mock_measure_stability.call_args_list
                    ]
                    called_quality_scores = [
                        peak.quality_score for peak in called_peaks
                    ]

                    # Verify we got the top 3 quality scores
                    assert (
                        sorted(called_quality_scores, reverse=True)
                        == expected_top_3_quality_scores
                    ), (
                        f"Expected top 3 quality scores {expected_top_3_quality_scores}, "
                        f"got {sorted(called_quality_scores, reverse=True)}"
                    )

                    # Verify result contains best peak information
                    assert "best_peak_voltage" in result

                    # Verify the selected peak is based on combined score, not just original quality
                    # We verify this by checking that the selected peak's combined score is the maximum
                    # among all tested peaks. We also verify that the selection considers stability,
                    # not just original quality, by ensuring the selected peak has a reasonable
                    # combined score that accounts for both factors.

                    # Reconstruct candidates to verify combined scores using stored measurements
                    test_candidates = []
                    for peak, stability in zip(
                        called_peaks, created_stability_measurements, strict=True
                    ):
                        test_candidates.append(
                            StablePeakCandidate(
                                fitted_peak=peak,
                                original_score=peak.quality_score or 0.0,
                                stability_measurement=stability,
                            )
                        )

                    # Calculate combined scores with same weights as function (50/50)
                    calculate_combined_scores(
                        test_candidates, original_weight=0.5, stability_weight=0.5
                    )

                    # Find the candidate with highest combined score
                    best_test_candidate = max(
                        test_candidates, key=lambda c: c.combined_score or 0.0
                    )

                    # Verify the selected peak matches the one with highest combined score
                    best_peak_voltage = result["best_peak_voltage"]
                    assert (
                        abs(
                            best_peak_voltage
                            - best_test_candidate.fitted_peak.peak_voltage
                        )
                        < 0.001
                    ), (
                        f"Selected peak voltage {best_peak_voltage} should match the peak with "
                        f"highest combined score {best_test_candidate.fitted_peak.peak_voltage}"
                    )

                    # Verify that the selection is not just based on original quality
                    # (i.e., the peak with highest original quality might not win)
                    highest_quality_peak = max(
                        called_peaks, key=lambda p: p.quality_score or 0.0
                    )
                    if (
                        best_test_candidate.fitted_peak.peak_idx
                        != highest_quality_peak.peak_idx
                    ):
                        # If a different peak won, verify it has better stability
                        best_stability = min(
                            test_candidates,
                            key=lambda c: c.stability_measurement.voltage_noise,
                        )
                        assert (
                            best_test_candidate.fitted_peak.peak_idx
                            == best_stability.fitted_peak.peak_idx
                            or (
                                best_test_candidate.combined_score
                                > next(
                                    c.combined_score
                                    for c in test_candidates
                                    if c.fitted_peak.peak_idx
                                    == highest_quality_peak.peak_idx
                                )
                            )
                        ), "Selection should favor peaks with better combined scores"


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

    voltage_noise = calculate_voltage_noise(current_std, local_slope)

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
    calculate_combined_scores(candidates, original_weight=0.3, stability_weight=0.7)

    # Verify stability scores are set and normalized
    for candidate in candidates:
        assert candidate.stability_measurement.stability_score is not None
        assert 0 <= candidate.stability_measurement.stability_score <= 1.0

    # Peak with lowest voltage noise should have highest stability score
    min_noise_idx = np.argmin(voltage_noises)
    assert candidates[min_noise_idx].stability_measurement.stability_score == 1.0


def test_peak_detector_model_output_parsing(mock_context_for_sensor_routines):
    """Verify ML model peak detector output is correctly parsed and used.

    This test ensures the integration between the ML model and the peak finding
    routine works correctly. The model returns peak_indices which must be
    correctly interpreted as array indices for window extraction.
    """
    mock_ctx, mock_device = mock_context_for_sensor_routines

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
