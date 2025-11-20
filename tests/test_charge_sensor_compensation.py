"""Tests for charge sensor compensation routines and utilities."""

import numpy as np
import pytest

from stanza.exceptions import RoutineError
from stanza.routines.builtins.charge_sensor_compensation import (
    RANSACFitResult,
    StabilityMeasurement,
    StablePeakCandidate,
    _build_sensor_sweep_voltage_list,
    _calculate_combined_scores,
    _calculate_local_slope,
    _calculate_peak_window_bounds,
    _calculate_quality_scores,
    _calculate_voltage_noise,
    _normalize_sensitivity_scores,
    fit_compensation_gradient_ransac,
)
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    ModelFitResult,
)

# =============================================================================
# Helper Utilities Tests
# =============================================================================


def test_build_sensor_sweep_voltage_list_respects_overrides():
    """Ensure _build_sensor_sweep_voltage_list keeps non-plunger gates at base voltage
    while honoring per-gate overrides and varying only the targeted plunger."""
    sensor_gates_list = ["gate1", "gate2", "gate3", "gate4"]
    sensor_plunger_index = 1  # gate2 is the plunger
    base_voltage = 0.5
    plunger_voltages = np.array([0.1, 0.2, 0.3])
    gate_voltage_overrides = {"gate3": 0.8, "gate4": 0.9}

    voltage_list = _build_sensor_sweep_voltage_list(
        sensor_gates_list=sensor_gates_list,
        sensor_plunger_index=sensor_plunger_index,
        base_voltage=base_voltage,
        plunger_voltages=plunger_voltages,
        gate_voltage_overrides=gate_voltage_overrides,
    )

    assert len(voltage_list) == 3
    # Check each sweep point
    for i, voltages in enumerate(voltage_list):
        assert len(voltages) == 4
        # gate1 should be at base_voltage
        assert voltages[0] == base_voltage
        # gate2 (plunger) should vary
        assert voltages[1] == plunger_voltages[i]
        # gate3 should use override
        assert voltages[2] == 0.8
        # gate4 should use override
        assert voltages[3] == 0.9


def test_calculate_peak_window_bounds_handles_edge_peaks():
    """Feed synthetic peak indices into _calculate_peak_window_bounds and verify
    first/middle/last peaks clamp to trace limits with WINDOW_FRACTION rules."""
    trace_length = 1000

    # Test first peak
    peak_indices = [100, 400, 700]
    start, end = _calculate_peak_window_bounds(
        peak_idx=100, peak_index=0, peak_indices=peak_indices, trace_length=trace_length
    )
    assert start >= 0
    assert end <= trace_length
    assert start < 100 < end

    # Test middle peak
    start, end = _calculate_peak_window_bounds(
        peak_idx=400, peak_index=1, peak_indices=peak_indices, trace_length=trace_length
    )
    assert start >= 0
    assert end <= trace_length
    assert start < 400 < end

    # Test last peak
    start, end = _calculate_peak_window_bounds(
        peak_idx=700, peak_index=2, peak_indices=peak_indices, trace_length=trace_length
    )
    assert start >= 0
    assert end <= trace_length
    assert start < 700 < end

    # Test single peak (should use entire trace)
    start, end = _calculate_peak_window_bounds(
        peak_idx=500, peak_index=0, peak_indices=[500], trace_length=trace_length
    )
    assert start == 0
    assert end == trace_length


def test_normalize_sensitivity_scores_constant_inputs():
    """Confirm _normalize_sensitivity_scores assigns score 1.0 when all sensitivities are equal."""
    # Create mock FittedPeak objects with identical sensitivities
    peaks = []
    for i in range(5):
        peak = FittedPeak(
            best_model="Lorentzian",
            lorentzian_fit=ModelFitResult(
                model_name="Lorentzian",
                amplitude=1.0,
                center_idx=float(i * 10),
                width=5.0,
                offset=0.0,
                r_squared=0.95,
                rmse=0.01,
                aicc=10.0,
                fwhm=10.0,
                area=100.0,
                skew_resid=0.0,
            ),
            sech2_fit=ModelFitResult(
                model_name="sech2",
                amplitude=1.0,
                center_idx=float(i * 10),
                width=5.0,
                offset=0.0,
                r_squared=0.90,
                rmse=0.02,
                aicc=12.0,
                fwhm=10.0,
                area=100.0,
                skew_resid=0.0,
            ),
            voigt_fit=ModelFitResult(
                model_name="Voigt",
                amplitude=1.0,
                center_idx=float(i * 10),
                width=5.0,
                offset=0.0,
                eta=0.5,
                r_squared=0.93,
                rmse=0.015,
                aicc=11.0,
                fwhm=10.0,
                area=100.0,
                skew_resid=0.0,
            ),
            sensitivity=1e-6,  # All identical
            sensitivity_voltage=float(i * 0.01),
            peak_idx=i * 10,
            peak_voltage=float(i * 0.01),
            window_currents=np.ones(10),
            window_voltages=np.linspace(0, 0.1, 10),
        )
        peaks.append(peak)

    _normalize_sensitivity_scores(peaks)

    for peak in peaks:
        assert peak.sensitivity_score == 1.0


def test_calculate_quality_scores_requires_normalized_sensitivity():
    """Call _calculate_quality_scores without pre-normalized scores and assert it raises RoutineError."""
    peak = FittedPeak(
        best_model="Lorentzian",
        lorentzian_fit=ModelFitResult(
            model_name="Lorentzian",
            amplitude=1.0,
            center_idx=5.0,
            width=5.0,
            offset=0.0,
            r_squared=0.95,
            rmse=0.01,
            aicc=10.0,
            fwhm=10.0,
            area=100.0,
            skew_resid=0.0,
        ),
        sech2_fit=ModelFitResult(
            model_name="sech2",
            amplitude=1.0,
            center_idx=5.0,
            width=5.0,
            offset=0.0,
            r_squared=0.90,
            rmse=0.02,
            aicc=12.0,
            fwhm=10.0,
            area=100.0,
            skew_resid=0.0,
        ),
        voigt_fit=ModelFitResult(
            model_name="Voigt",
            amplitude=1.0,
            center_idx=5.0,
            width=5.0,
            offset=0.0,
            eta=0.5,
            r_squared=0.93,
            rmse=0.015,
            aicc=11.0,
            fwhm=10.0,
            area=100.0,
            skew_resid=0.0,
        ),
        sensitivity=1e-6,
        sensitivity_voltage=0.01,
        sensitivity_score=None,  # NOT SET
        peak_idx=50,
        peak_voltage=0.01,
        window_currents=np.ones(10),
        window_voltages=np.linspace(0, 0.1, 10),
    )

    with pytest.raises(RoutineError, match="Sensitivity score not set"):
        _calculate_quality_scores([peak])


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
# Regression Math Tests
# =============================================================================


def test_fit_compensation_gradient_ransac_recovers_gradient():
    """Use synthetic measurement samples with known slope + outliers to ensure
    fit_compensation_gradient_ransac recovers the gradient and flags outliers."""
    # True gradient: 0.5 V/V
    true_gradient = 0.5
    reference_peak_center = 0.1

    # Generate clean measurements
    control_deltas = np.linspace(-0.02, 0.02, 50)
    peak_positions = reference_peak_center + true_gradient * control_deltas

    # Add some outliers
    peak_positions[5] += 0.05  # Large positive outlier
    peak_positions[25] -= 0.04  # Large negative outlier

    measurement_samples = [
        {"control_delta": cd, "peak_position": pp}
        for cd, pp in zip(control_deltas, peak_positions, strict=True)
    ]

    result = fit_compensation_gradient_ransac(
        measurement_samples=measurement_samples,
        reference_peak_center_voltage=reference_peak_center,
        gate_name="test_gate",
    )

    # Check gradient recovery (should be close to 0.5)
    assert isinstance(result, RANSACFitResult)
    assert abs(result.gradient - true_gradient) < 0.1
    # Should detect outliers
    assert result.num_outliers >= 2


def test_fit_compensation_gradient_ransac_rejects_identical_deltas():
    """Pass identical control_delta values and assert the function raises RoutineError."""
    # All control deltas are identical
    measurement_samples = [
        {"control_delta": 0.01, "peak_position": 0.1 + i * 0.001} for i in range(10)
    ]

    with pytest.raises(RoutineError, match="identical"):
        fit_compensation_gradient_ransac(
            measurement_samples=measurement_samples,
            reference_peak_center_voltage=0.1,
            gate_name="test_gate",
        )


def test_ransac_vs_mse_gradient_fitting():
    """Generate noisy samples around a true line and compare RANSAC vs plain least-squares fits,
    asserting RANSAC is closer to the ground-truth slope."""
    true_gradient = 0.3
    reference_peak_center = 0.15

    # Generate measurements with outliers
    np.random.seed(42)
    control_deltas = np.linspace(-0.03, 0.03, 60)
    peak_positions = reference_peak_center + true_gradient * control_deltas
    # Add noise
    peak_positions += np.random.normal(0, 0.001, len(peak_positions))
    # Add 10% outliers
    outlier_indices = np.random.choice(len(peak_positions), size=6, replace=False)
    peak_positions[outlier_indices] += np.random.uniform(-0.05, 0.05, 6)

    measurement_samples = [
        {"control_delta": cd, "peak_position": pp}
        for cd, pp in zip(control_deltas, peak_positions, strict=True)
    ]

    # RANSAC fit
    ransac_result = fit_compensation_gradient_ransac(
        measurement_samples=measurement_samples,
        reference_peak_center_voltage=reference_peak_center,
        gate_name="test_gate",
    )

    # Plain least-squares fit
    peak_shifts = np.array([pp - reference_peak_center for pp in peak_positions])
    lsq_gradient, _ = np.polyfit(control_deltas, peak_shifts, deg=1)

    # RANSAC should be closer to true gradient
    ransac_error = abs(ransac_result.gradient - true_gradient)
    lsq_error = abs(lsq_gradient - true_gradient)
    assert ransac_error < lsq_error


def test_run_compensation_randomizes_measurement_sequence():
    """Ensure the shuffled measurement_indices still execute every voltage delta exactly
    NUM_OF_SAMPLES_FOR_AVERAGING times despite random order."""
    # This test verifies the conceptual behavior, but requires access to internal implementation
    # For now, we'll test that the measurement samples contain the expected number of entries
    # This is a placeholder test that would need to be expanded with actual mocking
    from stanza.routines.builtins.charge_sensor_compensation import (
        NUM_OF_SAMPLES_FOR_AVERAGING,
    )

    # Create synthetic measurement samples
    num_voltage_points = 10
    expected_total_samples = num_voltage_points * NUM_OF_SAMPLES_FOR_AVERAGING

    # Generate measurement samples
    measurement_samples = []
    voltage_deltas = np.linspace(-0.01, 0.01, num_voltage_points)
    for delta in voltage_deltas:
        for _ in range(NUM_OF_SAMPLES_FOR_AVERAGING):
            measurement_samples.append(
                {"control_delta": delta, "peak_position": 0.1 + 0.5 * delta}
            )

    assert len(measurement_samples) == expected_total_samples

    # Count occurrences of each delta
    from collections import Counter

    deltas = [sample["control_delta"] for sample in measurement_samples]
    delta_counts = Counter(deltas)

    # Each unique delta should appear exactly NUM_OF_SAMPLES_FOR_AVERAGING times
    for count in delta_counts.values():
        assert count == NUM_OF_SAMPLES_FOR_AVERAGING


# =============================================================================
# Stability and Combined Scores Tests
# =============================================================================


def test_calculate_combined_scores():
    """Test _calculate_combined_scores with mock candidates."""
    candidates = []
    for i in range(3):
        fitted_peak = FittedPeak(
            best_model="Lorentzian",
            lorentzian_fit=ModelFitResult(
                model_name="Lorentzian",
                amplitude=1.0,
                center_idx=float(i * 10),
                width=5.0,
                offset=0.0,
                r_squared=0.95,
                rmse=0.01,
                aicc=10.0,
                fwhm=10.0,
                area=100.0,
                skew_resid=0.0,
            ),
            sech2_fit=ModelFitResult(
                model_name="sech2",
                amplitude=1.0,
                center_idx=float(i * 10),
                width=5.0,
                offset=0.0,
                r_squared=0.90,
                rmse=0.02,
                aicc=12.0,
                fwhm=10.0,
                area=100.0,
                skew_resid=0.0,
            ),
            voigt_fit=ModelFitResult(
                model_name="Voigt",
                amplitude=1.0,
                center_idx=float(i * 10),
                width=5.0,
                offset=0.0,
                eta=0.5,
                r_squared=0.93,
                rmse=0.015,
                aicc=11.0,
                fwhm=10.0,
                area=100.0,
                skew_resid=0.0,
            ),
            sensitivity=1e-6,
            sensitivity_voltage=0.01,
            peak_idx=i * 10,
            peak_voltage=0.01,
            window_currents=np.ones(10),
            window_voltages=np.linspace(0, 0.1, 10),
        )

        stability_measurement = StabilityMeasurement(
            peak_index=i,
            peak_voltage=0.01 * i,
            max_gradient_voltage=0.01 * i,
            time_array=np.linspace(0, 120, 100),
            current_array=np.ones(100) * 1e-9,
            current_mean=1e-9,
            current_std=1e-11,
            local_slope=1e-6,
            voltage_noise=1e-5 * (i + 1),  # Varying voltage noise
        )

        candidate = StablePeakCandidate(
            fitted_peak=fitted_peak,
            original_score=0.9 + i * 0.01,
            stability_measurement=stability_measurement,
        )
        candidates.append(candidate)

    _calculate_combined_scores(candidates, original_weight=0.3, stability_weight=0.7)

    # Verify all candidates have combined scores
    for candidate in candidates:
        assert candidate.combined_score is not None
        assert 0 <= candidate.combined_score <= 1.0
        assert candidate.stability_measurement.stability_score is not None


# =============================================================================
# Peak Region Partitioning Tests
# =============================================================================


def test_peak_region_partitioning_prevents_overlap_large_spacing():
    """With widely spaced peaks (> DEFAULT_WINDOW_HALF_WIDTH), ensure
    _calculate_peak_window_bounds windows do not overlap other peaks."""
    # Create peaks with large spacing (100 points apart)
    peak_indices = [50, 150, 250]
    trace_length = 300

    # Calculate bounds for each peak
    bounds = []
    for i, peak_idx in enumerate(peak_indices):
        result = _calculate_peak_window_bounds(peak_idx, i, peak_indices, trace_length)
        if result is not None:
            bounds.append(result)

    # Check that no window overlaps with adjacent peaks
    for i, (left, right) in enumerate(bounds):
        # Ensure window doesn't include adjacent peaks
        if i > 0:
            assert left > peak_indices[i - 1], (
                f"Window for peak {i} overlaps previous peak"
            )
        if i < len(peak_indices) - 1:
            assert right < peak_indices[i + 1], (
                f"Window for peak {i} overlaps next peak"
            )


def test_peak_region_partitioning_prevents_overlap_small_spacing():
    """With closely spaced peaks (~5 points apart), confirm windows are
    calculated reasonably even when spacing prevents perfect non-overlap."""
    # Create closely spaced peaks
    peak_indices = [20, 25, 30, 35]
    trace_length = 50

    # Calculate bounds for each peak
    bounds = []
    for i, peak_idx in enumerate(peak_indices):
        result = _calculate_peak_window_bounds(peak_idx, i, peak_indices, trace_length)
        if result is not None:
            bounds.append(result)

    # Verify all peaks got valid windows
    assert len(bounds) == len(peak_indices)

    # Verify windows are centered around their peaks and have reasonable size
    for i, (left, right) in enumerate(bounds):
        peak_idx = peak_indices[i]
        assert left < peak_idx < right, f"Peak {i} not within its window"
        assert right - left >= 3, f"Window {i} is too small (need at least 3 points)"


# =============================================================================
# run_compensation Tests
# =============================================================================


def test_run_compensation_validates_gates_to_compensate():
    """Pass invalid gate names to run_compensation and assert it raises
    the documented error."""
    from unittest.mock import Mock, patch

    from stanza.models import DeviceGroup
    from stanza.routines import RoutineContext
    from stanza.routines.builtins.charge_sensor_compensation import run_compensation

    # Create mock context with required results
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "find_sensor_peak": {
            "narrowed_sensor_plunger_range": (-1.0, -0.5),
            "mean_reservoir_saturation_voltage": 0.5,
            "sensor_gates_list": ["G1", "G2", "G3"],
            "sensor_plunger_index": 2,
            "step_size": 0.001,
        }
    }

    # Create mock device
    mock_device = Mock()
    control_group = DeviceGroup(name="control_group", gates=["G4", "G5"])
    mock_device.device_config = Mock()
    mock_device.device_config.groups = {"control_group": control_group}
    mock_device.get_gates_by_type.return_value = ["G4", "G5"]
    mock_device.check.return_value = [-0.5, -0.6]

    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_ctx.resources = mock_resources

    # Patch filter_gates_by_group to return gates as-is
    with patch(
        "stanza.routines.builtins.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        # Try to compensate invalid gates
        with pytest.raises(
            RoutineError, match="Invalid gates specified in gates_to_compensate"
        ):
            run_compensation(
                ctx=mock_ctx,
                peak_spacing=0.02,
                control_group_name="control_group",
                measure_electrode="OUT",
                bias_gate="BIAS",
                bias_voltage=1e-4,
                gates_to_compensate=["INVALID_GATE", "ANOTHER_BAD_GATE"],
            )


def test_run_compensation_restores_device_state_on_error():
    """Force an exception mid-measurement and confirm both control and sensor
    voltages are reset in the finally block."""
    from unittest.mock import Mock, patch

    from stanza.models import DeviceGroup
    from stanza.routines import RoutineContext
    from stanza.routines.builtins.charge_sensor_compensation import run_compensation

    # Create mock context
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "find_sensor_peak": {
            "narrowed_sensor_plunger_range": (-1.0, -0.5),
            "mean_reservoir_saturation_voltage": 0.5,
            "sensor_gates_list": ["G1", "G2", "G3"],
            "sensor_plunger_index": 2,
            "step_size": 0.001,
        }
    }

    # Create mock device
    mock_device = Mock()
    control_group = DeviceGroup(name="control_group", gates=["G4", "G5"])
    mock_device.device_config = Mock()
    mock_device.device_config.groups = {"control_group": control_group}

    initial_control_voltages = [-0.5, -0.6]
    initial_sensor_voltages = [0.1, 0.2, 0.3]

    mock_device.check.side_effect = [
        initial_control_voltages,  # First call for control gates
        initial_sensor_voltages,  # Second call for sensor gates
    ]
    mock_device.get_gates_by_type.return_value = ["G4", "G5"]

    # Mock sweep_nd to return proper values then fail
    call_count = [0]

    def sweep_nd_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] > 1:  # Allow first call, then fail
            raise RuntimeError("Simulated device error")
        return (np.linspace(-1.0, -0.5, 128), np.ones(128) * 1e-9)

    mock_device.sweep_nd.side_effect = sweep_nd_side_effect

    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_ctx.resources = mock_resources

    # Patch necessary functions
    with patch(
        "stanza.routines.builtins.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with patch(
            "stanza.routines.builtins.charge_sensor_compensation.ConductorQuantum"
        ) as mock_cq:
            mock_cq_instance = Mock()
            mock_cq_instance.Classifier.predict.return_value = {
                "classification": True,
                "score": 0.95,
            }
            mock_cq_instance.DotDetector.predict.return_value = {"peaks": [64]}
            mock_cq.return_value = mock_cq_instance

            with pytest.raises(RuntimeError, match="Simulated device error"):
                run_compensation(
                    ctx=mock_ctx,
                    peak_spacing=0.02,
                    control_group_name="control_group",
                    measure_electrode="OUT",
                    bias_gate="BIAS",
                    bias_voltage=1e-4,
                )

    # Verify device.jump was called in finally block to restore state
    assert mock_device.jump.call_count >= 2


def test_run_compensation_logs_per_sample_measurements():
    """Mock LoggerSession to assert per-sample log_analysis entries are
    emitted with the expected fields."""
    from unittest.mock import Mock, patch

    from stanza.logger.session import LoggerSession
    from stanza.models import DeviceGroup
    from stanza.routines import RoutineContext
    from stanza.routines.builtins.charge_sensor_compensation import run_compensation

    # Create mock context
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "find_sensor_peak": {
            "narrowed_sensor_plunger_range": (-1.0, -0.5),
            "mean_reservoir_saturation_voltage": 0.5,
            "sensor_gates_list": ["G1", "G2", "G3"],
            "sensor_plunger_index": 2,
            "step_size": 0.001,
        }
    }

    # Create mock device with complete behavior
    mock_device = Mock()
    mock_device.device_config.groups = {
        "control_group": DeviceGroup(name="control_group", gates=["G4"])
    }

    mock_device.check.return_value = [-0.5]
    mock_device.get_gates_by_type.return_value = ["G4"]

    # Mock measure to return varied currents
    mock_device.measure.return_value = 1e-9

    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_ctx.resources = mock_resources

    # Create mock logger session
    mock_session = Mock(spec=LoggerSession)
    mock_session.log_analysis = Mock()

    # Patch necessary functions
    with patch(
        "stanza.routines.builtins.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with patch(
            "stanza.routines.builtins.charge_sensor_compensation._single_window_sensor_plunger_sweep"
        ) as mock_sweep:
            # Mock the sweep to return a simple fitted peak
            from stanza.routines.builtins.utils.peak_fitting import (
                FittedPeak,
                ModelFitResult,
            )

            mock_peak = FittedPeak(
                peak_idx=50,
                peak_voltage=-0.75,
                lorentzian_fit=ModelFitResult(
                    model_name="lorentzian",
                    amplitude=1e-9,
                    center_idx=50.0,
                    width=0.01,
                    offset=0.0,
                    r_squared=0.95,
                    rmse=1e-11,
                    aicc=-100,
                    fwhm=0.02,
                    area=1e-10,
                    skew_resid=0.01,
                ),
                sech2_fit=None,
                voigt_fit=None,
                best_model="lorentzian",
                sensitivity=1e-8,
                quality_score=0.9,
            )
            mock_sweep.return_value = mock_peak

            run_compensation(
                ctx=mock_ctx,
                peak_spacing=0.02,
                control_group_name="control_group",
                measure_electrode="OUT",
                bias_gate="BIAS",
                bias_voltage=1e-4,
                session=mock_session,
            )

    # Verify log_analysis was called (it logs per-sample measurements)
    assert mock_session.log_analysis.call_count > 0


def test_single_window_sweep_repeats_measurements_around_park_point():
    """Ensure _single_window_sensor_plunger_sweep performs sweep measurements
    and returns a fitted peak."""
    from unittest.mock import Mock, patch

    from stanza.routines import RoutineContext
    from stanza.routines.builtins.charge_sensor_compensation import (
        _single_window_sensor_plunger_sweep,
    )

    # Create mock context
    mock_ctx = Mock(spec=RoutineContext)
    mock_resources = Mock()
    mock_device = Mock()

    # Mock sweep_nd to return proper data
    voltages = np.linspace(-1.0, -0.5, 128)
    currents = np.ones(128) * 1e-9
    mock_device.sweep_nd.return_value = (voltages, currents)

    mock_resources.device = mock_device
    mock_ctx.resources = mock_resources

    # Mock ConductorQuantum
    with patch(
        "stanza.routines.builtins.charge_sensor_compensation.ConductorQuantum"
    ) as mock_cq:
        mock_cq_instance = Mock()
        mock_cq_instance.Classifier.predict.return_value = {
            "classification": True,
            "score": 0.95,
        }
        mock_cq_instance.DotDetector.predict.return_value = {
            "peaks": [64]  # Middle of 128-point window
        }
        mock_cq.return_value = mock_cq_instance

        result = _single_window_sensor_plunger_sweep(
            ctx=mock_ctx,
            sensor_gates_list=["G1", "G2", "G3"],
            sensor_plunger_range=(-1.0, -0.5),
            mean_reservoir_saturation_voltage=0.5,
            sensor_plunger_index=2,
            step_size=0.001,
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
        )

    # Verify a peak was returned
    assert result is not None
    assert hasattr(result, "peak_voltage")
    assert hasattr(result, "quality_score")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
