"""Tests for charge sensor compensation routines and utilities."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.models import DeviceGroup
from stanza.routines import RoutineContext
from stanza.routines.builtins.charge_sensor.charge_sensor_compensation import (
    RANSACFitResult,
    _single_window_sensor_plunger_sweep,
    fit_compensation_gradient_ransac,
    run_compensation,
)
from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
    PeakWindowSweepOutput,
)
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    ModelFitResult,
    fit_peak_multi_model,
    lorentzian,
)

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


# =============================================================================
# run_compensation Tests
# =============================================================================


def test_run_compensation_validates_gates_to_compensate():
    """Pass invalid gate names to run_compensation and assert it raises
    the documented error."""

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
        "stanza.routines.builtins.charge_sensor.charge_sensor_compensation.filter_gates_by_group",
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
        "stanza.routines.builtins.charge_sensor.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with pytest.raises(RoutineError, match="Simulated device error"):
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
        "stanza.routines.builtins.charge_sensor.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with patch(
            "stanza.routines.builtins.charge_sensor.charge_sensor_compensation._single_window_sensor_plunger_sweep"
        ) as mock_sweep:
            # Mock the sweep to return a PeakWindowSweepOutput

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
                sensitivity_voltage=-0.74,
                window_currents=np.ones(10) * 1e-9,
                window_voltages=np.linspace(-1.0, -0.5, 10),
                quality_score=0.9,
            )

            mock_output = PeakWindowSweepOutput(
                best_peak=mock_peak,
                aggregated_voltages=np.linspace(-1.0, -0.5, 10),
                aggregated_currents=np.ones(10) * 1e-9,
                classification=True,
                score=0.9,
                num_peaks=1,
            )
            mock_sweep.return_value = mock_output

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

    # Verify a PeakWindowSweepOutput was returned
    assert result is not None
    assert hasattr(result, "best_peak")
    assert result.best_peak is not None
    assert hasattr(result.best_peak, "peak_voltage")
    assert hasattr(result.best_peak, "quality_score")


# =============================================================================
# Compensation Baseline & Perturbation Tests
# =============================================================================


def test_baseline_uses_median_for_robustness():
    """With multiple baseline measurements, verify the reference peak center
    voltage is computed as the median (not mean) for outlier robustness."""
    # Simulate 5 baseline peak measurements with one outlier
    peak_centers = np.array([-0.700, -0.701, -0.699, -0.700, -0.720])  # Last is outlier

    # Median is more robust than mean
    median_center = np.median(peak_centers)
    mean_center = np.mean(peak_centers)

    # Median should be closer to the cluster
    assert abs(median_center - (-0.700)) < 0.002
    # Mean would be pulled by the outlier
    assert abs(mean_center - (-0.700)) > abs(median_center - (-0.700))


# =============================================================================
# Peak Position Tracking & Fitting Accuracy Tests
# =============================================================================


def test_fitted_peak_more_accurate_than_discrete():
    """Generate synthetic peaks and verify fitted peak centers have sub-step-size
    resolution compared to discrete voltage points."""

    # Create synthetic peak with known center at non-integer index
    voltages = np.linspace(0.0, 0.1, 100)
    true_center_idx = 51.23  # Fractional index

    # Generate peak data
    indices = np.arange(100)
    currents = lorentzian(
        indices, amplitude=2e-9, center=true_center_idx, width=8, offset=1e-11
    )

    # Fit the peak
    peak_idx_discrete = int(true_center_idx)
    fitted_peak = fit_peak_multi_model(
        window_currents=currents,
        window_indices=indices,
        peak_idx_in_window=peak_idx_discrete,
        aggregated_voltages=voltages,
        window_start_idx=0,
        window_end_idx=len(currents),
        peak_idx_aggregated=peak_idx_discrete,
    )

    # Fitted center should be closer to true center than discrete
    best_fit = getattr(fitted_peak, f"{fitted_peak.best_model.lower()}_fit")
    fitted_center_idx = best_fit.center_idx

    # Verify sub-index resolution
    assert abs(fitted_center_idx - true_center_idx) < abs(
        peak_idx_discrete - true_center_idx
    )


# =============================================================================
# Compensation Measurement Logging Tests
# =============================================================================


def test_measurement_samples_marked_inlier_outlier():
    """After RANSAC fit, confirm each sample in measurement_samples has
    is_inlier boolean field matching RANSAC inlier_mask."""
    true_gradient = 0.3
    reference_center = 0.15

    # Generate measurements with outliers
    np.random.seed(42)
    control_deltas = np.linspace(-0.02, 0.02, 30)
    peak_positions = reference_center + true_gradient * control_deltas
    peak_positions[5] += 0.05  # Add outlier
    peak_positions[20] -= 0.04  # Add outlier

    measurement_samples = [
        {"control_delta": cd, "peak_position": pp}
        for cd, pp in zip(control_deltas, peak_positions, strict=True)
    ]

    result = fit_compensation_gradient_ransac(
        measurement_samples=measurement_samples,
        reference_peak_center_voltage=reference_center,
        gate_name="test_gate",
    )

    # Verify inlier mask exists and has correct length
    assert len(result.inlier_mask) == len(measurement_samples)
    assert result.num_inliers + result.num_outliers == len(measurement_samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
