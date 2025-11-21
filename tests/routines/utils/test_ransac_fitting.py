"""Tests for RANSAC fitting utilities."""

import numpy as np
import pytest

from stanza.exceptions import RoutineError
from stanza.routines.builtins.charge_sensor.charge_sensor_compensation import (
    fit_compensation_gradient_ransac,
)
from stanza.routines.builtins.utils.ransac_fitting import RANSACFitResult


def test_fit_compensation_gradient_ransac_recovers_gradient():
    """Use synthetic measurement samples with known slope + outliers to ensure
    fit_compensation_gradient_ransac recovers the gradient and flags outliers."""
    true_gradient = 0.5
    reference_peak_center = 0.1

    control_deltas = np.linspace(-0.02, 0.02, 50)
    peak_positions = reference_peak_center + true_gradient * control_deltas

    peak_positions[5] += 0.05
    peak_positions[25] -= 0.04

    measurement_samples = [
        {"control_delta": cd, "peak_position": pp}
        for cd, pp in zip(control_deltas, peak_positions, strict=True)
    ]

    result = fit_compensation_gradient_ransac(
        measurement_samples=measurement_samples,
        reference_peak_center_voltage=reference_peak_center,
        gate_name="test_gate",
    )

    assert isinstance(result, RANSACFitResult)
    assert abs(result.gradient - true_gradient) < 0.1
    assert result.num_outliers >= 2


def test_fit_compensation_gradient_ransac_rejects_identical_deltas():
    """Pass identical control_delta values and assert the function raises RoutineError."""
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

    np.random.seed(42)
    control_deltas = np.linspace(-0.03, 0.03, 60)
    peak_positions = reference_peak_center + true_gradient * control_deltas
    peak_positions += np.random.normal(0, 0.001, len(peak_positions))
    outlier_indices = np.random.choice(len(peak_positions), size=6, replace=False)
    peak_positions[outlier_indices] += np.random.uniform(-0.05, 0.05, 6)

    measurement_samples = [
        {"control_delta": cd, "peak_position": pp}
        for cd, pp in zip(control_deltas, peak_positions, strict=True)
    ]

    ransac_result = fit_compensation_gradient_ransac(
        measurement_samples=measurement_samples,
        reference_peak_center_voltage=reference_peak_center,
        gate_name="test_gate",
    )

    peak_shifts = np.array([pp - reference_peak_center for pp in peak_positions])
    lsq_gradient, _ = np.polyfit(control_deltas, peak_shifts, deg=1)

    ransac_error = abs(ransac_result.gradient - true_gradient)
    lsq_error = abs(lsq_gradient - true_gradient)
    assert ransac_error < lsq_error
