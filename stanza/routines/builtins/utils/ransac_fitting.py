"""
RANSAC regression fitting utilities for robust gradient estimation.

This module provides RANSAC-based fitting functions for robustly estimating
linear relationships in the presence of outliers, particularly for compensation
gradient calculations in quantum dot devices.
"""

# Standard library imports
import logging
from dataclasses import dataclass

# Third-party imports
import numpy as np

try:
    from sklearn.linear_model import RANSACRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# First-party imports
from stanza.exceptions import RoutineError

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class RANSACFitResult:
    """
    Result from RANSAC regression fit for compensation gradient calculation.

    Contains the fitted gradient (slope), intercept, and detailed outlier information
    for all individual measurements.
    """

    gradient: float  # Fitted slope (compensation gradient)
    intercept: float  # Fitted y-intercept (drift term)
    inlier_mask: np.ndarray  # Boolean array marking inliers (True) vs outliers (False)
    num_inliers: int  # Total number of inlier measurements
    num_outliers: int  # Total number of outlier measurements
    outlier_indices: list[int]  # Indices of outlier measurements
    outlier_voltages: list[float]  # Control voltage deltas for outliers
    outlier_peak_shifts: list[float]  # Peak shifts for outliers
    all_control_deltas: np.ndarray  # All control voltage deltas (X values)
    all_peak_shifts: np.ndarray  # All peak shifts (y values)


def fit_compensation_gradient_ransac(
    measurement_samples: list[dict[str, float]],
    reference_peak_center_voltage: float,
    gate_name: str,
) -> RANSACFitResult:
    """
    Fit compensation gradient using RANSAC regression on raw measurement samples.

    This function robustly estimates the compensation gradient by fitting a line
    through all individual measurements (typically 100 points: 10 voltage points ×
    10 samples each) while automatically detecting and rejecting outliers from
    occasional bad measurements.

    Args:
        measurement_samples: List of measurement records, each containing:
            - "control_delta": Control gate voltage change from baseline (V)
            - "peak_position": Measured sensor peak center voltage (V)
        reference_peak_center_voltage: Baseline sensor peak voltage (V)
        gate_name: Name of gate being measured (for logging)

    Returns:
        RANSACFitResult containing fitted gradient, intercept, and outlier details

    Raises:
        RoutineError: If RANSAC fitting fails or data is invalid
    """
    # Extract all raw measurement pairs (control_delta, peak_shift)
    all_control_deltas = np.array(
        [sample["control_delta"] for sample in measurement_samples]
    )
    all_peak_shifts = np.array(
        [
            sample["peak_position"] - reference_peak_center_voltage
            for sample in measurement_samples
        ]
    )

    # Validate data
    if len(all_control_deltas) == 0:
        raise RoutineError("No measurement samples provided for RANSAC fitting")

    if np.allclose(all_control_deltas, all_control_deltas[0]):
        raise RoutineError(
            f"Cannot compute compensation gradient for gate {gate_name}: "
            "all control voltage deltas are identical"
        )

    # Reshape for sklearn (expects 2D array)
    X = all_control_deltas.reshape(-1, 1)
    y = all_peak_shifts

    try:
        ransac = RANSACRegressor(
            min_samples=2,  # Minimum points to fit a line (slope + intercept)
            residual_threshold=None,  # Auto-detect based on MAD
            max_trials=1000,  # Sufficient for 100-point dataset
            random_state=42,  # For reproducibility
        )
        ransac.fit(X, y)

        # Extract gradient (slope) and intercept from fitted model
        gradient = float(ransac.estimator_.coef_[0])
        intercept = float(ransac.estimator_.intercept_)

        # Store inlier mask and counts
        inlier_mask = ransac.inlier_mask_
        num_inliers = int(np.sum(inlier_mask))
        num_outliers = len(inlier_mask) - num_inliers

        # Identify outlier points for detailed logging
        outlier_indices = [
            i for i, is_inlier in enumerate(inlier_mask) if not is_inlier
        ]
        outlier_voltages = [float(all_control_deltas[i]) for i in outlier_indices]
        outlier_peak_shifts = [float(all_peak_shifts[i]) for i in outlier_indices]

        # Log outlier detection results
        if num_outliers > 0:
            logger.warning(
                "RANSAC detected %d outlier(s) out of %d points for gate %s",
                num_outliers,
                len(inlier_mask),
                gate_name,
            )
        else:
            logger.info(
                "RANSAC fit for gate %s: all %d points are inliers",
                gate_name,
                num_inliers,
            )

        return RANSACFitResult(
            gradient=gradient,
            intercept=intercept,
            inlier_mask=inlier_mask,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            outlier_indices=outlier_indices,
            outlier_voltages=outlier_voltages,
            outlier_peak_shifts=outlier_peak_shifts,
            all_control_deltas=all_control_deltas,
            all_peak_shifts=all_peak_shifts,
        )

    except Exception as exc:
        raise RoutineError(
            f"Failed to compute compensation gradient using RANSAC for gate {gate_name}: {exc}"
        ) from exc
