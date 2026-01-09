"""Stability analysis utilities for charge sensor peak selection.

This module provides functions for measuring and analyzing peak stability,
including local slope calculation, voltage noise estimation, and combined
scoring of peak candidates based on quality and stability metrics.
"""

import logging
import time

import numpy as np

from stanza.device import Device
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.routines.builtins.charge_sensor.utils.constants import (
    DEFAULT_SETTLING_TIME_S,
)
from stanza.routines.builtins.charge_sensor.utils.types import (
    StabilityMeasurement,
    StablePeakCandidate,
)
from stanza.routines.builtins.utils.peak_fitting import FittedPeak

logger = logging.getLogger(__name__)


def calculate_local_slope(
    voltages: np.ndarray,
    currents: np.ndarray,
    target_voltage: float,
    window_points: int = 5,
) -> float:
    """
    Calculate local dI/dV slope around a target voltage using linear regression.

    Performs a local linear fit to the I-V data around the specified voltage
    to estimate the gradient (dI/dV) at that point.

    Args:
        voltages: Voltage array from sweep (V)
        currents: Current array from sweep (A)
        target_voltage: Voltage at which to calculate slope (V)
        window_points: Number of points on each side of target for linear fit (default: 5)

    Returns:
        Local slope dI/dV at target voltage (A/V)

    Raises:
        RoutineError: If target voltage is not in the voltage array or fit fails
    """
    idx = int(np.argmin(np.abs(voltages - target_voltage)))

    start_idx = max(0, idx - window_points)
    end_idx = min(len(voltages), idx + window_points + 1)

    v_window = voltages[start_idx:end_idx]
    i_window = currents[start_idx:end_idx]

    if len(v_window) < 3:
        raise RoutineError(
            f"Insufficient points for local slope calculation at voltage {target_voltage}V. "
            f"Only {len(v_window)} points available."
        )

    try:
        slope, _ = np.polyfit(v_window, i_window, deg=1)
        return float(slope)
    except Exception as e:
        raise RoutineError(
            f"Failed to calculate local slope at voltage {target_voltage}V: {e}"
        ) from e


def calculate_voltage_noise(current_std: float, local_slope: float) -> float:
    """
    Calculate effective voltage noise from current noise and local slope.

    Converts current noise (σᵢ) into equivalent voltage noise (σᵥ) using
    the local slope (dI/dV) of the peak:
        σᵥ = σᵢ / |dI/dV|

    Args:
        current_std: Standard deviation of current during hold (A)
        local_slope: Local dI/dV gradient (A/V)

    Returns:
        Voltage noise σᵥ (V)

    Raises:
        RoutineError: If local_slope is too close to zero (would cause division by zero)
    """
    abs_slope = abs(local_slope)

    if abs_slope < 1e-12:
        raise RoutineError(
            f"Local slope magnitude too small ({abs_slope:.2e} A/V) for voltage noise calculation. "
            "Cannot convert current noise to voltage noise with near-zero gradient."
        )

    voltage_noise = current_std / abs_slope
    return float(voltage_noise)


def calculate_combined_scores(
    candidates: list["StablePeakCandidate"],
    original_weight: float = 0.5,
    stability_weight: float = 0.5,
) -> None:
    """
    Calculate combined scores for all peak candidates using weighted scoring.

    Combines original peak quality scores with stability scores. Original scores
    are min-max normalized to [0, 1]. Stability scores are computed as 1/voltage_noise
    and normalized by dividing by max. Higher stability scores correspond to lower
    voltage noise (more stable peaks).

    Modifies candidates in-place by setting:
    - stability_measurement.stability_score (normalized)
    - combined_score (weighted combination)

    Args:
        candidates: List of StablePeakCandidate objects
        original_weight: Weight for original quality score (default: 0.5)
        stability_weight: Weight for stability score (default: 0.5)

    Raises:
        RoutineError: If weights don't sum to 1.0 or no candidates provided
    """
    if not candidates:
        raise RoutineError("Cannot calculate combined scores: no candidates provided")

    if not np.isclose(original_weight + stability_weight, 1.0):
        raise RoutineError(
            f"Weights must sum to 1.0, got {original_weight} + {stability_weight} = "
            f"{original_weight + stability_weight}"
        )

    original_scores = [c.original_score for c in candidates]
    voltage_noises = [c.stability_measurement.voltage_noise for c in candidates]

    for i, vn in enumerate(voltage_noises):
        if vn <= 0 or not np.isfinite(vn):
            raise RoutineError(
                f"Invalid voltage noise for candidate {i}: {vn}. "
                "Voltage noise must be positive and finite."
            )

    min_orig = min(original_scores)
    max_orig = max(original_scores)
    orig_range = max_orig - min_orig

    if orig_range > 0:
        normalized_original = [
            (score - min_orig) / orig_range for score in original_scores
        ]
    else:
        normalized_original = [1.0] * len(candidates)

    stability_scores_raw = [1.0 / vn for vn in voltage_noises]

    max_stab = max(stability_scores_raw)

    normalized_stability = [score / max_stab for score in stability_scores_raw]

    for i, candidate in enumerate(candidates):
        candidate.stability_measurement.stability_score = normalized_stability[i]
        candidate.combined_score = (
            original_weight * normalized_original[i]
            + stability_weight * normalized_stability[i]
        )


def measure_peak_stability(
    device: Device,
    peak: FittedPeak,
    peak_index: int,
    sensor_gates_list: list[str],
    sensor_plunger_gate: str,
    mean_reservoir_saturation_voltage: float,
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    aggregated_voltages: np.ndarray,
    aggregated_currents: np.ndarray,
    hold_time_seconds: float = 120.0,
    settling_time_s: float = DEFAULT_SETTLING_TIME_S,
    session: LoggerSession | None = None,
) -> "StabilityMeasurement":
    """
    Measure peak stability by holding at max-gradient voltage for 2 minutes.

    Moves the device to the peak's maximum gradient point (optimal sensing point),
    holds for the specified duration while continuously measuring current vs. time,
    and computes noise metrics to quantify peak position stability.

    Args:
        device: Device control interface
        peak: FittedPeak object containing peak parameters
        peak_index: Index of this peak in the candidate list (for logging)
        sensor_gates_list: List of all sensor gate names
        sensor_plunger_gate: Name of the sensor plunger gate
        mean_reservoir_saturation_voltage: Voltage for non-plunger sensor gates (V)
        measure_electrode: Electrode to measure current from
        bias_gate: Name of the bias gate (contact) to apply bias voltage
        bias_voltage: Voltage to apply to bias gate during measurements (V)
        aggregated_voltages: Voltage array from original sweep (for local slope calc)
        aggregated_currents: Current array from original sweep (for local slope calc)
        hold_time_seconds: Duration to hold and measure (default: 120.0 seconds)
        settling_time_s: Time to wait for device settling (default: 3.0s)
        session: Logger session for saving measurements

    Returns:
        StabilityMeasurement containing time-series data and noise metrics

    Raises:
        RoutineError: If measurement fails or noise calculation fails
    """

    logger.info(
        "Starting stability measurement for peak %d at max-gradient voltage %.6fV",
        peak_index + 1,
        peak.sensitivity_voltage,
    )

    device.jump({bias_gate: bias_voltage}, wait_for_settling=True)
    time.sleep(settling_time_s)

    device_state = dict.fromkeys(sensor_gates_list, mean_reservoir_saturation_voltage)
    device_state[sensor_plunger_gate] = peak.sensitivity_voltage

    device.jump(device_state, wait_for_settling=True)
    time.sleep(settling_time_s)

    logger.info(
        "Holding at max-gradient point for %.1f seconds, recording current...",
        hold_time_seconds,
    )

    time_array = []
    current_array = []
    start_time = time.time()

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= hold_time_seconds:
                break

            current = device.measure(measure_electrode)
            time_array.append(elapsed)
            current_array.append(current)

            time.sleep(0.01)

    except Exception as e:
        raise RoutineError(
            f"Failed to measure stability for peak {peak_index + 1}: {e}"
        ) from e

    time_array_np = np.array(time_array, dtype=np.float64)
    current_array_np = np.array(current_array, dtype=np.float64)

    if len(time_array_np) == 0 or len(current_array_np) == 0:
        raise RoutineError(
            f"Failed to collect any measurements for peak {peak_index + 1} during "
            f"{hold_time_seconds}s hold. Check device connectivity and measurement configuration."
        )

    logger.info(
        "Stability measurement complete: collected %d samples over %.1f seconds "
        "(avg rate: %.1f Hz)",
        len(time_array_np),
        time_array_np[-1] if len(time_array_np) > 0 else 0.0,
        len(time_array_np) / hold_time_seconds if hold_time_seconds > 0 else 0.0,
    )

    current_mean = float(np.mean(current_array_np))
    current_std = float(np.std(current_array_np))

    logger.info(
        "Current statistics: mean = %.3e A, std = %.3e A (%.2f%% of mean)",
        current_mean,
        current_std,
        100.0 * current_std / abs(current_mean) if abs(current_mean) > 0 else 0.0,
    )

    try:
        local_slope = calculate_local_slope(
            voltages=aggregated_voltages,
            currents=aggregated_currents,
            target_voltage=peak.sensitivity_voltage,
            window_points=5,
        )
        logger.info("Local slope dI/dV at max-gradient point: %.3e A/V", local_slope)
    except Exception as e:
        raise RoutineError(
            f"Failed to calculate local slope for peak {peak_index + 1}: {e}"
        ) from e

    try:
        voltage_noise = calculate_voltage_noise(current_std, local_slope)
        logger.info(
            "Voltage noise σᵥ = σᵢ / |dI/dV| = %.3e V / %.3e A/V = %.3e V",
            current_std,
            abs(local_slope),
            voltage_noise,
        )
    except Exception as e:
        raise RoutineError(
            f"Failed to calculate voltage noise for peak {peak_index + 1}: {e}"
        ) from e

    stability_measurement = StabilityMeasurement(
        peak_index=peak_index,
        peak_voltage=float(peak.peak_voltage),
        max_gradient_voltage=float(peak.sensitivity_voltage),
        time_array=time_array_np,
        current_array=current_array_np,
        current_mean=current_mean,
        current_std=current_std,
        local_slope=local_slope,
        voltage_noise=voltage_noise,
    )

    if session:
        session.log_analysis(
            name=f"stability_measurement_peak_{peak_index + 1}",
            data={
                "peak_index": peak_index + 1,
                "peak_voltage": float(peak.peak_voltage),
                "max_gradient_voltage": float(peak.sensitivity_voltage),
                "hold_time_seconds": float(hold_time_seconds),
                "num_samples": int(len(time_array_np)),
                "sampling_rate_hz": float(
                    len(time_array_np) / hold_time_seconds
                    if hold_time_seconds > 0
                    else 0.0
                ),
                "time_array": time_array_np.tolist(),
                "current_array": current_array_np.tolist(),
                "current_mean": current_mean,
                "current_std": current_std,
                "current_min": float(np.min(current_array_np)),
                "current_max": float(np.max(current_array_np)),
                "local_slope": local_slope,
                "voltage_noise": voltage_noise,
            },
        )

    return stability_measurement
