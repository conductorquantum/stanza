# Standard library imports
import logging
import time
from dataclasses import dataclass
from typing import Any

# Third-party imports
import numpy as np
from conductorquantum import ConductorQuantum

from stanza.device import Device

# First-party imports
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.routines import RoutineContext, routine
from stanza.routines.builtins.charge_sensor.constants import (
    COULOMB_CLASSIFIER_MODEL,
    DEFAULT_SETTLING_TIME_S,
    DEFAULT_WINDOW_HALF_WIDTH,
    INITIAL_WINDOW_MULTIPLIER,
    ML_MODEL_INPUT_SIZE,
    PEAK_DETECTOR_MODEL,
    REFINED_STEP_MULTIPLIER,
    WINDOW_FRACTION,
)
from stanza.routines.builtins.utils.group_handling import filter_gates_by_group
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    calculate_quality_score,
    fit_peak_multi_model,
)

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class StabilityMeasurement:
    """
    Result from a 2-minute stability measurement at a peak's max-gradient point.

    Contains time-series current data, noise metrics, and the computed voltage noise
    score that quantifies peak position stability.
    """

    peak_index: int  # Index of the peak in the original peak list
    peak_voltage: float  # Center voltage of the peak (V)
    max_gradient_voltage: (
        float  # Voltage at maximum gradient where measurement was taken (V)
    )
    time_array: np.ndarray  # Time points during the 2-minute hold (seconds)
    current_array: np.ndarray  # Measured current values over time (A)
    current_mean: float  # Mean current during hold (A)
    current_std: float  # Standard deviation of current (A) - σᵢ
    local_slope: float  # Local dI/dV from linear fit around max-gradient point (A/V)
    voltage_noise: float  # Effective voltage jitter: σᵥ = σᵢ / |dI/dV| (V)
    stability_score: float | None = (
        None  # Normalized stability score (higher is better)
    )


@dataclass
class StablePeakCandidate:
    """
    A peak candidate with both original quality score and stability measurements.

    Combines the initial peak quality metrics with stability measurements to
    enable selection of the most stable peak for charge sensing.
    """

    fitted_peak: FittedPeak  # Original fitted peak from multi-model fitting
    original_score: float  # Original quality score from peak fitting
    stability_measurement: StabilityMeasurement  # Stability data from 2-min hold
    combined_score: float | None = (
        None  # Final weighted score (50% original + 50% stability)
    )


@dataclass
class PeakWindowSweepOutput:
    """
    Result from a single-window barrier sweep with Lorentzian peak fitting.

    This model contains the best fitted peak from a refined barrier sweep,
    along with the full trace data and neighboring peak information for context.
    Used for both baseline measurements and gate compensation sweeps.
    """

    # Best fitted peak information
    best_peak: FittedPeak  # The best fitted peak selected from this sweep

    # Full trace data
    aggregated_voltages: np.ndarray  # Complete voltage array from the sweep
    aggregated_currents: np.ndarray  # Complete current array from the sweep

    # Metadata
    classification: bool  # Whether Coulomb blockade was detected
    score: float  # Classification confidence score
    num_peaks: int  # Total number of peaks detected and fitted

    # Neighboring peak context (for park point calculations)
    prev_peak_voltage: float | None = None  # Voltage of spatially previous peak
    next_peak_voltage: float | None = None  # Voltage of spatially next peak


@dataclass
class SensorDotPlungerSweepOutput:
    """
    Result from a single sensor plunger voltage sweep measurement.

    This stores results from continuous aggregation: each result contains the cumulative
    aggregated trace up to that window, the interpolated 128-point trace used for
    classification, and peak indices in the aggregated trace space.
    """

    sensor_plunger_voltage: float  # Start voltage of the window
    classification: bool  # Coulomb blockade detected (on interpolated trace)
    score: float  # Classification confidence score
    peak_indices: list[int]  # Detected peak locations in AGGREGATED trace space
    num_peaks: int  # Number of peaks detected
    aggregated_voltages: np.ndarray  # Full aggregated voltage array up to this window
    aggregated_currents: np.ndarray  # Full aggregated current array up to this window

    # Park point search range boundaries (voltage positions of neighboring peaks)
    best_peak_voltage: float | None = None  # Voltage of the selected best peak center
    best_peak_max_gradient_voltage: float | None = (
        None  # Voltage at max gradient (steepest slope) of best peak
    )
    # Voltage of spatially previous peak (lower peak_idx), None = use trace start
    prev_peak_voltage: float | None = None
    # Voltage of spatially next peak (higher peak_idx), None = use trace end
    next_peak_voltage: float | None = None


def build_sensor_sweep_voltage_list(
    sensor_gates_list: list[str],
    sensor_plunger_index: int,
    base_voltage: float,
    plunger_voltages: np.ndarray,
    gate_voltage_overrides: dict[str, float] | None = None,
) -> list[list[float]]:
    """
    Build voltage list for sensor sweep with one varying plunger gate.

    Creates voltage arrays where all gates are held at base_voltage except
    the sensor plunger, which steps through plunger_voltages.

    Args:
        sensor_gates_list: List of all sensor gate names
        sensor_plunger_index: Index of plunger gate in sensor_gates_list
        base_voltage: Fixed voltage for all non-plunger gates
        plunger_voltages: Array of voltages to sweep on plunger gate
        gate_voltage_overrides: Optional dict of {gate_name: voltage} to override
                                specific gates instead of using base_voltage

    Returns:
        List of voltage arrays ready for device.sweep_nd()
    """
    number_of_gates = len(sensor_gates_list)
    base_voltages = np.full(number_of_gates, base_voltage)

    # Apply gate voltage overrides if provided
    if gate_voltage_overrides:
        for gate_name, voltage in gate_voltage_overrides.items():
            if gate_name in sensor_gates_list:
                gate_index = sensor_gates_list.index(gate_name)
                base_voltages[gate_index] = voltage

    voltage_list = []
    for plunger_voltage in plunger_voltages:
        gate_voltages = base_voltages.copy()
        gate_voltages[sensor_plunger_index] = plunger_voltage
        voltage_list.append(gate_voltages.tolist())

    return voltage_list


def _fit_and_log_single_peak(
    peak_idx: int,
    peak_number: int,
    start_idx: int,
    end_idx: int,
    aggregated_currents: np.ndarray,
    aggregated_voltages: np.ndarray,
    analysis_session: LoggerSession | None,
) -> FittedPeak | None:
    """
    Fit a single peak and log its analysis results.

    Args:
        peak_idx: Index of peak in aggregated trace
        peak_number: Sequential number of peak (1-based, for logging)
        start_idx: Start index of window in aggregated trace
        end_idx: End index of window in aggregated trace
        aggregated_currents: Full aggregated current trace
        aggregated_voltages: Full aggregated voltage trace
        analysis_session: Logging session for saving analysis results

    Returns:
        FittedPeak object if successful, None otherwise
    """
    # Extract window data
    window_currents = aggregated_currents[start_idx:end_idx]
    window_indices = np.arange(len(window_currents))
    peak_idx_in_window = peak_idx - start_idx

    # Safety check
    if peak_idx_in_window < 0 or peak_idx_in_window >= len(window_currents):
        return None

    try:
        # Fit all three models (Lorentzian, sech², pseudo-Voigt) and select best by AICc
        fitted_peak = fit_peak_multi_model(
            window_currents=window_currents,
            window_indices=window_indices,
            peak_idx_in_window=peak_idx_in_window,
            aggregated_voltages=aggregated_voltages,
            window_start_idx=start_idx,
            window_end_idx=end_idx,
            peak_idx_aggregated=peak_idx,
        )

        # Log each peak's multi-model analysis
        if analysis_session:
            analysis_session.log_analysis(
                name=f"peak_{peak_number}_multi_model_fit",
                data={
                    "peak_number": peak_number,
                    "peak_idx": fitted_peak.peak_idx,
                    "best_model": fitted_peak.best_model,
                    "quality_score": fitted_peak.quality_score,
                    "sensitivity": fitted_peak.sensitivity,
                    "sensitivity_voltage": fitted_peak.sensitivity_voltage,
                    "peak_voltage": fitted_peak.peak_voltage,
                    # Lorentzian fit details
                    "lorentzian": {
                        "amplitude": fitted_peak.lorentzian_fit.amplitude,
                        "center_idx": fitted_peak.lorentzian_fit.center_idx,
                        "width": fitted_peak.lorentzian_fit.width,
                        "offset": fitted_peak.lorentzian_fit.offset,
                        "r_squared": fitted_peak.lorentzian_fit.r_squared,
                        "rmse": fitted_peak.lorentzian_fit.rmse,
                        "aicc": fitted_peak.lorentzian_fit.aicc,
                        "fwhm": fitted_peak.lorentzian_fit.fwhm,
                        "area": fitted_peak.lorentzian_fit.area,
                        "skew_resid": fitted_peak.lorentzian_fit.skew_resid,
                    },
                    # sech² fit details
                    "sech2": {
                        "amplitude": fitted_peak.sech2_fit.amplitude,
                        "center_idx": fitted_peak.sech2_fit.center_idx,
                        "width": fitted_peak.sech2_fit.width,
                        "offset": fitted_peak.sech2_fit.offset,
                        "r_squared": fitted_peak.sech2_fit.r_squared,
                        "rmse": fitted_peak.sech2_fit.rmse,
                        "aicc": fitted_peak.sech2_fit.aicc,
                        "fwhm": fitted_peak.sech2_fit.fwhm,
                        "area": fitted_peak.sech2_fit.area,
                        "skew_resid": fitted_peak.sech2_fit.skew_resid,
                    },
                    # Voigt fit details
                    "voigt": {
                        "amplitude": fitted_peak.voigt_fit.amplitude,
                        "center_idx": fitted_peak.voigt_fit.center_idx,
                        "width": fitted_peak.voigt_fit.width,
                        "offset": fitted_peak.voigt_fit.offset,
                        "eta": fitted_peak.voigt_fit.eta,
                        "r_squared": fitted_peak.voigt_fit.r_squared,
                        "rmse": fitted_peak.voigt_fit.rmse,
                        "aicc": fitted_peak.voigt_fit.aicc,
                        "fwhm": fitted_peak.voigt_fit.fwhm,
                        "area": fitted_peak.voigt_fit.area,
                        "skew_resid": fitted_peak.voigt_fit.skew_resid,
                    },
                },
            )

        return fitted_peak

    except Exception as e:
        logger.warning("Failed to fit peak at index %s: %s", peak_idx, e)
        return None


def calculate_peak_window_bounds(
    peak_idx: int,
    peak_index: int,
    peak_indices: list[int],
    trace_length: int,
) -> tuple[int, int] | None:
    """
    Calculate window boundaries for fitting a peak based on its position.

    Uses spacing between peaks to determine appropriate window size, handling
    edge cases for first, middle, and last peaks.

    Args:
        peak_idx: Index of the current peak in the aggregated trace
        peak_index: Position of peak in the sorted peak_indices list (0-based)
        peak_indices: Sorted list of all peak indices
        trace_length: Total length of the aggregated trace

    Returns:
        Tuple of (start_idx, end_idx) for the window, or None if invalid
    """
    if len(peak_indices) == 1:
        # Single peak: use entire trace
        start_idx = 0
        end_idx = trace_length
    elif peak_index == 0:
        # First peak
        distance_to_next = peak_indices[peak_index + 1] - peak_idx
        distance_to_start = peak_idx

        start_idx = (
            int(peak_idx - WINDOW_FRACTION * distance_to_start)
            if distance_to_start < DEFAULT_WINDOW_HALF_WIDTH
            else peak_idx - DEFAULT_WINDOW_HALF_WIDTH
        )
        end_idx = (
            int(peak_idx + WINDOW_FRACTION * distance_to_next)
            if distance_to_next < DEFAULT_WINDOW_HALF_WIDTH
            else peak_idx + DEFAULT_WINDOW_HALF_WIDTH
        )
    elif peak_index == len(peak_indices) - 1:
        # Last peak
        distance_to_end = trace_length - peak_idx
        distance_to_previous = peak_idx - peak_indices[peak_index - 1]

        start_idx = (
            int(peak_idx - distance_to_previous * WINDOW_FRACTION)
            if distance_to_previous < DEFAULT_WINDOW_HALF_WIDTH
            else peak_idx - DEFAULT_WINDOW_HALF_WIDTH
        )
        end_idx = (
            int(peak_idx + distance_to_end * WINDOW_FRACTION)
            if distance_to_end < DEFAULT_WINDOW_HALF_WIDTH
            else peak_idx + DEFAULT_WINDOW_HALF_WIDTH
        )
    else:
        # Middle peaks
        distance_to_next = peak_indices[peak_index + 1] - peak_idx
        distance_to_previous = peak_idx - peak_indices[peak_index - 1]

        start_idx = (
            int(peak_idx - WINDOW_FRACTION * distance_to_previous)
            if distance_to_previous < DEFAULT_WINDOW_HALF_WIDTH
            else peak_idx - DEFAULT_WINDOW_HALF_WIDTH
        )
        end_idx = (
            int(peak_idx + WINDOW_FRACTION * distance_to_next)
            if distance_to_next < DEFAULT_WINDOW_HALF_WIDTH
            else peak_idx + DEFAULT_WINDOW_HALF_WIDTH
        )

    # Ensure valid window bounds
    start_idx = int(max(0, start_idx))
    end_idx = int(min(trace_length, end_idx))

    # Ensure peak is properly within window (not at edges)
    if start_idx >= peak_idx:
        start_idx = int(max(0, peak_idx - 5))  # Ensure at least 5 points before peak
    if end_idx <= peak_idx:
        end_idx = int(
            min(trace_length, peak_idx + 5)
        )  # Ensure at least 5 points after peak

    # Final validation: ensure peak is within window
    if peak_idx < start_idx or peak_idx >= end_idx:
        return None

    return (start_idx, end_idx)


def _calculate_local_slope(
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
    # Find index closest to target voltage
    idx = int(np.argmin(np.abs(voltages - target_voltage)))

    # Define window boundaries
    start_idx = max(0, idx - window_points)
    end_idx = min(len(voltages), idx + window_points + 1)

    # Extract window data
    v_window = voltages[start_idx:end_idx]
    i_window = currents[start_idx:end_idx]

    # Validate window size
    if len(v_window) < 3:
        raise RoutineError(
            f"Insufficient points for local slope calculation at voltage {target_voltage}V. "
            f"Only {len(v_window)} points available."
        )

    # Perform linear regression: I = slope * V + intercept
    try:
        # Use numpy polyfit for simple linear regression (degree 1)
        slope, _ = np.polyfit(v_window, i_window, deg=1)
        return float(slope)
    except Exception as e:
        raise RoutineError(
            f"Failed to calculate local slope at voltage {target_voltage}V: {e}"
        ) from e


def _calculate_voltage_noise(current_std: float, local_slope: float) -> float:
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

    if abs_slope < 1e-12:  # Avoid division by near-zero slopes
        raise RoutineError(
            f"Local slope magnitude too small ({abs_slope:.2e} A/V) for voltage noise calculation. "
            "Cannot convert current noise to voltage noise with near-zero gradient."
        )

    voltage_noise = current_std / abs_slope
    return float(voltage_noise)


def _normalize_sensitivity_scores(fitted_peaks: list[FittedPeak]) -> None:
    """
    Normalize sensitivity scores across all fitted peaks using min-max scaling.

    Args:
        fitted_peaks: List of FittedPeak objects to normalize
    """
    if not fitted_peaks:
        return

    sensitivities = [peak.sensitivity for peak in fitted_peaks]
    min_sens = min(sensitivities)
    max_sens = max(sensitivities)
    sens_range = max_sens - min_sens

    if sens_range > 0:
        for peak in fitted_peaks:
            peak.sensitivity_score = (peak.sensitivity - min_sens) / sens_range
    else:
        # All peaks have same sensitivity
        for peak in fitted_peaks:
            peak.sensitivity_score = 1.0


def calculate_quality_scores(fitted_peaks: list[FittedPeak]) -> None:
    """
    Calculate quality scores for all fitted peaks using normalized sensitivity scores.

    Args:
        fitted_peaks: List of FittedPeak objects (must have sensitivity_score set)
    """
    for peak in fitted_peaks:
        # Get best fit model
        best_fit = getattr(peak, f"{peak.best_model.lower()}_fit")

        # Get y_max from the window for normalization
        y_max = (
            float(np.max(np.abs(peak.window_currents)))
            if len(peak.window_currents) > 0
            else 1.0
        )

        # Validate sensitivity_score is set before quality calculation
        if peak.sensitivity_score is None:
            raise RoutineError(
                f"Sensitivity score not set for peak at index {peak.peak_idx}. "
                "Ensure _normalize_sensitivity_scores() is called first."
            )
        peak.quality_score = calculate_quality_score(
            r_squared=best_fit.r_squared,
            rmse=best_fit.rmse,
            y_max=y_max,
            skew=best_fit.skew_resid,
            sensitivity_score=peak.sensitivity_score,
        )


def analyze_find_first_peak_voltages(
    aggregated_currents: np.ndarray,
    aggregated_voltages: np.ndarray,
    peak_indices_aggregated: list[int],
    analysis_session: LoggerSession | None,
) -> list[FittedPeak]:
    """
    Analyze and fit multiple peaks in aggregated current trace.

    Args:
        aggregated_currents: Aggregated current trace data
        aggregated_voltages: Aggregated voltage trace data
        peak_indices_aggregated: List of peak indices in aggregated trace
        analysis_session: Logging session for saving analysis results

    Returns:
        List of FittedPeak objects with quality metrics
    """
    fitted_peaks = []
    # Remove duplicate peak indices
    peak_indices_aggregated = sorted(set(peak_indices_aggregated))

    # Fit each peak
    for i, peak_idx in enumerate(peak_indices_aggregated):
        # Calculate window boundaries for this peak
        window_bounds = calculate_peak_window_bounds(
            peak_idx=peak_idx,
            peak_index=i,
            peak_indices=peak_indices_aggregated,
            trace_length=len(aggregated_currents),
        )

        if window_bounds is None:
            continue

        start_idx, end_idx = window_bounds

        # Fit peak and log analysis
        fitted_peak = _fit_and_log_single_peak(
            peak_idx=peak_idx,
            peak_number=i + 1,
            start_idx=start_idx,
            end_idx=end_idx,
            aggregated_currents=aggregated_currents,
            aggregated_voltages=aggregated_voltages,
            analysis_session=analysis_session,
        )

        if fitted_peak is not None:
            fitted_peaks.append(fitted_peak)

    # Log summary of peak fitting
    if analysis_session:
        analysis_session.log_analysis(
            name="peak_fitting_summary",
            data={
                "num_peaks_detected": len(peak_indices_aggregated),
                "num_peaks_fitted": len(fitted_peaks),
                "fitted_peak_indices": [int(p.peak_idx) for p in fitted_peaks],
            },
        )

    # Normalize sensitivity scores and calculate quality scores
    if fitted_peaks:
        _normalize_sensitivity_scores(fitted_peaks)
        calculate_quality_scores(fitted_peaks)

    return fitted_peaks


def _calculate_combined_scores(
    candidates: list[StablePeakCandidate],
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

    # Extract scores for normalization
    original_scores = [c.original_score for c in candidates]
    voltage_noises = [c.stability_measurement.voltage_noise for c in candidates]

    # Validate voltage noises are all positive and non-zero
    for i, vn in enumerate(voltage_noises):
        if vn <= 0 or not np.isfinite(vn):
            raise RoutineError(
                f"Invalid voltage noise for candidate {i}: {vn}. "
                "Voltage noise must be positive and finite."
            )

    # Min-max normalize original scores to [0, 1]
    min_orig = min(original_scores)
    max_orig = max(original_scores)
    orig_range = max_orig - min_orig

    if orig_range > 0:
        normalized_original = [
            (score - min_orig) / orig_range for score in original_scores
        ]
    else:
        # All original scores identical
        normalized_original = [1.0] * len(candidates)

    # Convert voltage noise to stability score (invert: lower noise = higher score)
    # stability_score = 1 / voltage_noise, then normalize by dividing by max
    # This preserves relative ratios: best peak = 1.0, others = their_score / max_score
    # Note: voltage_noises are already validated to be positive and finite
    stability_scores_raw = [1.0 / vn for vn in voltage_noises]

    max_stab = max(stability_scores_raw)

    # Normalize by dividing by max (preserves relative ratios)
    normalized_stability = [score / max_stab for score in stability_scores_raw]

    # Calculate combined scores and update candidates
    for i, candidate in enumerate(candidates):
        candidate.stability_measurement.stability_score = normalized_stability[i]
        candidate.combined_score = (
            original_weight * normalized_original[i]
            + stability_weight * normalized_stability[i]
        )


def _measure_peak_stability(
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
    session: LoggerSession | None = None,
) -> StabilityMeasurement:
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

    # Apply bias voltage to bias gate
    device.jump({bias_gate: bias_voltage}, wait_for_settling=True)
    time.sleep(DEFAULT_SETTLING_TIME_S)

    # Construct voltage state at max-gradient point
    device_state = dict.fromkeys(sensor_gates_list, mean_reservoir_saturation_voltage)
    device_state[sensor_plunger_gate] = peak.sensitivity_voltage

    # Move to max-gradient point
    device.jump(device_state, wait_for_settling=True)
    time.sleep(DEFAULT_SETTLING_TIME_S)

    # Record current vs. time during hold
    logger.info(
        "Holding at max-gradient point for %.1f seconds, recording current...",
        hold_time_seconds,
    )

    time_array = []
    current_array = []
    start_time = time.time()

    try:
        # Continuously measure current until hold time expires
        # Use device default sampling - measure as fast as possible
        while True:
            elapsed = time.time() - start_time
            if elapsed >= hold_time_seconds:
                break

            # Measure current at max-gradient point
            current = device.measure(measure_electrode)
            time_array.append(elapsed)
            current_array.append(current)

            # Small sleep to avoid overwhelming the device (adjust based on device capabilities)
            # If device has built-in rate limiting, this can be removed
            time.sleep(0.01)  # 10ms -> ~100 Hz sampling rate

    except Exception as e:
        raise RoutineError(
            f"Failed to measure stability for peak {peak_index + 1}: {e}"
        ) from e

    # Convert to numpy arrays
    time_array_np = np.array(time_array, dtype=np.float64)
    current_array_np = np.array(current_array, dtype=np.float64)

    # Validate that we collected data
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

    # Calculate statistics
    current_mean = float(np.mean(current_array_np))
    current_std = float(np.std(current_array_np))

    logger.info(
        "Current statistics: mean = %.3e A, std = %.3e A (%.2f%% of mean)",
        current_mean,
        current_std,
        100.0 * current_std / abs(current_mean) if abs(current_mean) > 0 else 0.0,
    )

    # Calculate local slope at max-gradient voltage from original sweep data
    try:
        local_slope = _calculate_local_slope(
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

    # Calculate voltage noise: σᵥ = σᵢ / |dI/dV|
    try:
        voltage_noise = _calculate_voltage_noise(current_std, local_slope)
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

    # Create StabilityMeasurement object
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

    # Log stability measurement data
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


def many_window_barrier_sweep(  # pylint: disable=too-many-locals,too-many-statements
    ctx: RoutineContext,
    sensor_gates_list: list[str],
    sensor_plunger_range: tuple[float, float],
    window_size: float,
    current_trace_number_of_points: int,
    mean_reservoir_saturation_voltage: float,
    sensor_plunger_index: int,
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    session: LoggerSession | None = None,
    gate_voltage_overrides: dict[str, float] | None = None,
) -> SensorDotPlungerSweepOutput:
    """
    Run a many window sensor plunger voltage sweep to detect single dot formation.

    Sets all gates except the sensor plunger to their saturation voltages, then
    sweeps the sensor plunger at sequential voltage windows to identify single dot
    formation using coulomb blockade classification and peak detection.

    Args:
        ctx: Routine context containing device and models client
        sensor_gates_list: List of gate names in sensor group
        sensor_plunger_range: (min, max) voltage range for sensor plunger
        window_size: Voltage range per measurement window
        current_trace_number_of_points: Points per current trace window
        mean_reservoir_saturation_voltage: Voltage for all gates except sensor plunger
        sensor_plunger_index: Index of sensor plunger in sensor_gates_list
        measure_electrode: Electrode to measure current from
        bias_gate: Name of the bias gate (contact) to apply bias voltage
        bias_voltage: Voltage to apply to bias gate during measurements
        session: Logger session for measurements and analysis
        gate_voltage_overrides: Optional dict of {gate_name: voltage} to override
                                specific gates instead of using mean_reservoir_saturation_voltage

    Returns:
        SensorDotPlungerSweepOutput containing the result of the sweep.
    """
    client: ConductorQuantum = ctx.resources.models_client
    device: Device = ctx.resources.device

    # Apply bias voltage to bias gate
    device.jump({bias_gate: bias_voltage}, wait_for_settling=True)
    time.sleep(DEFAULT_SETTLING_TIME_S)
    # Calculate sequential sweep parameters
    min_v, max_v = sensor_plunger_range

    # Generate start voltages for sequential windows from saturation to cutoff
    sensor_plunger_start_voltages = np.arange(min_v, max_v, window_size)

    # Initialize aggregated trace arrays
    aggregated_voltages = np.array([], dtype=np.float32)
    aggregated_currents = np.array([], dtype=np.float32)
    fitted_peaks = []  # Track fitted peaks across all windows
    last_classification = False  # Track last classification result
    last_peak_indices = []  # Track last peak indices

    max_v_float = float(max_v)
    for idx, sp_start_voltage in enumerate(sensor_plunger_start_voltages):
        # Generate sensor plunger voltages for this window
        # Check if the end voltage is within the range
        sp_start_voltage_float = float(sp_start_voltage)
        end_voltage_candidate = sp_start_voltage_float + window_size
        sp_end_voltage = float(np.minimum(end_voltage_candidate, max_v_float))

        sp_sweep_voltages = np.linspace(
            sp_start_voltage_float,
            sp_end_voltage,
            current_trace_number_of_points,
            endpoint=False,
        )

        # Build voltage list for sweep_nd using helper function
        voltage_list = build_sensor_sweep_voltage_list(
            sensor_gates_list=sensor_gates_list,
            sensor_plunger_index=sensor_plunger_index,
            base_voltage=mean_reservoir_saturation_voltage,
            plunger_voltages=sp_sweep_voltages,
            gate_voltage_overrides=gate_voltage_overrides,
        )

        # Set device to first voltage point and allow settling time to avoid current spikes
        first_voltage_point = dict(
            zip(sensor_gates_list, voltage_list[0], strict=False)
        )
        device.jump(first_voltage_point, wait_for_settling=True)
        time.sleep(DEFAULT_SETTLING_TIME_S)

        # Perform current trace measurement using device.sweep_nd
        _, current_trace = device.sweep_nd(
            gate_electrodes=sensor_gates_list,
            voltages=voltage_list,
            measure_electrode=measure_electrode,
            session=session,
        )

        # Append to aggregated trace
        aggregated_voltages = np.concatenate([aggregated_voltages, sp_sweep_voltages])
        aggregated_currents = np.concatenate([aggregated_currents, current_trace])

        # Run coulomb blockade classifier on aggregated trace
        # Model handles sliding windows internally for inputs > 128
        try:
            coulomb_result = client.models.execute(
                model=COULOMB_CLASSIFIER_MODEL, data=aggregated_currents
            ).output
        except Exception as e:
            raise RoutineError(f"ML model execution failed: {e}") from e
        classification = bool(coulomb_result["classification"])
        score = float(coulomb_result["score"])

        # Detect peaks if classification is positive
        if classification:
            # Model handles sliding windows internally for inputs > 1024
            try:
                aggregated_peak_indices = client.models.execute(
                    model=PEAK_DETECTOR_MODEL, data=aggregated_currents
                ).output["peak_indices"]
            except Exception as e:
                raise RoutineError(f"ML model execution failed: {e}") from e

            # Analyze and fit peaks
            fitted_peaks = analyze_find_first_peak_voltages(
                aggregated_currents,
                aggregated_voltages,
                aggregated_peak_indices,
                session,
            )

            # Store last successful classification info
            last_classification = classification
            last_peak_indices = aggregated_peak_indices

        # Log analysis for this window
        if session:
            session.log_analysis(
                name=f"sensor_plunger_window_{idx}",
                data={
                    "window_idx": idx,
                    "sensor_plunger_start_voltage": sp_start_voltage_float,
                    "sensor_plunger_end_voltage": sp_end_voltage,
                    "classification": classification,
                    "score": score,
                    "num_peaks": len(aggregated_peak_indices) if classification else 0,
                },
            )

    # Process fitted peaks and select the best one
    if not fitted_peaks:
        raise RoutineError("No peaks were successfully fitted during the sweep")

    # Normalize sensitivity scores and recalculate quality scores with normalized values
    _normalize_sensitivity_scores(fitted_peaks)
    calculate_quality_scores(fitted_peaks)

    # Sort peaks by quality score (descending)
    def key_func(p: FittedPeak) -> float:
        return p.quality_score or 0.0

    sorted_peaks: list[FittedPeak] = sorted(fitted_peaks, key=key_func, reverse=True)

    # Select the best peak by quality score
    best_peak = sorted_peaks[0]
    if best_peak.quality_score is None:
        raise RoutineError("Quality score not calculated for best peak")

    # Find spatially neighboring peaks (by peak_idx position in aggregated trace)
    # Previous peak: highest peak_idx < best_peak.peak_idx
    # Next peak: lowest peak_idx > best_peak.peak_idx
    prev_peak = None
    next_peak = None

    for peak in fitted_peaks:
        if peak.peak_idx < best_peak.peak_idx:
            if prev_peak is None or peak.peak_idx > prev_peak.peak_idx:
                prev_peak = peak
        elif peak.peak_idx > best_peak.peak_idx:
            if next_peak is None or peak.peak_idx < next_peak.peak_idx:
                next_peak = peak

    # Extract voltages for park point search range boundaries
    prev_peak_voltage: float | None = (
        float(prev_peak.peak_voltage) if prev_peak is not None else None
    )
    next_peak_voltage: float | None = (
        float(next_peak.peak_voltage) if next_peak is not None else None
    )

    # Create SensorDotPlungerSweepOutput with the best peak's data
    result = SensorDotPlungerSweepOutput(
        sensor_plunger_voltage=float(best_peak.peak_voltage),
        classification=last_classification,
        score=float(best_peak.quality_score),  # Use quality_score for best peak ranking
        peak_indices=last_peak_indices,
        num_peaks=len(fitted_peaks),
        aggregated_voltages=aggregated_voltages,
        aggregated_currents=aggregated_currents,
        best_peak_voltage=float(best_peak.peak_voltage),
        best_peak_max_gradient_voltage=float(
            best_peak.sensitivity_voltage
        ),  # Sensitivity voltage (max gradient point)
        prev_peak_voltage=prev_peak_voltage,
        next_peak_voltage=next_peak_voltage,
    )

    return result


@routine
def find_sensor_peak(  # pylint: disable=too-many-locals
    ctx: RoutineContext,
    peak_spacing: float,
    sensor_group_name: str,
    sensor_plunger_gate: str,
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    zero_control_side: bool = False,
    gate_voltage_overrides: dict[str, float] | None = None,
    session: LoggerSession | None = None,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> dict[str, Any]:
    """
    Find the optimal charge sensor operating point by sweeping sensor plunger.

    This routine performs a multi-window sweep of the sensor plunger gate to identify
    the best Coulomb blockade peak for charge sensing. It analyzes peaks using multi-model
    fitting (Lorentzian, sech², pseudo-Voigt) and selects the peak with the highest quality
    score. The routine also calculates a narrowed voltage range around the best peak for
    subsequent high-resolution measurements.

    Args:
        ctx: Routine context containing device resources and previous results. Requires:
             - ctx.results["global_accumulation"]["global_turn_on_voltage"]
             - ctx.results["finger_gate_characterization"][sensor_plunger_gate]
        peak_spacing: Expected peak spacing in volts (e.g., 0.020 for 20mV)
        sensor_group_name: Name of sensor side group (e.g., "side_B")
        sensor_plunger_gate: Name of the sensor plunger gate on sensor side
        measure_electrode: Electrode to measure current from (e.g., "OUT_B")
        bias_gate: Name of the bias gate (contact) to apply bias voltage (e.g., "IN_A_B")
        bias_voltage: Voltage to apply to bias gate during measurements (V)
        zero_control_side: If True, set control group gates to 0V before sweep.
            If False, maintain current control voltages. Shared reservoirs always
            set to sensor group's global turn-on voltage unless overridden. (default: True)
        gate_voltage_overrides: Optional dict of {gate_name: voltage} to override
                                specific sensor group gates (e.g., shared reservoirs)
                                instead of using global_turn_on_voltage (default: None)
        session: Logger session for measurements and analysis

    Returns:
        dict: Contains:
            - best_peak_voltage: Voltage at the center of the best peak (V)
            - best_peak_max_gradient_voltage: Voltage at maximum gradient (optimal sensing point) (V)
            - narrowed_sensor_plunger_range: (min, max) narrowed voltage range for high-res sweeps (V)
            - prev_peak_voltage: Voltage of previous peak or fallback value (V)
            - next_peak_voltage: Voltage of next peak or fallback value (V)
            - mean_reservoir_saturation_voltage: Saturation voltage used for sensor gates (V)
            - sensor_gates_list: List of sensor gate names
            - sensor_plunger_index: Index of sensor plunger in sensor_gates_list
            - step_size: Calculated step size for narrowed sweeps (V)
            - sensor_park_point: Dict of all sensor gate voltages at park point {gate: voltage}

    Raises:
        RoutineError: If required previous results are missing or peak finding fails

    Notes:
        - Uses 2x peak_spacing for initial window size
        - Automatically sets device to optimal sensing point after finding peak
        - Calculates narrowed range based on neighboring peaks or fallback spacing
        - Control group handling: specific gates zeroed or maintained, reservoirs
          always set to sensor's global turn-on
    """
    if peak_spacing <= 0:
        raise RoutineError("peak_spacing must be greater than 0")

    # Hardcode per requirements
    current_trace_number_of_points = ML_MODEL_INPUT_SIZE

    # Get device
    device = ctx.resources.device

    # Get groups from device config
    sensor_group = device.device_config.groups[sensor_group_name]
    sensor_gates = list(sensor_group.gates)
    sensor_gates = filter_gates_by_group(ctx, sensor_gates)

    # Get global turn-on voltage from global_accumulation results
    global_accumulation_results = ctx.results.get(
        f"global_accumulation_{sensor_group_name}",
        ctx.results.get("global_accumulation", {}),
    )
    global_turn_on_voltage = global_accumulation_results.get("global_turn_on_voltage")

    if global_turn_on_voltage is None:
        raise RoutineError(
            f"global_turn_on_voltage not found in ctx.results for group '{sensor_group_name}'. "
            "Please run global_accumulation routine first for this group."
        )

    # Use global turn-on voltage as the saturation voltage for all gates
    mean_reservoir_saturation_voltage = float(global_turn_on_voltage)

    # Get sensor plunger gate parameters from health check results
    finger_gate_char = ctx.results.get(
        f"finger_gate_characterization_{sensor_group_name}",
        ctx.results.get("finger_gate_characterization", {}),
    )
    if not finger_gate_char:
        raise RoutineError(
            f"finger_gate_characterization not found in ctx.results for group '{sensor_group_name}'. "
            "Please run finger_gate_characterization routine first for this group."
        )
    # Extract nested "finger_gate_characterization" dict if present
    # (finger_gate_characterization routine returns {"finger_gate_characterization": {...}})
    if "finger_gate_characterization" in finger_gate_char:
        finger_gate_char = finger_gate_char["finger_gate_characterization"]
    if sensor_plunger_gate not in finger_gate_char:
        raise RoutineError(
            f"Sensor plunger gate '{sensor_plunger_gate}' not found in "
            "finger_gate_characterization results."
        )

    sensor_plunger_cutoff_voltage = finger_gate_char[sensor_plunger_gate][
        "cutoff_voltage"
    ]
    sensor_plunger_saturation_voltage = finger_gate_char[sensor_plunger_gate][
        "saturation_voltage"
    ]

    sensor_plunger_range = (
        sensor_plunger_saturation_voltage,
        sensor_plunger_cutoff_voltage,
    )

    # Get the index of the sensor plunger in the sensor gates list
    sensor_gates_list = list(sensor_gates)
    sensor_plunger_index = sensor_gates_list.index(sensor_plunger_gate)

    # Handle control side gates before sensor sweep
    # Identify all gates and separate control from sensor
    all_control_gates = device.control_gates
    sensor_gates_set = set(sensor_gates_list)
    control_gates = [g for g in all_control_gates if g not in sensor_gates_set]

    # Set control side state (if there are any control gates)
    # Note: Shared reservoirs are in sensor_gates_list and handled by sensor sweep
    if control_gates:
        if zero_control_side:
            # Set control gates to 0V
            logger.info("Setting control gates to 0V: %s", control_gates)
            control_state = dict.fromkeys(control_gates, 0.0)
        else:
            # Maintain current control voltages
            current_control_voltages = device.check(control_gates)
            control_state = dict(
                zip(control_gates, current_control_voltages, strict=False)
            )
            logger.info(
                "Maintaining control gates at current voltages: %s",
                control_state,
            )

        # Apply control state
        device.jump(control_state, wait_for_settling=True)

    # Use 2x peak spacing for initial multi-window sweep
    window_size = peak_spacing * INITIAL_WINDOW_MULTIPLIER

    sensor_plunger_sweep_output = many_window_barrier_sweep(
        ctx=ctx,
        sensor_gates_list=sensor_gates_list,
        sensor_plunger_range=sensor_plunger_range,
        window_size=window_size,
        current_trace_number_of_points=current_trace_number_of_points,
        mean_reservoir_saturation_voltage=mean_reservoir_saturation_voltage,
        sensor_plunger_index=sensor_plunger_index,
        measure_electrode=measure_electrode,
        bias_gate=bias_gate,
        bias_voltage=bias_voltage,
        session=session,
        gate_voltage_overrides=gate_voltage_overrides,
    )

    # Handle cases with fewer than 3 peaks by using fallback boundaries
    best_peak_voltage = sensor_plunger_sweep_output.best_peak_voltage
    if best_peak_voltage is None:
        raise RoutineError("Best peak voltage not found in sensor sweep output")
    best_peak_max_gradient_voltage = (
        sensor_plunger_sweep_output.best_peak_max_gradient_voltage
    )
    if best_peak_max_gradient_voltage is None:
        raise RoutineError(
            "Best peak max gradient voltage not found in sensor sweep output"
        )

    # Set fallback values for missing neighboring peaks
    if sensor_plunger_sweep_output.prev_peak_voltage is None:
        prev_peak_voltage_fallback = best_peak_voltage - peak_spacing
        logger.warning(
            "No previous peak found. Using fallback: best_peak - peak_spacing = %.6fV",
            prev_peak_voltage_fallback,
        )
    else:
        prev_peak_voltage_fallback = sensor_plunger_sweep_output.prev_peak_voltage

    if sensor_plunger_sweep_output.next_peak_voltage is None:
        next_peak_voltage_fallback = best_peak_voltage + peak_spacing
        logger.warning(
            "No next peak found. Using fallback: best_peak + peak_spacing = %.6fV",
            next_peak_voltage_fallback,
        )
    else:
        next_peak_voltage_fallback = sensor_plunger_sweep_output.next_peak_voltage

    # Reconstruct the voltage configuration at the max gradient point
    # (optimal charge sensing point)
    # During the sweep, all gates were at mean_reservoir_saturation_voltage
    # except sensor plunger
    sensor_dot_state = dict.fromkeys(
        sensor_gates_list, mean_reservoir_saturation_voltage
    )
    sensor_dot_state[sensor_plunger_gate] = best_peak_max_gradient_voltage

    device.jump(sensor_dot_state, wait_for_settling=True)

    step_size_used = window_size / current_trace_number_of_points
    new_step_size = REFINED_STEP_MULTIPLIER * step_size_used

    # Calculate narrowed range using fallback values if neighboring peaks are missing
    number_of_points_between_previous_and_best_peak = (
        WINDOW_FRACTION
        * (best_peak_voltage - prev_peak_voltage_fallback)
        / new_step_size
    )
    number_of_points_between_best_and_next_peak = (
        WINDOW_FRACTION
        * (next_peak_voltage_fallback - best_peak_voltage)
        / new_step_size
    )

    start_of_range = (
        prev_peak_voltage_fallback
        if number_of_points_between_previous_and_best_peak < DEFAULT_WINDOW_HALF_WIDTH
        else best_peak_voltage - DEFAULT_WINDOW_HALF_WIDTH * new_step_size
    )
    end_of_range = (
        next_peak_voltage_fallback
        if number_of_points_between_best_and_next_peak < DEFAULT_WINDOW_HALF_WIDTH
        else best_peak_voltage + DEFAULT_WINDOW_HALF_WIDTH * new_step_size
    )

    narrowed_sensor_plunger_range = (start_of_range, end_of_range)
    logger.info(
        "Narrowed sweep range: %sV to %sV",
        narrowed_sensor_plunger_range[0],
        narrowed_sensor_plunger_range[1],
    )

    # Log park point analysis
    if session:
        session.log_analysis(
            name="sensor_park_point",
            data={
                "sensor_park_point": sensor_dot_state,
                "sensor_plunger_gate": sensor_plunger_gate,
                "sensor_plunger_park_voltage": float(best_peak_max_gradient_voltage),
                "other_gates_voltage": float(mean_reservoir_saturation_voltage),
                "best_peak_center_voltage": float(best_peak_voltage),
                "best_peak_max_gradient_voltage": float(best_peak_max_gradient_voltage),
                "narrowed_sensor_plunger_range": narrowed_sensor_plunger_range,
                "prev_peak_voltage": float(prev_peak_voltage_fallback),
                "next_peak_voltage": float(next_peak_voltage_fallback),
                "step_size": float(new_step_size),
            },
        )

    result = {
        "best_peak_voltage": float(best_peak_voltage),
        "best_peak_max_gradient_voltage": float(best_peak_max_gradient_voltage),
        "narrowed_sensor_plunger_range": narrowed_sensor_plunger_range,
        "prev_peak_voltage": float(prev_peak_voltage_fallback),
        "next_peak_voltage": float(next_peak_voltage_fallback),
        "mean_reservoir_saturation_voltage": float(mean_reservoir_saturation_voltage),
        "sensor_gates_list": sensor_gates_list,
        "sensor_plunger_index": sensor_plunger_index,
        "step_size": float(new_step_size),
        "sensor_park_point": sensor_dot_state,
    }

    return result


@routine
def find_stable_sensor_peak(  # pylint: disable=too-many-locals,too-many-statements
    ctx: RoutineContext,
    peak_spacing: float,
    sensor_group_name: str,
    sensor_plunger_gate: str,
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    zero_control_side: bool = False,
    gate_voltage_overrides: dict[str, float] | None = None,
    top_n_peaks: int = 3,
    hold_time_seconds: float = 120.0,
    session: LoggerSession | None = None,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> dict[str, Any]:
    """
    Find the most stable charge sensor operating point with 2-minute stability testing.

    This routine extends find_sensor_peak by measuring peak stability. After identifying
    Coulomb blockade peaks, it tests the top N peaks by holding at each peak's
    max-gradient point for 2 minutes while recording current vs. time. Peak position
    stability is quantified as voltage noise (σᵥ = σᵢ / |dI/dV|), and the final peak
    is selected by combining original quality score (50%) with stability score (50%).

    Args:
        ctx: Routine context containing device resources and previous results. Requires:
             - ctx.results["global_accumulation"]["global_turn_on_voltage"]
             - ctx.results["finger_gate_characterization"][sensor_plunger_gate]
        peak_spacing: Expected peak spacing in volts (e.g., 0.020 for 20mV)
        sensor_group_name: Name of sensor side group (e.g., "side_B")
        sensor_plunger_gate: Name of the sensor plunger gate on sensor side
        measure_electrode: Electrode to measure current from (e.g., "OUT_B")
        bias_gate: Name of the bias gate (contact) to apply bias voltage (e.g., "IN_A_B")
        bias_voltage: Voltage to apply to bias gate during measurements (V)
        zero_control_side: If True, set control group gates to 0V before sweep.
            If False, maintain current control voltages. (default: False)
        gate_voltage_overrides: Optional dict of {gate_name: voltage} to override
                                specific sensor group gates (default: None)
        top_n_peaks: Number of top-scoring peaks to test for stability (default: 3)
        hold_time_seconds: Duration to hold at each peak for stability measurement (default: 120.0)
        session: Logger session for measurements and analysis

    Returns:
        dict: Same format as find_sensor_peak, containing:
            - best_peak_voltage: Voltage at center of most stable peak (V)
            - best_peak_max_gradient_voltage: Voltage at maximum gradient (V)
            - narrowed_sensor_plunger_range: (min, max) voltage range (V)
            - prev_peak_voltage: Previous peak voltage or fallback (V)
            - next_peak_voltage: Next peak voltage or fallback (V)
            - mean_reservoir_saturation_voltage: Saturation voltage (V)
            - sensor_gates_list: List of sensor gate names
            - sensor_plunger_index: Index of plunger in sensor_gates_list
            - step_size: Refined step size for narrowed sweeps (V)
            - sensor_park_point: Gate voltages at optimal point {gate: voltage}

    Raises:
        RoutineError: If required previous results are missing or peak finding fails

    Notes:
        - Uses same peak detection as find_sensor_peak (ML-based with multi-model fitting)
        - Tests top N peaks (by original quality score) for stability
        - Selects best peak using combined score: 50% original + 50% stability
        - Total runtime: ~(top_n_peaks * hold_time_seconds) longer than find_sensor_peak
        - For top_n_peaks=3 and hold_time_seconds=120: adds ~6 minutes to routine
    """
    if peak_spacing <= 0:
        raise RoutineError("peak_spacing must be greater than 0")

    if top_n_peaks < 1:
        raise RoutineError("top_n_peaks must be at least 1")

    if hold_time_seconds <= 0:
        raise RoutineError("hold_time_seconds must be greater than 0")

    # Hardcode per requirements
    current_trace_number_of_points = ML_MODEL_INPUT_SIZE

    # Get device
    device = ctx.resources.device

    # Get groups from device config
    sensor_group = device.device_config.groups[sensor_group_name]
    sensor_gates = list(sensor_group.gates)
    sensor_gates = filter_gates_by_group(ctx, sensor_gates)

    # Get global turn-on voltage from global_accumulation results
    global_accumulation_results = ctx.results.get(
        f"global_accumulation_{sensor_group_name}",
        ctx.results.get("global_accumulation", {}),
    )
    global_turn_on_voltage = global_accumulation_results.get("global_turn_on_voltage")

    if global_turn_on_voltage is None:
        raise RoutineError(
            f"global_turn_on_voltage not found in ctx.results for group '{sensor_group_name}'. "
            "Please run global_accumulation routine first for this group."
        )

    # Use global turn-on voltage as the saturation voltage for all gates
    mean_reservoir_saturation_voltage = float(global_turn_on_voltage)

    # Get sensor plunger gate parameters from health check results
    finger_gate_char = ctx.results.get(
        f"finger_gate_characterization_{sensor_group_name}",
        ctx.results.get("finger_gate_characterization", {}),
    )
    if not finger_gate_char:
        raise RoutineError(
            f"finger_gate_characterization not found in ctx.results for group '{sensor_group_name}'. "
            "Please run finger_gate_characterization routine first for this group."
        )
    # Extract nested "finger_gate_characterization" dict if present
    if "finger_gate_characterization" in finger_gate_char:
        finger_gate_char = finger_gate_char["finger_gate_characterization"]
    if sensor_plunger_gate not in finger_gate_char:
        raise RoutineError(
            f"Sensor plunger gate '{sensor_plunger_gate}' not found in "
            "finger_gate_characterization results."
        )

    sensor_plunger_cutoff_voltage = finger_gate_char[sensor_plunger_gate][
        "cutoff_voltage"
    ]
    sensor_plunger_saturation_voltage = finger_gate_char[sensor_plunger_gate][
        "saturation_voltage"
    ]

    sensor_plunger_range = (
        sensor_plunger_saturation_voltage,
        sensor_plunger_cutoff_voltage,
    )

    # Get the index of the sensor plunger in the sensor gates list
    sensor_gates_list = list(sensor_gates)
    sensor_plunger_index = sensor_gates_list.index(sensor_plunger_gate)

    # Handle control side gates before sensor sweep
    all_control_gates = device.control_gates
    sensor_gates_set = set(sensor_gates_list)
    control_gates = [g for g in all_control_gates if g not in sensor_gates_set]

    # Set control side state (if there are any control gates)
    if control_gates:
        if zero_control_side:
            logger.info("Setting control gates to 0V: %s", control_gates)
            control_state = dict.fromkeys(control_gates, 0.0)
        else:
            current_control_voltages = device.check(control_gates)
            control_state = dict(
                zip(control_gates, current_control_voltages, strict=False)
            )
            logger.info(
                "Maintaining control gates at current voltages: %s",
                control_state,
            )

        # Apply control state
        device.jump(control_state, wait_for_settling=True)

    # Use 2x peak spacing for initial multi-window sweep
    window_size = peak_spacing * INITIAL_WINDOW_MULTIPLIER

    # Run initial peak detection sweep (same as find_sensor_peak)
    logger.info("Running initial peak detection sweep...")
    sensor_plunger_sweep_output = many_window_barrier_sweep(
        ctx=ctx,
        sensor_gates_list=sensor_gates_list,
        sensor_plunger_range=sensor_plunger_range,
        window_size=window_size,
        current_trace_number_of_points=current_trace_number_of_points,
        mean_reservoir_saturation_voltage=mean_reservoir_saturation_voltage,
        sensor_plunger_index=sensor_plunger_index,
        measure_electrode=measure_electrode,
        bias_gate=bias_gate,
        bias_voltage=bias_voltage,
        session=session,
        gate_voltage_overrides=gate_voltage_overrides,
    )

    # Extract aggregated trace data for stability measurements
    aggregated_voltages = sensor_plunger_sweep_output.aggregated_voltages
    aggregated_currents = sensor_plunger_sweep_output.aggregated_currents

    # Get all fitted peaks from the sweep (they're already scored and sorted)
    # We need to re-run the peak detection to get all peaks, not just the best one
    # The many_window_barrier_sweep returns the best peak, but we need all peaks

    # Re-analyze to get all fitted peaks
    logger.info("Re-analyzing sweep to extract all fitted peaks...")
    try:
        peak_indices = sensor_plunger_sweep_output.peak_indices
        if not peak_indices:
            raise RoutineError("No peaks detected during initial sweep")

        fitted_peaks = analyze_find_first_peak_voltages(
            aggregated_currents=aggregated_currents,
            aggregated_voltages=aggregated_voltages,
            peak_indices_aggregated=peak_indices,
            analysis_session=session,
        )

        if not fitted_peaks:
            raise RoutineError("No peaks were successfully fitted")

        logger.info("Found %d fitted peaks", len(fitted_peaks))

    except Exception as e:
        raise RoutineError(f"Failed to analyze peaks: {e}") from e

    # Sort peaks by quality score (descending) and select top N
    sorted_peaks = sorted(
        fitted_peaks, key=lambda p: p.quality_score or 0.0, reverse=True
    )
    num_peaks_to_test = min(top_n_peaks, len(sorted_peaks))
    peaks_to_test = sorted_peaks[:num_peaks_to_test]

    logger.info(
        "Testing stability of top %d peak(s) (out of %d total)",
        num_peaks_to_test,
        len(sorted_peaks),
    )

    # Measure stability for each of the top N peaks
    candidates = []
    for i, peak in enumerate(peaks_to_test):
        logger.info(
            "Testing peak %d/%d: center=%.6fV, max_grad=%.6fV, quality=%.4f",
            i + 1,
            num_peaks_to_test,
            peak.peak_voltage,
            peak.sensitivity_voltage,
            peak.quality_score or 0.0,
        )

        # Measure stability
        stability_measurement = _measure_peak_stability(
            device=device,
            peak=peak,
            peak_index=i,
            sensor_gates_list=sensor_gates_list,
            sensor_plunger_gate=sensor_plunger_gate,
            mean_reservoir_saturation_voltage=mean_reservoir_saturation_voltage,
            measure_electrode=measure_electrode,
            bias_gate=bias_gate,
            bias_voltage=bias_voltage,
            aggregated_voltages=aggregated_voltages,
            aggregated_currents=aggregated_currents,
            hold_time_seconds=hold_time_seconds,
            session=session,
        )

        # Create candidate
        candidate = StablePeakCandidate(
            fitted_peak=peak,
            original_score=peak.quality_score or 0.0,
            stability_measurement=stability_measurement,
        )
        candidates.append(candidate)

    # Calculate combined scores (50% original + 50% stability)
    _calculate_combined_scores(candidates, original_weight=0.5, stability_weight=0.5)

    # Log combined scoring analysis
    if session:
        session.log_analysis(
            name="combined_scoring_summary",
            data={
                "num_candidates": len(candidates),
                "candidates": [
                    {
                        "peak_index": i + 1,
                        "peak_voltage": c.fitted_peak.peak_voltage,
                        "max_gradient_voltage": c.fitted_peak.sensitivity_voltage,
                        "original_score": c.original_score,
                        "voltage_noise": c.stability_measurement.voltage_noise,
                        "stability_score": c.stability_measurement.stability_score,
                        "combined_score": c.combined_score,
                    }
                    for i, c in enumerate(candidates)
                ],
            },
        )

    # Select best candidate by combined score
    best_candidate = max(candidates, key=lambda c: c.combined_score or 0.0)
    best_peak = best_candidate.fitted_peak

    logger.info(
        "Selected best stable peak: center=%.6fV, max_grad=%.6fV, "
        "original_score=%.4f, stability_score=%.4f, combined_score=%.4f",
        best_peak.peak_voltage,
        best_peak.sensitivity_voltage,
        best_candidate.original_score,
        best_candidate.stability_measurement.stability_score or 0.0,
        best_candidate.combined_score or 0.0,
    )

    # Extract peak voltages for range calculation
    best_peak_voltage = float(best_peak.peak_voltage)
    best_peak_max_gradient_voltage = float(best_peak.sensitivity_voltage)

    # Find spatially neighboring peaks for range calculation
    prev_peak = None
    next_peak = None
    for peak in fitted_peaks:
        if peak.peak_idx < best_peak.peak_idx:
            if prev_peak is None or peak.peak_idx > prev_peak.peak_idx:
                prev_peak = peak
        elif peak.peak_idx > best_peak.peak_idx:
            if next_peak is None or peak.peak_idx < next_peak.peak_idx:
                next_peak = peak

    # Set fallback values for missing neighboring peaks
    if prev_peak is None:
        prev_peak_voltage_fallback = best_peak_voltage - peak_spacing
        logger.warning(
            "No previous peak found. Using fallback: best_peak - peak_spacing = %.6fV",
            prev_peak_voltage_fallback,
        )
    else:
        prev_peak_voltage_fallback = float(prev_peak.peak_voltage)

    if next_peak is None:
        next_peak_voltage_fallback = best_peak_voltage + peak_spacing
        logger.warning(
            "No next peak found. Using fallback: best_peak + peak_spacing = %.6fV",
            next_peak_voltage_fallback,
        )
    else:
        next_peak_voltage_fallback = float(next_peak.peak_voltage)

    # Reconstruct voltage configuration at max gradient point
    sensor_dot_state = dict.fromkeys(
        sensor_gates_list, mean_reservoir_saturation_voltage
    )
    sensor_dot_state[sensor_plunger_gate] = best_peak_max_gradient_voltage

    # Jump to optimal sensing point
    device.jump(sensor_dot_state, wait_for_settling=True)

    # Calculate narrowed range and step size (same logic as find_sensor_peak)
    step_size_used = window_size / current_trace_number_of_points
    new_step_size = REFINED_STEP_MULTIPLIER * step_size_used

    number_of_points_between_previous_and_best_peak = (
        WINDOW_FRACTION
        * (best_peak_voltage - prev_peak_voltage_fallback)
        / new_step_size
    )
    number_of_points_between_best_and_next_peak = (
        WINDOW_FRACTION
        * (next_peak_voltage_fallback - best_peak_voltage)
        / new_step_size
    )

    start_of_range = (
        prev_peak_voltage_fallback
        if number_of_points_between_previous_and_best_peak < DEFAULT_WINDOW_HALF_WIDTH
        else best_peak_voltage - DEFAULT_WINDOW_HALF_WIDTH * new_step_size
    )
    end_of_range = (
        next_peak_voltage_fallback
        if number_of_points_between_best_and_next_peak < DEFAULT_WINDOW_HALF_WIDTH
        else best_peak_voltage + DEFAULT_WINDOW_HALF_WIDTH * new_step_size
    )

    narrowed_sensor_plunger_range = (start_of_range, end_of_range)
    logger.info(
        "Narrowed sweep range: %sV to %sV",
        narrowed_sensor_plunger_range[0],
        narrowed_sensor_plunger_range[1],
    )

    # Log final selection
    if session:
        session.log_analysis(
            name="stable_sensor_peak_selection",
            data={
                "best_peak_voltage": best_peak_voltage,
                "best_peak_max_gradient_voltage": best_peak_max_gradient_voltage,
                "narrowed_sensor_plunger_range": narrowed_sensor_plunger_range,
                "prev_peak_voltage": prev_peak_voltage_fallback,
                "next_peak_voltage": next_peak_voltage_fallback,
                "step_size": new_step_size,
                "original_score": best_candidate.original_score,
                "voltage_noise": best_candidate.stability_measurement.voltage_noise,
                "stability_score": best_candidate.stability_measurement.stability_score,
                "combined_score": best_candidate.combined_score,
                "sensor_park_point": sensor_dot_state,
            },
        )

    # Return same format as find_sensor_peak
    result = {
        "best_peak_voltage": best_peak_voltage,
        "best_peak_max_gradient_voltage": best_peak_max_gradient_voltage,
        "narrowed_sensor_plunger_range": narrowed_sensor_plunger_range,
        "prev_peak_voltage": prev_peak_voltage_fallback,
        "next_peak_voltage": next_peak_voltage_fallback,
        "mean_reservoir_saturation_voltage": mean_reservoir_saturation_voltage,
        "sensor_gates_list": sensor_gates_list,
        "sensor_plunger_index": sensor_plunger_index,
        "step_size": new_step_size,
        "sensor_park_point": sensor_dot_state,
    }

    return result
