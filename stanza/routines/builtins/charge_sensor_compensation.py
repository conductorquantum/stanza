"""
Charge sensor compensation routines for quantum dot devices.

This module provides automated charge sensor compensation gradient calculation
for quantum devices using peak fitting and ML-based Coulomb blockade detection.

Physical Context:
-----------------
In quantum dot devices, gate electrodes control the electrostatic potential
landscape. The "charge sensor" is a quantum dot configured to operate near
a Coulomb blockade peak, where conductance changes rapidly with electron
number. Control gates can unintentionally shift the sensor's operating point
through capacitive coupling (cross-talk).

Compensation gradients quantify this capacitive coupling between gates:
    gradient = dV_sensor_peak / dV_control_gate

These gradients enable real-time correction of sensor gate voltages when
control gates change, maintaining optimal charge sensing fidelity throughout
device operation.

The module includes two main routines:

1. find_sensor_peak: Locates optimal charge sensing operating point
   - Sweeps sensor plunger to identify Coulomb blockade peaks
   - Uses ML-based classification and multi-model peak fitting
   - Returns peak location and narrowed voltage range for high-resolution sweeps

2. run_compensation: Calculates compensation gradients for control gates
   - Measures how each control gate voltage affects sensor peak position
   - Returns gradient matrix for cross-talk compensation in tuneup sequences
"""

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
from stanza.models import GateType
from stanza.routines import RoutineContext, routine
from stanza.routines.builtins.utils.group_handling import filter_gates_by_group
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    calculate_quality_score,
    fit_peak_multi_model,
)

# Configure logger
logger = logging.getLogger(__name__)

# Configuration constants

# Default settling time before sweeps to avoid current spikes
DEFAULT_SETTLING_TIME_S = 5

# ML model input constraint: the coulomb blockade classifier model requires
# exactly 128 points per input window for inference
ML_MODEL_INPUT_SIZE = 128

# Use 2x peak spacing for initial multi-window sweep to ensure we capture
# full peak width plus sufficient context for accurate ML classification
INITIAL_WINDOW_MULTIPLIER = 2

PERTURBATION_DIVISOR = 10

# Use 80% of inter-peak distance for peak fitting windows to avoid
# overlapping windows while maximizing window size for better fit quality
# (empirically determined to balance resolution vs. overlap)
WINDOW_FRACTION = 0.8

# Default half-width (in points) for peak fitting windows when peaks are
# widely spaced. 128 points provides sufficient context for Lorentzian/sech²/
# Voigt fitting while avoiding edge effects
DEFAULT_WINDOW_HALF_WIDTH = 128

# Use 0.5x (half) of the initial step size for refined narrowed-range sweeps
# to improve peak center localization accuracy
REFINED_STEP_MULTIPLIER = 0.5


# Number of samples to average for each gate compensation measurement
NUM_OF_SAMPLES_FOR_AVERAGING = 10

# ML model constants
COULOMB_CLASSIFIER_MODEL = "coulomb-blockade-classifier-v3"
PEAK_DETECTOR_MODEL = "coulomb-blockade-peak-detector-v2"


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


def _build_sensor_sweep_voltage_list(
    sensor_gates_list: list[str],
    sensor_plunger_index: int,
    base_voltage: float,
    plunger_voltages: np.ndarray,
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

    Returns:
        List of voltage arrays ready for device.sweep_nd()
    """
    number_of_gates = len(sensor_gates_list)
    base_voltages = np.full(number_of_gates, base_voltage)

    voltage_list = []
    for plunger_voltage in plunger_voltages:
        gate_voltages = base_voltages.copy()
        gate_voltages[sensor_plunger_index] = plunger_voltage
        voltage_list.append(gate_voltages.tolist())

    return voltage_list


def analyze_single_window_barrier_sweep(
    aggregated_currents: np.ndarray,
    aggregated_voltages: np.ndarray,
    analysis_session: LoggerSession | None,
) -> FittedPeak | None:
    """
    Analyze single-window barrier sweep using multi-model peak fitting.

    Fits three peak models (Lorentzian, sech², pseudo-Voigt) to the entire
    current trace, selects best model by AICc, and calculates comprehensive
    quality metrics for charge sensor characterization.

    Args:
        aggregated_currents: Current trace data
        aggregated_voltages: Voltage trace data
        analysis_session: Logging session for saving analysis results

    Returns:
        FittedPeak object with multi-model fit results and quality metrics
    """
    # Extract window data - treat entire trace as single window
    window_currents = aggregated_currents
    window_indices = np.arange(len(aggregated_currents))
    center_index = int(len(aggregated_currents) / 2)

    # Fit all three models and select best
    fitted_peak = fit_peak_multi_model(
        window_currents=window_currents,
        window_indices=window_indices,
        peak_idx_in_window=center_index,
        aggregated_voltages=aggregated_voltages,
        window_start_idx=0,
        window_end_idx=int(len(aggregated_currents)),
        peak_idx_aggregated=center_index,
    )

    # For single peak, sensitivity_score is always 1.0 (no other peaks to compare)
    # Use helper function for consistency
    fitted_peak.sensitivity_score = 1.0
    fitted_peaks_list = [fitted_peak]
    _calculate_quality_scores(fitted_peaks_list)

    # Log analysis results with all three model fits
    if analysis_session:
        analysis_session.log_analysis(
            name="peak_multi_model_fit",
            data={
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


def _calculate_peak_window_bounds(
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


def _calculate_quality_scores(fitted_peaks: list[FittedPeak]) -> None:
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
        window_bounds = _calculate_peak_window_bounds(
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
        _calculate_quality_scores(fitted_peaks)

    return fitted_peaks


def _single_window_sensor_plunger_sweep(
    ctx: RoutineContext,
    sensor_gates_list: list[str],
    sensor_plunger_range: tuple[float, float],
    mean_reservoir_saturation_voltage: float,
    sensor_plunger_index: int,
    step_size: float,
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    session: LoggerSession | None = None,
) -> PeakWindowSweepOutput:
    """
    Run a single-window sensor plunger voltage sweep with multi-model peak fitting.

    Performs a refined sweep of the sensor plunger within a narrow voltage range
    to precisely characterize Coulomb blockade peaks using three models (Lorentzian,
    sech², pseudo-Voigt). Best model is selected by AICc, and comprehensive quality
    metrics (R², RMSE, skew, DW, FWHM, area) are calculated for automated peak
    ranking and charge sensor operating point selection.

    Used for both baseline measurements and gate compensation sweeps.

    Args:
        ctx: Routine context containing device resources
        sensor_gates_list: List of gate names in sensor group
        sensor_plunger_range: (min, max) voltage range for sensor plunger
        mean_reservoir_saturation_voltage: Voltage for all gates except sensor plunger
        sensor_plunger_index: Index of sensor plunger in sensor_gates_list
        step_size: Voltage increment between points
        measure_electrode: Electrode to measure current from
        bias_gate: Name of the bias gate (contact) to apply bias voltage
        bias_voltage: Voltage to apply to bias gate during measurements
        session: Logger session for logging measurements and analysis

    Returns:
        PeakWindowSweepOutput containing the best fitted peak (with all
        three model fits and quality metrics) and trace data.
    """
    device = ctx.resources.device

    # Apply bias voltage to bias gate
    device.jump({bias_gate: bias_voltage}, wait_for_settling=True)
    time.sleep(DEFAULT_SETTLING_TIME_S)

    # Calculate sequential sweep parameters
    min_v, max_v = sensor_plunger_range

    # Initialize aggregated trace arrays
    aggregated_voltages = np.array([], dtype=np.float32)
    aggregated_currents = np.array([], dtype=np.float32)
    last_classification = False  # Track last classification result

    # Generate sensor plunger voltages with matching resolution
    # Calculate number of points needed based on step_size to match initial
    # sweep resolution
    num_points = int(np.ceil((max_v - min_v) / step_size)) + 1
    sp_sweep_voltages = np.linspace(min_v, max_v, num_points, endpoint=True)

    # Build voltage list for sweep_nd using helper function
    voltage_list = _build_sensor_sweep_voltage_list(
        sensor_gates_list=sensor_gates_list,
        sensor_plunger_index=sensor_plunger_index,
        base_voltage=mean_reservoir_saturation_voltage,
        plunger_voltages=sp_sweep_voltages,
    )

    if not voltage_list:
        raise RoutineError("Sensor sweep voltage list is empty - cannot start sweep.")

    # Set device to first voltage point and allow settling time to avoid current spikes
    first_voltage_point = {
        gate: float(voltage)
        for gate, voltage in zip(sensor_gates_list, voltage_list[0], strict=False)
    }
    sensor_plunger_gate = sensor_gates_list[sensor_plunger_index]
    first_plunger_voltage = first_voltage_point[sensor_plunger_gate]
    if not np.isclose(first_plunger_voltage, sp_sweep_voltages[0]):
        logger.debug(
            "Adjusting first voltage point for %s from %.6fV to %.6fV to match sweep start",
            sensor_plunger_gate,
            first_plunger_voltage,
            sp_sweep_voltages[0],
        )
        first_voltage_point[sensor_plunger_gate] = float(sp_sweep_voltages[0])

    device.jump(first_voltage_point, wait_for_settling=True)
    time.sleep(DEFAULT_SETTLING_TIME_S)

    # Perform current trace measurement
    try:
        _, current_trace = device.sweep_nd(
            gate_electrodes=sensor_gates_list,
            voltages=voltage_list,
            measure_electrode=measure_electrode,
            session=session,
        )

        # Append to aggregated trace
        aggregated_voltages = np.concatenate([aggregated_voltages, sp_sweep_voltages])
        aggregated_currents = np.concatenate([aggregated_currents, current_trace])

        fitted_peak = analyze_single_window_barrier_sweep(
            aggregated_currents, aggregated_voltages, session
        )

    except Exception as e:
        raise RoutineError(
            f"Error in _single_window_sensor_plunger_sweep: {str(e)}"
        ) from e

    # Validate fitted peak results
    if fitted_peak is None:
        raise RoutineError("Failed to fit peak in single window sensor plunger sweep")
    if fitted_peak.quality_score is None:
        raise RoutineError(
            "Peak quality score not calculated - check peak fitting logic"
        )

    # Create PeakWindowSweepOutput with the best peak's data
    result = PeakWindowSweepOutput(
        best_peak=fitted_peak,
        aggregated_voltages=aggregated_voltages,
        aggregated_currents=aggregated_currents,
        classification=last_classification,
        score=float(fitted_peak.quality_score),  # Use quality_score for consistency
        num_peaks=1,
    )

    return result


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
        voltage_list = _build_sensor_sweep_voltage_list(
            sensor_gates_list=sensor_gates_list,
            sensor_plunger_index=sensor_plunger_index,
            base_voltage=mean_reservoir_saturation_voltage,
            plunger_voltages=sp_sweep_voltages,
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
    _calculate_quality_scores(fitted_peaks)

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
            set to sensor group's global turn-on voltage. (default: True)
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
def run_compensation(  # pylint: disable=too-many-locals,too-many-statements
    ctx: RoutineContext,
    peak_spacing: float,
    control_group_name: str,
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    zero_control_side: bool = False,
    gates_to_compensate: list[str] | None = None,
    session: LoggerSession | None = None,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> dict[str, float]:
    """
    Calculate compensation gradients for charge sensor gates.

    This routine measures how control gate voltages affect the sensor peak position
    and calculates compensation gradients to maintain optimal charge sensing. It
    performs baseline measurements with control gates at a specified state, then
    sweeps each control gate individually while measuring peak shifts.

    Args:
        ctx: Routine context containing device resources and previous results. Requires:
             - ctx.results["find_sensor_peak"]: Results from find_sensor_peak routine
        peak_spacing: Expected peak spacing in volts (e.g., 0.020 for 20mV)
        control_group_name: Name of control side group (e.g., "side_A")
        measure_electrode: Electrode to measure current from (e.g., "OUT_B")
        bias_gate: Name of the bias gate (contact) to apply bias voltage (e.g., "IN_A_B")
        bias_voltage: Voltage to apply to bias gate during measurements (V)
        zero_control_side: If True, measure gradients relative to 0V baseline.
            If False, measure gradients relative to current control voltages.
            Useful for measuring compensation at non-zero operating points. (default: True)
        gates_to_compensate: Optional list of gate names to measure compensation for.
            If provided, only these gates will be tested. Must be valid non-reservoir
            gates (plunger or barrier) from the control group. Applied after automatic
            type and group filtering. If None, all eligible gates are tested. (default: None)
        session: Logger session for measurements and analysis

    Returns:
        dict: Dictionary mapping gate names to compensation gradients (V/V)

    Raises:
        RoutineError: If find_sensor_peak results are missing or invalid

    Notes:
        - Requires find_sensor_peak to be run first
        - Tests 10 voltage points per gate in symmetric range around baseline
        - Voltage perturbations applied relative to baseline (0V or current voltages)
        - Automatically resets to initial state after testing
        - For non-linear cross-talk, use zero_control_side=False to measure at
          actual operating point
    """
    if peak_spacing <= 0:
        raise RoutineError("peak_spacing must be greater than 0")

    # Get device
    device = ctx.resources.device

    # Get groups from device config
    control_group = device.device_config.groups[control_group_name]
    control_gates = list(control_group.gates)
    control_gates = filter_gates_by_group(ctx, control_gates)

    # Get results from find_sensor_peak routine
    find_sensor_peak_results = ctx.results.get("find_sensor_peak", {})
    if not find_sensor_peak_results:
        raise RoutineError(
            "find_sensor_peak results not found in ctx.results. "
            "Please run find_sensor_peak routine first."
        )

    # Extract values from find_sensor_peak results
    narrowed_sensor_plunger_range = find_sensor_peak_results[
        "narrowed_sensor_plunger_range"
    ]
    mean_reservoir_saturation_voltage = find_sensor_peak_results[
        "mean_reservoir_saturation_voltage"
    ]
    sensor_gates_list = find_sensor_peak_results["sensor_gates_list"]
    sensor_plunger_index = find_sensor_peak_results["sensor_plunger_index"]
    sensor_gate_key = sensor_gates_list[sensor_plunger_index]
    new_step_size = find_sensor_peak_results["step_size"]

    voltage_range = 0.8 * peak_spacing
    # Create symmetric voltage points around zero, excluding zero itself
    # to avoid division by zero
    half_n = PERTURBATION_DIVISOR // 2
    voltage_differences = np.concatenate(
        [
            np.linspace(-voltage_range, -voltage_range / half_n, half_n),
            np.linspace(voltage_range / half_n, voltage_range, half_n),
        ]
    )
    rng = np.random.default_rng()

    # Get non-reservoir gates from control side
    # Apply filter_gates_by_group to honor device exclusions
    all_plungers = device.get_gates_by_type(GateType.PLUNGER)
    all_barriers = device.get_gates_by_type(GateType.BARRIER)
    all_plungers = filter_gates_by_group(ctx, all_plungers)
    all_barriers = filter_gates_by_group(ctx, all_barriers)
    control_non_reservoir_gates = [
        g for g in (all_plungers + all_barriers) if g in control_gates
    ]

    # Apply additional filtering if gates_to_compensate is specified
    if gates_to_compensate is not None:
        # Validate that all requested gates are in the eligible set
        invalid_gates = [
            g for g in gates_to_compensate if g not in control_non_reservoir_gates
        ]
        if invalid_gates:
            raise RoutineError(
                f"Invalid gates specified in gates_to_compensate: {invalid_gates}. "
                f"Must be non-reservoir gates from control group "
                f"'{control_group_name}'. "
                f"Eligible gates: {control_non_reservoir_gates}"
            )
        # Filter to only include requested gates
        control_non_reservoir_gates = [
            g for g in control_non_reservoir_gates if g in gates_to_compensate
        ]

    # Capture initial device state for cleanup in finally block
    initial_control_voltages = device.check(control_non_reservoir_gates)
    initial_sensor_voltages = device.check(sensor_gates_list)

    # Determine baseline control state
    if zero_control_side:
        logger.info("Acquiring baseline measurement with control gates at 0V.")
        baseline_control_state = dict.fromkeys(control_non_reservoir_gates, 0.0)
    else:
        logger.info(
            "Acquiring baseline measurement with control gates at current voltages: %s",
            dict(
                zip(control_non_reservoir_gates, initial_control_voltages, strict=False)
            ),
        )
        baseline_control_state = dict(
            zip(control_non_reservoir_gates, initial_control_voltages, strict=False)
        )

    try:
        # Set control side gates to baseline state
        device.jump(baseline_control_state, wait_for_settling=True)
        time.sleep(DEFAULT_SETTLING_TIME_S)
        # Perform baseline sweep 5 times and average for better estimate
        try:
            baseline_sensitivity_voltages = []
            for _ in range(NUM_OF_SAMPLES_FOR_AVERAGING):
                baseline_sweep_output = _single_window_sensor_plunger_sweep(
                    ctx=ctx,
                    sensor_gates_list=sensor_gates_list,
                    sensor_plunger_range=narrowed_sensor_plunger_range,
                    mean_reservoir_saturation_voltage=mean_reservoir_saturation_voltage,
                    sensor_plunger_index=sensor_plunger_index,
                    step_size=new_step_size,
                    measure_electrode=measure_electrode,
                    bias_gate=bias_gate,
                    bias_voltage=bias_voltage,
                    session=session,
                )
                baseline_sensitivity_voltages.append(
                    baseline_sweep_output.best_peak.sensitivity_voltage
                )

            reference_max_gradient_voltage = float(
                np.mean(baseline_sensitivity_voltages)
            )
        except Exception as e:
            raise RoutineError(f"Error in baseline measurement: {str(e)}") from e
        sensor_park_point_voltages = dict.fromkeys(
            sensor_gates_list, mean_reservoir_saturation_voltage
        )
        sensor_park_point_voltages[sensor_gate_key] = reference_max_gradient_voltage
        compensation_gradients_dict = {}
        per_gate_details = {}  # Store detailed arrays for each gate
        for gate in control_non_reservoir_gates:
            num_deltas = len(voltage_differences)
            measurement_indices = np.repeat(
                np.arange(num_deltas, dtype=int), NUM_OF_SAMPLES_FOR_AVERAGING
            )
            rng.shuffle(measurement_indices)
            per_delta_measurements: dict[int, list[float]] = {
                idx: [] for idx in range(num_deltas)
            }
            measurement_voltage_sequence: list[float] = []
            measurement_samples: list[dict[str, float]] = []

            total_measurements = len(measurement_indices)
            for counter, delta_index in enumerate(measurement_indices, start=1):
                voltage_difference = float(voltage_differences[delta_index])
                measurement_voltage_sequence.append(voltage_difference)
                print(
                    f"Measurement {counter} of {total_measurements} for gate {gate}: "
                    f"{voltage_difference:+.6f} V"
                )

                # Apply voltage perturbation relative to baseline
                device_state = baseline_control_state.copy()
                device_state[gate] = baseline_control_state[gate] + voltage_difference
                device.jump(device_state, wait_for_settling=True)
                time.sleep(DEFAULT_SETTLING_TIME_S)

                # Single sensor plunger sweep for this sample
                iteration_sweep_output = _single_window_sensor_plunger_sweep(
                    ctx=ctx,
                    sensor_gates_list=sensor_gates_list,
                    sensor_plunger_range=narrowed_sensor_plunger_range,
                    mean_reservoir_saturation_voltage=mean_reservoir_saturation_voltage,
                    sensor_plunger_index=sensor_plunger_index,
                    step_size=new_step_size,
                    measure_electrode=measure_electrode,
                    bias_gate=bias_gate,
                    bias_voltage=bias_voltage,
                    session=session,
                )

                best_peak = iteration_sweep_output.best_peak
                sensitivity_voltage = float(best_peak.sensitivity_voltage)
                per_delta_measurements[delta_index].append(sensitivity_voltage)
                # Log per-sample deltas explicitly (with clear names + aliases)
                peak_shift = float(sensitivity_voltage - reference_max_gradient_voltage)
                sample_record = {
                    # Plunger delta (control gate change relative to baseline)
                    "control_delta": voltage_difference,
                    "delta_plunger": voltage_difference,  # alias for clarity
                    # Peak location at this sample and its delta vs baseline
                    "peak_position": sensitivity_voltage,
                    "peak_shift": peak_shift,
                    "delta_peak": peak_shift,  # alias for clarity
                }
                measurement_samples.append(sample_record)

                # Emit per-sample analysis immediately so each measurement is logged in real time
                if session:
                    session.log_analysis(
                        name=f"compensation_measurement_sample_{gate}",
                        data={
                            "gate": gate,
                            "counter": counter,
                            "total_measurements": total_measurements,
                            **sample_record,
                        },
                    )

            peak_positions = np.empty_like(voltage_differences, dtype=np.float64)
            for idx in range(num_deltas):
                measurements = per_delta_measurements[idx]
                if len(measurements) != NUM_OF_SAMPLES_FOR_AVERAGING:
                    raise RoutineError(
                        "Incomplete measurement set: expected "
                        f"{NUM_OF_SAMPLES_FOR_AVERAGING} samples for voltage difference "
                        f"{voltage_differences[idx]:+.6f} V, got {len(measurements)}"
                    )
                peak_positions[idx] = float(np.mean(measurements))
            peak_positions_difference = peak_positions - reference_max_gradient_voltage
            # Keep per-point gradients for diagnostics, but use least squares with a
            # free intercept to avoid noise amplification and absorb slow drift.
            per_point_gradients = peak_positions_difference / voltage_differences

            if np.allclose(voltage_differences, voltage_differences[0]):
                raise RoutineError(
                    "Cannot compute compensation gradient: voltage_differences are identical"
                )

            design_matrix = np.column_stack(
                (voltage_differences, np.ones_like(voltage_differences))
            )
            try:
                least_squares_gradient, drift_intercept = np.linalg.lstsq(
                    design_matrix, peak_positions_difference, rcond=None
                )[0]
            except np.linalg.LinAlgError as exc:
                raise RoutineError(
                    "Failed to compute compensation gradient: singular fit matrix"
                ) from exc

            least_squares_gradient = float(least_squares_gradient)
            drift_intercept = float(drift_intercept)
            compensation_gradients_dict[gate] = least_squares_gradient

            # Store detailed arrays for this gate for later analysis/logging
            per_gate_details[gate] = {
                "peak_positions": peak_positions,
                "peak_positions_difference": peak_positions_difference,
                "per_point_gradients": per_point_gradients,
                "least_squares_gradient": least_squares_gradient,
                "drift_intercept": drift_intercept,
                "mean_per_point_gradient": float(np.mean(per_point_gradients)),
                "mean_gradient": least_squares_gradient,
                "peak_vs_gate_deltas": [
                    {
                        "control_delta": float(control_delta),
                        "peak_position": float(peak_position),
                        "peak_shift": float(peak_shift),
                    }
                    for control_delta, peak_position, peak_shift in zip(
                        voltage_differences,
                        peak_positions,
                        peak_positions_difference,
                        strict=False,
                    )
                ],
                "measurement_voltage_sequence": list(measurement_voltage_sequence),
                "measurement_samples": measurement_samples,
            }

            # Reset this gate back to baseline before moving to next gate
            reset_state = {gate: baseline_control_state[gate]}
            device.jump(reset_state, wait_for_settling=True)
            time.sleep(DEFAULT_SETTLING_TIME_S)

        logger.info("Compensation gradients: %s", compensation_gradients_dict)

        # Save compensation gradient analysis to disk using passed session
        # Log per-gate gradient details
        if session:
            for gate, details in per_gate_details.items():
                session.log_analysis(
                    name=f"compensation_gradient_{gate}",
                    data={
                        "gate": gate,
                        "mean_gradient": float(details["mean_gradient"]),
                        "num_measurements": len(voltage_differences),
                        "peak_positions": details["peak_positions"].tolist(),
                        "peak_positions_difference": details[
                            "peak_positions_difference"
                        ].tolist(),
                        "per_point_gradients": details["per_point_gradients"].tolist(),
                        "least_squares_gradient": details["least_squares_gradient"],
                        "drift_intercept": details["drift_intercept"],
                        "mean_per_point_gradient": details["mean_per_point_gradient"],
                        "peak_vs_gate_deltas": details["peak_vs_gate_deltas"],
                        "measurement_voltage_sequence": details[
                            "measurement_voltage_sequence"
                        ],
                        "measurement_samples": details["measurement_samples"],
                        "voltage_differences": voltage_differences.tolist(),
                        "num_deltas": len(voltage_differences),
                        "samples_per_delta": NUM_OF_SAMPLES_FOR_AVERAGING,
                        "total_samples": int(
                            len(voltage_differences) * NUM_OF_SAMPLES_FOR_AVERAGING
                        ),
                    },
                )

            # Log overall summary
            session.log_analysis(
                name="compensation_gradient_summary",
                data={
                    "gradients": {
                        k: float(v) for k, v in compensation_gradients_dict.items()
                    },
                    "compensation_gradients": compensation_gradients_dict,
                    "sensor_park_point_voltages": sensor_park_point_voltages,
                    "sensor_plunger_key": sensor_gate_key,
                    "sensor_plunger_ranges": narrowed_sensor_plunger_range,
                },
            )

        result = {
            "compensation_gradients": compensation_gradients_dict,
            "sensor_park_point_voltages": sensor_park_point_voltages,
            "sensor_plunger_key": sensor_gate_key,
            "sensor_plunger_ranges": narrowed_sensor_plunger_range,
        }
        return result

    finally:
        # Restore initial device state for both control and sensor gates
        logger.info("Restoring initial device state")
        device.jump(
            dict(
                zip(control_non_reservoir_gates, initial_control_voltages, strict=False)
            ),
            wait_for_settling=True,
        )
        device.jump(
            dict(zip(sensor_gates_list, initial_sensor_voltages, strict=False)),
            wait_for_settling=True,
        )
