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
from sklearn.linear_model import RANSACRegressor

# First-party imports
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.models import GateType
from stanza.routines import RoutineContext, routine
from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
    PeakWindowSweepOutput,
    build_sensor_sweep_voltage_list,
    calculate_quality_scores,
)
from stanza.routines.builtins.charge_sensor.constants import (
    DEFAULT_SETTLING_TIME_S,
    MULTIPLER_OF_PEAK_SPACING,
    NUM_OF_SAMPLES_FOR_AVERAGING,
    PERTURBATION_DIVISOR,
)
from stanza.routines.builtins.utils.group_handling import filter_gates_by_group
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    fit_peak_multi_model,
)

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
    calculate_quality_scores(fitted_peaks_list)

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
    voltage_list = build_sensor_sweep_voltage_list(
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

    # Get results from peak-finding routine
    # Check for find_stable_sensor_peak first as it provides more robust peaks
    # by measuring current stability at multiple candidate peaks over time.
    # Fall back to find_sensor_peak for backward compatibility.
    find_sensor_peak_results = ctx.results.get("find_stable_sensor_peak")
    if find_sensor_peak_results is None:
        find_sensor_peak_results = ctx.results.get("find_sensor_peak", {})

    if not find_sensor_peak_results:
        raise RoutineError(
            "Peak-finding results not found in ctx.results. "
            "Please run either find_stable_sensor_peak or find_sensor_peak routine first."
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

    voltage_range = MULTIPLER_OF_PEAK_SPACING * peak_spacing
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
        # Perform baseline sweep NUM_OF_SAMPLES_FOR_AVERAGING times and take median for robust estimate
        try:
            baseline_sensitivity_voltages = []
            baseline_peak_center_voltages = []
            total_baseline_measurements = NUM_OF_SAMPLES_FOR_AVERAGING
            for baseline_idx in range(total_baseline_measurements):
                logger.info(
                    "Baseline measurement %d of %d for sensor plunger %s",
                    baseline_idx + 1,
                    total_baseline_measurements,
                    sensor_gate_key,
                )
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
                baseline_peak_center_voltages.append(
                    baseline_sweep_output.best_peak.peak_voltage
                )
                if session:
                    session.log_analysis(
                        name="baseline_measurement_sample",
                        data={
                            "sensor_plunger_gate": sensor_gate_key,
                            "iteration": baseline_idx + 1,
                            "total_iterations": total_baseline_measurements,
                            "sensitivity_voltage": float(
                                baseline_sweep_output.best_peak.sensitivity_voltage
                            ),
                            "peak_center_voltage": float(
                                baseline_sweep_output.best_peak.peak_voltage
                            ),
                        },
                    )

            # Reference for parking (max gradient) - use median for robustness to outliers
            # (consistent with RANSAC approach for gradient fitting)
            reference_max_gradient_voltage = float(
                np.median(baseline_sensitivity_voltages)
            )
            # Reference for gradient calculation (peak center) - use median for robustness
            reference_peak_center_voltage = float(
                np.median(baseline_peak_center_voltages)
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
                peak_center_voltage = float(best_peak.peak_voltage)
                per_delta_measurements[delta_index].append(peak_center_voltage)
                # Log per-sample deltas explicitly (with clear names + aliases)
                peak_shift = float(peak_center_voltage - reference_peak_center_voltage)
                sample_record = {
                    # Plunger delta (control gate change relative to baseline)
                    "control_delta": voltage_difference,
                    "delta_plunger": voltage_difference,  # alias for clarity
                    # Peak location at this sample and its delta vs baseline (using center)
                    "peak_position": peak_center_voltage,
                    "peak_shift": peak_shift,
                    "delta_peak": peak_shift,  # alias for clarity
                    "sensitivity_voltage": float(best_peak.sensitivity_voltage),
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
            peak_positions_difference = peak_positions - reference_peak_center_voltage
            # Keep per-point gradients for diagnostics, but use RANSAC regression
            # to robustly fit the gradient while rejecting outliers from bad measurements.
            per_point_gradients = peak_positions_difference / voltage_differences

            # Use RANSAC to robustly fit gradient through all individual measurements
            ransac_result = fit_compensation_gradient_ransac(
                measurement_samples=measurement_samples,
                reference_peak_center_voltage=reference_peak_center_voltage,
                gate_name=gate,
            )

            # Extract results
            least_squares_gradient = ransac_result.gradient
            drift_intercept = ransac_result.intercept
            inlier_mask = ransac_result.inlier_mask
            num_inliers = ransac_result.num_inliers
            num_outliers = ransac_result.num_outliers
            outlier_indices = ransac_result.outlier_indices
            outlier_voltages = ransac_result.outlier_voltages
            outlier_peak_shifts = ransac_result.outlier_peak_shifts
            all_control_deltas = ransac_result.all_control_deltas
            all_peak_shifts = ransac_result.all_peak_shifts

            compensation_gradients_dict[gate] = least_squares_gradient

            # Mark each measurement sample with its inlier status
            for i, sample in enumerate(measurement_samples):
                sample["is_inlier"] = bool(inlier_mask[i])

            # Calculate per-averaged-point inlier ratios
            # For each of the 10 voltage points, count how many of its 10 samples were inliers
            per_voltage_inlier_counts = []
            for idx in range(num_deltas):
                # Find which of the 100 measurements correspond to this voltage point
                samples_for_this_voltage = [
                    i
                    for i, sample in enumerate(measurement_samples)
                    if np.isclose(sample["control_delta"], voltage_differences[idx])
                ]
                num_inliers_for_voltage = sum(
                    inlier_mask[i] for i in samples_for_this_voltage
                )
                per_voltage_inlier_counts.append(num_inliers_for_voltage)

            # Store detailed arrays for this gate for later analysis/logging
            per_gate_details[gate] = {
                "peak_positions": peak_positions,
                "peak_positions_difference": peak_positions_difference,
                "per_point_gradients": per_point_gradients,
                "least_squares_gradient": least_squares_gradient,
                "drift_intercept": drift_intercept,
                "mean_per_point_gradient": float(np.mean(per_point_gradients)),
                "mean_gradient": least_squares_gradient,
                # RANSAC-specific fields (based on 100 raw measurements)
                "inlier_mask": inlier_mask.tolist(),
                "num_inliers": num_inliers,
                "num_outliers": num_outliers,
                "outlier_indices": outlier_indices,
                "outlier_voltages": outlier_voltages,
                "outlier_peak_shifts": outlier_peak_shifts,
                "all_control_deltas": all_control_deltas.tolist(),
                "all_peak_shifts": all_peak_shifts.tolist(),
                "peak_vs_gate_deltas": [
                    {
                        "control_delta": float(control_delta),
                        "peak_position": float(peak_position),
                        "peak_shift": float(peak_shift),
                        "num_inliers": int(inlier_count),
                        "inlier_fraction": float(
                            inlier_count / NUM_OF_SAMPLES_FOR_AVERAGING
                        ),
                    }
                    for control_delta, peak_position, peak_shift, inlier_count in zip(
                        voltage_differences,
                        peak_positions,
                        peak_positions_difference,
                        per_voltage_inlier_counts,
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
                        # RANSAC-specific fields (based on 100 raw measurements)
                        "regression_method": "ransac",
                        "inlier_mask": details["inlier_mask"],
                        "num_inliers": details["num_inliers"],
                        "num_outliers": details["num_outliers"],
                        "outlier_indices": details["outlier_indices"],
                        "outlier_voltages": details["outlier_voltages"],
                        "outlier_peak_shifts": details["outlier_peak_shifts"],
                        "all_control_deltas": details["all_control_deltas"],
                        "all_peak_shifts": details["all_peak_shifts"],
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
