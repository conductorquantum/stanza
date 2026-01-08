"""Charge sensor compensation routines for quantum dot devices.

This module calculates compensation gradients that quantify how control gate
voltages affect the sensor's operating point through capacitive cross-talk.
Gradients enable real-time correction to maintain optimal charge sensing.

The run_compensation routine measures gradients by: (1) establishing a baseline
peak position, (2) applying voltage perturbations to control gates, (3) measuring
peak shifts using multi-model peak fitting, and (4) using RANSAC regression to
robustly fit gradients while rejecting outliers.
"""

import logging
import time
from typing import Any

import numpy as np

from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.models import GateType
from stanza.routines import RoutineContext, routine
from stanza.routines.builtins.charge_sensor.utils.constants import (
    DEFAULT_SETTLING_TIME_S,
    MULTIPLIER_OF_PEAK_SPACING,
    NUM_OF_SAMPLES_FOR_AVERAGING,
    PERTURBATION_DIVISOR,
)
from stanza.routines.builtins.charge_sensor.utils.sweeps import (
    build_sensor_sweep_voltage_list,
)
from stanza.routines.builtins.charge_sensor.utils.types import (
    PeakWindowSweepOutput,
)
from stanza.routines.builtins.utils.group_handling import filter_gates_by_group
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    calculate_quality_scores,
    fit_peak_multi_model,
)
from stanza.routines.builtins.utils.ransac_fitting import (
    fit_compensation_gradient_ransac,
)

logger = logging.getLogger(__name__)


def _analyze_single_window_barrier_sweep(
    aggregated_currents: np.ndarray,
    aggregated_voltages: np.ndarray,
    analysis_session: LoggerSession | None,
) -> FittedPeak | None:
    """Analyze single-window barrier sweep using multi-model peak fitting.

    Fits three peak models (Lorentzian, sech², pseudo-Voigt) and selects best by AICc.

    Args:
        aggregated_currents: Current trace data
        aggregated_voltages: Voltage trace data
        analysis_session: Logging session for analysis results

    Returns:
        FittedPeak object with multi-model fit results and quality metrics
    """
    window_currents = aggregated_currents
    window_indices = np.arange(len(aggregated_currents))
    center_index = int(len(aggregated_currents) / 2)

    fitted_peak = fit_peak_multi_model(
        window_currents=window_currents,
        window_indices=window_indices,
        peak_idx_in_window=center_index,
        aggregated_voltages=aggregated_voltages,
        window_start_idx=0,
        window_end_idx=int(len(aggregated_currents)),
        peak_idx_aggregated=center_index,
    )

    fitted_peak.sensitivity_score = 1.0
    fitted_peaks_list = [fitted_peak]
    calculate_quality_scores(fitted_peaks_list)

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
    """Run a single-window sensor plunger voltage sweep with multi-model peak fitting.

    Performs a refined sweep within a narrow voltage range to characterize Coulomb
    blockade peaks using three models (Lorentzian, sech², pseudo-Voigt).

    Args:
        ctx: Routine context containing device resources
        sensor_gates_list: List of gate names in sensor group
        sensor_plunger_range: (min, max) voltage range for sensor plunger
        mean_reservoir_saturation_voltage: Voltage for all gates except sensor plunger
        sensor_plunger_index: Index of sensor plunger in sensor_gates_list
        step_size: Voltage increment between points
        measure_electrode: Electrode to measure current from
        bias_gate: Name of the bias gate to apply bias voltage
        bias_voltage: Voltage to apply to bias gate during measurements
        session: Logger session for measurements and analysis

    Returns:
        PeakWindowSweepOutput containing the best fitted peak and trace data
    """
    device = ctx.resources.device

    device.jump({bias_gate: bias_voltage}, wait_for_settling=True)
    time.sleep(DEFAULT_SETTLING_TIME_S)

    min_v, max_v = sensor_plunger_range

    aggregated_voltages = np.array([], dtype=np.float32)
    aggregated_currents = np.array([], dtype=np.float32)
    last_classification = False

    num_points = int(np.ceil((max_v - min_v) / step_size)) + 1
    sp_sweep_voltages = np.linspace(min_v, max_v, num_points, endpoint=True)

    voltage_list = build_sensor_sweep_voltage_list(
        sensor_gates_list=sensor_gates_list,
        sensor_plunger_index=sensor_plunger_index,
        base_voltage=mean_reservoir_saturation_voltage,
        plunger_voltages=sp_sweep_voltages,
    )

    if not voltage_list:
        raise RoutineError("Sensor sweep voltage list is empty - cannot start sweep.")

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

    try:
        _, current_trace = device.sweep_nd(
            gate_electrodes=sensor_gates_list,
            voltages=voltage_list,
            measure_electrode=measure_electrode,
            session=session,
        )
        aggregated_voltages = np.concatenate([aggregated_voltages, sp_sweep_voltages])
        aggregated_currents = np.concatenate([aggregated_currents, current_trace])

        fitted_peak = _analyze_single_window_barrier_sweep(
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
    return PeakWindowSweepOutput(
        best_peak=fitted_peak,
        aggregated_voltages=aggregated_voltages,
        aggregated_currents=aggregated_currents,
        classification=last_classification,
        score=float(fitted_peak.quality_score),  # Use quality_score for consistency
        num_peaks=1,
    )


@routine
def run_compensation(
    ctx: RoutineContext,
    peak_spacing: float,
    control_group_name: str,
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    zero_control_side: bool = False,
    gates_to_compensate: list[str] | None = None,
    session: LoggerSession | None = None,
    seed: int | None = None,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> dict[str, float]:
    """Calculate compensation gradients for control gates affecting charge sensor.

    Measures how control gate voltages affect sensor peak position and calculates
    gradients using RANSAC regression. Performs baseline measurements, then sweeps
    each control gate while measuring peak shifts.

    Args:
        ctx: Routine context with device resources. Requires find_sensor_peak results.
        peak_spacing: Expected peak spacing in volts (e.g., 0.020 for 20mV)
        control_group_name: Name of control side group (e.g., "side_A")
        measure_electrode: Electrode to measure current from (e.g., "OUT_B")
        bias_gate: Name of the bias gate to apply bias voltage
        bias_voltage: Voltage to apply to bias gate (V)
        zero_control_side: If True, measure gradients relative to 0V baseline (default: False)
        gates_to_compensate: Optional list of gate names to test (default: None, tests all)
        session: Logger session for measurements and analysis
        seed: Random seed for reproducible measurement ordering (default: None, non-deterministic)

    Returns:
        dict: Dictionary mapping gate names to compensation gradients (V/V)

    Raises:
        RoutineError: If find_sensor_peak results are missing or invalid
    """
    if peak_spacing <= 0:
        raise RoutineError("peak_spacing must be greater than 0")

    device = ctx.resources.device

    control_group = device.device_config.groups[control_group_name]
    control_gates = list(control_group.gates)
    control_gates = filter_gates_by_group(ctx, control_gates)

    find_sensor_peak_results = ctx.results.get("find_stable_sensor_peak")
    if find_sensor_peak_results is None:
        find_sensor_peak_results = ctx.results.get("find_sensor_peak", {})

    if not find_sensor_peak_results:
        raise RoutineError(
            "Peak-finding results not found in ctx.results. "
            "Please run either find_stable_sensor_peak or find_sensor_peak routine first."
        )
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

    voltage_range = MULTIPLIER_OF_PEAK_SPACING * peak_spacing
    half_n = PERTURBATION_DIVISOR // 2
    voltage_differences = np.concatenate(
        [
            np.linspace(-voltage_range, -voltage_range / half_n, half_n),
            np.linspace(voltage_range / half_n, voltage_range, half_n),
        ]
    )
    rng = np.random.default_rng(seed)

    all_plungers = device.get_gates_by_type(GateType.PLUNGER)
    all_barriers = device.get_gates_by_type(GateType.BARRIER)
    all_plungers = filter_gates_by_group(ctx, all_plungers)
    all_barriers = filter_gates_by_group(ctx, all_barriers)
    control_non_reservoir_gates = [
        g for g in (all_plungers + all_barriers) if g in control_gates
    ]

    if gates_to_compensate is not None:
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
        control_non_reservoir_gates = [
            g for g in control_non_reservoir_gates if g in gates_to_compensate
        ]

    initial_control_voltages = device.check(control_non_reservoir_gates)
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

    device.jump(baseline_control_state, wait_for_settling=True)
    time.sleep(DEFAULT_SETTLING_TIME_S)
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

        reference_max_gradient_voltage = float(np.median(baseline_sensitivity_voltages))
        reference_peak_center_voltage = float(np.median(baseline_peak_center_voltages))
    except Exception as e:
        raise RoutineError(f"Error in baseline measurement: {str(e)}") from e
    sensor_park_point_voltages = dict.fromkeys(
        sensor_gates_list, mean_reservoir_saturation_voltage
    )
    sensor_park_point_voltages[sensor_gate_key] = reference_max_gradient_voltage
    compensation_gradients_dict = {}
    per_gate_details = {}
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
        # Get voltage limits for this gate to validate perturbations
        if gate not in device.channel_configs:
            raise RoutineError(
                f"Gate '{gate}' not found in device channel configurations. "
                "Cannot validate voltage limits."
            )
        gate_voltage_range = device.channel_configs[gate].voltage_range
        min_voltage, max_voltage = gate_voltage_range
        if min_voltage is None or max_voltage is None:
            raise RoutineError(
                f"Voltage limits not configured for gate '{gate}'. "
                "Cannot safely apply voltage perturbations."
            )

        for counter, delta_index in enumerate(measurement_indices, start=1):
            voltage_difference = float(voltage_differences[delta_index])
            measurement_voltage_sequence.append(voltage_difference)

            # Calculate resulting voltage and validate against safety limits
            resulting_voltage = baseline_control_state[gate] + voltage_difference
            if resulting_voltage < min_voltage or resulting_voltage > max_voltage:
                raise RoutineError(
                    f"Voltage perturbation would exceed safety limits for gate '{gate}'. "
                    f"Baseline: {baseline_control_state[gate]:.6f}V, "
                    f"Perturbation: {voltage_difference:+.6f}V, "
                    f"Result: {resulting_voltage:.6f}V, "
                    f"Valid range: [{min_voltage:.6f}V, {max_voltage:.6f}V]. "
                    f"Consider reducing peak_spacing or MULTIPLIER_OF_PEAK_SPACING."
                )

            device_state = baseline_control_state.copy()
            device_state[gate] = resulting_voltage
            device.jump(device_state, wait_for_settling=True)
            time.sleep(DEFAULT_SETTLING_TIME_S)

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
            peak_shift = float(peak_center_voltage - reference_peak_center_voltage)
            sample_record = {
                "control_delta": voltage_difference,
                "delta_plunger": voltage_difference,
                "peak_position": peak_center_voltage,
                "peak_shift": peak_shift,
                "delta_peak": peak_shift,
                "sensitivity_voltage": float(best_peak.sensitivity_voltage),
            }
            measurement_samples.append(sample_record)

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
        per_point_gradients = peak_positions_difference / voltage_differences

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

        for i, sample in enumerate(measurement_samples):
            sample["is_inlier"] = bool(inlier_mask[i])

        per_voltage_inlier_counts = []
        for idx in range(num_deltas):
            samples_for_this_voltage = [
                i
                for i, sample in enumerate(measurement_samples)
                if np.isclose(sample["control_delta"], voltage_differences[idx])
            ]
            num_inliers_for_voltage = sum(
                inlier_mask[i] for i in samples_for_this_voltage
            )
            per_voltage_inlier_counts.append(num_inliers_for_voltage)

        per_gate_details[gate] = {
            "peak_positions": peak_positions,
            "peak_positions_difference": peak_positions_difference,
            "per_point_gradients": per_point_gradients,
            "least_squares_gradient": least_squares_gradient,
            "drift_intercept": drift_intercept,
            "mean_per_point_gradient": float(np.mean(per_point_gradients)),
            "mean_gradient": least_squares_gradient,
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

        reset_state = {gate: baseline_control_state[gate]}
        device.jump(reset_state, wait_for_settling=True)
        time.sleep(DEFAULT_SETTLING_TIME_S)

    logger.info("Compensation gradients: %s", compensation_gradients_dict)

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
