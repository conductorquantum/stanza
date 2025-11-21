"""
Charge sensor peak finding routines for quantum dot devices.

This module provides automated charge sensor peak detection and characterization
for quantum dot devices using ML-based classification, multi-model peak fitting,
and stability analysis. The routines identify optimal operating points for charge
sensing by locating Coulomb blockade peaks with high sensitivity and stability.

Physical Context:
-----------------
In quantum dot devices, a charge sensor is a quantum dot configured to operate
near a Coulomb blockade peak, where conductance changes rapidly with electron
number. This high sensitivity makes it ideal for detecting charge state changes
in nearby control quantum dots. The sensor's performance depends critically on
operating at the optimal point on the Coulomb blockade peak.

The optimal operating point balances two competing factors:
1. Sensitivity: Maximum current change per unit charge (steepest slope)
2. Stability: Minimum voltage noise from current fluctuations

Peak Finding Methodology:
--------------------------
The find_sensor_peak routine uses a multi-stage approach:

1. Wide-Range Sweep: Performs a coarse sweep of the sensor plunger gate across
   a wide voltage range to identify candidate Coulomb blockade peaks using
   ML-based peak detection.

2. Peak Fitting: For each detected peak, extracts a window around the peak and
   fits multiple models (Lorentzian, Sech², Voigt) to determine precise peak
   parameters including:
   - Peak center voltage
   - Peak width (FWHM)
   - Peak amplitude
   - Sensitivity (maximum gradient point)

3. Quality Scoring: Calculates a composite quality score for each peak based on:
   - Fit quality (R², RMSE, skew residual): 70% weight
   - Normalized sensitivity: 20% weight
   - Fit statistics: 10% weight

4. Peak Selection: Selects the peak with the highest quality score as the
   optimal operating point.

Stability Analysis:
--------------------
The find_stable_sensor_peak routine extends peak finding with stability testing:

1. Candidate Selection: Identifies the top N peaks (typically 3) by quality score

2. Stability Measurement: For each candidate:
   - Positions device at the peak's maximum gradient point (highest sensitivity)
   - Performs a 2-minute hold measurement
   - Records time-series current data
   - Calculates voltage noise: σ_V = σ_I / |dI/dV|

3. Combined Scoring: Computes a final score combining:
   - Original quality score (30% weight)
   - Stability score (70% weight)

4. Optimal Peak Selection: Returns the peak with the highest combined score,
   ensuring both high sensitivity and low noise.

ML-Based Peak Detection:
------------------------
The routine uses machine learning models for robust peak detection:
- Peak Detector Model: Identifies candidate peak locations in current traces
- Coulomb Classifier Model: Validates that detected features are genuine
  Coulomb blockade peaks rather than noise or artifacts

This ML-based approach provides robust peak detection even in noisy measurements
and handles various peak shapes and sizes automatically.

Integration with Charge Sensor Workflow:
----------------------------------------
This module is the first stage in a three-stage charge sensor workflow:

1. find_sensor_peak: Locates optimal charge sensing operating point
   - Returns peak location, sensitivity voltage, and narrowed voltage range
   - Provides quality metrics for peak assessment

2. run_compensation: Calculates compensation gradients for control gates
   - Uses the narrowed range from find_sensor_peak for high-resolution sweeps
   - Measures how control gates affect sensor peak position

3. charge_sensor_csd_readout: Performs compensated 2D sweeps
   - Uses gradients from run_compensation for feedforward compensation
   - Maintains sensor at optimal operating point during control gate sweeps
"""

# Standard library imports
import logging
from typing import Any

# Third-party imports
# First-party imports
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.routines import RoutineContext, routine
from stanza.routines.builtins.charge_sensor.utils.constants import (
    DEFAULT_WINDOW_HALF_WIDTH,
    INITIAL_WINDOW_MULTIPLIER,
    ML_MODEL_INPUT_SIZE,
    REFINED_STEP_MULTIPLIER,
    WINDOW_FRACTION,
)
from stanza.routines.builtins.charge_sensor.utils.peak_stability import (
    calculate_combined_scores,
    measure_peak_stability,
)
from stanza.routines.builtins.charge_sensor.utils.sweeps import (
    many_window_barrier_sweep,
)
from stanza.routines.builtins.charge_sensor.utils.types import (
    StablePeakCandidate,
)
from stanza.routines.builtins.utils.group_handling import filter_gates_by_group
from stanza.routines.builtins.utils.peak_fitting import (
    analyze_find_first_peak_voltages,
)

# Configure logger
logger = logging.getLogger(__name__)


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
        stability_measurement = measure_peak_stability(
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
    calculate_combined_scores(candidates, original_weight=0.5, stability_weight=0.5)

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
