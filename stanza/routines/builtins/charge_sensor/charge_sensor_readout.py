"""Charge sensor readout routines for quantum dot devices.

This module performs compensated 2D sweeps of control gates while maintaining
a constant sensor operating point through real-time compensation. The sensor
plunger voltage is adjusted using feedforward (gradient-based) and optional
feedback (current-based) correction to cancel capacitive cross-talk.

Compensation formula: V_sensor = V_initial + sum(gradient_i * delta_V_i) - beta * I_error
Optional adaptive learning updates gradients: gradient += (gamma / delta_V) * I_error
"""

import logging
import time
from typing import Any

import numpy as np

from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.routines import RoutineContext, routine
from stanza.routines.builtins.utils.group_handling import filter_gates_by_group

logger = logging.getLogger(__name__)

# Default settling time before sweeps to avoid current spikes
DEFAULT_SETTLING_TIME_S = 2.0


def _calculate_compensated_voltages(
    control_plunger_gates: list[str],
    control_plunger_ranges: dict[str, tuple[float, float]],
    compensation_gradients: dict[str, float],
    initial_control_voltages: dict[str, float],
    initial_sensor_voltage: float,
    charge_sensor_plunger_gate: str,
    sweep_resolution: int,
) -> tuple[list[list[float]], list[float], list[str]]:
    """Pre-compute compensated sensor voltages for all 2D sweep points.

    Args:
        control_plunger_gates: List of 2 control plunger gate names
        control_plunger_ranges: Dict mapping gate names to (start, end) voltage tuples
        compensation_gradients: Dict mapping gate names to dV_sensor/dV_control gradients
        initial_control_voltages: Initial voltages for control gates
        initial_sensor_voltage: Initial sensor plunger voltage
        charge_sensor_plunger_gate: Name of sensor plunger gate to compensate
        sweep_resolution: Number of points per dimension

    Returns:
        Tuple of (voltages_with_compensation, compensation_applied, gate_electrodes)
    """
    if len(control_plunger_gates) != 2:
        raise RoutineError(
            f"Expected exactly 2 control plunger gates, got {len(control_plunger_gates)}"
        )

    g1_name, g2_name = control_plunger_gates[0], control_plunger_gates[1]
    g1_start, g1_end = control_plunger_ranges[g1_name]
    g2_start, g2_end = control_plunger_ranges[g2_name]

    g1_voltages = np.linspace(g1_start, g1_end, sweep_resolution)
    g2_voltages = np.linspace(g2_start, g2_end, sweep_resolution)

    voltages_with_compensation = []
    compensation_applied = []

    current_sensor_voltage = initial_sensor_voltage
    previous_control_voltages = {
        g1_name: initial_control_voltages[g1_name],
        g2_name: initial_control_voltages[g2_name],
    }

    for row_idx, v_g1 in enumerate(g1_voltages):
        if row_idx % 2 == 0:
            row_voltages = g2_voltages
        else:
            row_voltages = g2_voltages[::-1]

        for v_g2 in row_voltages:
            delta_v1 = float(v_g1) - previous_control_voltages[g1_name]
            delta_v2 = float(v_g2) - previous_control_voltages[g2_name]

            compensation_update = 0.0
            if g1_name in compensation_gradients:
                compensation_update += compensation_gradients[g1_name] * delta_v1
            if g2_name in compensation_gradients:
                compensation_update += compensation_gradients[g2_name] * delta_v2

            current_sensor_voltage += compensation_update

            voltages_with_compensation.append(
                [float(v_g1), float(v_g2), float(current_sensor_voltage)]
            )

            cumulative_compensation = current_sensor_voltage - initial_sensor_voltage
            compensation_applied.append(float(cumulative_compensation))

            previous_control_voltages[g1_name] = float(v_g1)
            previous_control_voltages[g2_name] = float(v_g2)

    gate_electrodes = [g1_name, g2_name, charge_sensor_plunger_gate]
    return voltages_with_compensation, compensation_applied, gate_electrodes


@routine
def charge_sensor_csd_readout(  # pylint: disable=too-many-locals,too-many-statements
    ctx: RoutineContext,
    charge_sensor_group_name: str,
    control_group_name: str,
    sensor_park_point_voltages: dict[str, float],
    charge_sensor_plunger_gate: str,
    initial_control_voltages: dict[str, float],
    control_plunger_ranges: dict[str, tuple[float, float]],
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    compensation_gradients: dict[str, float] | None = None,
    sweep_resolution: int = 48,
    num_sweep_repetitions: int = 10,
    beta: float | None = None,
    max_feedback_correction: float = 1.0,
    gamma_factors: dict[str, float] | None = None,
    max_adaptive_gradient: float = 2.0,
    session: LoggerSession | None = None,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> dict[str, Any]:
    """Perform charge sensor CSD readout sweep with optional compensation.

    Sweeps control group plunger gates while measuring current through the charge sensor.
    Optionally compensates sensor plunger voltage to maintain constant operating point.

    Args:
        ctx: Routine context containing device resources
        charge_sensor_group_name: Name of charge sensor group (e.g., "side_A")
        control_group_name: Name of control group being swept (e.g., "side_B")
        sensor_park_point_voltages: Voltages for all charge sensor gates (V)
        charge_sensor_plunger_gate: Name of sensor plunger gate to compensate (e.g., "G3")
        initial_control_voltages: Initial voltages for ALL control group gates (V)
        control_plunger_ranges: Voltage ranges for exactly 2 control plunger gates {gate: (start, end)}
        measure_electrode: Electrode to measure current from (e.g., "OUT_A")
        bias_gate: Name of the bias gate to apply bias voltage
        bias_voltage: Voltage to apply to bias gate (V)
        compensation_gradients: Optional gradients {gate: dV_sensor/dV_control} (default: None)
        sweep_resolution: Number of points per dimension (default: 48)
        num_sweep_repetitions: Number of times to repeat sweep (default: 10)
        beta: Optional proportional feedback gain (V/A) for current-based correction (default: None)
        max_feedback_correction: Maximum feedback correction (default: 1.0)
        gamma_factors: Optional per-gate adaptation gains {gate: gamma} (default: None)
        max_adaptive_gradient: Maximum allowed magnitude for adaptive gradients (V/V) (default: 2.0)
        session: Logger session for measurements and analysis

    Returns:
        dict: Contains voltage_measurements, current_measurements, compensation_applied,
              control_plunger_gates, compensation_gradients, beta, gamma_factors, and
              related metadata

    Raises:
        RoutineError: If validation fails or sweep encounters errors
    """
    # Validate inputs
    if sweep_resolution <= 0:
        raise RoutineError("sweep_resolution must be greater than 0")
    if num_sweep_repetitions <= 0:
        raise RoutineError("num_sweep_repetitions must be greater than 0")
    if not sensor_park_point_voltages:
        raise RoutineError("sensor_park_point_voltages cannot be empty")
    if not initial_control_voltages:
        raise RoutineError("initial_control_voltages cannot be empty")
    if len(control_plunger_ranges) != 2:
        raise RoutineError(
            f"control_plunger_ranges must contain exactly 2 gates, got {len(control_plunger_ranges)}"
        )
    if charge_sensor_plunger_gate not in sensor_park_point_voltages:
        raise RoutineError(
            f"Sensor plunger gate '{charge_sensor_plunger_gate}' not found "
            "in sensor_park_point_voltages"
        )
    if max_feedback_correction <= 0:
        raise RoutineError(
            f"max_feedback_correction must be positive, got {max_feedback_correction}"
        )
    if max_adaptive_gradient <= 0:
        raise RoutineError(
            f"max_adaptive_gradient must be positive, got {max_adaptive_gradient}"
        )

    # Validate gamma_factors if provided
    gradient_adaptation_enabled = gamma_factors is not None
    if gradient_adaptation_enabled:
        if not compensation_gradients:
            raise RoutineError(
                "gamma_factors requires compensation_gradients to be provided. "
                "Gradient adaptation needs initial gradient values to adapt from."
            )
        # Validate gamma_factors keys match control plunger gates
        control_plunger_gates_temp = list(control_plunger_ranges.keys())
        for gate in control_plunger_gates_temp:
            if gate not in gamma_factors:  # type: ignore
                raise RoutineError(
                    f"Missing gamma factor for control plunger '{gate}'. "
                    "gamma_factors must include all control plunger gates."
                )
        # Validate gamma values are non-negative
        for gate, gamma in gamma_factors.items():  # type: ignore
            if gamma < 0:
                raise RoutineError(
                    f"gamma factor for gate '{gate}' must be non-negative, got {gamma}"
                )

    # Determine if compensation is enabled
    compensation_enabled = compensation_gradients is not None

    # Determine if current feedback is enabled
    feedback_enabled = beta is not None and beta != 0.0

    # Validate compensation gradients if enabled
    control_plunger_gates = list(control_plunger_ranges.keys())

    # Validate that control plungers are in initial_control_voltages
    for gate in control_plunger_gates:
        if gate not in initial_control_voltages:
            raise RoutineError(
                f"Control plunger '{gate}' not found in initial_control_voltages. "
                "Please provide initial voltage for this gate."
            )

    if compensation_enabled:
        for gate in control_plunger_gates:
            if gate not in compensation_gradients:  # type: ignore
                raise RoutineError(
                    f"Missing compensation gradient for control plunger '{gate}'. "
                    "Please provide gradient in compensation_gradients dict."
                )

    device = ctx.resources.device

    charge_sensor_group = device.device_config.groups[charge_sensor_group_name]
    control_group = device.device_config.groups[control_group_name]

    charge_sensor_gates = list(charge_sensor_group.gates)
    control_gates = list(control_group.gates)

    charge_sensor_gates = filter_gates_by_group(ctx, charge_sensor_gates)
    control_gates = filter_gates_by_group(ctx, control_gates)
    logger.info("Charge sensor readout configuration:")
    logger.info("  Charge sensor group: %s", charge_sensor_group_name)
    logger.info("  Control group: %s", control_group_name)
    logger.info("  Control plungers: %s", control_plunger_gates)
    logger.info("  Sensor plunger: %s", charge_sensor_plunger_gate)
    logger.info("  Measure electrode: %s", measure_electrode)
    logger.info("  Sweep resolution: %dx%d", sweep_resolution, sweep_resolution)
    logger.info("  Number of repetitions: %d", num_sweep_repetitions)
    logger.info("  Compensation enabled: %s", compensation_enabled)
    if compensation_enabled:
        logger.info("  Compensation gradients: %s", compensation_gradients)
    logger.info("  Gradient adaptation enabled: %s", gradient_adaptation_enabled)
    if gradient_adaptation_enabled:
        logger.info("  Gamma factors: %s", gamma_factors)
        logger.info("  Max adaptive gradient: %.6f V/V", max_adaptive_gradient)
    logger.info("  Current feedback enabled: %s", feedback_enabled)
    if feedback_enabled:
        logger.info("  Beta (feedback gain): %.6e V/A", beta)
        logger.info("  Max feedback correction: %.6f V", max_feedback_correction)

    voltage_dict = {}

    for gate, voltage in sensor_park_point_voltages.items():
        voltage_dict[gate] = float(voltage)

    for gate, voltage in initial_control_voltages.items():
        if gate in voltage_dict:
            logger.warning(
                "Gate %s appears in both sensor and control groups. "
                "Using control voltage %.6fV (overriding sensor voltage %.6fV)",
                gate,
                voltage,
                voltage_dict[gate],
            )
        voltage_dict[gate] = float(voltage)

    initial_control_plunger_voltages = {}
    for gate, (start, _end) in control_plunger_ranges.items():
        voltage_dict[gate] = float(start)
        initial_control_plunger_voltages[gate] = float(start)

    logger.info("Setting initial gate voltages...")
    device.jump(voltage_dict, wait_for_settling=True)
    device.jump({bias_gate: bias_voltage}, wait_for_settling=True)

    logger.info("Waiting %d seconds for device settling...", DEFAULT_SETTLING_TIME_S)
    time.sleep(DEFAULT_SETTLING_TIME_S)

    park_point_current = device.measure(measure_electrode)
    logger.info("Baseline park point current: %.6e A", park_point_current)

    initial_sensor_voltage = float(
        sensor_park_point_voltages[charge_sensor_plunger_gate]
    )

    if gradient_adaptation_enabled:
        adaptive_gradients = compensation_gradients.copy()  # type: ignore
        gradient_history: list[dict[str, Any]] | None = []
        delta_v_threshold = 1e-9
        logger.info(
            "Gradient adaptation initialized with delta_v_threshold=%.3e V",
            delta_v_threshold,
        )
        logger.info("Initial gradients: %s", adaptive_gradients)
    else:
        adaptive_gradients = None
        gradient_history = None
        delta_v_threshold = None

    if compensation_enabled and not gradient_adaptation_enabled:
        (
            voltages_with_compensation,
            compensation_applied,
            gate_electrodes,
        ) = _calculate_compensated_voltages(
            control_plunger_gates=control_plunger_gates,
            control_plunger_ranges=control_plunger_ranges,
            compensation_gradients=compensation_gradients,  # type: ignore
            initial_control_voltages=initial_control_plunger_voltages,
            initial_sensor_voltage=initial_sensor_voltage,
            charge_sensor_plunger_gate=charge_sensor_plunger_gate,
            sweep_resolution=sweep_resolution,
        )
        logger.info(
            "Compensation range: %.6f to %.6f V (serpentine scan pattern)",
            min(compensation_applied),
            max(compensation_applied),
        )
    elif not compensation_enabled:
        logger.info(
            "Compensation disabled - sensor plunger held constant (serpentine scan pattern)"
        )
        g1_name, g2_name = control_plunger_gates[0], control_plunger_gates[1]
        g1_voltages = np.linspace(*control_plunger_ranges[g1_name], sweep_resolution)
        g2_voltages = np.linspace(*control_plunger_ranges[g2_name], sweep_resolution)

        voltages_list = []
        for row_idx, v_g1 in enumerate(g1_voltages):
            row_voltages = g2_voltages if row_idx % 2 == 0 else g2_voltages[::-1]
            for v_g2 in row_voltages:
                voltages_list.append([float(v_g1), float(v_g2)])

        voltages_with_compensation = voltages_list
        compensation_applied = [0.0] * len(voltages_list)
        gate_electrodes = control_plunger_gates
    else:
        logger.info(
            "Adaptive gradient mode - compensation computed dynamically (serpentine scan pattern)"
        )
        g1_name, g2_name = control_plunger_gates[0], control_plunger_gates[1]
        g1_voltages = np.linspace(*control_plunger_ranges[g1_name], sweep_resolution)
        g2_voltages = np.linspace(*control_plunger_ranges[g2_name], sweep_resolution)

        control_voltages_grid = []
        for row_idx, v_g1 in enumerate(g1_voltages):
            row_voltages = g2_voltages if row_idx % 2 == 0 else g2_voltages[::-1]
            for v_g2 in row_voltages:
                control_voltages_grid.append([float(v_g1), float(v_g2)])

        voltages_with_compensation = control_voltages_grid
        compensation_applied = []
        gate_electrodes = [g1_name, g2_name, charge_sensor_plunger_gate]

    # Log sweep range info
    logger.info(
        "Starting 2D sweep: %s (%.6fV to %.6fV), %s (%.6fV to %.6fV)",
        control_plunger_gates[0],
        control_plunger_ranges[control_plunger_gates[0]][0],
        control_plunger_ranges[control_plunger_gates[0]][1],
        control_plunger_gates[1],
        control_plunger_ranges[control_plunger_gates[1]][0],
        control_plunger_ranges[control_plunger_gates[1]][1],
    )

    # Perform multiple sweeps and average
    currents_list = []
    feedback_corrections_list = []

    # Initialize clipping event counters
    sensor_clipping_events_pre_feedback = 0
    gradient_clipping_events = 0

    for i in range(num_sweep_repetitions):
        logger.info("Starting sweep %d of %d...", i + 1, num_sweep_repetitions)

        # Reset to initial position before each sweep
        reset_dict = {}
        if compensation_enabled:
            reset_dict[charge_sensor_plunger_gate] = initial_sensor_voltage
        for gate in control_plunger_gates:
            reset_dict[gate] = initial_control_plunger_voltages[gate]

        device.jump(reset_dict, wait_for_settling=True)
        time.sleep(DEFAULT_SETTLING_TIME_S)

        voltage_measurements_rep = []
        current_measurements_rep = []
        feedback_corrections_rep = []

        if gradient_adaptation_enabled:
            previous_control_voltages = {
                g: initial_control_plunger_voltages[g] for g in control_plunger_gates
            }
            current_sensor_voltage = initial_sensor_voltage
            gradient_update_count = dict.fromkeys(control_plunger_gates, 0)

        for point_idx, voltage_point in enumerate(voltages_with_compensation):
            if gradient_adaptation_enabled:
                assert adaptive_gradients is not None
                assert gradient_history is not None
                assert delta_v_threshold is not None
                control_v1, control_v2 = voltage_point[0], voltage_point[1]
                g1_name, g2_name = (
                    control_plunger_gates[0],
                    control_plunger_gates[1],
                )

                delta_v1 = control_v1 - previous_control_voltages[g1_name]
                delta_v2 = control_v2 - previous_control_voltages[g2_name]

                compensation_update = 0.0
                compensation_update += adaptive_gradients[g1_name] * delta_v1
                compensation_update += adaptive_gradients[g2_name] * delta_v2

                current_sensor_voltage += compensation_update

                voltage_dict = {
                    g1_name: control_v1,
                    g2_name: control_v2,
                    charge_sensor_plunger_gate: current_sensor_voltage,
                }

                cumulative_compensation = (
                    current_sensor_voltage - initial_sensor_voltage
                )
                compensation_applied.append(float(cumulative_compensation))

                previous_control_voltages[g1_name] = control_v1
                previous_control_voltages[g2_name] = control_v2
            else:
                voltage_dict = dict(zip(gate_electrodes, voltage_point, strict=False))

            if charge_sensor_plunger_gate in voltage_dict:
                sensor_voltage = voltage_dict[charge_sensor_plunger_gate]

                min_voltage, max_voltage = device.channel_configs[
                    charge_sensor_plunger_gate
                ].voltage_range

                if min_voltage is None:
                    raise RoutineError(
                        f"No minimum voltage limit configured for {charge_sensor_plunger_gate}"
                    )
                if max_voltage is None:
                    raise RoutineError(
                        f"No maximum voltage limit configured for {charge_sensor_plunger_gate}"
                    )

                if sensor_voltage < min_voltage or sensor_voltage > max_voltage:
                    original_voltage = sensor_voltage
                    sensor_voltage = float(
                        np.clip(sensor_voltage, min_voltage, max_voltage)
                    )
                    voltage_dict[charge_sensor_plunger_gate] = sensor_voltage
                    sensor_clipping_events_pre_feedback += 1
                    logger.warning(
                        "Sensor voltage for %s out of bounds: %.6fV clipped to %.6fV (valid range: %.1f to %.1fV). "
                        "Point %d, Rep %d. Consider reducing gamma or max_adaptive_gradient.",
                        charge_sensor_plunger_gate,
                        original_voltage,
                        sensor_voltage,
                        min_voltage,
                        max_voltage,
                        point_idx,
                        i,
                    )

            device.jump(voltage_dict, wait_for_settling=True)

            current = device.measure(measure_electrode)
            current_error_pre = current - park_point_current

            feedback_correction = 0.0
            if feedback_enabled:
                assert beta is not None
                feedback_correction_raw = -beta * current_error_pre

                current_sensor_v = voltage_dict[charge_sensor_plunger_gate]
                min_v, max_v = device.channel_configs[
                    charge_sensor_plunger_gate
                ].voltage_range

                if min_v is None:
                    raise RoutineError(
                        f"No minimum voltage limit configured for {charge_sensor_plunger_gate}"
                    )
                if max_v is None:
                    raise RoutineError(
                        f"No maximum voltage limit configured for {charge_sensor_plunger_gate}"
                    )

                max_correction_up = max_v - current_sensor_v
                max_correction_down = current_sensor_v - min_v

                feedback_correction = float(
                    np.clip(
                        feedback_correction_raw,
                        -max_correction_down,
                        max_correction_up,
                    )
                )

                voltage_dict[charge_sensor_plunger_gate] += feedback_correction
                if gradient_adaptation_enabled:
                    current_sensor_voltage += feedback_correction

                device.jump(voltage_dict, wait_for_settling=True)
                current = device.measure(measure_electrode)

            if gradient_adaptation_enabled:
                assert adaptive_gradients is not None
                assert gradient_history is not None
                assert delta_v_threshold is not None
                assert gamma_factors is not None
                if abs(delta_v1) > delta_v_threshold:
                    assert current_error_pre is not None
                    gradient_update = (
                        gamma_factors[g1_name] / delta_v1
                    ) * current_error_pre

                    old_gradient = adaptive_gradients[g1_name]
                    adaptive_gradients[g1_name] += gradient_update

                    adaptive_gradients[g1_name] = float(
                        np.clip(
                            adaptive_gradients[g1_name],
                            -max_adaptive_gradient,
                            max_adaptive_gradient,
                        )
                    )

                    if abs(old_gradient + gradient_update) > max_adaptive_gradient:
                        gradient_clipping_events += 1
                        logger.warning(
                            "Adaptive gradient for %s clipped: %.6f -> %.6f (limit: ±%.6f V/V)",
                            g1_name,
                            old_gradient + gradient_update,
                            adaptive_gradients[g1_name],
                            max_adaptive_gradient,
                        )

                    gradient_update_count[g1_name] += 1
                    gradient_history.append(
                        {
                            "point_index": point_idx,
                            "repetition": i,
                            "gate": g1_name,
                            "delta_v": float(delta_v1),
                            "current_error_pre_feedback": float(current_error_pre),
                            "gradient_update": float(gradient_update),
                            "new_gradient": float(adaptive_gradients[g1_name]),
                        }
                    )

                if abs(delta_v2) > delta_v_threshold:
                    assert current_error_pre is not None
                    gradient_update = (
                        gamma_factors[g2_name] / delta_v2
                    ) * current_error_pre

                    old_gradient = adaptive_gradients[g2_name]
                    adaptive_gradients[g2_name] += gradient_update

                    adaptive_gradients[g2_name] = float(
                        np.clip(
                            adaptive_gradients[g2_name],
                            -max_adaptive_gradient,
                            max_adaptive_gradient,
                        )
                    )

                    if abs(old_gradient + gradient_update) > max_adaptive_gradient:
                        gradient_clipping_events += 1
                        logger.warning(
                            "Adaptive gradient for %s clipped: %.6f -> %.6f (limit: ±%.6f V/V)",
                            g2_name,
                            old_gradient + gradient_update,
                            adaptive_gradients[g2_name],
                            max_adaptive_gradient,
                        )

                    gradient_update_count[g2_name] += 1
                    gradient_history.append(
                        {
                            "point_index": point_idx,
                            "repetition": i,
                            "gate": g2_name,
                            "delta_v": float(delta_v2),
                            "current_error_pre_feedback": float(current_error_pre),
                            "gradient_update": float(gradient_update),
                            "new_gradient": float(adaptive_gradients[g2_name]),
                        }
                    )

            actual_voltages = [voltage_dict[g] for g in control_plunger_gates]
            voltage_measurements_rep.append(actual_voltages)
            current_measurements_rep.append(current)
            feedback_corrections_rep.append(feedback_correction)

        if gradient_adaptation_enabled:
            logger.info(
                "Repetition %d gradient updates: %s",
                i + 1,
                gradient_update_count,
            )
            logger.info("Repetition %d final gradients: %s", i + 1, adaptive_gradients)

        voltage_measurements = voltage_measurements_rep
        current_measurements = current_measurements_rep
        currents_list.append(current_measurements)
        feedback_corrections_list.append(feedback_corrections_rep)

        g1_name, g2_name = control_plunger_gates

        if session is not None:
            session.log_sweep(
                name="charge_sensor_csd_readout",
                x_data=voltage_measurements,
                y_data=current_measurements,
                x_label=", ".join(control_plunger_gates),
                y_label="current",
                metadata={"repetition": i + 1},
            )

    # Average currents across all sweeps
    average_currents = np.mean(currents_list, axis=0)

    # Average feedback corrections across all sweeps
    if feedback_enabled:
        average_feedback_corrections = np.mean(feedback_corrections_list, axis=0)
    else:
        average_feedback_corrections = np.zeros(len(average_currents))

    # Subtract baseline to get differential current signal
    differential_currents = average_currents - park_point_current

    logger.info("Sweep completed: %d measurements acquired", len(differential_currents))
    logger.info(
        "Differential current range: %.3e to %.3e A",
        np.min(differential_currents),
        np.max(differential_currents),
    )

    logger.info("Clipping events summary:")
    logger.info(
        "  Sensor voltage clipping (pre-feedback): %d events",
        sensor_clipping_events_pre_feedback,
    )
    if gradient_adaptation_enabled:
        logger.info("  Gradient clipping: %d events", gradient_clipping_events)

    if feedback_enabled:
        logger.info(
            "Feedback correction range: %.6f to %.6f V",
            np.min(average_feedback_corrections),
            np.max(average_feedback_corrections),
        )
        logger.info(
            "Feedback correction mean: %.6f V (std: %.6f V)",
            np.mean(average_feedback_corrections),
            np.std(average_feedback_corrections),
        )

    if gradient_adaptation_enabled:
        assert compensation_gradients is not None
        assert adaptive_gradients is not None
        assert gradient_history is not None
        logger.info("Gradient adaptation summary:")
        logger.info("  Initial gradients: %s", compensation_gradients)
        logger.info("  Final gradients: %s", adaptive_gradients)
        for gate in control_plunger_gates:
            initial_grad = compensation_gradients[gate]
            final_grad = adaptive_gradients[gate]
            change = final_grad - initial_grad
            percent_change = (change / initial_grad * 100) if initial_grad != 0 else 0
            logger.info(
                "  %s: %.6f -> %.6f (change: %.6f, %.2f%%)",
                gate,
                initial_grad,
                final_grad,
                change,
                percent_change,
            )
        logger.info("  Total gradient updates: %d", len(gradient_history))

    voltage_measurements_2d = [[v[0], v[1]] for v in voltage_measurements]

    # Log sweep data
    if session:
        session.log_sweep(
            name="charge_sensor_csd_readout",
            x_data=voltage_measurements_2d,
            y_data=differential_currents.tolist(),
            x_label=", ".join(control_plunger_gates),
            y_label="differential_current",
            metadata={
                "compensation_enabled": compensation_enabled,
                "feedback_enabled": feedback_enabled,
                "beta": float(beta) if beta is not None else None,
                "sensor_plunger": charge_sensor_plunger_gate,
                "measure_electrode": measure_electrode,
                "gate_electrodes": gate_electrodes,
                "num_repetitions": num_sweep_repetitions,
                "park_point_current": float(park_point_current),
            },
        )

        session.log_analysis(
            name="charge_sensor_csd_readout_summary",
            data={
                "control_plunger_gates": control_plunger_gates,
                "sweep_resolution": sweep_resolution,
                "total_measurements": len(differential_currents),
                "current_min": float(np.min(differential_currents)),
                "current_max": float(np.max(differential_currents)),
                "current_mean": float(np.mean(differential_currents)),
                "current_std": float(np.std(differential_currents)),
                "compensation_enabled": compensation_enabled,
                "feedback_enabled": feedback_enabled,
                "beta": float(beta) if beta is not None else None,
                "measure_electrode": measure_electrode,
                "num_repetitions": num_sweep_repetitions,
                "park_point_current": float(park_point_current),
                "sensor_clipping_events_pre_feedback": sensor_clipping_events_pre_feedback,
                "gradient_clipping_events": gradient_clipping_events,
            },
        )

        if compensation_enabled:
            session.log_analysis(
                name="charge_sensor_csd_compensation_summary",
                data={
                    "compensation_min": float(np.min(compensation_applied)),
                    "compensation_max": float(np.max(compensation_applied)),
                    "compensation_mean": float(np.mean(compensation_applied)),
                    "compensation_std": float(np.std(compensation_applied)),
                },
            )

        if feedback_enabled:
            assert beta is not None
            session.log_analysis(
                name="charge_sensor_csd_feedback_summary",
                data={
                    "feedback_min": float(np.min(average_feedback_corrections)),
                    "feedback_max": float(np.max(average_feedback_corrections)),
                    "feedback_mean": float(np.mean(average_feedback_corrections)),
                    "feedback_std": float(np.std(average_feedback_corrections)),
                    "beta": float(beta),
                },
            )

        if gradient_adaptation_enabled:
            assert adaptive_gradients is not None
            assert gradient_history is not None
            assert compensation_gradients is not None
            assert gamma_factors is not None
            session.log_analysis(
                name="charge_sensor_csd_gradient_adaptation_summary",
                data={
                    "initial_gradients": {
                        k: float(v) for k, v in compensation_gradients.items()
                    },
                    "final_gradients": {
                        k: float(v) for k, v in adaptive_gradients.items()
                    },
                    "gradient_changes": {
                        gate: float(
                            adaptive_gradients[gate] - compensation_gradients[gate]
                        )
                        for gate in control_plunger_gates
                    },
                    "gamma_factors": {k: float(v) for k, v in gamma_factors.items()},
                    "total_updates": len(gradient_history),
                },
            )

            session.log_analysis(
                name="charge_sensor_csd_gradient_history",
                data={"gradient_history": gradient_history},
            )

    result = {
        "voltage_measurements": voltage_measurements_2d,
        "current_measurements": differential_currents.tolist(),
        "compensation_applied": compensation_applied,
        "control_plunger_gates": control_plunger_gates,
        "control_plunger_ranges": control_plunger_ranges,
        "sweep_resolution": sweep_resolution,
        "measure_electrode": measure_electrode,
        "bias_voltage": float(bias_voltage),
        "bias_gate": bias_gate,
        "initial_sensor_plunger_voltage": (
            initial_sensor_voltage if compensation_enabled else None
        ),
        "sensor_park_point_voltages": sensor_park_point_voltages,
        "initial_control_voltages": initial_control_voltages,
        "compensation_gradients": compensation_gradients,
        "compensation_enabled": compensation_enabled,
        "park_point_current": float(park_point_current),
        "num_repetitions": num_sweep_repetitions,
        "beta": beta,
        "feedback_enabled": feedback_enabled,
        "feedback_corrections": average_feedback_corrections.tolist(),
        "sensor_clipping_events_pre_feedback": sensor_clipping_events_pre_feedback,
        "gradient_clipping_events": gradient_clipping_events,
        # Gradient adaptation fields
        "gamma_factors": gamma_factors,
        "gradient_adaptation_enabled": gradient_adaptation_enabled,
        "initial_gradients": compensation_gradients
        if gradient_adaptation_enabled
        else None,
        "final_gradients": adaptive_gradients if gradient_adaptation_enabled else None,
        "gradient_history": gradient_history if gradient_adaptation_enabled else None,
    }

    return result
