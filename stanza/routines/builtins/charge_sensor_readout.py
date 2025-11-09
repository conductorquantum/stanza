"""
Charge sensor readout routine for quantum dot devices.

This module provides compensated charge sensor readout functionality for quantum
devices, allowing measurement of control gate effects while maintaining constant
sensor operating point through real-time compensation.

Physical Context:
-----------------
In quantum dot devices, a charge sensor (quantum dot near Coulomb blockade) can
detect charge state changes in nearby control dots. However, changing control gate
voltages also shifts the sensor's operating point through capacitive cross-talk.

This routine sweeps control gates while dynamically compensating the sensor plunger
voltage to maintain optimal charge sensing fidelity throughout the measurement.

Compensation formula:
    V_sensor_compensated = V_sensor_initial + sum(gradient_i * delta_V_control_i)

where gradient_i = dV_sensor/dV_control_i from the run_compensation routine.
"""

# Standard library imports
import logging
import time
from typing import Any

# Third-party imports
import numpy as np

# First-party imports
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.models import GateType
from stanza.routines import RoutineContext, routine
from stanza.routines.builtins.utils.group_handling import filter_gates_by_group

# Configure logger
logger = logging.getLogger(__name__)

# Default settling time before sweeps to avoid current spikes
DEFAULT_SETTLING_TIME_S = 10


@routine
def charge_sensor_readout(  # pylint: disable=too-many-locals,too-many-statements
    ctx: RoutineContext,
    charge_sensor_park_voltages: dict[str, float],
    charge_sensor_group_name: str,
    control_group_name: str,
    charge_sensor_plunger_gate: str,
    measure_electrode: str,
    bias_gate: str,
    bias_voltage: float,
    compensation_gradients: dict[str, float] | None = None,
    sweep_resolution: int = 100,
    session: LoggerSession | None = None,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> dict[str, Any]:
    """
    Perform charge sensor readout sweep with control gates.

    This routine sweeps control group plunger gates while measuring current through
    the charge sensor. The charge sensor plunger voltage can be optionally compensated
    to maintain a constant sensor operating point as control voltages change.

    Physical context:
    - Charge sensor group: Acts as sensor, measures current changes
    - Control group: Gates being swept to modify quantum dot states
    - Compensation (optional): Adjusts sensor plunger to cancel cross-talk from control gates

    Args:
        ctx: Routine context containing device resources and previous results. Requires:
             - ctx.results[f"finger_gate_characterization_{control_group_name}"]
             - ctx.results[f"global_accumulation_{control_group_name}"]
        charge_sensor_park_voltages: Initial voltages for all charge sensor gates
        charge_sensor_group_name: Name of charge sensor group (e.g., "side_A")
        control_group_name: Name of control group being swept (e.g., "side_B")
        charge_sensor_plunger_gate: Name of sensor plunger gate to compensate
        measure_electrode: Electrode to measure current from (e.g., "OUT_A")
        bias_gate: Name of the bias gate (contact) to apply bias voltage
        bias_voltage: Voltage to apply to bias gate during measurements (V)
        compensation_gradients: Optional gradients {gate: dV_sensor/dV_control} from
            run_compensation routine. If None or empty, no compensation is applied.
        sweep_resolution: Number of points per dimension for 2D sweep (default 100)
        session: Logger session for measurements and analysis
        **kwargs: Additional keyword arguments (for config compatibility)

    Returns:
        dict: Contains:
            - voltage_measurements: List of [G1_voltage, G2_voltage] pairs
            - current_measurements: List of measured currents (A)
            - compensation_applied: List of compensation voltages applied to sensor (V)
            - control_plunger_gates: List of control plunger gate names
            - sweep_voltages: Dict mapping gate names to voltage arrays
            - sweep_resolution: Number of points per dimension
            - measure_electrode: Name of measurement electrode
            - initial_sensor_plunger_voltage: Initial charge sensor plunger voltage (V)
            - charge_sensor_park_voltages: Dict of initial sensor gate voltages
            - compensation_gradients: Dict of compensation gradients used (or None)
            - compensation_enabled: Boolean indicating if compensation was applied
            - global_turn_on_voltage: Global turn-on voltage used for control gates (V)

    Raises:
        RoutineError: If required previous results are missing or sweep fails

    Notes:
        - Control plungers swept from transition to cutoff voltage
        - Control barriers/reservoirs set to global turn-on saturation
        - If compensation_gradients provided: sensor plunger compensated at each sweep point
        - If compensation_gradients is None or empty: sensor plunger held constant
        - Compensation formula: V_compensated = V_initial + sum(gradient_i * delta_V_i)
    """
    # Validate inputs
    if sweep_resolution <= 0:
        raise RoutineError("sweep_resolution must be greater than 0")
    if not charge_sensor_park_voltages:
        raise RoutineError("charge_sensor_park_voltages cannot be empty")
    if charge_sensor_plunger_gate not in charge_sensor_park_voltages:
        raise RoutineError(
            f"Sensor plunger gate '{charge_sensor_plunger_gate}' not found "
            "in charge_sensor_park_voltages"
        )

    # Determine if compensation is enabled
    compensation_enabled = bool(compensation_gradients)

    # Get device
    device = ctx.resources.device

    # Get control group plunger gates
    control_group = device.device_config.groups[control_group_name]
    control_gates = list(control_group.gates)
    control_gates = filter_gates_by_group(ctx, control_gates)

    all_plungers = device.get_gates_by_type(GateType.PLUNGER)
    all_barriers = device.get_gates_by_type(GateType.BARRIER)
    all_reservoirs = device.get_gates_by_type(GateType.RESERVOIR)
    all_plungers = filter_gates_by_group(ctx, all_plungers)
    all_barriers = filter_gates_by_group(ctx, all_barriers)
    all_reservoirs = filter_gates_by_group(ctx, all_reservoirs)

    control_plungers = [g for g in all_plungers if g in control_gates]
    control_barriers = [g for g in all_barriers if g in control_gates]
    control_reservoirs = [g for g in all_reservoirs if g in control_gates]

    if len(control_plungers) != 2:
        raise RoutineError(
            f"Expected exactly 2 control plunger gates, found {len(control_plungers)}: "
            f"{control_plungers}"
        )

    # Extract voltage parameters for control group from cached results
    finger_results_key = f"finger_gate_characterization_{control_group_name}"
    finger_results = ctx.results.get(finger_results_key)
    if not finger_results:
        raise RoutineError(
            f"finger_gate_characterization not found for '{control_group_name}'. "
            "Please run finger_gate_characterization routine first."
        )

    # Extract nested dict if present
    if "finger_gate_characterization" in finger_results:
        finger_results = finger_results["finger_gate_characterization"]

    global_results_key = f"global_accumulation_{control_group_name}"
    global_results = ctx.results.get(global_results_key)
    if not global_results:
        raise RoutineError(
            f"global_accumulation not found for '{control_group_name}'. "
            "Please run global_accumulation routine first."
        )

    global_turn_on = global_results.get("global_turn_on_voltage")
    if global_turn_on is None:
        raise RoutineError(
            f"global_turn_on_voltage not found in global_accumulation results "
            f"for '{control_group_name}'"
        )

    # Extract transition and cutoff voltages for control plungers
    plunger_params = {}
    for gate in control_plungers:
        if gate not in finger_results:
            raise RoutineError(
                f"No characterization data found for control plunger '{gate}'"
            )

        gate_data = finger_results[gate]
        transition_v = gate_data["transition_voltage"]
        cutoff_v = gate_data["cutoff_voltage"]
        plunger_params[gate] = {
            "transition": float(transition_v),
            "cutoff": float(cutoff_v),
        }

    # Validate compensation gradients if compensation is enabled
    if compensation_enabled:
        for gate in control_plungers:
            if gate not in compensation_gradients:  # type: ignore
                raise RoutineError(
                    f"Missing compensation gradient for control plunger '{gate}'. "
                    "Please provide gradient in compensation_gradients dict."
                )

    # Log configuration
    logger.info("Charge sensor readout configuration:")
    logger.info("  Charge sensor group: %s", charge_sensor_group_name)
    logger.info("  Control group: %s", control_group_name)
    logger.info("  Control plungers: %s", control_plungers)
    logger.info("  Control barriers: %s", control_barriers)
    logger.info("  Control reservoirs: %s", control_reservoirs)
    logger.info("  Sensor plunger: %s", charge_sensor_plunger_gate)
    logger.info("  Measure electrode: %s", measure_electrode)
    logger.info("  Sweep resolution: %dx%d", sweep_resolution, sweep_resolution)
    logger.info("  Compensation enabled: %s", compensation_enabled)
    if compensation_enabled:
        logger.info("  Compensation gradients: %s", compensation_gradients)

    # Capture initial device state for cleanup
    initial_voltages = device.check(device.control_gates)

    try:
        # Set charge sensor gates to park voltages
        device.jump(charge_sensor_park_voltages, wait_for_settling=True)

        # Set control barriers and reservoirs to global turn-on
        control_voltage_dict = {}
        for gate in control_barriers + control_reservoirs:
            control_voltage_dict[gate] = float(global_turn_on)

        # Set control plungers to initial sweep position (transition voltage)
        initial_control_voltages = {}
        for gate in control_plungers:
            initial_voltage = plunger_params[gate]["transition"]
            control_voltage_dict[gate] = initial_voltage
            initial_control_voltages[gate] = initial_voltage

        device.jump(control_voltage_dict, wait_for_settling=True)

        # Apply bias voltage
        device.jump({bias_gate: bias_voltage}, wait_for_settling=True)

        # Allow settling time
        time.sleep(DEFAULT_SETTLING_TIME_S)

        # Define sweep ranges for control plungers
        g1_name, g2_name = control_plungers[0], control_plungers[1]
        g1_voltages = np.linspace(
            plunger_params[g1_name]["transition"],
            plunger_params[g1_name]["cutoff"],
            sweep_resolution,
        )
        g2_voltages = np.linspace(
            plunger_params[g2_name]["transition"],
            plunger_params[g2_name]["cutoff"],
            sweep_resolution,
        )

        logger.info(
            "Starting compensated 2D sweep: %s (%.4fV to %.4fV), %s (%.4fV to %.4fV)",
            g1_name,
            g1_voltages[0],
            g1_voltages[-1],
            g2_name,
            g2_voltages[0],
            g2_voltages[-1],
        )

        # Prepare data storage
        voltage_measurements = []
        current_measurements = []
        compensation_applied = []

        # Store initial sensor plunger voltage
        initial_sensor_voltage = float(
            charge_sensor_park_voltages[charge_sensor_plunger_gate]
        )

        # Manual 2D sweep with compensation
        total_points = sweep_resolution * sweep_resolution
        point_counter = 0

        for i, v_g1 in enumerate(g1_voltages):
            for j, v_g2 in enumerate(g2_voltages):
                point_counter += 1

                # Calculate total compensation if enabled
                total_compensation = 0.0
                if compensation_enabled:
                    # Calculate control voltage changes from initial position
                    control_voltage_changes = {
                        g1_name: float(v_g1 - initial_control_voltages[g1_name]),
                        g2_name: float(v_g2 - initial_control_voltages[g2_name]),
                    }

                    # Calculate total compensation using linear formula
                    for gate_name, delta_v in control_voltage_changes.items():
                        if gate_name in compensation_gradients:  # type: ignore
                            gradient = compensation_gradients[gate_name]  # type: ignore
                            total_compensation += gradient * delta_v

                # Calculate compensated sensor voltage
                compensated_sensor_voltage = initial_sensor_voltage + total_compensation

                # Set control plunger voltages
                device.jump(
                    {g1_name: float(v_g1), g2_name: float(v_g2)},
                    wait_for_settling=False,
                )

                # Set sensor plunger voltage (compensated if enabled, constant if disabled)
                if compensation_enabled or point_counter == 1:
                    # Always set on first point, then only if compensation enabled
                    device.jump(
                        {charge_sensor_plunger_gate: compensated_sensor_voltage},
                        wait_for_settling=True,
                    )
                else:
                    # Just wait for control gates to settle
                    device.jump({}, wait_for_settling=True)

                # Measure current
                current = device.measure(measure_electrode)

                # Store data
                voltage_measurements.append([float(v_g1), float(v_g2)])
                current_measurements.append(float(current))
                compensation_applied.append(float(total_compensation))

                # Progress logging every 10%
                if point_counter % (total_points // 10) == 0 or point_counter == total_points:
                    progress = 100.0 * point_counter / total_points
                    logger.info(
                        "Progress: %.0f%% (%d/%d points)",
                        progress,
                        point_counter,
                        total_points,
                    )

        logger.info("Sweep completed: %d measurements acquired", total_points)
        logger.info(
            "Current range: %.3e to %.3e A",
            min(current_measurements),
            max(current_measurements),
        )
        if compensation_enabled:
            logger.info(
                "Compensation range: %.6f to %.6f V",
                min(compensation_applied),
                max(compensation_applied),
            )
        else:
            logger.info("Compensation: disabled (sensor plunger held constant)")

        # Log analysis results
        if session:
            analysis_data = {
                "control_plunger_gates": control_plungers,
                "sweep_resolution": sweep_resolution,
                "total_measurements": total_points,
                "current_min": float(min(current_measurements)),
                "current_max": float(max(current_measurements)),
                "current_mean": float(np.mean(current_measurements)),
                "compensation_enabled": compensation_enabled,
                "initial_sensor_voltage": initial_sensor_voltage,
                "measure_electrode": measure_electrode,
            }
            if compensation_enabled:
                analysis_data.update(
                    {
                        "compensation_min": float(min(compensation_applied)),
                        "compensation_max": float(max(compensation_applied)),
                        "compensation_mean": float(np.mean(compensation_applied)),
                    }
                )
            session.log_analysis(
                name="charge_sensor_readout_summary",
                data=analysis_data,
            )

        # Return results
        result = {
            "voltage_measurements": voltage_measurements,
            "current_measurements": current_measurements,
            "compensation_applied": compensation_applied,
            "control_plunger_gates": control_plungers,
            "sweep_voltages": {
                g1_name: g1_voltages.tolist(),
                g2_name: g2_voltages.tolist(),
            },
            "sweep_resolution": sweep_resolution,
            "measure_electrode": measure_electrode,
            "initial_sensor_plunger_voltage": initial_sensor_voltage,
            "charge_sensor_park_voltages": charge_sensor_park_voltages,
            "compensation_gradients": compensation_gradients,
            "compensation_enabled": compensation_enabled,
            "global_turn_on_voltage": float(global_turn_on),
        }

        return result

    finally:
        # Restore initial device state
        logger.info("Restoring initial device state")
        device.jump(
            dict(zip(device.control_gates, initial_voltages, strict=False)),
            wait_for_settling=True,
        )
