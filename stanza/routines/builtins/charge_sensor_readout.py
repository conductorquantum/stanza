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

import matplotlib.pyplot as plt

# Third-party imports
import numpy as np

# First-party imports
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.routines import RoutineContext, routine
from stanza.routines.builtins.utils.group_handling import filter_gates_by_group

# Configure logger
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
    """
    Pre-compute compensated sensor voltages for all 2D sweep points.

    Args:
        control_plunger_gates: List of 2 control plunger gate names
        control_plunger_ranges: Dict mapping gate names to (start, end) voltage tuples
        compensation_gradients: Dict mapping gate names to dV_sensor/dV_control gradients
        initial_control_voltages: Initial voltages for control gates
        initial_sensor_voltage: Initial sensor plunger voltage
        charge_sensor_plunger_gate: Name of sensor plunger gate to compensate
        sweep_resolution: Number of points per dimension

    Returns:
        Tuple of (voltages_with_compensation, compensation_applied, gate_electrodes):
        - voltages_with_compensation: List of [G1, G2, sensor] or [G1, G2] voltage arrays
        - compensation_applied: List of compensation deltas applied at each point (V)
        - gate_electrodes: List of gate names in order [G1, G2, sensor] or [G1, G2]
    """
    if len(control_plunger_gates) != 2:
        raise RoutineError(
            f"Expected exactly 2 control plunger gates, got {len(control_plunger_gates)}"
        )

    # Create voltage arrays for each control plunger
    g1_name, g2_name = control_plunger_gates[0], control_plunger_gates[1]
    g1_start, g1_end = control_plunger_ranges[g1_name]
    g2_start, g2_end = control_plunger_ranges[g2_name]

    g1_voltages = np.linspace(g1_start, g1_end, sweep_resolution)
    g2_voltages = np.linspace(g2_start, g2_end, sweep_resolution)

    voltages_with_compensation = []
    compensation_applied = []

    for v_g1 in g1_voltages:
        for v_g2 in g2_voltages:
            # Calculate voltage changes from initial position
            voltage_changes = {
                g1_name: float(v_g1) - initial_control_voltages[g1_name],
                g2_name: float(v_g2) - initial_control_voltages[g2_name],
            }

            # Calculate total compensation
            total_compensation = 0.0
            for gate_name, delta_v in voltage_changes.items():
                if gate_name in compensation_gradients:
                    gradient = compensation_gradients[gate_name]
                    total_compensation += gradient * delta_v

            # Calculate compensated sensor voltage
            compensated_sensor_voltage = initial_sensor_voltage + total_compensation

            # Store [G1, G2, sensor_plunger] for sweep_nd
            voltages_with_compensation.append(
                [float(v_g1), float(v_g2), float(compensated_sensor_voltage)]
            )
            compensation_applied.append(float(total_compensation))

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
    session: LoggerSession | None = None,
    **kwargs: Any,  # pylint: disable=unused-argument
) -> dict[str, Any]:
    """
    Perform charge sensor CSD readout sweep with optional compensation.

    This routine sweeps control group plunger gates while measuring current through
    the charge sensor. The charge sensor plunger voltage can be optionally compensated
    to maintain a constant sensor operating point as control voltages change.

    Physical context:
    - Charge sensor group: Acts as sensor, measures current changes
    - Control group: Gates being swept to modify quantum dot states
    - Compensation (optional): Adjusts sensor plunger to cancel cross-talk from control gates

    Args:
        ctx: Routine context containing device resources
        charge_sensor_group_name: Name of charge sensor group (e.g., "side_A")
        control_group_name: Name of control group being swept (e.g., "side_B")
        sensor_park_point_voltages: Voltages for all charge sensor gates (V).
            Typically obtained from run_compensation routine results.
        charge_sensor_plunger_gate: Name of sensor plunger gate to compensate (e.g., "G3")
        initial_control_voltages: Initial voltages for ALL control group gates (V).
            This includes barriers, reservoirs, and plungers. Plungers will be
            overridden to start of sweep range.
        control_plunger_ranges: Voltage ranges for exactly 2 control plunger gates
            to sweep. Format: {gate_name: (start_V, end_V)}
        measure_electrode: Electrode to measure current from (e.g., "OUT_A")
        bias_gate: Name of the bias gate (contact) to apply bias voltage (e.g., "IN_A_B")
        bias_voltage: Voltage to apply to bias gate during measurements (V)
        compensation_gradients: Optional gradients {gate: dV_sensor/dV_control} from
            run_compensation routine. If None, no compensation is applied (sensor
            plunger held constant). (default: None)
        sweep_resolution: Number of points per dimension for 2D sweep (default: 48)
        num_sweep_repetitions: Number of times to repeat sweep for averaging (default: 10)
        session: Logger session for measurements and analysis
        **kwargs: Additional keyword arguments (for config compatibility)

    Returns:
        dict: Contains:
            - voltage_measurements: List of [G1_voltage, G2_voltage] pairs
            - current_measurements: List of measured differential currents (A)
            - compensation_applied: List of compensation voltages applied to sensor (V)
            - control_plunger_gates: List of control plunger gate names
            - control_plunger_ranges: Dict mapping gate names to (start, end) tuples
            - sweep_resolution: Number of points per dimension
            - measure_electrode: Name of measurement electrode
            - initial_sensor_plunger_voltage: Initial charge sensor plunger voltage (V)
            - sensor_park_point_voltages: Dict of sensor gate voltages used
            - initial_control_voltages: Dict of initial control gate voltages used
            - compensation_gradients: Dict of compensation gradients used (or None)
            - compensation_enabled: Boolean indicating if compensation was applied
            - park_point_current: Baseline current before sweep (A)
            - num_repetitions: Number of sweep repetitions performed

    Raises:
        RoutineError: If validation fails or sweep encounters errors

    Notes:
        - Control plungers swept from start to end of specified ranges
        - If compensation_gradients provided: sensor plunger compensated at each sweep point
        - If compensation_gradients is None: sensor plunger held constant
        - Compensation formula: V_compensated = V_initial + sum(gradient_i * delta_V_i)
        - For shared gates (e.g., reservoirs): initial_control_voltages takes precedence
        - Returns differential current (measured - baseline) for better signal quality

    Example:
        ```python
        # Get compensation data from previous routine
        comp_results = ctx.results.get("run_compensation")
        sensor_park_voltages = comp_results["sensor_park_point_voltages"]
        comp_gradients = comp_results["compensation_gradients"]

        # Run compensated readout
        result = charge_sensor_csd_readout(
            ctx=ctx,
            charge_sensor_group_name="side_A",
            control_group_name="side_B",
            sensor_park_point_voltages=sensor_park_voltages,
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G7": -2.55, "G8": -2.0, "G9": -2.75, "G10": -1.74, "G11": -2.55},
            control_plunger_ranges={"G8": (-2.0, -1.90), "G10": (-1.74, -1.70)},
            measure_electrode="OUT_A",
            bias_gate="IN_A_B",
            bias_voltage=1e-4,
            compensation_gradients=comp_gradients,
            sweep_resolution=48,
        )
        ```
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

    # Determine if compensation is enabled
    compensation_enabled = compensation_gradients is not None

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

    # Get device
    device = ctx.resources.device

    # Get groups from device config and filter by active group
    charge_sensor_group = device.device_config.groups[charge_sensor_group_name]
    control_group = device.device_config.groups[control_group_name]

    charge_sensor_gates = list(charge_sensor_group.gates)
    control_gates = list(control_group.gates)

    charge_sensor_gates = filter_gates_by_group(ctx, charge_sensor_gates)
    control_gates = filter_gates_by_group(ctx, control_gates)

    # Log configuration
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

    # Capture initial device state for cleanup
    initial_voltages = device.check(device.control_gates)

    try:
        # Build voltage dictionary for device setup
        voltage_dict = {}

        # 1. Set all sensor gates from sensor_park_point_voltages
        for gate, voltage in sensor_park_point_voltages.items():
            voltage_dict[gate] = float(voltage)

        # 2. Set all control gates from initial_control_voltages
        # (overwrites shared gates with warning)
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

        # 3. Override control plungers to start of sweep range
        initial_control_plunger_voltages = {}
        for gate, (start, _end) in control_plunger_ranges.items():
            voltage_dict[gate] = float(start)
            initial_control_plunger_voltages[gate] = float(start)

        # Apply initial voltages
        logger.info("Setting initial gate voltages...")
        device.jump(voltage_dict, wait_for_settling=True)
        # Apply bias voltage
        device.jump({bias_gate: bias_voltage}, wait_for_settling=True)
        time.sleep(DEFAULT_SETTLING_TIME_S)

        # Allow settling time
        logger.info(
            "Waiting %d seconds for device settling...", DEFAULT_SETTLING_TIME_S
        )
        time.sleep(DEFAULT_SETTLING_TIME_S)

        # Measure baseline park point current
        park_point_current = device.measure(measure_electrode)
        logger.info("Baseline park point current: %.6e A", park_point_current)

        # Get initial sensor voltage (needed for both modes)
        initial_sensor_voltage = float(
            sensor_park_point_voltages[charge_sensor_plunger_gate]
        )

        # Pre-compute compensated voltages if compensation enabled
        if compensation_enabled:
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
                "Compensation range: %.6f to %.6f V",
                min(compensation_applied),
                max(compensation_applied),
            )
        else:
            # No compensation: sweep only control gates, sensor held constant
            logger.info("Compensation disabled - sensor plunger held constant")
            g1_name, g2_name = control_plunger_gates[0], control_plunger_gates[1]
            g1_voltages = np.linspace(
                *control_plunger_ranges[g1_name], sweep_resolution
            )
            g2_voltages = np.linspace(
                *control_plunger_ranges[g2_name], sweep_resolution
            )

            voltages_list = []
            for v_g1 in g1_voltages:
                for v_g2 in g2_voltages:
                    voltages_list.append([float(v_g1), float(v_g2)])

            voltages_with_compensation = voltages_list
            compensation_applied = [0.0] * len(voltages_list)
            gate_electrodes = control_plunger_gates

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
        for i in range(num_sweep_repetitions):
            logger.info("Starting sweep %d of %d...", i + 1, num_sweep_repetitions)

            # Reset to initial position before each sweep
            reset_dict = {}
            if compensation_enabled:
                reset_dict[charge_sensor_plunger_gate] = initial_sensor_voltage  # type: ignore
            for gate in control_plunger_gates:
                reset_dict[gate] = initial_control_plunger_voltages[gate]

            device.jump(reset_dict, wait_for_settling=True)
            time.sleep(DEFAULT_SETTLING_TIME_S)

            # Perform sweep
            voltage_measurements, current_measurements = device.sweep_nd(
                gate_electrodes=gate_electrodes,
                voltages=voltages_with_compensation,
                measure_electrode=measure_electrode,
                session=None,  # Don't log individual sweeps
            )

            currents_list.append(current_measurements)
            plt.imshow(
                np.array(current_measurements).reshape(
                    sweep_resolution, sweep_resolution
                ),
                cmap="viridis",
                origin="lower",
            )
            plt.colorbar(label="Current (A)")
            plt.xlabel("G10 Voltage (V)")
            plt.ylabel("G8 Voltage (V)")
            plt.title("Compensated Charge Sensor Readout")
            plt.savefig(f"current_measurements_{i}.png")
            plt.close()

            session.log_sweep(
                name="charge_sensor_csd_readout",
                x_data=voltage_measurements,
                y_data=current_measurements,
                x_label=gate_electrodes,
                y_label="current",
                metadata={"repetition": i + 1},
            )

        # Average currents across all sweeps
        average_currents = np.mean(currents_list, axis=0)

        # Subtract baseline to get differential current signal
        differential_currents = average_currents - park_point_current

        logger.info(
            "Sweep completed: %d measurements acquired", len(differential_currents)
        )
        logger.info(
            "Differential current range: %.3e to %.3e A",
            np.min(differential_currents),
            np.max(differential_currents),
        )

        # Extract 2D control gate voltages for logging (exclude sensor compensation dimension)
        voltage_measurements_2d = [[v[0], v[1]] for v in voltage_measurements]

        # Log sweep data
        if session:
            session.log_sweep(
                name="charge_sensor_csd_readout",
                x_data=voltage_measurements_2d,
                y_data=differential_currents.tolist(),
                x_label=control_plunger_gates,
                y_label="differential_current",
                metadata={
                    "compensation_enabled": compensation_enabled,
                    "sensor_plunger": charge_sensor_plunger_gate,
                    "measure_electrode": measure_electrode,
                    "gate_electrodes": gate_electrodes,
                    "num_repetitions": num_sweep_repetitions,
                    "park_point_current": float(park_point_current),
                },
            )

            # Log analysis summary
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
                    "measure_electrode": measure_electrode,
                    "num_repetitions": num_sweep_repetitions,
                    "park_point_current": float(park_point_current),
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

        # Return results
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
                initial_sensor_voltage if compensation_enabled else None  # type: ignore
            ),
            "sensor_park_point_voltages": sensor_park_point_voltages,
            "initial_control_voltages": initial_control_voltages,
            "compensation_gradients": compensation_gradients,
            "compensation_enabled": compensation_enabled,
            "park_point_current": float(park_point_current),
            "num_repetitions": num_sweep_repetitions,
        }

        return result

    finally:
        # Restore initial device state
        logger.info("Restoring initial device state")
        device.jump(
            dict(zip(device.control_gates, initial_voltages, strict=False)),
            wait_for_settling=True,
        )
