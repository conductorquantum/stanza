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
    V_sensor_compensated = V_sensor_initial + sum(gradient_i * delta_V_control_i) - beta * I_error

where:
    - gradient_i = dV_sensor/dV_control_i from the run_compensation routine
    - beta = proportional feedback gain (V/A)
    - I_error = I_measured - I_park_point (current error from baseline)
"""

# Standard library imports
import logging
import time
from collections.abc import Sequence
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

    # Initialize walking state
    current_sensor_voltage = initial_sensor_voltage
    previous_control_voltages = {
        g1_name: initial_control_voltages[g1_name],
        g2_name: initial_control_voltages[g2_name],
    }

    # Use serpentine (boustrophedon) pattern to eliminate voltage jumps at row boundaries
    for row_idx, v_g1 in enumerate(g1_voltages):
        # Alternate sweep direction each row
        if row_idx % 2 == 0:
            # Even rows: sweep left-to-right
            row_voltages = g2_voltages
        else:
            # Odd rows: sweep right-to-left (reverse)
            row_voltages = g2_voltages[::-1]

        for v_g2 in row_voltages:
            # Calculate voltage changes from previous point (walking state)
            delta_v1 = float(v_g1) - previous_control_voltages[g1_name]
            delta_v2 = float(v_g2) - previous_control_voltages[g2_name]

            # Calculate compensation update
            compensation_update = 0.0
            if g1_name in compensation_gradients:
                compensation_update += compensation_gradients[g1_name] * delta_v1
            if g2_name in compensation_gradients:
                compensation_update += compensation_gradients[g2_name] * delta_v2

            # Update sensor voltage (walking state)
            current_sensor_voltage += compensation_update

            # Store [G1, G2, sensor_plunger] for sweep_nd
            voltages_with_compensation.append(
                [float(v_g1), float(v_g2), float(current_sensor_voltage)]
            )

            # Track cumulative compensation from initial point (for logging)
            cumulative_compensation = current_sensor_voltage - initial_sensor_voltage
            compensation_applied.append(float(cumulative_compensation))

            # Update previous voltages
            previous_control_voltages[g1_name] = float(v_g1)
            previous_control_voltages[g2_name] = float(v_g2)

    gate_electrodes = [g1_name, g2_name, charge_sensor_plunger_gate]
    return voltages_with_compensation, compensation_applied, gate_electrodes


def _reshape_serpentine_grid(
    values: Sequence[float], sweep_resolution: int
) -> np.ndarray:
    """
    Convert a 1D list of serpentine-sweep values into a 2D grid where each row
    runs left-to-right in the same voltage order.

    Args:
        values: Flat measurement array collected in serpentine order
        sweep_resolution: Number of points per sweep dimension
    """
    grid = np.array(values, dtype=float).reshape(sweep_resolution, sweep_resolution)
    # Odd rows were acquired right-to-left; flip them to align increasing axis order.
    grid[1::2] = grid[1::2, ::-1]
    return grid


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
        beta: Optional proportional feedback gain (V/A) for current-based sensor voltage
            correction. If None, no feedback is applied. When provided, sensor plunger
            voltage is adjusted at each measurement point to maintain current near park
            point: delta_V = -beta * (I_measured - I_park_point). (default: None)
        max_feedback_correction: Safety limit on per-point feedback correction magnitude
            (V). Prevents runaway adjustments. Only used when beta is not None. (default: 0.1)
        gamma_factors: Optional per-gate adaptation gains {gate: gamma} for dynamic gradient
            updates (default: None). When provided, compensation gradients are updated at each
            measurement point using: A_C[x+1] = A_C[x] + (gamma/ΔV) * i_S[x], where i_S is the
            current error (I_measured - I_park_point) and ΔV is the voltage step change. Units:
            [V/A] (related to inverse transconductance × learning rate). Requires compensation_gradients
            to be provided (cannot adapt from zero). Default 0.0 disables adaptation. Must include
            all control plunger gates. Gradients only updated when |ΔV| > 1e-9 V to avoid division
            by zero.
        max_adaptive_gradient: Maximum allowed magnitude for adaptive gradients (V/V). Adaptive
            gradients are clipped to [-max, +max] after each update to prevent unbounded growth
            that could drive sensor voltage out of hardware limits. Only applies when gamma_factors
            is enabled. (default: 2.0)
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
            - beta: Proportional feedback gain used (V/A) or None
            - feedback_enabled: Boolean indicating if current feedback was applied
            - feedback_corrections: List of feedback corrections applied at each point (V)
            - max_feedback_correction: Maximum allowed feedback correction (V)
            - gamma_factors: Dict of gamma adaptation gains used (or None)
            - gradient_adaptation_enabled: Boolean indicating if gradient adaptation was used
            - initial_gradients: Dict of initial gradient values before adaptation (or None)
            - final_gradients: Dict of final adapted gradient values (or None)
            - gradient_history: List of dicts tracking all gradient updates with fields:
                point_index, repetition, gate, delta_v, current_error_pre_feedback,
                gradient_update, new_gradient (or None if adaptation disabled). Note:
                gradient updates use pre-feedback current error to learn compensation
                independent of beta feedback corrections.

    Raises:
        RoutineError: If validation fails or sweep encounters errors

    Notes:
        - Control plungers swept from start to end of specified ranges
        - If compensation_gradients provided: sensor plunger compensated at each sweep point
        - If compensation_gradients is None: sensor plunger held constant
        - If beta provided: additional per-point feedback applied based on current error
        - If gamma_factors provided: compensation gradients adapt during sweep

        Compensation and feedback formulas:
        - Sensor voltage update: V_PS[x+1] = V_PS[x] + ΔV_1·A_C1[x] + ΔV_2·A_C2[x] - β·i_S[x]
        - Gradient adaptation: A_C[x+1] = A_C[x] + (γ/ΔV)·i_S_pre[x]
        where i_S_pre = I_measured - I_park_point (pre-feedback current error)

        - Gradient updates use PRE-feedback error to learn compensation independently
        - Beta feedback is applied after gradient update, then current is re-measured
        - Gradient updates only occur when |ΔV| > 1e-9 V (avoid division by zero in raster scan)
        - Adaptive gradients persist across all repetitions (continuous learning)
        - For shared gates (e.g., reservoirs): initial_control_voltages takes precedence
        - Returns differential current (measured - baseline) for better signal quality
        - With adaptation: gradients evolve throughout all sweeps to minimize current errors

    Example:
        ```python
        # Get compensation data from previous routine
        comp_results = ctx.results.get("run_compensation")
        sensor_park_voltages = comp_results["sensor_park_point_voltages"]
        comp_gradients = comp_results["compensation_gradients"]

        # Run compensated readout (static gradients)
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

        # Run with adaptive gradients (gamma_factors enables real-time gradient updates)
        result_adaptive = charge_sensor_csd_readout(
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
            gamma_factors={"G8": 1e-6, "G10": 1e-6},  # Enable adaptation with learning rate
            beta=-1e5,  # Optional: add proportional feedback
            sweep_resolution=48,
        )

        # Access adapted gradients
        print(f"Initial gradients: {result_adaptive['initial_gradients']}")
        print(f"Final gradients: {result_adaptive['final_gradients']}")
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
    logger.info("  Gradient adaptation enabled: %s", gradient_adaptation_enabled)
    if gradient_adaptation_enabled:
        logger.info("  Gamma factors: %s", gamma_factors)
        logger.info("  Max adaptive gradient: %.6f V/V", max_adaptive_gradient)
    logger.info("  Current feedback enabled: %s", feedback_enabled)
    if feedback_enabled:
        logger.info("  Beta (feedback gain): %.6e V/A", beta)
        logger.info("  Max feedback correction: %.6f V", max_feedback_correction)

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

        # Initialize gradient adaptation tracking if enabled
        if gradient_adaptation_enabled:
            # Copy gradients for adaptation (will be modified during sweep)
            # NOTE: adaptive_gradients persist across ALL repetitions - this is intentional
            # for dynamic charge sensing where gradients evolve continuously throughout
            # the entire measurement to minimize current errors
            adaptive_gradients = compensation_gradients.copy()  # type: ignore
            gradient_history = []
            # Epsilon threshold to avoid division by near-zero voltage changes
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

        # Pre-compute compensated voltages if compensation enabled (non-adaptive only)
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
            # No compensation: sweep only control gates, sensor held constant
            logger.info(
                "Compensation disabled - sensor plunger held constant (serpentine scan pattern)"
            )
            g1_name, g2_name = control_plunger_gates[0], control_plunger_gates[1]
            g1_voltages = np.linspace(
                *control_plunger_ranges[g1_name], sweep_resolution
            )
            g2_voltages = np.linspace(
                *control_plunger_ranges[g2_name], sweep_resolution
            )

            # Use serpentine pattern
            voltages_list = []
            for row_idx, v_g1 in enumerate(g1_voltages):
                # Alternate sweep direction each row
                row_voltages = g2_voltages if row_idx % 2 == 0 else g2_voltages[::-1]
                for v_g2 in row_voltages:
                    voltages_list.append([float(v_g1), float(v_g2)])

            voltages_with_compensation = voltages_list
            compensation_applied = [0.0] * len(voltages_list)
            gate_electrodes = control_plunger_gates
        else:
            # Adaptive gradient mode: prepare control voltage grid only
            # Compensation will be computed dynamically during sweep
            logger.info(
                "Adaptive gradient mode - compensation computed dynamically (serpentine scan pattern)"
            )
            g1_name, g2_name = control_plunger_gates[0], control_plunger_gates[1]
            g1_voltages = np.linspace(
                *control_plunger_ranges[g1_name], sweep_resolution
            )
            g2_voltages = np.linspace(
                *control_plunger_ranges[g2_name], sweep_resolution
            )

            # Create control voltage grid with serpentine pattern (no pre-computed compensation)
            control_voltages_grid = []
            for row_idx, v_g1 in enumerate(g1_voltages):
                # Alternate sweep direction each row
                row_voltages = g2_voltages if row_idx % 2 == 0 else g2_voltages[::-1]
                for v_g2 in row_voltages:
                    control_voltages_grid.append([float(v_g1), float(v_g2)])

            voltages_with_compensation = control_voltages_grid
            compensation_applied = []  # Will be populated during sweep
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

            # Perform sweep with per-point feedback if enabled
            voltage_measurements_rep = []
            current_measurements_rep = []
            feedback_corrections_rep = []

            # Initialize tracking for adaptive gradients
            if gradient_adaptation_enabled:
                # Track previous control voltages for delta calculation
                previous_control_voltages = {
                    g: initial_control_plunger_voltages[g]
                    for g in control_plunger_gates
                }
                # Track sensor voltage state
                current_sensor_voltage = initial_sensor_voltage
                # Counters for gradient updates
                gradient_update_count = dict.fromkeys(control_plunger_gates, 0)

            for point_idx, voltage_point in enumerate(voltages_with_compensation):
                if gradient_adaptation_enabled:
                    # Extract control voltages from voltage_point
                    control_v1, control_v2 = voltage_point[0], voltage_point[1]
                    g1_name, g2_name = (
                        control_plunger_gates[0],
                        control_plunger_gates[1],
                    )

                    # Calculate voltage changes from previous point
                    delta_v1 = control_v1 - previous_control_voltages[g1_name]
                    delta_v2 = control_v2 - previous_control_voltages[g2_name]

                    # Calculate compensation update based on control voltage changes
                    compensation_update = 0.0
                    compensation_update += adaptive_gradients[g1_name] * delta_v1  # type: ignore
                    compensation_update += adaptive_gradients[g2_name] * delta_v2  # type: ignore

                    # Update sensor voltage: V_PS[x+1] = V_PS[x] + ΔV_1·A_C1[x] + ΔV_2·A_C2[x]
                    current_sensor_voltage += compensation_update

                    # Build voltage dict with computed sensor voltage
                    voltage_dict = {
                        g1_name: control_v1,
                        g2_name: control_v2,
                        charge_sensor_plunger_gate: current_sensor_voltage,
                    }

                    # Track cumulative compensation from initial sensor voltage
                    cumulative_compensation = (
                        current_sensor_voltage - initial_sensor_voltage
                    )
                    compensation_applied.append(float(cumulative_compensation))

                    # Update previous control voltages
                    previous_control_voltages[g1_name] = control_v1
                    previous_control_voltages[g2_name] = control_v2
                else:
                    # Non-adaptive mode: use pre-computed voltages
                    voltage_dict = dict(
                        zip(gate_electrodes, voltage_point, strict=False)
                    )

                # Validate and clip sensor voltage if needed
                if charge_sensor_plunger_gate in voltage_dict:
                    sensor_voltage = voltage_dict[charge_sensor_plunger_gate]

                    # Get actual voltage range from device gate configuration
                    min_voltage, max_voltage = device.channel_configs[
                        charge_sensor_plunger_gate
                    ].voltage_range

                    # Handle None values in voltage_range (use conservative hardware limits)
                    if min_voltage is None:
                        min_voltage = -10.0
                        logger.warning(
                            "No minimum voltage limit configured for %s, using -10V",
                            charge_sensor_plunger_gate,
                        )
                    if max_voltage is None:
                        max_voltage = 10.0
                        logger.warning(
                            "No maximum voltage limit configured for %s, using +10V",
                            charge_sensor_plunger_gate,
                        )

                    # Check and clip if out of bounds
                    if sensor_voltage < min_voltage or sensor_voltage > max_voltage:
                        original_voltage = sensor_voltage
                        sensor_voltage = float(
                            np.clip(sensor_voltage, min_voltage, max_voltage)
                        )
                        voltage_dict[charge_sensor_plunger_gate] = sensor_voltage
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

                # Apply voltages
                device.jump(voltage_dict, wait_for_settling=True)

                # Measure current
                current = device.measure(measure_electrode)
                current_error_pre = current - park_point_current  # Pre-feedback error

                # Apply proportional feedback if enabled
                feedback_correction = 0.0
                if feedback_enabled:
                    feedback_correction = -beta * current_error_pre  # type: ignore

                    # Apply safety clamp
                    feedback_correction = float(
                        np.clip(
                            feedback_correction,
                            -max_feedback_correction,
                            max_feedback_correction,
                        )
                    )

                    # Adjust sensor voltage and re-apply
                    voltage_dict[charge_sensor_plunger_gate] += feedback_correction
                    if gradient_adaptation_enabled:
                        current_sensor_voltage += (
                            feedback_correction  # Track updated state
                        )

                    # Validate and clip sensor voltage after feedback correction
                    sensor_voltage_fb = voltage_dict[charge_sensor_plunger_gate]

                    # Get actual voltage range from device gate configuration
                    min_voltage, max_voltage = device.channel_configs[
                        charge_sensor_plunger_gate
                    ].voltage_range

                    # Handle None values in voltage_range (use conservative hardware limits)
                    if min_voltage is None:
                        min_voltage = -10.0
                    if max_voltage is None:
                        max_voltage = 10.0

                    if (
                        sensor_voltage_fb < min_voltage
                        or sensor_voltage_fb > max_voltage
                    ):
                        original_voltage_fb = sensor_voltage_fb
                        sensor_voltage_fb = float(
                            np.clip(sensor_voltage_fb, min_voltage, max_voltage)
                        )
                        voltage_dict[charge_sensor_plunger_gate] = sensor_voltage_fb
                        # Also update tracked state if in adaptive mode
                        if gradient_adaptation_enabled:
                            current_sensor_voltage = sensor_voltage_fb
                        logger.warning(
                            "Sensor voltage for %s after feedback out of bounds: %.6fV clipped to %.6fV (valid range: %.1f to %.1fV). "
                            "Point %d, Rep %d. Consider reducing beta or max_feedback_correction.",
                            charge_sensor_plunger_gate,
                            original_voltage_fb,
                            sensor_voltage_fb,
                            min_voltage,
                            max_voltage,
                            point_idx,
                            i,
                        )

                    device.jump(voltage_dict, wait_for_settling=True)

                    # Re-measure current after feedback adjustment
                    current = device.measure(measure_electrode)
                    # Post-feedback error (for logging/diagnostics only)

                # Update adaptive gradients based on PRE-feedback current error
                if gradient_adaptation_enabled:
                    # Update gradients only when respective gate voltage changed
                    if abs(delta_v1) > delta_v_threshold:  # type: ignore
                        gradient_update = (
                            gamma_factors[g1_name] / delta_v1
                        ) * current_error_pre  # type: ignore

                        # Store old gradient before update for logging
                        old_gradient = adaptive_gradients[g1_name]  # type: ignore
                        adaptive_gradients[g1_name] += gradient_update  # type: ignore

                        # Clip adaptive gradient to prevent unbounded growth
                        adaptive_gradients[g1_name] = float(
                            np.clip(
                                adaptive_gradients[g1_name],  # type: ignore
                                -max_adaptive_gradient,
                                max_adaptive_gradient,
                            )
                        )  # type: ignore

                        # Log if clipping occurred
                        if abs(old_gradient + gradient_update) > max_adaptive_gradient:
                            logger.warning(
                                "Adaptive gradient for %s clipped: %.6f -> %.6f (limit: ±%.6f V/V)",
                                g1_name,
                                old_gradient + gradient_update,
                                adaptive_gradients[g1_name],  # type: ignore
                                max_adaptive_gradient,
                            )

                        gradient_update_count[g1_name] += 1
                        # Log to gradient history
                        gradient_history.append(  # type: ignore
                            {
                                "point_index": point_idx,
                                "repetition": i,
                                "gate": g1_name,
                                "delta_v": float(delta_v1),
                                "current_error_pre_feedback": float(current_error_pre),
                                "gradient_update": float(gradient_update),
                                "new_gradient": float(adaptive_gradients[g1_name]),  # type: ignore
                            }
                        )

                    if abs(delta_v2) > delta_v_threshold:  # type: ignore
                        gradient_update = (
                            gamma_factors[g2_name] / delta_v2
                        ) * current_error_pre  # type: ignore

                        # Store old gradient before update for logging
                        old_gradient = adaptive_gradients[g2_name]  # type: ignore
                        adaptive_gradients[g2_name] += gradient_update  # type: ignore

                        # Clip adaptive gradient to prevent unbounded growth
                        adaptive_gradients[g2_name] = float(
                            np.clip(
                                adaptive_gradients[g2_name],  # type: ignore
                                -max_adaptive_gradient,
                                max_adaptive_gradient,
                            )
                        )  # type: ignore

                        # Log if clipping occurred
                        if abs(old_gradient + gradient_update) > max_adaptive_gradient:
                            logger.warning(
                                "Adaptive gradient for %s clipped: %.6f -> %.6f (limit: ±%.6f V/V)",
                                g2_name,
                                old_gradient + gradient_update,
                                adaptive_gradients[g2_name],  # type: ignore
                                max_adaptive_gradient,
                            )

                        gradient_update_count[g2_name] += 1
                        # Log to gradient history
                        gradient_history.append(  # type: ignore
                            {
                                "point_index": point_idx,
                                "repetition": i,
                                "gate": g2_name,
                                "delta_v": float(delta_v2),
                                "current_error_pre_feedback": float(current_error_pre),
                                "gradient_update": float(gradient_update),
                                "new_gradient": float(adaptive_gradients[g2_name]),  # type: ignore
                            }
                        )

                # Record actual voltages (control gates only) and current
                actual_voltages = [voltage_dict[g] for g in control_plunger_gates]
                voltage_measurements_rep.append(actual_voltages)
                current_measurements_rep.append(current)
                feedback_corrections_rep.append(feedback_correction)

            # Log gradient updates for this repetition
            if gradient_adaptation_enabled:
                logger.info(
                    "Repetition %d gradient updates: %s",
                    i + 1,
                    gradient_update_count,
                )
                logger.info(
                    "Repetition %d final gradients: %s", i + 1, adaptive_gradients
                )

            voltage_measurements = voltage_measurements_rep
            current_measurements = current_measurements_rep
            currents_list.append(current_measurements)
            feedback_corrections_list.append(feedback_corrections_rep)
            current_grid = _reshape_serpentine_grid(
                current_measurements, sweep_resolution
            )
            g1_name, g2_name = control_plunger_gates
            g1_start, g1_end = control_plunger_ranges[g1_name]
            g2_start, g2_end = control_plunger_ranges[g2_name]
            plt.imshow(
                current_grid,
                cmap="viridis",
                origin="lower",
                extent=[g2_start, g2_end, g1_start, g1_end],
                aspect="auto",
            )
            plt.colorbar(label="Current (A)")
            plt.xlabel(f"{g2_name} Voltage (V)")
            plt.ylabel(f"{g1_name} Voltage (V)")
            plt.title("Compensated Charge Sensor Readout")
            plt.savefig(f"current_measurements_{i}.png")
            plt.close()

            session.log_sweep(
                name="charge_sensor_csd_readout",
                x_data=voltage_measurements,
                y_data=current_measurements,
                x_label=control_plunger_gates,
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

        logger.info(
            "Sweep completed: %d measurements acquired", len(differential_currents)
        )
        logger.info(
            "Differential current range: %.3e to %.3e A",
            np.min(differential_currents),
            np.max(differential_currents),
        )
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

        # Log gradient adaptation summary
        if gradient_adaptation_enabled:
            logger.info("Gradient adaptation summary:")
            logger.info("  Initial gradients: %s", compensation_gradients)
            logger.info("  Final gradients: %s", adaptive_gradients)
            for gate in control_plunger_gates:
                initial_grad = compensation_gradients[gate]  # type: ignore
                final_grad = adaptive_gradients[gate]  # type: ignore
                change = final_grad - initial_grad
                percent_change = (
                    (change / initial_grad * 100) if initial_grad != 0 else 0
                )
                logger.info(
                    "  %s: %.6f -> %.6f (change: %.6f, %.2f%%)",
                    gate,
                    initial_grad,
                    final_grad,
                    change,
                    percent_change,
                )
            logger.info("  Total gradient updates: %d", len(gradient_history))  # type: ignore

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
                    "feedback_enabled": feedback_enabled,
                    "beta": float(beta) if beta is not None else None,
                    "max_feedback_correction": float(max_feedback_correction),
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
                    "feedback_enabled": feedback_enabled,
                    "beta": float(beta) if beta is not None else None,
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

            if feedback_enabled:
                session.log_analysis(
                    name="charge_sensor_csd_feedback_summary",
                    data={
                        "feedback_min": float(np.min(average_feedback_corrections)),
                        "feedback_max": float(np.max(average_feedback_corrections)),
                        "feedback_mean": float(np.mean(average_feedback_corrections)),
                        "feedback_std": float(np.std(average_feedback_corrections)),
                        "beta": float(beta),  # type: ignore
                        "max_feedback_correction": float(max_feedback_correction),
                    },
                )

            if gradient_adaptation_enabled:
                # Log gradient adaptation data
                session.log_analysis(
                    name="charge_sensor_csd_gradient_adaptation_summary",
                    data={
                        "initial_gradients": {
                            k: float(v)
                            for k, v in compensation_gradients.items()  # type: ignore
                        },
                        "final_gradients": {
                            k: float(v)
                            for k, v in adaptive_gradients.items()  # type: ignore
                        },
                        "gradient_changes": {
                            gate: float(
                                adaptive_gradients[gate] - compensation_gradients[gate]
                            )  # type: ignore
                            for gate in control_plunger_gates
                        },
                        "gamma_factors": {
                            k: float(v) for k, v in gamma_factors.items()
                        },  # type: ignore
                        "total_updates": len(gradient_history),  # type: ignore
                    },
                )

                # Log full gradient history
                session.log_analysis(
                    name="charge_sensor_csd_gradient_history",
                    data={"gradient_history": gradient_history},  # type: ignore
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
            "beta": beta,
            "feedback_enabled": feedback_enabled,
            "feedback_corrections": average_feedback_corrections.tolist(),
            "max_feedback_correction": max_feedback_correction,
            # Gradient adaptation fields
            "gamma_factors": gamma_factors,
            "gradient_adaptation_enabled": gradient_adaptation_enabled,
            "initial_gradients": compensation_gradients
            if gradient_adaptation_enabled
            else None,
            "final_gradients": adaptive_gradients
            if gradient_adaptation_enabled
            else None,
            "gradient_history": gradient_history
            if gradient_adaptation_enabled
            else None,
        }

        return result

    finally:
        # Restore initial device state
        logger.info("Restoring initial device state")
        device.jump(
            dict(zip(device.control_gates, initial_voltages, strict=False)),
            wait_for_settling=True,
        )
