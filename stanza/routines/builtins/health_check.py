"""Device health check routines for quantum dot devices operating in accumulation mode.

These routines are specifically designed for accumulation mode devices, where gates are
swept from depletion to accumulation to establish conductive channels. The algorithms
identify turn-on voltages where current begins to flow as gates transition from depleted
to accumulated states.

Voltage Bounds and Carrier Type Conventions
--------------------------------------------
The safe voltage bounds (max_safe_voltage_bound and min_safe_voltage_bound) represent the
extreme voltages that can be safely applied to gates without causing device damage or
unwanted leakage. The interpretation of these bounds depends on the charge carrier type:

For electron-based devices (n-channel, charge_carrier_type="electron"):
    - Accumulation occurs at POSITIVE gate voltages, which attract electrons to form
      a conductive channel
    - max_safe_voltage_bound is POSITIVE and represents the maximum safe positive voltage
    - min_safe_voltage_bound is NEGATIVE and represents the most negative voltage (depletion)
    - Current INCREASES as gate voltage sweeps from negative (depleted) to positive (accumulated)

For hole-based devices (p-channel, charge_carrier_type="hole"):
    - Accumulation occurs at NEGATIVE gate voltages, which attract holes to form
      a conductive channel
    - max_safe_voltage_bound is POSITIVE and represents the most positive voltage (depletion)
    - min_safe_voltage_bound is NEGATIVE and represents the maximum safe negative voltage
    - Current INCREASES as gate voltage sweeps from positive (depleted) to negative (accumulated)

In both cases, the routines sweep from depletion (low current) toward accumulation (high current),
but the voltage polarity is reversed between electron and hole devices. The charge_carrier_type
parameter determines which voltage bound is used for accumulation sweeps.

Health check tests are an improvement of:
    Kovach, T. et al. BATIS: Bootstrapping, Autonomous Testing, and
    Initialization System for Si/SiGe Multi-quantum Dot Devices. arXiv
    preprint arXiv:2412.07676 (2024). https://arxiv.org/abs/2412.07676
"""

import logging
import time
from typing import Any

import numpy as np

from stanza.analysis.criterion import fit_quality_criterion
from stanza.analysis.fitting import fit_pinchoff_parameters
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.models import GateType
from stanza.routines import RoutineContext, routine

logger = logging.getLogger(__name__)

DEFAULT_SETTLING_TIME_S = 10


@routine
def noise_floor_measurement(
    ctx: RoutineContext,
    measure_electrode: str,
    num_points: int = 100,
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Measure the noise floor of the device to establish baseline current measurement statistics.

    This routine performs repeated current measurements on a specified electrode to characterize
    the measurement noise floor. The resulting mean and standard deviation are used to establish
    minimum current thresholds for subsequent health check routines, helping to distinguish
    real signals from measurement noise.

    Args:
        ctx: Routine context containing device resources and previous results.
        measure_electrode: Name of the electrode to measure current from.
        num_points: Number of current measurements to collect for statistical analysis. Default is 100.
        session: Optional logger session for recording measurements and analysis results.

    Returns:
        dict: Contains:
            - current_mean: Mean of measured currents (A)
            - current_std: Standard deviation of measured currents (A)

    Notes:
        - All measurements are taken at the device's current voltage configuration
        - Results are automatically logged to session if provided
        - The current_std value is commonly used in leakage_test as min_current_threshold
    """
    currents = []
    ctx.resources.device.jump(
        dict.fromkeys(
            ctx.resources.device.control_gates + ctx.resources.device.control_contacts,
            0.0,
        )
    )
    for _ in range(num_points):
        current = ctx.resources.device.measure(measure_electrode)
        currents.append(current)
    current_mean, current_std = np.mean(currents), np.std(currents)
    if session:
        session.log_analysis(
            "noise_floor_measurement",
            {
                "measure_electrode": measure_electrode,
                "currents": currents,
                "current_mean": current_mean,
                "current_std": current_std,
            },
        )
    return {
        "current_mean": current_mean,
        "current_std": current_std,
    }


@routine
def leakage_test(
    ctx: RoutineContext,
    leakage_threshold_resistance: int,
    leakage_threshold_count: int = 0,
    num_points: int = 10,
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Test for leakage between control gates by measuring inter-gate resistance across voltage ranges.

    This routine systematically tests for electrical leakage between all pairs of control gates
    by sweeping each gate through its voltage range and measuring the current response on all
    other gates. Leakage is quantified as resistance (dV/dI) between gate pairs. Gates with
    resistance below the threshold indicate potential shorts or unwanted crosstalk.

    The test sweeps both positive (max_voltage_bound) and negative (min_voltage_bound) directions
    from the initial voltages to detect asymmetric leakage behavior. Both bounds are tested
    independently, so failure at one bound does not prevent testing the other.

    Note on voltage bounds: max_safe_voltage_bound is typically positive and min_safe_voltage_bound
    is typically negative, representing the extreme safe operating voltages. For electron devices,
    max (positive) is used for accumulation sweeps; for hole devices, min (negative) is used for
    accumulation sweeps. See module-level docstring for detailed explanation.

    Args:
        ctx: Routine context containing device resources. Uses ctx.results.get("current_std")
             if available from noise_floor_measurement, otherwise defaults to 1e-10 A.
        leakage_threshold_resistance: Maximum acceptable resistance (Ohms) between gate pairs.
                                     Lower values indicate more stringent leakage requirements.
        leakage_threshold_count: Maximum number of leaky gate pairs allowed before failing the test.
                                Default is 0 (any leakage causes failure).
        num_points: Number of voltage steps to test in each direction. Default is 10.
        session: Optional logger session for recording leakage matrices and analysis results.

    Returns:
        dict: Contains the delta_V tested for each voltage bound:
            - max_voltage_bound: The voltage offset tested for max bound (V)
            - min_voltage_bound: The voltage offset tested for min bound (V)
            If leakage exceeds threshold, returns the offset where it was first detected.
            If no leakage detected, returns the final offset tested.

    Raises:
        Exception: Re-raises any exceptions encountered during testing after logging.

    Notes:
        - Device is always returned to initial voltages in the finally block
        - Leakage matrix is symmetric, only upper triangle is checked
        - Both voltage bounds are tested regardless of failures
        - Current differences below min_current_threshold are skipped to avoid noise
        - Voltage bounds are determined from device channel configurations
    """

    leakage_threshold_resistance = int(leakage_threshold_resistance)
    leakage_threshold_count = int(leakage_threshold_count)
    num_points = int(num_points)

    control_gates = ctx.resources.device.control_gates
    control_gate_configs = {
        gate: ctx.resources.device.channel_configs[gate] for gate in control_gates
    }

    max_voltage_bound = min(
        control_gate_configs.values(), key=lambda x: x.voltage_range[1]
    ).voltage_range[1]
    min_voltage_bound = max(
        control_gate_configs.values(), key=lambda x: x.voltage_range[0]
    ).voltage_range[0]

    noise_floor_measurement_results = ctx.results.get("noise_floor_measurement", {})
    min_current_threshold = noise_floor_measurement_results.get("current_std", 1e-10)
    leakage_test_results = {}

    try:
        initial_currents = ctx.resources.device.measure(control_gates)
        initial_voltages = ctx.resources.device.check(control_gates)

        # Test max voltage bound
        max_delta_v, max_leaked = _test_single_voltage_bound(
            ctx,
            max_voltage_bound,
            control_gates,
            initial_voltages,
            initial_currents,
            leakage_threshold_resistance,
            leakage_threshold_count,
            min_current_threshold,
            num_points,
            session,
        )
        leakage_test_results["max_safe_voltage_bound"] = max_delta_v

        # Test min voltage bound
        min_delta_v, min_leaked = _test_single_voltage_bound(
            ctx,
            min_voltage_bound,
            control_gates,
            initial_voltages,
            initial_currents,
            leakage_threshold_resistance,
            leakage_threshold_count,
            min_current_threshold,
            num_points,
            session,
        )
        leakage_test_results["min_safe_voltage_bound"] = min_delta_v

        # Log success only if neither bound leaked
        if not max_leaked and not min_leaked:
            if session:
                session.log_analysis(
                    "leakage_test_success",
                    {
                        "gates": control_gates,
                        "max_safe_voltage_bound": max_delta_v,
                        "min_safe_voltage_bound": min_delta_v,
                        "leaky_gate_pairs": [],
                        "num_leaky_connections": 0,
                    },
                )

        return leakage_test_results

    except Exception as e:
        logger.error("Error in leakage test: %s", e)
        raise e

    finally:
        ctx.resources.device.jump(
            dict(zip(control_gates, initial_voltages, strict=False)),
            wait_for_settling=True,
        )


@routine
def global_accumulation(
    ctx: RoutineContext,
    measure_electrode: str,
    step_size: float,
    bias_gate: str,
    bias_voltage: float,
    charge_carrier_type: str,
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Determine the global turn-on voltage by sweeping all control gates simultaneously.

    This health check routine is specifically designed for accumulation mode devices.
    It sweeps all control gates together from minimum to maximum voltage while measuring
    current at a specified electrode. The sweep data is analyzed using a pinch-off heuristic
    to identify the voltage at which the device transitions from depletion to accumulation
    (turn-on). After finding this voltage, all gates are set to the cut-off voltage.

    This global characterization establishes a baseline operating point before individual
    gate characterization in subsequent routines.

    Args:
        ctx: Routine context containing device resources and previous results. Requires
             ctx.results["max_voltage_bound"] and ctx.results["min_voltage_bound"] from
             prior routines (typically leakage_test).
        measure_electrode: Name of the electrode to measure current from during the sweep.
        step_size: Voltage increment (V) between sweep points. Smaller values provide
                  higher resolution but increase measurement time.
        charge_carrier_type: The mobile charge particle type. Must be "electron" or "hole".
                           For electrons: sweeps from 0V toward max_safe_voltage_bound (positive).
                           For holes: sweeps from 0V toward min_safe_voltage_bound (negative).
        session: Optional logger session for recording sweep measurements and analysis results.

    Returns:
        dict: Contains:
            - global_turn_on_voltage: The cut-off voltage (cutoff_voltage) where current begins to flow through the device (V)

    Notes:
        - All control gates are swept together (global sweep)
        - Device is automatically set to cutoff_voltage after analysis
        - The cutoff_voltage value is used by subsequent characterization routines (reservoir, finger gates)
        - step_size is converted to num_points based on voltage range
        - For electron devices: sweeps toward positive voltages (accumulation at positive V)
        - For hole devices: sweeps toward negative voltages (accumulation at negative V)
    """
    charge_carrier_type = charge_carrier_type.lower()
    if charge_carrier_type not in ["electron", "hole"]:
        raise RoutineError("Charge carrier type is required for global accumulation")

    if step_size <= 0:
        raise RoutineError("Step size must be greater than 0")

    leakage_test_results = ctx.results.get("leakage_test", {})
    voltage_bound = leakage_test_results[
        "max_safe_voltage_bound"
        if charge_carrier_type == "electron"
        else "min_safe_voltage_bound"
    ]

    ctx.resources.device.jump({bias_gate: bias_voltage}, wait_for_settling=True)

    num_points = max(2, int(voltage_bound / step_size))
    sweep_voltages = np.linspace(0, voltage_bound, num_points)
    _, currents = ctx.resources.device.sweep_all(
        voltages=sweep_voltages,
        measure_electrode=measure_electrode,
        session=session,
    )
    try:
        turn_on_analysis = analyze_single_gate_heuristic(
            sweep_voltages, np.array(currents)
        )
    except Exception as e:
        raise RoutineError(f"Error in global_accumulation: {str(e)}") from e

    ctx.resources.device.jump(
        dict.fromkeys(
            ctx.resources.device.control_gates, turn_on_analysis["cutoff_voltage"]
        ),
        wait_for_settling=True,
    )

    if session:
        session.log_analysis(
            name="global_turn_on_voltage",
            data=turn_on_analysis,
        )

    return {
        "global_turn_on_voltage": turn_on_analysis["cutoff_voltage"],
    }


@routine
def reservoir_characterization(
    ctx: RoutineContext,
    measure_electrode: str,
    step_size: float,
    bias_gate: str,
    bias_voltage: float,
    charge_carrier_type: str = "electron",
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Characterize individual reservoir gates by sweeping each while holding others in accumulation.

    This health check routine is specifically designed for accumulation mode devices.
    It determines the cut-off voltage (cutoff_voltage) for each reservoir gate individually.
    For each reservoir under test, all other reservoirs are set to 120% of the global turn-on
    voltage (to ensure they're fully conducting), while the target reservoir is swept from
    minimum to maximum voltage. This isolates the behavior of each reservoir and identifies its
    individual pinch-off characteristics.

    Args:
        ctx: Routine context containing device resources and previous results. Requires:
             - ctx.results["max_safe_voltage_bound"]: Maximum voltage for sweeps
             - ctx.results["min_safe_voltage_bound"]: Minimum voltage for sweeps
             - ctx.results["global_turn_on_voltage"]: From global_accumulation routine
        measure_electrode: Name of the electrode to measure current from during sweeps.
        step_size: Voltage increment (V) between sweep points. Smaller values provide
                  higher resolution but increase measurement time.
        charge_carrier_type: The mobile charge particle type. Must be "electron" or "hole". Default is "electron".
                           For electrons: sweeps from depletion (10% of min_safe_voltage_bound, negative)
                                        toward accumulation (max_safe_voltage_bound, positive).
                           For holes: sweeps from depletion (10% of max_safe_voltage_bound, positive)
                                    toward accumulation (min_safe_voltage_bound, negative).
        session: Optional logger session for recording sweep measurements and analysis results.

    Returns:
        dict: Contains:
            - reservoir_characterization: Dictionary mapping each reservoir name to its
              cut-off voltage (cutoff_voltage) in volts.

    Notes:
        - Each reservoir is tested sequentially
        - Other reservoirs are biased at min(1.2 * global_turn_on_voltage, max_voltage_bound)
        - Target reservoir starts at 0V before sweeping
        - 10 second settling time is used between voltage changes
        - Pinch-off analysis may raise ValueError if curve fit fails
        - Sweep direction depends on carrier type: electrons sweep toward positive, holes toward negative
    """
    charge_carrier_type = charge_carrier_type.lower()
    if charge_carrier_type not in ["electron", "hole"]:
        raise RoutineError(
            "Charge carrier type is required for reservoir characterization"
        )

    if step_size <= 0:
        raise RoutineError("Step size must be greater than 0")

    ctx.resources.device.jump({bias_gate: bias_voltage}, wait_for_settling=True)

    leakage_test_results = ctx.results.get("leakage_test", {})
    global_accumulation_results = ctx.results.get("global_accumulation", {})

    max_safe_voltage_bound = leakage_test_results["max_safe_voltage_bound"]
    min_safe_voltage_bound = leakage_test_results["min_safe_voltage_bound"]

    voltage_left_bound = (
        0.1 * min_safe_voltage_bound
        if charge_carrier_type == "electron"
        else max_safe_voltage_bound
    )
    voltage_right_bound = (
        max_safe_voltage_bound
        if charge_carrier_type == "electron"
        else min_safe_voltage_bound
    )
    voltage_cut_off = 1.2 * global_accumulation_results["global_turn_on_voltage"]

    global_turn_on_voltage = (
        min(voltage_cut_off, max_safe_voltage_bound)
        if charge_carrier_type == "electron"
        else max(voltage_cut_off, min_safe_voltage_bound)
    )

    reservoir_characterization_results = {}

    reservoirs = ctx.resources.device.get_gates_by_type(GateType.RESERVOIR)
    for reservoir in reservoirs:
        other_reservoirs = [r for r in reservoirs if r != reservoir]
        ctx.resources.device.jump(
            dict.fromkeys(other_reservoirs, global_turn_on_voltage),
            wait_for_settling=True,
        )
        time.sleep(DEFAULT_SETTLING_TIME_S)
        ctx.resources.device.jump(
            {reservoir: voltage_left_bound}, wait_for_settling=True
        )
        time.sleep(DEFAULT_SETTLING_TIME_S)

        num_points = max(2, int(abs(voltage_right_bound / step_size)))
        voltages, currents = ctx.resources.device.sweep_1d(
            reservoir,
            np.linspace(voltage_left_bound, voltage_right_bound, num_points),
            measure_electrode,
            session,
        )
        try:
            reservoir_analysis = analyze_single_gate_heuristic(
                np.array(voltages), np.array(currents)
            )
        except Exception as e:
            raise RoutineError(f"Error in reservoir_characterization: {str(e)}") from e

        reservoir_characterization_results[reservoir] = reservoir_analysis

        if session:
            session.log_analysis(
                name=f"reservoir_characterization_{reservoir}",
                data=reservoir_analysis,
            )

    return {
        "reservoir_characterization": reservoir_characterization_results,
    }


@routine
def finger_gate_characterization(
    ctx: RoutineContext,
    measure_electrode: str,
    step_size: float,
    bias_gate: str,
    bias_voltage: float,
    charge_carrier_type: str = "electron",
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Characterize individual finger gates by sweeping each while holding others in accumulation.

    This health check routine is specifically designed for accumulation mode devices.
    It determines the cut-off voltage for each finger gate individually. For each finger gate
    under test, all other finger gates are set to 120% of the global turn-on voltage (to ensure
    they're fully accumulated), while the target gate is swept from minimum to maximum voltage.
    This isolates the behavior of each finger gate and identifies its individual pinch-off
    characteristics.

    Args:
        ctx: Routine context containing device resources and previous results. Requires:
             - ctx.results["max_safe_voltage_bound"]: Maximum voltage for sweeps
             - ctx.results["min_safe_voltage_bound"]: Minimum voltage for sweeps
             - ctx.results["global_turn_on_voltage"]: From global_accumulation routine
        measure_electrode: Name of the electrode to measure current from during sweeps.
        step_size: Voltage increment (V) between sweep points. Smaller values provide
                  higher resolution but increase measurement time.
        charge_carrier_type: The mobile charge particle type. Must be "electron" or "hole". Default is "electron".
                           For electrons: sweeps from depletion (10% of min_safe_voltage_bound, negative)
                                        toward accumulation (max_safe_voltage_bound, positive).
                           For holes: sweeps from depletion (10% of max_safe_voltage_bound, positive)
                                    toward accumulation (min_safe_voltage_bound, negative).
        session: Optional logger session for recording sweep measurements and analysis results.

    Returns:
        dict: Contains:
            - finger_gate_characterization: Dictionary mapping each finger gate name to its
              cut-off voltage (cutoff_voltage) in volts.

    Notes:
        - Each finger gate is tested sequentially
        - Other finger gates are biased at min(1.2 * global_turn_on_voltage, max_voltage_bound)
        - Target gate starts at 0V before sweeping
        - 10 second settling time is used after setting target gate to 0V
        - Pinch-off analysis may raise ValueError if curve fit fails
        - Sweep direction depends on carrier type: electrons sweep toward positive, holes toward negative
    """
    charge_carrier_type = charge_carrier_type.lower()
    if charge_carrier_type not in ["electron", "hole"]:
        raise RoutineError(
            "Charge carrier type is required for finger gate characterization"
        )

    if step_size <= 0:
        raise RoutineError("Step size must be greater than 0")

    ctx.resources.device.jump({bias_gate: bias_voltage}, wait_for_settling=True)

    leakage_test_results = ctx.results.get("leakage_test", {})
    global_accumulation_results = ctx.results.get("global_accumulation", {})

    max_safe_voltage_bound = leakage_test_results["max_safe_voltage_bound"]
    min_safe_voltage_bound = leakage_test_results["min_safe_voltage_bound"]

    voltage_left_bound = (
        0.1 * min_safe_voltage_bound
        if charge_carrier_type == "electron"
        else max_safe_voltage_bound
    )
    voltage_right_bound = (
        max_safe_voltage_bound
        if charge_carrier_type == "electron"
        else min_safe_voltage_bound
    )
    voltage_cut_off = 1.2 * global_accumulation_results["global_turn_on_voltage"]

    global_turn_on_voltage = (
        min(voltage_cut_off, max_safe_voltage_bound)
        if charge_carrier_type == "electron"
        else max(voltage_cut_off, min_safe_voltage_bound)
    )

    finger_gate_characterization_results = {}

    plunger_gates = ctx.resources.device.get_gates_by_type(GateType.PLUNGER)
    barrier_gates = ctx.resources.device.get_gates_by_type(GateType.BARRIER)
    finger_gates = plunger_gates + barrier_gates

    for gate in finger_gates:
        other_gates = [g for g in finger_gates if g != gate]
        ctx.resources.device.jump(
            dict.fromkeys(other_gates, global_turn_on_voltage), wait_for_settling=True
        )
        ctx.resources.device.jump({gate: 0.0}, wait_for_settling=True)
        time.sleep(DEFAULT_SETTLING_TIME_S)
        num_points = max(2, int(abs(voltage_right_bound / step_size)))
        voltages, currents = ctx.resources.device.sweep_1d(
            gate,
            np.linspace(voltage_left_bound, voltage_right_bound, num_points),
            measure_electrode,
            session,
        )
        try:
            finger_gate_analysis = analyze_single_gate_heuristic(
                np.array(voltages), np.array(currents)
            )
        except Exception as e:
            raise RoutineError(f"Error in finger_gate_characterization: {e}") from e

        finger_gate_characterization_results[gate] = finger_gate_analysis

        if session:
            session.log_analysis(
                name=f"finger_gate_characterization_{gate}",
                data=finger_gate_analysis,
            )

    return {
        "finger_gate_characterization": finger_gate_characterization_results,
    }


def _calculate_leakage_matrix(delta_V: float, current_diff: np.ndarray) -> np.ndarray:
    """Calculate leakage resistance matrix from voltage change and current differences.

    Args:
        delta_V: Voltage change applied (V).
        current_diff: Array of current differences for each gate (A).

    Returns:
        Leakage resistance matrix (Ohms) where leakage_matrix[i,j] = |delta_V / (I_j - I_j_prev)|
        when gate i was swept. Inf values indicate no measurable current change.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        leakage_matrix = np.abs(delta_V / current_diff)
        leakage_matrix[current_diff == 0] = np.inf
    return leakage_matrix


def _check_leakage_threshold(
    leakage_matrix: np.ndarray,
    leakage_threshold_resistance: int,
    leakage_threshold_count: int,
    control_gates: list[str],
    delta_V: float,
    session: LoggerSession | None,
) -> bool:
    """Check if leakage matrix exceeds threshold and log if it does.

    Args:
        leakage_matrix: Resistance matrix between gate pairs (Ohms).
        leakage_threshold_resistance: Maximum acceptable resistance threshold (Ohms).
        leakage_threshold_count: Maximum number of leaky gate pairs allowed.
        control_gates: List of control gate names for reporting.
        delta_V: Current voltage offset being tested (V).
        session: Optional logger session for recording leakage analysis.

    Returns:
        True if leakage threshold is exceeded, False otherwise.
    """
    is_leaky = leakage_matrix < leakage_threshold_resistance
    upper_triangle = np.triu(is_leaky, k=1)
    num_leaky_connections = np.count_nonzero(upper_triangle)

    if num_leaky_connections > leakage_threshold_count:
        leaky_pairs = np.where(upper_triangle)
        leaky_gate_pairs = list(zip(leaky_pairs[0], leaky_pairs[1], strict=False))

        if session:
            session.log_analysis(
                "leaky_gate_pairs",
                {
                    "gates": control_gates,
                    "leaky_gate_pairs": leaky_gate_pairs,
                    "num_leaky_connections": num_leaky_connections,
                    "delta_V": delta_V,
                },
            )

        logger.error(
            f"Leakage test failed: Found {num_leaky_connections} leaky connections "
            f"(threshold: {leakage_threshold_count}). Leaky gate pairs: {leaky_gate_pairs}"
        )
        return True

    return False


def _test_single_voltage_bound(
    ctx: RoutineContext,
    voltage_bound: float,
    control_gates: list[str],
    initial_voltages: list[float],
    initial_currents: list[float],
    leakage_threshold_resistance: int,
    leakage_threshold_count: int,
    min_current_threshold: float,
    num_points: int,
    session: LoggerSession | None,
) -> tuple[float, bool]:
    """Test for leakage at a single voltage bound by sweeping gates incrementally.

    Args:
        ctx: Routine context containing device resources.
        voltage_bound: Target voltage bound to test (V).
        control_gates: List of control gate names to test.
        initial_voltages: Initial voltage for each gate (V).
        initial_currents: Initial current for each gate (A).
        leakage_threshold_resistance: Maximum acceptable resistance between gates (Ohms).
        leakage_threshold_count: Maximum number of leaky gate pairs allowed.
        min_current_threshold: Minimum current change to consider for resistance calculation (A).
        num_points: Number of voltage steps to test.
        session: Optional logger session for recording measurements.

    Returns:
        Tuple of (delta_V, leaked) where:
            - delta_V: The voltage offset tested (V). If leakage detected, the offset where it occurred.
            - leaked: True if leakage threshold was exceeded, False otherwise.
    """
    initial_currents_array = np.array(initial_currents)
    prev_voltages_dict = dict(zip(control_gates, initial_voltages, strict=False))

    for delta_V in np.linspace(0, voltage_bound, num_points):
        ctx.resources.device.jump(prev_voltages_dict, wait_for_settling=True)
        currents_matrix = []
        for gate in control_gates:
            current_voltage = prev_voltages_dict[gate] + delta_V
            ctx.resources.device.jump({gate: current_voltage}, wait_for_settling=True)

            currents_array = ctx.resources.device.measure(control_gates)
            current_diff = currents_array - initial_currents_array

            if np.isclose(current_voltage - prev_voltages_dict[gate], 0, atol=1e-10):
                logger.warning(
                    f"Voltage is too close to previous voltage for {gate}, skipping leakage matrix calculation"
                )
                break
            elif np.max(np.abs(current_diff)) < min_current_threshold:
                logger.warning(
                    f"Current change too low for {gate}, skipping leakage matrix calculation"
                )
                break

            ctx.resources.device.jump(
                {gate: prev_voltages_dict[gate]}, wait_for_settling=True
            )
            currents_matrix.append(currents_array)

        if len(currents_matrix) != len(control_gates):
            continue

        currents_diff = np.array(currents_matrix) - initial_currents_array
        leakage_matrix = _calculate_leakage_matrix(delta_V, currents_diff)

        if session:
            session.log_measurement(
                "leakage_matrix",
                {
                    "gates": control_gates,
                    "voltages": ctx.resources.device.check(control_gates),
                    "delta_V": delta_V,
                    "leakage_matrix": leakage_matrix,
                },
            )

        # Check if leakage threshold exceeded
        if _check_leakage_threshold(
            leakage_matrix,
            leakage_threshold_resistance,
            leakage_threshold_count,
            control_gates,
            delta_V,
            session,
        ):
            # Reset to initial voltages before returning
            ctx.resources.device.jump(
                dict(zip(control_gates, initial_voltages, strict=False)),
                wait_for_settling=True,
            )
            return delta_V, True

    return delta_V, False


def analyze_single_gate_heuristic(
    voltages: np.ndarray, currents: np.ndarray
) -> dict[str, Any]:
    """Fit gate sweep data to extract cut-off voltage using a heuristic model.

    This analysis function is specifically designed for accumulation mode devices,
    where the sweep progresses from depletion to accumulation and the algorithm
    identifies the cut-off voltage at which current begins to flow.

    Args:
        voltages: Array of gate voltages (V) from the sweep.
        currents: Array of measured currents (A) corresponding to each voltage.

    Returns:
        dict: Contains cutoff_voltage (cut-off voltage) and other fit parameters.

    Raises:
        ValueError: If the curve fit quality is poor based on R² and NRMSE metrics.
    """

    pinchoff_fit = fit_pinchoff_parameters(voltages, currents, percent_threshold=0.05)
    y_pred = pinchoff_fit.fit_curve(voltages)
    is_good_fit = fit_quality_criterion(voltages, currents, y_pred)

    if not is_good_fit:
        raise ValueError("Curve fit quality is poor (low R² or high NRMSE)")

    return {
        "cutoff_voltage": pinchoff_fit.v_cut_off,
        "transition_voltage": pinchoff_fit.v_transition,
        "saturation_voltage": pinchoff_fit.v_saturation,
        "popt": pinchoff_fit.popt,
        "pcov": pinchoff_fit.pcov,
    }
