"""Sweep utilities for charge sensor measurements.

This module provides functions for performing voltage sweeps on charge sensor
devices, including building voltage lists for sensor plunger sweeps and
executing multi-window barrier sweeps with ML-based peak detection.
"""

import logging

import numpy as np
from conductorquantum import ConductorQuantum

from stanza.device import Device
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.routines import RoutineContext
from stanza.routines.builtins.charge_sensor.utils.constants import (
    COULOMB_CLASSIFIER_MODEL,
    DEFAULT_SETTLING_TIME_S,
    PEAK_DETECTOR_MODEL,
)
from stanza.routines.builtins.charge_sensor.utils.types import (
    SensorDotPlungerSweepOutput,
)
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    analyze_find_first_peak_voltages,
    calculate_quality_scores,
    normalize_sensitivity_scores,
)

logger = logging.getLogger(__name__)


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
    settling_time_s: float = DEFAULT_SETTLING_TIME_S,
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
        settling_time_s: Time to wait for device settling (default: 3.0s)
        session: Logger session for measurements and analysis
        gate_voltage_overrides: Optional dict of {gate_name: voltage} to override
                                specific gates instead of using mean_reservoir_saturation_voltage

    Returns:
        SensorDotPlungerSweepOutput containing the result of the sweep.
    """
    import time

    client: ConductorQuantum = ctx.resources.models_client
    device: Device = ctx.resources.device

    device.jump({bias_gate: bias_voltage}, wait_for_settling=True)
    time.sleep(settling_time_s)
    min_v, max_v = sensor_plunger_range

    sensor_plunger_start_voltages = np.arange(min_v, max_v, window_size)

    aggregated_voltages = np.array([], dtype=np.float32)
    aggregated_currents = np.array([], dtype=np.float32)
    fitted_peaks = []
    last_classification = False
    last_peak_indices = []

    max_v_float = float(max_v)
    for idx, sp_start_voltage in enumerate(sensor_plunger_start_voltages):
        sp_start_voltage_float = float(sp_start_voltage)
        end_voltage_candidate = sp_start_voltage_float + window_size
        sp_end_voltage = float(np.minimum(end_voltage_candidate, max_v_float))

        sp_sweep_voltages = np.linspace(
            sp_start_voltage_float,
            sp_end_voltage,
            current_trace_number_of_points,
            endpoint=False,
        )

        voltage_list = build_sensor_sweep_voltage_list(
            sensor_gates_list=sensor_gates_list,
            sensor_plunger_index=sensor_plunger_index,
            base_voltage=mean_reservoir_saturation_voltage,
            plunger_voltages=sp_sweep_voltages,
            gate_voltage_overrides=gate_voltage_overrides,
        )

        first_voltage_point = dict(
            zip(sensor_gates_list, voltage_list[0], strict=False)
        )
        device.jump(first_voltage_point, wait_for_settling=True)
        time.sleep(settling_time_s)

        _, current_trace = device.sweep_nd(
            gate_electrodes=sensor_gates_list,
            voltages=voltage_list,
            measure_electrode=measure_electrode,
            session=session,
        )

        aggregated_voltages = np.concatenate([aggregated_voltages, sp_sweep_voltages])
        aggregated_currents = np.concatenate([aggregated_currents, current_trace])

        try:
            coulomb_result = client.models.execute(
                model=COULOMB_CLASSIFIER_MODEL, data=aggregated_currents
            ).output
        except Exception as e:
            raise RoutineError(f"ML model execution failed: {e}") from e
        classification = bool(coulomb_result["classification"])
        score = float(coulomb_result["score"])

        if classification:
            try:
                aggregated_peak_indices = client.models.execute(
                    model=PEAK_DETECTOR_MODEL, data=aggregated_currents
                ).output["peak_indices"]
            except Exception as e:
                raise RoutineError(f"ML model execution failed: {e}") from e

            fitted_peaks = analyze_find_first_peak_voltages(
                aggregated_currents,
                aggregated_voltages,
                aggregated_peak_indices,
                session,
            )

            last_classification = classification
            last_peak_indices = aggregated_peak_indices

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

    if not fitted_peaks:
        raise RoutineError("No peaks were successfully fitted during the sweep")

    normalize_sensitivity_scores(fitted_peaks)
    calculate_quality_scores(fitted_peaks)

    def key_func(p: FittedPeak) -> float:
        return p.quality_score or 0.0

    sorted_peaks: list[FittedPeak] = sorted(fitted_peaks, key=key_func, reverse=True)

    best_peak = sorted_peaks[0]
    if best_peak.quality_score is None:
        raise RoutineError("Quality score not calculated for best peak")

    prev_peak = None
    next_peak = None

    for peak in fitted_peaks:
        if peak.peak_idx < best_peak.peak_idx:
            if prev_peak is None or peak.peak_idx > prev_peak.peak_idx:
                prev_peak = peak
        elif peak.peak_idx > best_peak.peak_idx:
            if next_peak is None or peak.peak_idx < next_peak.peak_idx:
                next_peak = peak

    prev_peak_voltage: float | None = (
        float(prev_peak.peak_voltage) if prev_peak is not None else None
    )
    next_peak_voltage: float | None = (
        float(next_peak.peak_voltage) if next_peak is not None else None
    )

    result = SensorDotPlungerSweepOutput(
        sensor_plunger_voltage=float(best_peak.peak_voltage),
        classification=last_classification,
        score=float(best_peak.quality_score),
        peak_indices=last_peak_indices,
        num_peaks=len(fitted_peaks),
        aggregated_voltages=aggregated_voltages,
        aggregated_currents=aggregated_currents,
        best_peak_voltage=float(best_peak.peak_voltage),
        best_peak_max_gradient_voltage=float(best_peak.sensitivity_voltage),
        prev_peak_voltage=prev_peak_voltage,
        next_peak_voltage=next_peak_voltage,
    )

    return result
