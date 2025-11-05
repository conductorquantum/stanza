import logging
import uuid
from typing import Any

import numpy as np

from stanza.logger.session import LoggerSession
from stanza.models import GateType
from stanza.routines.builtins.simple_tuner.utils import generate_linear_sweep
from stanza.routines.core import RoutineContext, routine

logger = logging.getLogger(__name__)


@routine
def compute_peak_spacing(
    ctx: RoutineContext,
    gates: list[str],
    measure_electrode: str,
    min_peak_spacing: float,
    max_peak_spacing: float,
    current_trace_points: int,
    plunger_x_bounds: tuple[float, float],
    plunger_y_bounds: tuple[float, float],
    reservoir_voltages: dict[str, float],
    barrier_voltages: dict[str, float],
    max_number_of_samples: int = 30,
    number_of_samples_for_scale_computation: int = 10,
    seed: int = 42,
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> float:
    """Compute the peak spacing of the device by testing random sweep traces.

    Args:
        ctx: The context object
        gates: List of gate electrode names
        measure_electrode: Electrode to measure current from
        min_peak_spacing: Minimum peak spacing to test (V)
        max_peak_spacing: Maximum peak spacing to test (V)
        current_trace_points: Number of points in each sweep
        plunger_x_bounds: (min, max) voltage bounds for X plunger
        plunger_y_bounds: (min, max) voltage bounds for Y plunger
        reservoir_voltages: Fixed voltages for reservoir gates
        barrier_voltages: Fixed voltages for barrier gates
        max_number_of_samples: Maximum attempts per scale
        number_of_samples_for_scale_computation: Target samples per scale
        seed: Random seed
        session: Logger session for measurements

    Returns:
        Computed peak spacing in volts
    """
    device = ctx.resources.device
    np.random.seed(seed)

    plunger_gates = device.get_gate_by_type(GateType.PLUNGER)
    reservoir_gates = device.get_gate_by_type(GateType.RESERVOIR)
    barrier_gates = device.get_gate_by_type(GateType.BARRIER)

    plunger_idx = [i for i, g in enumerate(gates) if g in plunger_gates]
    reservoir_idx = [i for i, g in enumerate(gates) if g in reservoir_gates]
    barrier_idx = [i for i, g in enumerate(gates) if g in barrier_gates]

    scales = np.linspace(min_peak_spacing, max_peak_spacing, 10)
    px_min, px_max = plunger_x_bounds
    py_min, py_max = plunger_y_bounds

    for scale in scales:
        logger.info(f"Testing scale {scale:.4f} V")
        samples_collected = 0
        attempts = 0

        while (
            samples_collected < number_of_samples_for_scale_computation
            and attempts < max_number_of_samples
        ):
            attempts += 1

            # Random direction and start point
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            start = np.array(
                [np.random.uniform(px_min, px_max), np.random.uniform(py_min, py_max)]
            )

            # Check if sweep stays in bounds
            total_dist = scale * (current_trace_points - 1)
            end = start + direction * total_dist
            if not (px_min <= end[0] <= px_max and py_min <= end[1] <= py_max):
                continue

            # Build voltage array
            sweep_voltages = generate_linear_sweep(
                start, direction, total_dist, current_trace_points
            )
            voltages = np.zeros((current_trace_points, len(gates)))

            voltages[:, plunger_idx] = sweep_voltages
            voltages[:, reservoir_idx] = [
                reservoir_voltages[gates[i]] for i in reservoir_idx
            ]
            voltages[:, barrier_idx] = [barrier_voltages[gates[i]] for i in barrier_idx]

            # Measure
            currents = np.array(
                device.sweep_nd(gates, voltages.tolist(), measure_electrode)
            )

            if session:
                session.log_measurement(
                    "peak_spacing_current_trace",
                    {
                        "id": str(uuid.uuid4()),
                        "scale": float(scale),
                        "sample_index": samples_collected,
                        "start_point": start.tolist(),
                        "end_point": end.tolist(),
                        "currents": currents.tolist(),
                    },
                    metadata={
                        "gate_electrodes": gates,
                        "measure_electrode": measure_electrode,
                    },
                    routine_name="compute_peak_spacing",
                )

            # TODO: Implement peak detection and spacing computation
            # detected_spacing = detect_peaks_and_compute_spacing(currents, scale)
            # if detected_spacing is not None:
            #     peak_spacings.append(detected_spacing)

            samples_collected += 1

    # TODO: Return best scale based on peak spacing analysis
    return float(np.mean(scales))
