import logging
import uuid
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stanza.logger.session import LoggerSession
from stanza.models import GateType
from stanza.routines.builtins.simple_tuner.utils import (
    GateIndices,
    compute_peak_spacings,
    generate_linear_sweep,
    generate_random_sweep,
)
from stanza.routines.core import RoutineContext, routine

logger = logging.getLogger(__name__)


def _build_gate_indices(gates: list[str], device: Any) -> GateIndices:
    """Extract indices for each gate type from the gate list."""
    plunger_gates = device.get_gate_by_type(GateType.PLUNGER)
    reservoir_gates = device.get_gate_by_type(GateType.RESERVOIR)
    barrier_gates = device.get_gate_by_type(GateType.BARRIER)

    return GateIndices(
        plunger=[i for i, g in enumerate(gates) if g in plunger_gates],
        reservoir=[i for i, g in enumerate(gates) if g in reservoir_gates],
        barrier=[i for i, g in enumerate(gates) if g in barrier_gates],
    )


def _build_full_voltages(
    sweep_voltages: NDArray[np.float64],
    num_gates: int,
    gate_idx: GateIndices,
    gates: list[str],
    reservoir_voltages: dict[str, float],
    barrier_voltages: dict[str, float],
) -> NDArray[np.float64]:
    """Construct full voltage array from sweep voltages."""
    voltages = np.zeros((len(sweep_voltages), num_gates))
    voltages[:, gate_idx.plunger] = sweep_voltages
    for idx in gate_idx.reservoir:
        voltages[:, idx] = reservoir_voltages[gates[idx]]
    for idx in gate_idx.barrier:
        voltages[:, idx] = barrier_voltages[gates[idx]]
    return voltages


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
) -> float:
    """Compute peak spacing by analyzing Coulomb blockade patterns in random sweeps.

    Args:
        ctx: Routine context with device and models client
        gates: Gate electrode names
        measure_electrode: Current measurement electrode
        min_peak_spacing: Minimum voltage spacing to test (V)
        max_peak_spacing: Maximum voltage spacing to test (V)
        current_trace_points: Points per sweep trace
        plunger_x_bounds: (min, max) voltage bounds for X plunger
        plunger_y_bounds: (min, max) voltage bounds for Y plunger
        reservoir_voltages: Fixed voltages for reservoir gates
        barrier_voltages: Fixed voltages for barrier gates
        max_number_of_samples: Maximum sweep attempts per scale
        number_of_samples_for_scale_computation: Target successful samples per scale
        seed: Random seed for reproducibility
        session: Logger session for telemetry

    Returns:
        Median peak spacing in volts

    Raises:
        ValueError: If no valid peak spacings detected
    """
    np.random.seed(seed)

    device = ctx.resources.device
    client = ctx.resources.models_client
    gate_idx = _build_gate_indices(gates, device)
    scales = np.linspace(min_peak_spacing, max_peak_spacing, 10)

    peak_spacings: list[float] = []
    metadata = {"gate_electrodes": gates, "measure_electrode": measure_electrode}

    for scale in scales:
        logger.info(f"Testing scale {scale:.4f} V")

        for attempt in range(max_number_of_samples):
            sweep = generate_random_sweep(
                plunger_x_bounds, plunger_y_bounds, scale, current_trace_points
            )
            if not sweep:
                continue

            # Generate sweep voltages once
            sweep_voltages = generate_linear_sweep(
                sweep.start, sweep.direction, sweep.total_distance, current_trace_points
            )

            # Build full voltage array
            voltages = _build_full_voltages(
                sweep_voltages,
                len(gates),
                gate_idx,
                gates,
                reservoir_voltages,
                barrier_voltages,
            )

            # Measure
            currents = np.array(
                device.sweep_nd(gates, voltages.tolist(), measure_electrode)
            )

            measurement_id = str(uuid.uuid4())
            if session:
                session.log_measurement(
                    "peak_spacing_current_trace",
                    {
                        "id": measurement_id,
                        "scale": float(scale),
                        "sample_index": attempt,
                        "start_point": sweep.start.tolist(),
                        "end_point": sweep.end.tolist(),
                        "currents": currents.tolist(),
                    },
                    metadata=metadata,
                    routine_name="compute_peak_spacing",
                )

            # Classify
            if not client.models.execute(
                model="coulomb-blockade-classifier-v3", data=currents
            ).output["classification"]:
                if session:
                    session.log_analysis(
                        "peak_spacing_detection",
                        data={
                            "measurement_id": measurement_id,
                            "scale": float(scale),
                            "sample_number": attempt,
                            "success": False,
                            "reason": "no_coulomb_blockade_detected",
                        },
                        metadata=metadata,
                        routine_name="compute_peak_spacing",
                    )
                continue

            # Extract peaks
            peak_indices = np.array(
                client.models.execute(
                    model="coulomb-blockade-peak-detector-v1", data=currents
                ).output["peak_indices"]
            )

            # Compute spacings
            spacings = compute_peak_spacings(peak_indices, sweep_voltages)
            if spacings is None:
                if session:
                    session.log_analysis(
                        "peak_spacing_detection",
                        data={
                            "measurement_id": measurement_id,
                            "scale": float(scale),
                            "sample_number": attempt,
                            "num_peaks_detected": len(peak_indices),
                            "success": False,
                            "reason": "insufficient_peaks_for_spacing_calculation",
                        },
                        metadata=metadata,
                        routine_name="compute_peak_spacing",
                    )
                continue

            # Valid - record result
            mean_spacing = float(np.mean(spacings))
            peak_spacings.append(mean_spacing)

            if session:
                peak_positions = sweep_voltages[peak_indices]
                session.log_analysis(
                    "peak_spacing_detection",
                    data={
                        "measurement_id": measurement_id,
                        "scale": float(scale),
                        "sample_number": attempt,
                        "num_peaks_detected": len(peak_indices),
                        "peak_indices": peak_indices.tolist(),
                        "peak_voltages_x": peak_positions[:, 0].tolist(),
                        "peak_voltages_y": peak_positions[:, 1].tolist(),
                        "individual_peak_spacings": spacings.tolist(),
                        "mean_peak_spacing": mean_spacing,
                        "success": True,
                    },
                    metadata=metadata,
                    routine_name="compute_peak_spacing",
                )

            if len(peak_spacings) >= number_of_samples_for_scale_computation:
                break

    if not peak_spacings:
        raise ValueError("No peak spacings found.")

    result = round(float(np.median(peak_spacings)), 6)

    if session:
        session.log_analysis(
            "peak_spacing_computation_summary",
            {
                "scales_investigated": scales.tolist(),
                "num_successful_measurements": len(peak_spacings),
                "all_peak_spacings": peak_spacings,
                "peak_spacing": result,
                "std_peak_spacing": float(np.std(peak_spacings)),
                "min_peak_spacing": float(np.min(peak_spacings)),
                "max_peak_spacing": float(np.max(peak_spacings)),
            },
        )

    return result
