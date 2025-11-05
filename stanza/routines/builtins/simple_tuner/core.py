import logging
import uuid
from typing import Any

import numpy as np

from stanza.logger.session import LoggerSession
from stanza.models import GateType
from stanza.routines.builtins.simple_tuner.utils import generate_linear_sweep
from stanza.routines.core import routine

logger = logging.getLogger(__name__)


@routine
def compute_peak_spacing(  # type: ignore[no-untyped-def, return]
    ctx,
    gates: list[str],
    measure_electrode: str,
    min_peak_spacing: float,
    max_peak_spacing: float,
    current_trace_points: int,
    plunger_x_bounds: tuple[float, float],
    plunger_y_bounds: tuple[float, float],
    max_number_of_samples: int = 30,
    number_of_samples_for_scale_computation: int = 10,
    seed: int = 42,
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> float:
    """Compute the peak spacing of the device.

    Args:
        ctx: The context object
        device: The device object
        seed: The random seed
        min_peak_spacing: The minimum peak spacing
        max_peak_spacing: The maximum peak spacing
    """
    device = ctx.resources.device

    reservoir_characterization_results = ctx.resources.results.get(  # noqa: F841
        "reservoir_characterization", {}
    )

    # TODO: Fetch results from reservoir and finger gate characterization

    np.random.seed(seed)

    scales_to_investigate = np.linspace(min_peak_spacing, max_peak_spacing, 10)

    peak_spacings: list[float] = []

    for scale in scales_to_investigate:
        if len(peak_spacings) >= number_of_samples_for_scale_computation:
            break
        logger.info(f"Computing peak spacing for scale {scale:.4f} V")

        num_samples = 0
        while num_samples < max_number_of_samples:
            # Calculate total sweep dist (scale is per-point spacing)
            total_sweep_dist = scale * (current_trace_points - 1)

            # Get plunger ranges
            plunger_x_min, plunger_x_max = plunger_x_bounds
            plunger_y_min, plunger_y_max = plunger_y_bounds

            # Generate random unit vector direction
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])

            # Generate random start point within plunger ranges
            start_point = np.array(
                [
                    np.random.uniform(plunger_x_min, plunger_x_max),
                    np.random.uniform(plunger_y_min, plunger_y_max),
                ]
            )

            # Calculate end point
            end_point = start_point + direction * total_sweep_dist

            # Check if end point is within bounds
            if not (
                plunger_x_min <= end_point[0] <= plunger_x_max
                and plunger_y_min <= end_point[1] <= plunger_y_max
            ):
                continue

            # Generate trace in the random direction
            scale_finding_voltages = generate_linear_sweep(
                start_point,
                direction,
                total_sweep_dist,
                current_trace_points,
            )

            voltages = np.zeros((scale_finding_voltages.shape[0], len(gates)))

            plunger_indices = [
                i
                for i, gate in enumerate(gates)
                if gate in device.get_gate_by_type(GateType.PLUNGER)
            ]
            reservoir_indices = [
                i
                for i, gate in enumerate(gates)
                if gate in device.get_gate_by_type(GateType.RESERVOIR)
            ]
            barrier_indices = [
                i
                for i, gate in enumerate(gates)
                if gate in device.get_gate_by_type(GateType.BARRIER)
            ]

            for i in plunger_indices:
                voltages[:, i] = scale_finding_voltages[:, i]

            for i in reservoir_indices:
                voltages[:, i] = fixed_reservoir_voltages[gates[i]]  # type: ignore[name-defined]  # noqa: F821

            for i in barrier_indices:
                voltages[:, i] = fixed_barrier_voltages[gates[i]]  # type: ignore[name-defined]  # noqa: F821

            current_trace = device.sweep_nd(
                gate_electrodes=gates,
                voltages=voltages.tolist(),
                measure_electrode=measure_electrode,
            )

            current_trace_arr = np.array(current_trace)

            session.log_measurement(  # type: ignore[union-attr]
                "peak_spacing_current_trace",
                {
                    "id": str(uuid.uuid4()),
                    "scale": float(scale),
                    "sample_index": num_samples,
                    "start_point": start_point.tolist(),
                    "end_point": end_point.tolist(),
                    "direction": direction.tolist(),
                    "total_sweep_distance": float(total_sweep_dist),
                    "voltages": voltages.tolist(),
                    "currents": current_trace_arr.tolist(),
                },
                metadata={
                    "gate_electrodes": gates,
                    "measure_electrode": measure_electrode,
                },
                routine_name="compute_peak_spacing",
            )
