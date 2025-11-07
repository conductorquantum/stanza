import logging
import uuid
from typing import Any

import numpy as np

from stanza.logger.session import LoggerSession
from stanza.routines.builtins.simple_tuner.grid_search import (
    GRID_SQUARE_MULTIPLIER,
    SearchSquare,
    generate_2d_sweep,
    generate_diagonal_sweep,
    generate_grid_corners,
    select_next_square,
)
from stanza.routines.builtins.simple_tuner.utils import (
    build_full_voltages,
    build_gate_indices,
    check_voltages_in_bounds,
    compute_peak_spacings,
    generate_linear_sweep,
    generate_random_sweep,
    get_gate_safe_bounds,
    get_plunger_gate_bounds,
    get_voltages,
)
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
    max_number_of_samples: int = 30,
    number_of_samples_for_scale_computation: int = 10,
    seed: int = 42,
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Compute peak spacing by analyzing Coulomb blockade patterns in random sweeps.

    Args:
        ctx: Routine context with device and models client
        gates: Gate electrode names
        measure_electrode: Current measurement electrode
        min_peak_spacing: Minimum voltage spacing to test (V)
        max_peak_spacing: Maximum voltage spacing to test (V)
        current_trace_points: Points per sweep trace
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
    results = ctx.results

    device.jump({"IN_A_B": 0.0005}, wait_for_settling=True)

    saturation_voltages = get_voltages(gates, "saturation_voltage", results)
    gate_idx = build_gate_indices(gates, device)
    plunger_gates = [gates[i] for i in gate_idx.plunger]
    plunger_gate_bounds = get_plunger_gate_bounds(plunger_gates, results)

    scales = np.linspace(min_peak_spacing, max_peak_spacing, 10)

    peak_spacings: list[float] = []
    metadata = {"gate_electrodes": gates, "measure_electrode": measure_electrode}

    for scale in scales:
        logger.info(f"Testing scale {scale:.4f} V")

        successful_measurements = 0
        total_attempts = 0

        while successful_measurements < max_number_of_samples:
            total_attempts += 1
            if total_attempts > max_number_of_samples * 10:  # Safety limit
                logger.warning(f"Exceeded max attempts for scale {scale:.4f} V")
                break

            x_bounds = plunger_gate_bounds[plunger_gates[0]]
            y_bounds = plunger_gate_bounds[plunger_gates[1]]
            sweep = generate_random_sweep(
                x_bounds, y_bounds, scale, current_trace_points
            )
            if not sweep:
                continue

            # Generate sweep voltages once
            sweep_voltages = generate_linear_sweep(
                sweep.start, sweep.direction, sweep.total_distance, current_trace_points
            )

            # Build full voltage array
            voltages = build_full_voltages(
                sweep_voltages,
                gates,
                gate_idx,
                saturation_voltages,
            )

            # Measure
            _, currents = device.sweep_nd(gates, voltages.tolist(), measure_electrode)
            currents = np.array(currents)

            measurement_id = str(uuid.uuid4())
            if session:
                session.log_measurement(
                    "peak_spacing_current_trace",
                    {
                        "id": measurement_id,
                        "scale": float(scale),
                        "sample_index": successful_measurements,
                        "start_point": sweep.start.tolist(),
                        "end_point": sweep.end.tolist(),
                        "currents": currents.tolist(),
                    },
                    metadata=metadata,
                    routine_name="compute_peak_spacing",
                )

            successful_measurements += 1

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
                            "sample_number": successful_measurements - 1,
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
                    model="coulomb-blockade-peak-detector-v2", data=currents
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
                            "sample_number": successful_measurements - 1,
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
                        "sample_number": successful_measurements - 1,
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

    return {
        "peak_spacing": result,
    }


@routine
def run_dqd_search_fixed_barriers(
    ctx: RoutineContext,
    gates: list[str],
    measure_electrode: str,
    current_trace_points: int = 128,
    low_res_csd_points: int = 25,
    high_res_csd_points: int = 50,
    max_samples: int | None = None,
    num_dqds_for_exit: int = 1,
    include_diagonals: bool = False,
    charge_carrier_type: str = "electrons",
    seed: int = 42,
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> list[SearchSquare]:
    """Run DQD search with fixed barrier voltages using adaptive grid sampling.

    Args:
        ctx: Routine context with device and models client
        gates: Gate electrode names
        measure_electrode: Current measurement electrode
        current_trace_points: Points in diagonal current trace
        low_res_csd_points: Points per axis in low-res charge stability diagram
        high_res_csd_points: Points per axis in high-res CSD
        max_samples: Maximum grid squares to sample (default: 50% of grid)
        num_dqds_for_exit: Exit after finding this many DQDs
        include_diagonals: Use 8-connected neighborhoods vs 4-connected
        charge_carrier_type: "electrons" or "holes" - determines sweep direction
        seed: Random seed for reproducibility
        session: Logger session for telemetry

    Returns:
        List of all DQD squares found, sorted by score descending
    """
    np.random.seed(seed)

    device = ctx.resources.device
    client = ctx.resources.models_client
    results = ctx.results

    saturation_voltages = get_voltages(gates, "saturation_voltage", ctx.results)
    safe_bounds = get_gate_safe_bounds(gates, ctx.results)
    peak_spacing = ctx.results.get("compute_peak_spacing")["peak_spacing"]
    gate_idx = build_gate_indices(gates, device)
    plunger_gates = [gates[i] for i in gate_idx.plunger]
    plunger_gate_bounds = get_plunger_gate_bounds(plunger_gates, results)

    # Setup grid
    square_size = peak_spacing * GRID_SQUARE_MULTIPLIER
    x_bounds = plunger_gate_bounds[plunger_gates[0]]
    y_bounds = plunger_gate_bounds[plunger_gates[1]]

    grid_corners, n_x, n_y = generate_grid_corners(x_bounds, y_bounds, square_size)
    total_squares = n_x * n_y

    if max_samples is None:
        max_samples = int(total_squares * 0.5)

    logger.info(
        f"Grid: {n_x}x{n_y} squares, size={square_size * 1000:.3f}mV, "
        f"safe_bounds=[{safe_bounds[0]:.3f}, {safe_bounds[1]:.3f}]V, "
        f"max_samples={max_samples}"
    )

    visited: list[SearchSquare] = []
    dqd_squares: list[SearchSquare] = []
    metadata = {"gates": gates, "measure_electrode": measure_electrode}

    for sample_idx in range(max_samples):
        # Select next square
        grid_idx: int
        if not visited:
            grid_idx = int(np.random.choice(total_squares))
        else:
            selected = select_next_square(
                visited, dqd_squares, n_x, n_y, include_diagonals
            )
            if selected is None:
                logger.info("All squares visited")
                break
            grid_idx = selected

        corner = grid_corners[grid_idx]

        # Pre-validate: check if square violates bounds
        test_sweep = generate_diagonal_sweep(
            corner, square_size, 8, charge_carrier_type
        )
        test_voltages = build_full_voltages(
            test_sweep, gates, gate_idx, saturation_voltages
        )
        if not check_voltages_in_bounds(test_voltages, safe_bounds):
            visited.append(
                SearchSquare(
                    grid_idx=grid_idx,
                    current_trace_currents=np.array([]),
                    current_trace_voltages=test_voltages,
                    current_trace_score=0.0,
                    current_trace_classification=False,
                    low_res_csd_currents=None,
                    low_res_csd_voltages=None,
                    low_res_csd_score=0.0,
                    low_res_csd_classification=False,
                    high_res_csd_currents=None,
                    high_res_csd_voltages=None,
                    high_res_csd_score=0.0,
                    high_res_csd_classification=False,
                )
            )
            continue

        # Stage 1: Current trace
        ct_sweep = generate_diagonal_sweep(
            corner, square_size, current_trace_points, charge_carrier_type
        )
        ct_voltages = build_full_voltages(
            ct_sweep, gates, gate_idx, saturation_voltages
        )
        _, ct_currents = device.sweep_nd(gates, ct_voltages.tolist(), measure_electrode)
        ct_currents = np.array(ct_currents)

        ct_id = str(uuid.uuid4())
        if session:
            session.log_measurement(
                "dqd_search_current_trace",
                {
                    "id": ct_id,
                    "sample_idx": sample_idx,
                    "grid_idx": grid_idx,
                    "voltages": ct_voltages.tolist(),
                    "currents": ct_currents.tolist(),
                },
                metadata=metadata,
                routine_name="run_dqd_search_fixed_barriers",
            )

        ct_result = client.models.execute(
            model="coulomb-blockade-classifier-v3", data=ct_currents
        ).output
        ct_classification = ct_result["classification"]
        ct_score = ct_result.get("score", 0.0)

        if session:
            session.log_analysis(
                "dqd_search_classification",
                {
                    "measurement_id": ct_id,
                    "measurement_type": "current_trace",
                    "classification": bool(ct_classification),
                    "score": float(ct_score),
                },
                metadata=metadata,
                routine_name="run_dqd_search_fixed_barriers",
            )

        # Initialize CSD variables
        lr_currents = lr_voltages = None
        hr_currents = hr_voltages = None
        lr_score = hr_score = 0.0
        lr_classification = hr_classification = False

        # Stage 2: Low-res CSD
        if ct_classification:
            lr_sweep = generate_2d_sweep(
                corner, square_size, low_res_csd_points, charge_carrier_type
            )
            lr_voltages = build_full_voltages(
                lr_sweep, gates, gate_idx, saturation_voltages
            )
            _, lr_currents = device.sweep_nd(
                gates,
                lr_voltages.reshape(-1, len(gates)).tolist(),
                measure_electrode,
            )
            lr_currents = np.array(lr_currents).reshape(
                low_res_csd_points, low_res_csd_points
            )

            lr_id = str(uuid.uuid4())
            if session:
                session.log_measurement(
                    "dqd_search_low_res_csd",
                    {
                        "id": lr_id,
                        "sample_idx": sample_idx,
                        "grid_idx": grid_idx,
                        "linked_current_trace_id": ct_id,
                        "voltages": lr_voltages.tolist(),
                        "currents": lr_currents.tolist(),
                    },
                    metadata=metadata,
                    routine_name="run_dqd_search_fixed_barriers",
                )

            lr_result = client.models.execute(
                model="csd-dqd-classifier-v1", data=lr_currents
            ).output
            lr_classification = lr_result["classification"]
            lr_score = lr_result.get("score", 0.0)

            if session:
                session.log_analysis(
                    "dqd_search_classification",
                    {
                        "measurement_id": lr_id,
                        "measurement_type": "low_res_csd",
                        "classification": bool(lr_classification),
                        "score": float(lr_score),
                    },
                    metadata=metadata,
                    routine_name="run_dqd_search_fixed_barriers",
                )

        # Stage 3: High-res CSD
        if lr_classification:
            hr_sweep = generate_2d_sweep(
                corner, square_size, high_res_csd_points, charge_carrier_type
            )
            hr_voltages = build_full_voltages(
                hr_sweep, gates, gate_idx, saturation_voltages
            )
            _, hr_currents = device.sweep_nd(
                gates,
                hr_voltages.reshape(-1, len(gates)).tolist(),
                measure_electrode,
            )
            hr_currents = np.array(hr_currents).reshape(
                high_res_csd_points, high_res_csd_points
            )

            hr_id = str(uuid.uuid4())
            if session:
                session.log_measurement(
                    "dqd_search_high_res_csd",
                    {
                        "id": hr_id,
                        "sample_idx": sample_idx,
                        "grid_idx": grid_idx,
                        "linked_low_res_csd_id": lr_id,
                        "voltages": hr_voltages.tolist(),
                        "currents": hr_currents.tolist(),
                    },
                    metadata=metadata,
                    routine_name="run_dqd_search_fixed_barriers",
                )

            hr_result = client.models.execute(
                model="csd-dqd-classifier-v1", data=hr_currents
            ).output
            hr_classification = hr_result["classification"]
            hr_score = hr_result.get("score", 0.0)

            if session:
                session.log_analysis(
                    "dqd_search_classification",
                    {
                        "measurement_id": hr_id,
                        "measurement_type": "high_res_csd",
                        "classification": bool(hr_classification),
                        "score": float(hr_score),
                    },
                    metadata=metadata,
                    routine_name="run_dqd_search_fixed_barriers",
                )

        # Record results
        square = SearchSquare(
            grid_idx=grid_idx,
            current_trace_currents=ct_currents,
            current_trace_voltages=ct_voltages,
            current_trace_score=ct_score,
            current_trace_classification=ct_classification,
            low_res_csd_currents=lr_currents,
            low_res_csd_voltages=lr_voltages,
            low_res_csd_score=lr_score,
            low_res_csd_classification=lr_classification,
            high_res_csd_currents=hr_currents,
            high_res_csd_voltages=hr_voltages,
            high_res_csd_score=hr_score,
            high_res_csd_classification=hr_classification,
        )

        visited.append(square)

        if square.is_dqd:
            dqd_squares.append(square)
            logger.info(
                f"Found DQD {len(dqd_squares)}/{num_dqds_for_exit} "
                f"(score={square.total_score:.3f})"
            )

            if len(dqd_squares) >= num_dqds_for_exit:
                logger.info("Exit condition met")
                break

    # Sort by score descending
    dqd_squares.sort(key=lambda sq: sq.total_score, reverse=True)

    if session:
        session.log_analysis(
            "dqd_search_summary",
            {
                "total_samples": len(visited),
                "num_dqds_found": len(dqd_squares),
                "grid_size": [n_x, n_y],
                "square_size_mv": square_size * 1000,
                "success": len(dqd_squares) >= num_dqds_for_exit,
            },
            metadata=metadata,
            routine_name="run_dqd_search_fixed_barriers",
        )

    return dqd_squares
