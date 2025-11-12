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
    get_global_turn_on_voltage,
    get_plunger_gate_bounds,
    get_voltages,
    log_classification,
    log_measurement,
    measure_and_classify,
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
    barrier_voltages: dict[str, float] | None = None,
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

    saturation_voltages = get_voltages(gates, "saturation_voltage", results)
    transition_voltages = get_voltages(gates, "transition_voltage", results)

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

            # Safety limit to prevent infinite loops
            if total_attempts > max_number_of_samples * 10:
                logger.warning(
                    f"Exceeded max attempts ({total_attempts}) for scale {scale:.4f} V"
                )
                break

            # Early exit if we have enough successful peak spacings across all scales
            if len(peak_spacings) >= number_of_samples_for_scale_computation:
                logger.info(
                    f"Collected {len(peak_spacings)} peak spacings, moving to next scale"
                )
                break

            x_bounds = plunger_gate_bounds[plunger_gates[0]]
            y_bounds = plunger_gate_bounds[plunger_gates[1]]
            sweep = generate_random_sweep(
                x_bounds, y_bounds, scale, current_trace_points
            )
            if not sweep:
                continue

            sweep_voltages = generate_linear_sweep(
                sweep.start, sweep.direction, sweep.total_distance, current_trace_points
            )

            voltages = build_full_voltages(
                sweep_voltages,
                gates,
                gate_idx,
                transition_voltages=transition_voltages,
                saturation_voltages=saturation_voltages,
                barrier_voltages=barrier_voltages,
            )

            currents, has_coulomb_blockade, cb_score = measure_and_classify(
                device,
                client,
                gates,
                voltages,
                measure_electrode,
                "coulomb-blockade-classifier-v3",
            )

            measurement_id = str(uuid.uuid4())
            log_measurement(
                session,
                "peak_spacing_current_trace",
                measurement_id,
                {
                    "scale": float(scale),
                    "sample_index": successful_measurements,
                    "start_point": sweep.start.tolist(),
                    "end_point": sweep.end.tolist(),
                    "currents": currents.tolist(),
                },
                metadata,
                "compute_peak_spacing",
            )
            log_classification(
                session,
                measurement_id,
                "coulomb_blockade",
                has_coulomb_blockade,
                cb_score,
                metadata,
                "compute_peak_spacing",
            )

            successful_measurements += 1

            if not has_coulomb_blockade:
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
    low_res_csd_points: int = 16,
    high_res_csd_points: int = 48,
    max_samples: int | None = None,
    num_dqds_for_exit: int = 1,
    include_diagonals: bool = False,
    seed: int = 42,
    session: LoggerSession | None = None,
    barrier_voltages: dict[str, float] | None = None,
    **kwargs: Any,
) -> dict[str, list[dict[str, Any]]]:
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
        seed: Random seed for reproducibility
        session: Logger session for telemetry
        barrier_voltages: Barrier voltages to use
        **kwargs: Additional arguments passed to sub-routines

    Returns:
        Dictionary with "dqd_squares" key containing list of all DQD squares found
    """
    np.random.seed(seed)

    device = ctx.resources.device
    client = ctx.resources.models_client
    results = ctx.results

    saturation_voltages = get_voltages(gates, "saturation_voltage", results)
    transition_voltages = get_voltages(gates, "transition_voltage", results)

    safe_bounds = get_gate_safe_bounds(results)
    peak_spacing = ctx.results.get("compute_peak_spacing")["peak_spacing"]
    gate_idx = build_gate_indices(gates, device)
    plunger_gates = [gates[i] for i in gate_idx.plunger]
    plunger_gate_bounds = get_plunger_gate_bounds(plunger_gates, results)

    square_size = peak_spacing * GRID_SQUARE_MULTIPLIER
    x_bounds = plunger_gate_bounds[plunger_gates[0]]
    y_bounds = plunger_gate_bounds[plunger_gates[1]]

    grid_corners, n_x, n_y = generate_grid_corners(x_bounds, y_bounds, square_size)
    total_squares = n_x * n_y

    if max_samples is None:
        max_samples = total_squares // 2

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
        test_sweep = generate_diagonal_sweep(corner, square_size, 8)
        test_voltages = build_full_voltages(
            test_sweep,
            gates,
            gate_idx,
            transition_voltages=transition_voltages,
            saturation_voltages=saturation_voltages,
            barrier_voltages=barrier_voltages,
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
        trace_sweep = generate_diagonal_sweep(corner, square_size, current_trace_points)
        trace_voltages = build_full_voltages(
            trace_sweep,
            gates,
            gate_idx,
            transition_voltages=transition_voltages,
            saturation_voltages=saturation_voltages,
            barrier_voltages=barrier_voltages,
        )
        trace_currents, trace_classification, trace_score = measure_and_classify(
            device,
            client,
            gates,
            trace_voltages,
            measure_electrode,
            "coulomb-blockade-classifier-v3",
        )

        trace_id = str(uuid.uuid4())
        log_measurement(
            session,
            "dqd_search_current_trace",
            trace_id,
            {
                "sample_idx": sample_idx,
                "grid_idx": grid_idx,
                "voltages": trace_voltages.tolist(),
                "currents": trace_currents.tolist(),
            },
            metadata,
            "run_dqd_search_fixed_barriers",
        )
        log_classification(
            session,
            trace_id,
            "current_trace",
            trace_classification,
            trace_score,
            metadata,
            "run_dqd_search_fixed_barriers",
        )

        # Initialize CSD variables
        low_res_currents = low_res_voltages = None
        high_res_currents = high_res_voltages = None
        low_res_score = high_res_score = 0.0
        low_res_classification = high_res_classification = False

        # Stage 2: Low-res CSD
        if trace_classification:
            low_res_sweep = generate_2d_sweep(corner, square_size, low_res_csd_points)
            low_res_voltages = build_full_voltages(
                low_res_sweep,
                gates,
                gate_idx,
                transition_voltages=transition_voltages,
                saturation_voltages=saturation_voltages,
                barrier_voltages=barrier_voltages,
            )
            low_res_currents, low_res_classification, low_res_score = (
                measure_and_classify(
                    device,
                    client,
                    gates,
                    low_res_voltages.reshape(-1, len(gates)),
                    measure_electrode,
                    "charge-stability-diagram-binary-classifier-v2-16x16",
                    reshape=(low_res_csd_points, low_res_csd_points),
                )
            )

            low_res_id = str(uuid.uuid4())
            log_measurement(
                session,
                "dqd_search_low_res_csd",
                low_res_id,
                {
                    "sample_idx": sample_idx,
                    "grid_idx": grid_idx,
                    "linked_current_trace_id": trace_id,
                    "voltages": low_res_voltages.tolist(),
                    "currents": low_res_currents.tolist(),
                },
                metadata,
                "run_dqd_search_fixed_barriers",
            )
            log_classification(
                session,
                low_res_id,
                "low_res_csd",
                low_res_classification,
                low_res_score,
                metadata,
                "run_dqd_search_fixed_barriers",
            )

        # Stage 3: High-res CSD
        if low_res_classification:
            high_res_sweep = generate_2d_sweep(corner, square_size, high_res_csd_points)
            high_res_voltages = build_full_voltages(
                high_res_sweep,
                gates,
                gate_idx,
                transition_voltages=transition_voltages,
                saturation_voltages=saturation_voltages,
                barrier_voltages=barrier_voltages,
            )
            high_res_currents, high_res_classification, high_res_score = (
                measure_and_classify(
                    device,
                    client,
                    gates,
                    high_res_voltages.reshape(-1, len(gates)),
                    measure_electrode,
                    "charge-stability-diagram-binary-classifier-v1-48x48",
                    reshape=(high_res_csd_points, high_res_csd_points),
                )
            )

            high_res_id = str(uuid.uuid4())
            log_measurement(
                session,
                "dqd_search_high_res_csd",
                high_res_id,
                {
                    "sample_idx": sample_idx,
                    "grid_idx": grid_idx,
                    "linked_low_res_csd_id": low_res_id,
                    "voltages": high_res_voltages.tolist(),
                    "currents": high_res_currents.tolist(),
                },
                metadata,
                "run_dqd_search_fixed_barriers",
            )
            log_classification(
                session,
                high_res_id,
                "high_res_csd",
                high_res_classification,
                high_res_score,
                metadata,
                "run_dqd_search_fixed_barriers",
            )

        # Record results
        square = SearchSquare(
            grid_idx=grid_idx,
            current_trace_currents=trace_currents,
            current_trace_voltages=trace_voltages,
            current_trace_score=trace_score,
            current_trace_classification=trace_classification,
            low_res_csd_currents=low_res_currents,
            low_res_csd_voltages=low_res_voltages,
            low_res_csd_score=low_res_score,
            low_res_csd_classification=low_res_classification,
            high_res_csd_currents=high_res_currents,
            high_res_csd_voltages=high_res_voltages,
            high_res_csd_score=high_res_score,
            high_res_csd_classification=high_res_classification,
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

    return {"dqd_squares": [sq.to_dict() for sq in dqd_squares]}


@routine
def run_dqd_search(
    ctx: RoutineContext,
    gates: list[str],
    measure_electrode: str,
    min_peak_spacing: float,
    max_peak_spacing: float,
    current_trace_points: int,
    outer_barrier_points: int = 5,
    inner_barrier_points: int = 5,
    num_dqds_for_exit: int = 1,
    session: LoggerSession | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run peak spacing and DQD search over barrier voltage sweeps.

    Sweeps outer barriers (0, 2) from global turn-on to mean transition voltage.
    Sweeps inner barrier (1) from transition to global turn-on voltage.
    Runs compute_peak_spacing and run_dqd_search_fixed_barriers at each point.

    Args:
        ctx: Routine context
        gates: Gate electrode names
        measure_electrode: Current measurement electrode
        outer_barrier_points: Number of sweep points for outer barriers
        inner_barrier_points: Number of sweep points for inner barrier
        min_peak_spacing: Minimum peak spacing for compute_peak_spacing
        max_peak_spacing: Maximum peak spacing for compute_peak_spacing
        num_dqds_for_exit: Number of DQDs for exit
        current_trace_points: Points per current trace
        session: Logger session
        **kwargs: Additional arguments passed to sub-routines

    Returns:
        Dict with run_dqd_search results
    """
    device = ctx.resources.device
    results = ctx.results

    # Get voltages
    global_turn_on = get_global_turn_on_voltage(results)
    transition_voltages = get_voltages(gates, "transition_voltage", results)

    # Get barrier gates
    gate_idx = build_gate_indices(gates, device)
    barrier_gates = [gates[i] for i in gate_idx.barrier]

    # Calculate mean transition voltage for outer barriers
    mean_outer_transition = np.mean(
        [transition_voltages[barrier_gates[0]], transition_voltages[barrier_gates[2]]]
    )

    # Create sweep ranges
    outer_voltages = np.linspace(
        global_turn_on, mean_outer_transition, outer_barrier_points
    )
    inner_voltages = np.linspace(
        transition_voltages[barrier_gates[1]], global_turn_on, inner_barrier_points
    )

    sweep_results = []

    for outer_v in outer_voltages:
        for inner_v in inner_voltages:
            # Set barrier voltages
            barrier_v = {
                barrier_gates[0]: outer_v,
                barrier_gates[1]: inner_v,
                barrier_gates[2]: outer_v,
            }

            logger.info(
                f"Sweeping barriers: outer={outer_v:.4f}V, inner={inner_v:.4f}V"
            )

            # Run peak spacing
            peak_result = compute_peak_spacing(
                ctx,
                gates,
                measure_electrode,
                min_peak_spacing,
                max_peak_spacing,
                current_trace_points,
                barrier_voltages=barrier_v,
                session=session,
                **kwargs,
            )
            ctx.results["compute_peak_spacing"] = peak_result

            # Run DQD search
            dqd_result = run_dqd_search_fixed_barriers(
                ctx,
                gates,
                measure_electrode,
                num_dqds_for_exit=num_dqds_for_exit,
                barrier_voltages=barrier_v,
                session=session,
                **kwargs,
            )
            ctx.results["run_dqd_search_fixed_barriers"] = dqd_result

            sweep_results.append(
                {
                    "outer_barrier_voltage": float(outer_v),
                    "inner_barrier_voltage": float(inner_v),
                    "peak_spacing": peak_result,
                    "dqd_squares": dqd_result["dqd_squares"],
                }
            )

            if len(dqd_result["dqd_squares"]) >= num_dqds_for_exit:
                logger.info(
                    f"Found {len(dqd_result['dqd_squares'])} DQDs at outer={outer_v:.4f}V, "
                    f"inner={inner_v:.4f}V - exiting barrier search"
                )
                return {"run_dqd_search": sweep_results}

    return {"run_dqd_search": sweep_results}
