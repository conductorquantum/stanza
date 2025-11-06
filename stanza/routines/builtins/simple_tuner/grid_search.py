"""Grid-based search utilities for DQD discovery."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Algorithm constants
HIGH_SCORE_THRESHOLD: float = 1.5  # Minimum score for priority exploration
GRID_SQUARE_MULTIPLIER: float = 4.0  # Peak spacing multiplier for grid size
DISTANCE_DECAY_FACTOR: float = 1.0  # Weight decay with distance in weighted selection


@dataclass
class SearchSquare:
    """Results from measuring a single grid square."""

    grid_idx: int
    current_trace_currents: NDArray[np.float64]
    current_trace_voltages: NDArray[np.float64]
    current_trace_score: float
    current_trace_classification: bool
    low_res_csd_currents: NDArray[np.float64] | None
    low_res_csd_voltages: NDArray[np.float64] | None
    low_res_csd_score: float
    low_res_csd_classification: bool
    high_res_csd_currents: NDArray[np.float64] | None
    high_res_csd_voltages: NDArray[np.float64] | None
    high_res_csd_score: float
    high_res_csd_classification: bool

    @property
    def total_score(self) -> float:
        return (
            self.current_trace_score + self.low_res_csd_score + self.high_res_csd_score
        )

    @property
    def is_dqd(self) -> bool:
        return self.high_res_csd_classification


def generate_grid_corners(
    plunger_x_bounds: tuple[float, float],
    plunger_y_bounds: tuple[float, float],
    square_size: float,
) -> tuple[NDArray[np.float64], int, int]:
    """Generate grid of square corners covering the voltage space.

    Args:
        plunger_x_bounds: (min, max) voltage bounds for X plunger
        plunger_y_bounds: (min, max) voltage bounds for Y plunger
        square_size: Side length of each grid square

    Returns:
        Tuple of (grid_corners, n_x, n_y) where:
        - grid_corners: (n_x*n_y, 2) array of bottom-left corners
        - n_x: Number of squares in X direction
        - n_y: Number of squares in Y direction
    """
    range_x = plunger_x_bounds[1] - plunger_x_bounds[0]
    range_y = plunger_y_bounds[1] - plunger_y_bounds[0]

    n_x = abs(int(np.floor(range_x / square_size)))
    n_y = abs(int(np.floor(range_y / square_size)))

    # Adjust bounds to fit integer number of squares
    adj_x_max = plunger_x_bounds[0] + n_x * square_size
    adj_y_max = plunger_y_bounds[0] + n_y * square_size

    x_corners = np.linspace(plunger_x_bounds[0], adj_x_max, n_x + 1)[:-1]
    y_corners = np.linspace(plunger_y_bounds[0], adj_y_max, n_y + 1)[:-1]

    grid_x, grid_y = np.meshgrid(x_corners, y_corners)
    corners = np.column_stack([grid_x.flatten(), grid_y.flatten()])

    return corners, n_x, n_y


def generate_diagonal_sweep(
    corner: NDArray[np.float64],
    size: float,
    num_points: int,
    charge_carrier_type: str = "electrons",
) -> NDArray[np.float64]:
    """Generate diagonal line through a square.

    Sweeps towards accumulation (electrons) or pinch-off (holes) depending
    on charge carrier type.

    Args:
        corner: (2,) bottom-left corner coordinates
        size: Square side length
        num_points: Number of points along diagonal
        charge_carrier_type: "electrons" (positive sweep) or "holes" (negative sweep)

    Returns:
        (num_points, 2) array of voltage coordinates
    """
    if charge_carrier_type == "electrons":
        t = np.linspace(0, size, num_points)
    else:  # holes - sweep in negative direction towards pinch-off
        t = np.linspace(0, -size, num_points)

    return corner + np.column_stack([t, t])


def generate_2d_sweep(
    corner: NDArray[np.float64],
    size: float,
    num_points: int,
    charge_carrier_type: str = "electrons",
) -> NDArray[np.float64]:
    """Generate 2D grid sweep over a square.

    Sweeps towards accumulation (electrons) or pinch-off (holes) depending
    on charge carrier type.

    Args:
        corner: (2,) bottom-left corner coordinates
        size: Square side length
        num_points: Number of points per axis
        charge_carrier_type: "electrons" (positive sweep) or "holes" (negative sweep)

    Returns:
        (num_points, num_points, 2) array of voltage coordinates
    """
    if charge_carrier_type == "electrons":
        t = np.linspace(0, size, num_points)
    else:  # holes - sweep in negative direction towards pinch-off
        t = np.linspace(0, -size, num_points)

    x_mesh, y_mesh = np.meshgrid(t, t)
    return np.stack([x_mesh + corner[0], y_mesh + corner[1]], axis=-1)


def get_neighboring_squares(
    grid_idx: int, n_x: int, n_y: int, include_diagonals: bool = False
) -> list[int]:
    """Get neighboring grid square indices.

    Args:
        grid_idx: Linear index of current grid square
        n_x: Number of grid squares in X direction
        n_y: Number of grid squares in Y direction
        include_diagonals: Use 8-connected (True) vs 4-connected (False)

    Returns:
        List of neighboring grid square indices
    """
    row, col = grid_idx // n_x, grid_idx % n_x
    neighbors = []

    directions = (
        [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        if include_diagonals
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )

    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < n_y and 0 <= c < n_x:
            neighbors.append(r * n_x + c)

    return neighbors


def get_grid_distance(idx1: int, idx2: int, n_x: int, n_y: int) -> int:
    """Calculate Manhattan distance between two grid squares.

    Manhattan distance is the sum of horizontal and vertical distances,
    appropriate for grid topology where moves are limited to cardinal directions.

    Args:
        idx1: Linear index of first square
        idx2: Linear index of second square
        n_x: Number of squares in X direction
        n_y: Number of squares in Y direction

    Returns:
        Manhattan distance (sum of horizontal and vertical distance)
    """
    row1, col1 = divmod(idx1, n_x)
    row2, col2 = divmod(idx2, n_x)
    return abs(row1 - row2) + abs(col1 - col2)


def select_weighted_by_score(
    candidates: list[int],
    successful_squares: list[SearchSquare],
    n_x: int,
    n_y: int,
    distance_decay: float = DISTANCE_DECAY_FACTOR,
) -> int:
    """Select from candidates weighted by proximity and score of successful squares.

    Exploits spatial correlation in voltage space: DQDs tend to cluster, so squares
    near high-scoring regions are more likely to contain DQDs. Weight combines:
    - Score magnitude (higher = more important)
    - Distance decay (closer = more important)

    Args:
        candidates: List of candidate square indices
        successful_squares: List of high-scoring squares
        n_x: Number of squares in X direction
        n_y: Number of squares in Y direction
        distance_decay: Multiplier for weights at unit distance (default: 1.0)

    Returns:
        Selected square index, chosen probabilistically by weights
    """
    weights = []
    for candidate_idx in candidates:
        weight = 0.0
        for success_square in successful_squares:
            score = success_square.total_score
            dist = get_grid_distance(candidate_idx, success_square.grid_idx, n_x, n_y)
            # Weight by score, decayed by distance: closer high-score regions dominate
            weight += score * distance_decay / (dist + 1.0)
        weights.append(weight)

    # Normalize and sample
    weights_array = np.array(weights)
    weights_normalized = weights_array / weights_array.sum()
    return int(np.random.choice(candidates, p=weights_normalized))


def select_next_square(
    visited: list[SearchSquare],
    dqd_squares: list[SearchSquare],
    n_x: int,
    n_y: int,
    include_diagonals: bool,
    score_threshold: float = HIGH_SCORE_THRESHOLD,
) -> int | None:
    """Select next grid square using hierarchical priority strategy.

    Priority hierarchy (exploitation → exploration):
    1. Neighbors of confirmed DQDs (weighted by score and proximity)
    2. Neighbors of high-scoring squares (weighted by score and proximity)
    3. Random unvisited square

    Uses weighted selection for priorities 1 and 2 to exploit spatial correlation:
    DQDs cluster in voltage space, so squares near high-scoring regions are
    preferentially sampled.

    Args:
        visited: List of already-visited squares
        dqd_squares: List of confirmed DQD squares
        n_x: Number of squares in X direction
        n_y: Number of squares in Y direction
        include_diagonals: Use 8-connected neighborhoods
        score_threshold: Minimum score for high-scoring squares

    Returns:
        Grid index to sample next, or None if all visited
    """
    total_squares = n_x * n_y
    visited_indices = {sq.grid_idx for sq in visited}
    unvisited = set(range(total_squares)) - visited_indices

    if not unvisited:
        return None

    # Priority 1: DQD neighbors (weighted by score and distance)
    if dqd_squares:
        candidates: set[int] = set()
        for sq in dqd_squares:
            neighbors = get_neighboring_squares(
                sq.grid_idx, n_x, n_y, include_diagonals
            )
            candidates.update(n for n in neighbors if n not in visited_indices)
        if candidates:
            return select_weighted_by_score(
                list(candidates), dqd_squares, n_x, n_y, DISTANCE_DECAY_FACTOR
            )

    # Priority 2: High-score neighbors (weighted by score and distance)
    high_score_squares = [sq for sq in visited if sq.total_score >= score_threshold]
    if high_score_squares:
        candidates_2: set[int] = set()
        for sq in high_score_squares:
            neighbors = get_neighboring_squares(
                sq.grid_idx, n_x, n_y, include_diagonals
            )
            candidates_2.update(n for n in neighbors if n not in visited_indices)
        if candidates_2:
            return select_weighted_by_score(
                list(candidates_2), high_score_squares, n_x, n_y, DISTANCE_DECAY_FACTOR
            )

    # Priority 3: Random exploration
    return int(np.random.choice(list(unvisited)))
