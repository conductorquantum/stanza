"""Grid-based search utilities for DQD discovery."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Algorithm constants
HIGH_SCORE_THRESHOLD: float = 1.5  # Minimum score for priority exploration
GRID_SQUARE_MULTIPLIER: float = 3.0 / np.sqrt(
    2
)  # Peak spacing multiplier for grid size
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""

        def _to_list(arr: NDArray[np.float64] | None) -> list[float] | None:
            return arr.tolist() if arr is not None else None

        return {
            "grid_idx": self.grid_idx,
            "current_trace_currents": self.current_trace_currents.tolist(),
            "current_trace_voltages": self.current_trace_voltages.tolist(),
            "current_trace_score": self.current_trace_score,
            "current_trace_classification": self.current_trace_classification,
            "low_res_csd_currents": _to_list(self.low_res_csd_currents),
            "low_res_csd_voltages": _to_list(self.low_res_csd_voltages),
            "low_res_csd_score": self.low_res_csd_score,
            "low_res_csd_classification": self.low_res_csd_classification,
            "high_res_csd_currents": _to_list(self.high_res_csd_currents),
            "high_res_csd_voltages": _to_list(self.high_res_csd_voltages),
            "high_res_csd_score": self.high_res_csd_score,
            "high_res_csd_classification": self.high_res_csd_classification,
            "total_score": self.total_score,
            "is_dqd": self.is_dqd,
        }


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
) -> NDArray[np.float64]:
    """Generate diagonal line through a square.

    Args:
        corner: (2,) bottom-left corner coordinates
        size: Square side length
        num_points: Number of points along diagonal

    Returns:
        (num_points, 2) array of voltage coordinates
    """
    t = np.linspace(0, size, num_points)
    return corner + np.column_stack([t, t])


def generate_2d_sweep(
    corner: NDArray[np.float64],
    size: float,
    num_points: int,
) -> NDArray[np.float64]:
    """Generate 2D grid sweep over a square.

    Args:
        corner: (2,) bottom-left corner coordinates
        size: Square side length
        num_points: Number of points per axis

    Returns:
        (num_points, num_points, 2) array of voltage coordinates
    """
    t = np.linspace(0, size, num_points)
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
    """Select candidate weighted by proximity and score of successful squares.

    DQDs cluster in voltage space, so weight candidates by nearby high scores.

    Args:
        candidates: Candidate square indices
        successful_squares: High-scoring squares
        n_x: Number of squares in X direction
        n_y: Number of squares in Y direction
        distance_decay: Weight decay per unit distance

    Returns:
        Selected square index
    """
    weights = []
    for candidate in candidates:
        weight = 0.0
        for square in successful_squares:
            dist = get_grid_distance(candidate, square.grid_idx, n_x, n_y)
            weight += square.total_score * distance_decay / (dist + 1.0)
        weights.append(weight)

    probabilities = np.array(weights)
    probabilities /= probabilities.sum()
    return int(np.random.choice(candidates, p=probabilities))


def select_next_square(
    visited: list[SearchSquare],
    dqd_squares: list[SearchSquare],
    n_x: int,
    n_y: int,
    include_diagonals: bool,
    score_threshold: float = HIGH_SCORE_THRESHOLD,
) -> int | None:
    """Select next grid square using hierarchical priority strategy.

    Priority: DQD neighbors > high-score neighbors > random unvisited.

    Args:
        visited: Already-visited squares
        dqd_squares: Confirmed DQD squares
        n_x: Number of squares in X direction
        n_y: Number of squares in Y direction
        include_diagonals: Use 8-connected vs 4-connected neighborhoods
        score_threshold: Minimum score for high-scoring squares

    Returns:
        Grid index to sample next, or None if all visited
    """
    total_squares = n_x * n_y
    visited_indices = {sq.grid_idx for sq in visited}
    unvisited = set(range(total_squares)) - visited_indices

    if not unvisited:
        return None

    def _get_unvisited_neighbors(squares: list[SearchSquare]) -> set[int]:
        candidates: set[int] = set()
        for sq in squares:
            neighbors = get_neighboring_squares(
                sq.grid_idx, n_x, n_y, include_diagonals
            )
            candidates.update(n for n in neighbors if n not in visited_indices)
        return candidates

    # Priority 1: DQD neighbors
    if dqd_squares:
        candidates = _get_unvisited_neighbors(dqd_squares)
        if candidates:
            return select_weighted_by_score(list(candidates), dqd_squares, n_x, n_y)

    # Priority 2: High-score neighbors
    high_score_squares = [sq for sq in visited if sq.total_score >= score_threshold]
    if high_score_squares:
        candidates = _get_unvisited_neighbors(high_score_squares)
        if candidates:
            return select_weighted_by_score(
                list(candidates), high_score_squares, n_x, n_y
            )

    # Priority 3: Random exploration
    return int(np.random.choice(list(unvisited)))
