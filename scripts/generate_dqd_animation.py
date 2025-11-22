#!/usr/bin/env python3
"""Generate an animated GIF showing the DQD search algorithm.

This script creates a visualization of the Double Quantum Dot (DQD) search
algorithm, demonstrating the three-stage adaptive grid search process and
intelligent square selection strategy.
"""

from dataclasses import dataclass
from enum import Enum

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Set style
plt.style.use("default")

# Constants from the actual algorithm
GRID_SIZE = (8, 8)  # n_x, n_y
HIGH_SCORE_THRESHOLD = 1.5
INCLUDE_DIAGONALS = False


class SquareState(Enum):
    """States a grid square can be in during search."""

    UNVISITED = 0
    CURRENT = 1  # Currently being measured
    STAGE1_FAIL = 2  # Failed current trace classification
    STAGE2_FAIL = 3  # Passed stage 1, failed low-res CSD
    STAGE3_FAIL = 4  # Passed stage 2, failed high-res CSD
    DQD_CONFIRMED = 5  # Passed all stages - DQD found!


@dataclass
class SearchSquareResult:
    """Result from measuring a single grid square."""

    grid_idx: int
    state: SquareState
    total_score: float
    current_trace_score: float
    low_res_score: float
    high_res_score: float


# Color scheme
COLORS = {
    SquareState.UNVISITED: "#f0f0f0",
    SquareState.CURRENT: "#00cccc",  # Cyan/Teal - currently measuring
    SquareState.STAGE1_FAIL: "#ff4444",  # Red - failed stage 1
    SquareState.STAGE2_FAIL: "#ff8844",  # Orange - failed stage 2
    SquareState.STAGE3_FAIL: "#ffcc44",  # Yellow - failed stage 3
    SquareState.DQD_CONFIRMED: "#44ff44",  # Green - DQD found!
}

EDGE_COLORS = {
    SquareState.UNVISITED: "#cccccc",
    SquareState.CURRENT: "#00cccc",
    SquareState.STAGE1_FAIL: "#ff4444",
    SquareState.STAGE2_FAIL: "#ff8844",
    SquareState.STAGE3_FAIL: "#ffcc44",
    SquareState.DQD_CONFIRMED: "#00ff00",
}


def get_neighboring_squares(
    grid_idx: int, n_x: int, n_y: int, include_diagonals: bool = False
) -> list[int]:
    """Get neighboring grid square indices (from actual algorithm)."""
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


def simulate_dqd_search(n_x: int, n_y: int, seed: int = 42) -> list[SearchSquareResult]:
    """Simulate a DQD search to generate animation frames.

    This creates a realistic search trajectory following the actual algorithm logic.
    """
    np.random.seed(seed)
    total_squares = n_x * n_y

    # Place some "true" DQDs in the grid (these would be found by the ML classifiers)
    true_dqd_locations = {18, 19, 26, 27}  # Cluster of DQDs in middle-left
    high_score_locations = {17, 25, 34, 35}  # Near-miss squares

    results: list[SearchSquareResult] = []
    visited_indices = set()
    dqd_squares = []

    # Helper to classify a square (simulated)
    def classify_square(idx: int) -> SearchSquareResult:
        # Simulate the three-stage classification
        base_score = np.random.uniform(0.2, 0.8)

        if idx in true_dqd_locations:
            # This is a real DQD - passes all stages
            return SearchSquareResult(
                grid_idx=idx,
                state=SquareState.DQD_CONFIRMED,
                total_score=2.7 + np.random.uniform(0, 0.3),
                current_trace_score=0.9 + np.random.uniform(0, 0.1),
                low_res_score=0.85 + np.random.uniform(0, 0.15),
                high_res_score=0.9 + np.random.uniform(0, 0.1),
            )
        elif idx in high_score_locations:
            # High score but not quite DQD
            stage = np.random.choice([2, 3], p=[0.3, 0.7])
            if stage == 2:
                return SearchSquareResult(
                    grid_idx=idx,
                    state=SquareState.STAGE2_FAIL,
                    total_score=1.5 + np.random.uniform(0, 0.3),
                    current_trace_score=0.75 + np.random.uniform(0, 0.15),
                    low_res_score=0.3 + np.random.uniform(0, 0.2),
                    high_res_score=0.0,
                )
            else:
                return SearchSquareResult(
                    grid_idx=idx,
                    state=SquareState.STAGE3_FAIL,
                    total_score=1.6 + np.random.uniform(0, 0.3),
                    current_trace_score=0.75 + np.random.uniform(0, 0.15),
                    low_res_score=0.65 + np.random.uniform(0, 0.15),
                    high_res_score=0.3 + np.random.uniform(0, 0.1),
                )
        else:
            # Random square - most fail early
            if np.random.random() < 0.7:  # 70% fail stage 1
                return SearchSquareResult(
                    grid_idx=idx,
                    state=SquareState.STAGE1_FAIL,
                    total_score=base_score,
                    current_trace_score=base_score,
                    low_res_score=0.0,
                    high_res_score=0.0,
                )
            elif np.random.random() < 0.8:  # Most of remaining fail stage 2
                return SearchSquareResult(
                    grid_idx=idx,
                    state=SquareState.STAGE2_FAIL,
                    total_score=0.9 + np.random.uniform(0, 0.3),
                    current_trace_score=0.7 + np.random.uniform(0, 0.2),
                    low_res_score=0.2 + np.random.uniform(0, 0.1),
                    high_res_score=0.0,
                )
            else:  # Few fail stage 3
                return SearchSquareResult(
                    grid_idx=idx,
                    state=SquareState.STAGE3_FAIL,
                    total_score=1.2 + np.random.uniform(0, 0.3),
                    current_trace_score=0.65 + np.random.uniform(0, 0.15),
                    low_res_score=0.5 + np.random.uniform(0, 0.15),
                    high_res_score=0.25 + np.random.uniform(0, 0.05),
                )

    # Start with random square
    current_idx = np.random.choice(total_squares)

    max_iterations = 35  # Stop after this many squares
    num_dqds_for_exit = 3  # Exit after finding 3 DQDs

    for _iteration in range(max_iterations):
        # Measure current square
        result = classify_square(current_idx)
        results.append(result)
        visited_indices.add(current_idx)

        if result.state == SquareState.DQD_CONFIRMED:
            dqd_squares.append(result)
            if len(dqd_squares) >= num_dqds_for_exit:
                break

        # Select next square using priority strategy
        unvisited = set(range(total_squares)) - visited_indices
        if not unvisited:
            break

        # Priority 1: DQD neighbors
        if dqd_squares:
            candidates = set()
            for sq in dqd_squares:
                neighbors = get_neighboring_squares(
                    sq.grid_idx, n_x, n_y, INCLUDE_DIAGONALS
                )
                candidates.update(n for n in neighbors if n not in visited_indices)
            if candidates:
                current_idx = np.random.choice(list(candidates))
                continue

        # Priority 2: High-score neighbors
        high_score_squares = [
            r for r in results if r.total_score >= HIGH_SCORE_THRESHOLD
        ]
        if high_score_squares:
            candidates = set()
            for sq in high_score_squares:
                neighbors = get_neighboring_squares(
                    sq.grid_idx, n_x, n_y, INCLUDE_DIAGONALS
                )
                candidates.update(n for n in neighbors if n not in visited_indices)
            if candidates:
                current_idx = np.random.choice(list(candidates))
                continue

        # Priority 3: Random exploration
        current_idx = np.random.choice(list(unvisited))

    return results


def create_animation(output_path: str = "dqd_search_animation.gif"):
    """Create the DQD search animation."""
    n_x, n_y = GRID_SIZE
    search_results = simulate_dqd_search(n_x, n_y)

    # Define voltage ranges for plunger gates (realistic experimental values)
    v_p1_min, v_p1_max = -2.0, -1.5  # 500 mV range for P1
    v_p2_min, v_p2_max = -1.9, -1.4  # 500 mV range for P2

    # Calculate voltage per grid square
    v_p1_per_square = (v_p1_max - v_p1_min) / n_x
    v_p2_per_square = (v_p2_max - v_p2_min) / n_y

    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 7))

    # Left: Grid visualization
    ax_grid = plt.subplot(1, 2, 1)
    ax_grid.set_xlim(v_p1_min, v_p1_max)
    ax_grid.set_ylim(v_p2_min, v_p2_max)
    ax_grid.set_aspect("equal")
    ax_grid.set_xlabel(
        "Plunger Gate 1 Voltage (V)", fontsize=13, color="black", labelpad=10
    )
    ax_grid.set_ylabel(
        "Plunger Gate 2 Voltage (V)", fontsize=13, color="black", labelpad=10
    )
    ax_grid.set_title(
        "DQD Search: Adaptive Grid Sampling", fontsize=15, color="black", pad=15
    )
    ax_grid.tick_params(colors="black")
    # Create voltage tick labels
    v_p1_ticks = np.linspace(v_p1_min, v_p1_max, n_x + 1)
    v_p2_ticks = np.linspace(v_p2_min, v_p2_max, n_y + 1)
    ax_grid.set_xticks(v_p1_ticks)
    ax_grid.set_yticks(v_p2_ticks)
    ax_grid.set_xticklabels([f"{v:.2f}" for v in v_p1_ticks])
    ax_grid.set_yticklabels([f"{v:.2f}" for v in v_p2_ticks])
    ax_grid.grid(False)

    # Right: Legend
    ax_legend = plt.subplot(1, 2, 2)
    ax_legend.axis("off")

    # Pre-create all grid squares (in voltage space)
    grid_squares = {}
    for idx in range(n_x * n_y):
        row, col = divmod(idx, n_x)
        # Convert grid indices to voltage positions
        v_p1 = v_p1_min + col * v_p1_per_square
        v_p2 = v_p2_min + row * v_p2_per_square
        rect = patches.Rectangle(
            (v_p1, v_p2),
            v_p1_per_square,
            v_p2_per_square,
            facecolor=COLORS[SquareState.UNVISITED],
            edgecolor="#cccccc",
            linewidth=0.5,
            zorder=1,
        )
        ax_grid.add_patch(rect)
        grid_squares[idx] = rect

    # Text for status message
    status_text = ax_legend.text(
        0.5, 0.85, "", ha="center", va="top", fontsize=14, color="black"
    )

    # Create legend with colored squares (centered)
    legend_y_start = 0.65
    legend_items = [
        (SquareState.UNVISITED, "Unvisited"),
        (SquareState.CURRENT, "Current"),
        (SquareState.STAGE1_FAIL, "Failed Stage 1: Current Trace"),
        (SquareState.STAGE2_FAIL, "Failed Stage 2: Low-Res CSD"),
        (SquareState.STAGE3_FAIL, "Failed Stage 3: High-Res CSD"),
        (SquareState.DQD_CONFIRMED, "DQD Confirmed"),
    ]

    legend_patches = []
    legend_texts = []

    for i, (state, label) in enumerate(legend_items):
        y_pos = legend_y_start - i * 0.11

        # Create colored square (centered)
        legend_rect = patches.Rectangle(
            (0.25, y_pos - 0.025),
            0.08,
            0.08,
            facecolor=COLORS[state],
            edgecolor="black",
            linewidth=1.5,
            transform=ax_legend.transAxes,
        )
        ax_legend.add_patch(legend_rect)
        legend_patches.append(legend_rect)

        # Add label text
        text = ax_legend.text(
            0.38,
            y_pos,
            label,
            ha="left",
            va="center",
            fontsize=12,
            color="black",
            transform=ax_legend.transAxes,
        )
        legend_texts.append(text)

    # Current square indicator (highlight border)
    current_indicator = patches.Rectangle(
        (0, 0),
        v_p1_per_square,
        v_p2_per_square,
        facecolor="none",
        edgecolor="#00cccc",
        linewidth=3,
        zorder=3,
        visible=False,
    )
    ax_grid.add_patch(current_indicator)

    # Animation state
    visited_states = {}

    def init():
        """Initialize animation."""
        return (
            list(grid_squares.values())
            + [status_text, current_indicator]
            + legend_patches
            + legend_texts
        )

    def update(frame):
        """Update animation frame."""
        # Hold on first and last frames
        if frame == 0:
            status_text.set_text("Initializing search...")
            return (
                list(grid_squares.values())
                + [status_text, current_indicator]
                + legend_patches
                + legend_texts
            )

        # Calculate which search step we're on
        step_idx = min(frame - 1, len(search_results) - 1)
        result = search_results[step_idx]

        # Update visited states
        if result.grid_idx not in visited_states:
            visited_states[result.grid_idx] = result

        # Update grid squares
        for idx, state_result in visited_states.items():
            square = grid_squares[idx]
            if idx == result.grid_idx and frame <= len(search_results):
                # Currently measuring this square
                square.set_facecolor(COLORS[SquareState.CURRENT])
                square.set_edgecolor("#cccccc")
                square.set_linewidth(0.5)
            else:
                # Show final state
                square.set_facecolor(COLORS[state_result.state])
                square.set_edgecolor("#cccccc")
                square.set_linewidth(0.5)

        # Update current square indicator
        if frame <= len(search_results):
            row, col = divmod(result.grid_idx, n_x)
            # Convert grid indices to voltage positions
            v_p1 = v_p1_min + col * v_p1_per_square
            v_p2 = v_p2_min + row * v_p2_per_square
            current_indicator.set_xy((v_p1, v_p2))
            current_indicator.set_visible(True)
        else:
            current_indicator.set_visible(False)

        # Count statistics
        dqd_count = sum(
            1 for s in visited_states.values() if s.state == SquareState.DQD_CONFIRMED
        )

        # Update status
        if frame <= len(search_results):
            status_text.set_text(
                f"Searching... (Step {step_idx + 1}/{len(search_results)})"
            )
        else:
            status_text.set_text(f"Search Complete! Found {dqd_count} DQDs")

        return (
            list(grid_squares.values())
            + [status_text, current_indicator]
            + legend_patches
            + legend_texts
        )

    # Create animation
    # Add extra frames at start and end for viewing
    n_frames = len(search_results) + 20  # Start pause + search steps + end pause

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=400,  # 400ms per frame
        blit=True,
        repeat=True,
    )

    # Save as GIF
    print("Generating animation... (this may take a minute)")
    writer = PillowWriter(fps=2.5)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"✓ Animation saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    import sys

    output_file = sys.argv[1] if len(sys.argv) > 1 else "dqd_search_animation.gif"

    print("=" * 60)
    print("DQD Search Algorithm Animation Generator")
    print("=" * 60)
    print()
    print("This animation demonstrates:")
    print("  • Three-stage adaptive classification")
    print("  • Intelligent square selection strategy")
    print("  • ML-based DQD detection")
    print()

    create_animation(output_file)

    print()
    print("=" * 60)
    print("Done! Use this animation in your documentation.")
    print("=" * 60)
