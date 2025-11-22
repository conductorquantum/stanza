#!/usr/bin/env python3
"""Generate an animated GIF showing Double Quantum Dot potential tuning.

This animation demonstrates how barrier voltages tune a double quantum dot (DQD)
potential landscape, showing the characteristic 'W' shape formed by two parabolic
wells separated by a barrier.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Set style
plt.style.use("default")

# Constants
NUM_FRAMES = 70


def parabolic_well(
    x: np.ndarray, center: float, depth: float, width: float
) -> np.ndarray:
    """Generate a parabolic potential well."""
    return depth * ((x - center) / width) ** 2


def barrier_potential(
    x: np.ndarray, center: float, height: float, width: float
) -> np.ndarray:
    """Generate a Gaussian barrier."""
    return height * np.exp(-((x - center) ** 2) / (2 * width**2))


def create_dqd_potential(
    x: np.ndarray,
    left_barrier_height: float,
    middle_barrier_height: float,
    right_barrier_height: float,
) -> np.ndarray:
    """Create double quantum dot potential landscape.

    Args:
        x: Position array
        left_barrier_height: Height of left outer barrier
        middle_barrier_height: Height of middle (interdot) barrier
        right_barrier_height: Height of right outer barrier

    Returns:
        Potential energy array
    """
    # Two parabolic wells
    left_well = parabolic_well(x, center=-0.3, depth=0.5, width=0.25)
    right_well = parabolic_well(x, center=0.3, depth=0.5, width=0.25)

    # Take the minimum to create the base double-well structure
    base_potential = np.minimum(left_well, right_well)

    # Add barriers
    left_barrier = barrier_potential(
        x, center=-0.6, height=left_barrier_height, width=0.15
    )
    middle_barrier = barrier_potential(
        x, center=0.0, height=middle_barrier_height, width=0.12
    )
    right_barrier = barrier_potential(
        x, center=0.6, height=right_barrier_height, width=0.15
    )

    # Combine
    potential = base_potential + left_barrier + middle_barrier + right_barrier

    return potential


def create_animation(output_path: str = "dqd_potential_animation.gif"):
    """Create the DQD potential tuning animation."""

    # Position array
    x = np.linspace(-1.0, 1.0, 500)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Configure axes
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel("Position (a.u.)", fontsize=13, color="black", labelpad=10)
    ax.set_ylabel("Potential Energy (a.u.)", fontsize=13, color="black", labelpad=10)
    ax.tick_params(colors="black", labelsize=11)
    ax.grid(True, alpha=0.2, color="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    # Add labels for regions
    ax.text(
        -0.75,
        1.05,
        "Left\nReservoir",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        alpha=0.8,
    )
    ax.text(
        0.75,
        1.05,
        "Right\nReservoir",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        alpha=0.8,
    )

    # Status text
    status_text = fig.text(
        0.5, 0.95, "", ha="center", va="top", fontsize=12, color="black"
    )

    # Storage for plot elements
    potential_line = None
    potential_fill = None
    barrier_annotations = []
    dot_markers = []

    def init():
        """Initialize animation."""
        return []

    def update(frame):
        """Update animation frame."""
        nonlocal potential_line, potential_fill, barrier_annotations, dot_markers

        # Clear previous elements
        if potential_line:
            potential_line.remove()
        if potential_fill:
            potential_fill.remove()
        for ann in barrier_annotations:
            ann.remove()
        barrier_annotations = []
        for marker in dot_markers:
            marker.remove()
        dot_markers = []

        # Animation phases - two distinct sequential steps
        if frame < 15:
            # Phase 1: Initial state
            inner_progress = 0.0
            outer_progress = 0.0
            status_text.set_text("Initial Configuration")
        elif frame < 35:
            # Phase 2: Raise inner barrier (outer barriers stay constant)
            inner_progress = (frame - 15) / 20.0
            outer_progress = 0.0
            status_text.set_text("Step 1: Inner Barrier (B₁) ↑")
        elif frame < 55:
            # Phase 3: Lower outer barriers (inner barrier stays constant)
            inner_progress = 1.0
            outer_progress = (frame - 35) / 20.0
            status_text.set_text("Step 2: Outer Barriers (B₀, B₂) ↓")
        else:
            # Phase 4: Hold at final configuration
            inner_progress = 1.0
            outer_progress = 1.0
            status_text.set_text("Final Configuration")

        # Barrier heights as a function of progress
        # Outer barriers: start high (0.8) and decrease to low (0.3)
        outer_barrier_height = 0.8 - outer_progress * 0.5
        # Inner barrier: start low (0.2) and increase to high (0.7)
        middle_barrier_height = 0.2 + inner_progress * 0.5

        # Generate potential
        potential = create_dqd_potential(
            x,
            left_barrier_height=outer_barrier_height,
            middle_barrier_height=middle_barrier_height,
            right_barrier_height=outer_barrier_height,
        )

        # Plot potential
        (potential_line,) = ax.plot(
            x, potential, color="#0077cc", linewidth=3, alpha=0.9
        )

        # Fill under the curve
        potential_fill = ax.fill_between(x, 0, potential, color="#0077cc", alpha=0.2)

        # Find the actual minima positions for each quantum dot
        # Left dot: search in the left well region
        left_region_mask = (x >= -0.5) & (x <= -0.1)
        left_region_potential = potential[left_region_mask]
        left_region_x = x[left_region_mask]
        left_min_idx = np.argmin(left_region_potential)
        left_dot_x = left_region_x[left_min_idx]
        left_dot_y = left_region_potential[left_min_idx]

        # Right dot: search in the right well region
        right_region_mask = (x >= 0.1) & (x <= 0.5)
        right_region_potential = potential[right_region_mask]
        right_region_x = x[right_region_mask]
        right_min_idx = np.argmin(right_region_potential)
        right_dot_x = right_region_x[right_min_idx]
        right_dot_y = right_region_potential[right_min_idx]

        # Add red dots at the actual minima
        (dot1_marker,) = ax.plot(
            left_dot_x,
            left_dot_y,
            "o",
            color="#ff0000",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=3,
        )
        (dot2_marker,) = ax.plot(
            right_dot_x,
            right_dot_y,
            "o",
            color="#ff0000",
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=3,
        )
        dot_markers.extend([dot1_marker, dot2_marker])

        # Add barrier height annotations
        # Left barrier
        left_ann = ax.annotate(
            f"B₀: {outer_barrier_height:.2f}",
            xy=(-0.6, outer_barrier_height + 0.15),
            fontsize=10,
            color="#cc8800",
            fontweight="bold",
            ha="center",
        )
        barrier_annotations.append(left_ann)

        # Middle barrier
        middle_ann = ax.annotate(
            f"B₁: {middle_barrier_height:.2f}",
            xy=(0.0, middle_barrier_height + 0.15),
            fontsize=10,
            color="#cc8800",
            fontweight="bold",
            ha="center",
        )
        barrier_annotations.append(middle_ann)

        # Right barrier
        right_ann = ax.annotate(
            f"B₂: {outer_barrier_height:.2f}",
            xy=(0.6, outer_barrier_height + 0.15),
            fontsize=10,
            color="#cc8800",
            fontweight="bold",
            ha="center",
        )
        barrier_annotations.append(right_ann)

        return (
            [potential_line, potential_fill, status_text]
            + barrier_annotations
            + dot_markers
        )

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=NUM_FRAMES,
        interval=75,  # 75ms per frame
        blit=False,
        repeat=True,
    )

    # Save as GIF
    print("Generating animation... (this may take a minute)")
    writer = PillowWriter(fps=13.33)  # ~75ms per frame
    anim.save(output_path, writer=writer, dpi=100)
    print(f"✓ Animation saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    import sys

    output_file = sys.argv[1] if len(sys.argv) > 1 else "dqd_potential_animation.gif"

    print("=" * 60)
    print("Double Quantum Dot Potential Animation Generator")
    print("=" * 60)
    print()
    print("This animation demonstrates:")
    print("  • Double quantum dot 'W' potential structure")
    print("  • Inner barrier tuning (tunnel coupling)")
    print("  • Outer barrier tuning (reservoir coupling)")
    print()

    create_animation(output_file)

    print()
    print("=" * 60)
    print("Done! Use this animation in your documentation.")
    print("=" * 60)
