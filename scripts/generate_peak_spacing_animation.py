#!/usr/bin/env python3
"""Generate an animated GIF showing the Peak Spacing Computation algorithm.

Peak spacing computation showing random diagonal sweeps (left) and Coulomb peak
detection (right). Inter-peak spacings are measured and the median determines
the grid square size.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Set style
plt.style.use("default")

# Constants
NUM_SWEEPS = 8
TRACE_POINTS = 128


@dataclass
class CoulombPeak:
    """A single Coulomb peak."""

    position: float  # Position along trace (0-1)
    height: float
    width: float


@dataclass
class Sweep:
    """A diagonal voltage sweep."""

    start: np.ndarray  # (2,) start position
    end: np.ndarray  # (2,) end position
    peaks: list[CoulombPeak]  # Coulomb peaks in this sweep
    has_sufficient_peaks: (
        bool  # Whether this sweep has enough peaks for spacing calculation
    )


def lorentzian(x: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    """Generate a Lorentzian peak - characteristic of Coulomb peaks."""
    return height / (1 + ((x - center) / width) ** 2)


def generate_sweep(
    v_p1_min: float,
    v_p1_max: float,
    v_p2_min: float,
    v_p2_max: float,
) -> Sweep:
    """Generate a random diagonal sweep with Coulomb peaks."""
    # Random start position
    start = np.array(
        [
            np.random.uniform(v_p1_min + 0.1, v_p1_max - 0.1),
            np.random.uniform(v_p2_min + 0.1, v_p2_max - 0.1),
        ]
    )

    # Random direction
    angle = np.random.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])

    # Random length
    sweep_length = np.random.uniform(0.15, 0.35)
    end = start + direction * sweep_length

    # Clip to bounds
    end[0] = np.clip(end[0], v_p1_min, v_p1_max)
    end[1] = np.clip(end[1], v_p2_min, v_p2_max)

    # Some sweeps have insufficient or no peaks (about 30% failure rate)
    rand = np.random.random()
    if rand < 0.15:  # 15% have no peaks (no Coulomb blockade)
        num_peaks = 0
        has_sufficient = False
    elif rand < 0.30:  # 15% have insufficient peaks (1-2 peaks)
        num_peaks = np.random.randint(1, 3)
        has_sufficient = False
    else:  # 70% have sufficient peaks (3-5 peaks)
        num_peaks = np.random.randint(3, 6)
        has_sufficient = True

    peaks = []
    for i in range(num_peaks):
        position = 0.2 + (0.6 / (num_peaks - 1)) * i if num_peaks > 1 else 0.5
        position += np.random.uniform(-0.05, 0.05)  # Add jitter
        position = np.clip(position, 0.15, 0.85)

        height = np.random.uniform(0.5, 0.8)
        width = np.random.uniform(0.012, 0.020)

        peaks.append(CoulombPeak(position, height, width))

    return Sweep(start, end, peaks, has_sufficient)


def generate_current_trace(
    peaks: list[CoulombPeak],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate current trace with Coulomb peaks."""
    x = np.linspace(0, 1, TRACE_POINTS)

    # Baseline current
    current = np.ones_like(x) * 0.1

    # Add each Coulomb peak
    peak_indices = []
    for peak in peaks:
        current += lorentzian(x, peak.position, peak.width, peak.height)
        # Find the index of the peak maximum
        peak_idx = np.argmin(np.abs(x - peak.position))
        peak_indices.append(peak_idx)

    # Add realistic noise
    current += np.random.normal(0, 0.015, TRACE_POINTS)

    # Clip to valid range
    current = np.clip(current, 0, 1)

    return x, current, peak_indices


def create_animation(output_path: str = "peak_spacing_animation.gif"):
    """Create the peak spacing computation animation."""

    # Define voltage ranges (matching DQD search)
    v_p1_min, v_p1_max = -2.0, -1.5
    v_p2_min, v_p2_max = -1.9, -1.4

    # Generate all sweeps
    np.random.seed(42)
    sweeps = [
        generate_sweep(v_p1_min, v_p1_max, v_p2_min, v_p2_max)
        for _ in range(NUM_SWEEPS)
    ]

    # Create figure with proper spacing
    fig = plt.figure(figsize=(14, 6.5))

    # Use gridspec for better control over layout
    gs = fig.add_gridspec(
        1, 2, left=0.08, right=0.98, bottom=0.12, top=0.82, wspace=0.25
    )

    # Left: Voltage space with sweeps
    ax_voltage = fig.add_subplot(gs[0, 0])
    ax_voltage.set_xlim(v_p1_min, v_p1_max)
    ax_voltage.set_ylim(v_p2_min, v_p2_max)
    ax_voltage.set_aspect("equal", adjustable="box")
    ax_voltage.set_xlabel(
        "Plunger Gate 1 Voltage (V)", fontsize=12, color="black", labelpad=8
    )
    ax_voltage.set_ylabel(
        "Plunger Gate 2 Voltage (V)", fontsize=12, color="black", labelpad=8
    )
    ax_voltage.set_title("Random Diagonal Sweeps", fontsize=14, color="black", pad=12)
    ax_voltage.tick_params(colors="black", labelsize=10)
    ax_voltage.grid(True, alpha=0.2, color="black", linewidth=0.5)

    # Right: Current trace
    ax_trace = fig.add_subplot(gs[0, 1])
    ax_trace.set_xlim(0, 1)
    ax_trace.set_ylim(0, 1.1)
    ax_trace.set_xlabel(
        "Diagonal Sweep Trace (V)", fontsize=12, color="black", labelpad=8
    )
    ax_trace.set_ylabel("Current (a.u.)", fontsize=12, color="black", labelpad=8)
    ax_trace.set_title("Coulomb Peak Detection", fontsize=14, color="black", pad=12)
    ax_trace.tick_params(colors="black", labelsize=10)
    ax_trace.grid(True, alpha=0.2, color="black", linewidth=0.5)

    # Status text with proper spacing from plots
    status_text = fig.text(
        0.5, 0.92, "", ha="center", va="top", fontsize=13, color="black"
    )

    # Storage for drawn elements
    completed_sweep_lines = []
    current_sweep_line = None
    current_sweep_segments = []
    trace_line = None
    peak_markers = []

    def init():
        """Initialize animation."""
        return []

    def update(frame):
        """Update animation frame."""
        nonlocal current_sweep_line, trace_line, peak_markers, current_sweep_segments

        # Clear current frame elements
        if trace_line:
            trace_line.remove()
            trace_line = None
        for marker in peak_markers:
            marker.remove()
        peak_markers = []
        if current_sweep_line:
            current_sweep_line.remove()
            current_sweep_line = None

        # Initialization frame
        if frame == 0:
            status_text.set_text("Peak Spacing Computation")
            return completed_sweep_lines + [status_text]

        # Calculate which sweep and progress within sweep
        frames_per_sweep = 25  # Frames to draw each sweep
        frames_per_pause = 5  # Pause between sweeps

        total_frames_per_sweep = frames_per_sweep + frames_per_pause
        sweep_idx = (frame - 1) // total_frames_per_sweep
        frame_in_sweep = (frame - 1) % total_frames_per_sweep

        # Check if we're done
        if sweep_idx >= len(sweeps):
            successful = sum(1 for s in sweeps if s.has_sufficient_peaks)
            status_text.set_text(
                f"Complete! {successful}/{len(sweeps)} sweeps successful"
            )
            ax_trace.clear()
            ax_trace.axis("off")
            ax_trace.text(
                0.5,
                0.5,
                f"{len(sweeps)} Random Sweeps\n{successful} Successful",
                ha="center",
                va="center",
                fontsize=20,
                color="#00aa00",
                transform=ax_trace.transAxes,
            )
            return completed_sweep_lines + [status_text]

        sweep = sweeps[sweep_idx]

        # Update status
        status_text.set_text(
            f"Sweep {sweep_idx + 1}/{len(sweeps)} | Detecting Coulomb Peaks"
        )

        # Animate drawing the sweep line
        if frame_in_sweep < frames_per_sweep:
            # Draw sweep incrementally
            progress = (frame_in_sweep + 1) / frames_per_sweep

            # Draw partial sweep line
            partial_end = sweep.start + (sweep.end - sweep.start) * progress
            (current_sweep_line,) = ax_voltage.plot(
                [sweep.start[0], partial_end[0]],
                [sweep.start[1], partial_end[1]],
                color="#0088ff",
                linewidth=3,
                alpha=0.9,
                zorder=2,
            )

            # Generate and show current trace up to this point
            x, current, peak_indices = generate_current_trace(sweep.peaks)

            # Only show trace up to current progress
            cutoff_idx = int(progress * TRACE_POINTS)
            # Use cyan for actively drawing sweep
            (trace_line,) = ax_trace.plot(
                x[:cutoff_idx],
                current[:cutoff_idx],
                color="#0088ff",
                linewidth=2.5,
                alpha=0.9,
            )

            # Show detected peaks if we've passed them
            for peak_idx in peak_indices:
                if peak_idx < cutoff_idx:
                    (marker,) = ax_trace.plot(
                        x[peak_idx],
                        current[peak_idx],
                        "x",
                        color="#ff0000",
                        markersize=15,
                        markeredgewidth=3,
                        zorder=3,
                    )
                    peak_markers.append(marker)

            return (
                completed_sweep_lines
                + [current_sweep_line, trace_line, status_text]
                + peak_markers
            )

        else:
            # Pause frame - show complete sweep and trace
            # Add completed sweep to permanent collection
            if frame_in_sweep == frames_per_sweep:
                # Color based on whether sweep was successful
                sweep_color = "#00aa00" if sweep.has_sufficient_peaks else "#cc0000"
                (completed_line,) = ax_voltage.plot(
                    [sweep.start[0], sweep.end[0]],
                    [sweep.start[1], sweep.end[1]],
                    color=sweep_color,
                    linewidth=2,
                    alpha=0.6,
                    zorder=1,
                )
                completed_sweep_lines.append(completed_line)

            # Show full trace with all peaks
            x, current, peak_indices = generate_current_trace(sweep.peaks)
            # Color trace based on success
            trace_color = "#00aa00" if sweep.has_sufficient_peaks else "#cc0000"
            (trace_line,) = ax_trace.plot(
                x, current, color=trace_color, linewidth=2.5, alpha=0.9
            )

            # Show all detected peaks (only if there are peaks)
            for peak_idx in peak_indices:
                (marker,) = ax_trace.plot(
                    x[peak_idx],
                    current[peak_idx],
                    "x",
                    color="#ff0000",
                    markersize=15,
                    markeredgewidth=3,
                    zorder=3,
                )
                peak_markers.append(marker)

            return completed_sweep_lines + [trace_line, status_text] + peak_markers

    # Create animation
    total_sweeps = len(sweeps)
    frames_per_sweep = 25
    frames_per_pause = 5
    n_frames = (
        1 + total_sweeps * (frames_per_sweep + frames_per_pause) + 15
    )  # init + sweeps + end pause

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=50,  # 50ms per frame
        blit=False,
        repeat=True,
    )

    # Save as GIF
    print("Generating animation... (this may take a minute)")
    writer = PillowWriter(fps=20)  # 50ms per frame = 20 fps
    anim.save(output_path, writer=writer, dpi=100)
    print(f"✓ Animation saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    import sys

    output_file = sys.argv[1] if len(sys.argv) > 1 else "peak_spacing_animation.gif"

    print("=" * 60)
    print("Peak Spacing Computation Animation Generator")
    print("=" * 60)
    print()
    print("This animation demonstrates:")
    print("  • Random diagonal sweeps through voltage space")
    print("  • Coulomb blockade peak detection")
    print("  • Peak spacing measurement")
    print()

    create_animation(output_file)

    print()
    print("=" * 60)
    print("Done! Use this animation in your documentation.")
    print("=" * 60)
