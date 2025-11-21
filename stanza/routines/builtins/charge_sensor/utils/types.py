"""Type definitions for charge sensor routines."""

from dataclasses import dataclass

import numpy as np

from stanza.routines.builtins.utils.peak_fitting import FittedPeak


@dataclass
class StabilityMeasurement:
    """
    Result from a 2-minute stability measurement at a peak's max-gradient point.

    Contains time-series current data, noise metrics, and the computed voltage noise
    score that quantifies peak position stability.
    """

    peak_index: int  # Index of the peak in the original peak list
    peak_voltage: float  # Center voltage of the peak (V)
    max_gradient_voltage: (
        float  # Voltage at maximum gradient where measurement was taken (V)
    )
    time_array: np.ndarray  # Time points during the 2-minute hold (seconds)
    current_array: np.ndarray  # Measured current values over time (A)
    current_mean: float  # Mean current during hold (A)
    current_std: float  # Standard deviation of current (A) - σᵢ
    local_slope: float  # Local dI/dV from linear fit around max-gradient point (A/V)
    voltage_noise: float  # Effective voltage jitter: σᵥ = σᵢ / |dI/dV| (V)
    stability_score: float | None = (
        None  # Normalized stability score (higher is better)
    )


@dataclass
class StablePeakCandidate:
    """
    A peak candidate with both original quality score and stability measurements.

    Combines the initial peak quality metrics with stability measurements to
    enable selection of the most stable peak for charge sensing.
    """

    fitted_peak: FittedPeak  # Original fitted peak from multi-model fitting
    original_score: float  # Original quality score from peak fitting
    stability_measurement: StabilityMeasurement  # Stability data from 2-min hold
    combined_score: float | None = (
        None  # Final weighted score (50% original + 50% stability)
    )


@dataclass
class PeakWindowSweepOutput:
    """
    Result from a single-window barrier sweep with Lorentzian peak fitting.

    This model contains the best fitted peak from a refined barrier sweep,
    along with the full trace data and neighboring peak information for context.
    Used for both baseline measurements and gate compensation sweeps.
    """

    # Best fitted peak information
    best_peak: FittedPeak  # The best fitted peak selected from this sweep

    # Full trace data
    aggregated_voltages: np.ndarray  # Complete voltage array from the sweep
    aggregated_currents: np.ndarray  # Complete current array from the sweep

    # Metadata
    classification: bool  # Whether Coulomb blockade was detected
    score: float  # Classification confidence score
    num_peaks: int  # Total number of peaks detected and fitted

    # Neighboring peak context (for park point calculations)
    prev_peak_voltage: float | None = None  # Voltage of spatially previous peak
    next_peak_voltage: float | None = None  # Voltage of spatially next peak


@dataclass
class SensorDotPlungerSweepOutput:
    """
    Result from a single sensor plunger voltage sweep measurement.

    This stores results from continuous aggregation: each result contains the cumulative
    aggregated trace up to that window, the interpolated 128-point trace used for
    classification, and peak indices in the aggregated trace space.
    """

    sensor_plunger_voltage: float  # Start voltage of the window
    classification: bool  # Coulomb blockade detected (on interpolated trace)
    score: float  # Classification confidence score
    peak_indices: list[int]  # Detected peak locations in AGGREGATED trace space
    num_peaks: int  # Number of peaks detected
    aggregated_voltages: np.ndarray  # Full aggregated voltage array up to this window
    aggregated_currents: np.ndarray  # Full aggregated current array up to this window

    # Park point search range boundaries (voltage positions of neighboring peaks)
    best_peak_voltage: float | None = None  # Voltage of the selected best peak center
    best_peak_max_gradient_voltage: float | None = (
        None  # Voltage at max gradient (steepest slope) of best peak
    )
    # Voltage of spatially previous peak (lower peak_idx), None = use trace start
    prev_peak_voltage: float | None = None
    # Voltage of spatially next peak (higher peak_idx), None = use trace end
    next_peak_voltage: float | None = None
