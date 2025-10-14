from dataclasses import dataclass

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from stanza.analysis.preprocessing import normalize

# Constants for parameter bounds computation
DEFAULT_C_MARGIN = 10.0
DEFAULT_BOUNDS = [(1e-8, 1.0), (-20.0, 20.0), (-10.0, 10.0)]


@dataclass
class PinchoffFitResult:
    """Result of pinchoff curve fitting.

    Attributes:
        v_pinch_off: Pinchoff voltage (where device turns off)
        v_transition: Transition voltage (midpoint of transition)
        v_cut_off: Conducting voltage (where device is fully on)
        popt: Fitted parameters [a, b, c] for pinchoff_curve in normalized space
        pcov: Covariance matrix of fitted parameters
        v_min: Minimum voltage value used for normalization
        v_max: Maximum voltage value used for normalization
        i_min: Minimum current value used for normalization
        i_max: Maximum current value used for normalization

    Note:
        popt parameters are fitted in normalized [0,1] space. To generate the
        fitted curve in original space, use the fit_curve() method or manually
        normalize voltages before applying pinchoff_curve().
    """

    v_pinch_off: float | None
    v_transition: float | None
    v_cut_off: float | None
    popt: np.ndarray
    pcov: np.ndarray

    v_min: float = 0.0
    v_max: float = 1.0
    i_min: float = 0.0
    i_max: float = 1.0

    def fit_curve(self, voltages: np.ndarray) -> np.ndarray:
        """Generate fitted curve in original voltage/current space.

        Args:
            voltages: Voltage array in original space

        Returns:
            Fitted current values in original space
        """
        v_norm = (voltages - self.v_min) / (self.v_max - self.v_min)
        i_norm = pinchoff_curve(v_norm, *self.popt)
        return i_norm * (self.i_max - self.i_min) + self.i_min


def pinchoff_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Smooth pinchoff curve with coefficients a, b, c

    From: Darulová, J. et al. Autonomous tuning and charge state
        detection of gate defined quantum dots. Physical Review,
        Applied 13, 054005 (2019).

    Args:
        x (np.ndarray): Input voltage
        a (float): Amplitude
        b (float): Slope
        c (float): Offset

    Returns:
        np.ndarray: Pinchoff current f(x) output
    """
    return a * (1 + np.tanh(b * x + c))


def derivative_extrema_indices(x: np.ndarray, y: np.ndarray) -> tuple[int, int, int]:
    """
    Return the indices of key voltages for pinchoff curves.

    v_pinch_off corresponds to the pinchoff state (low current)
    v_cut_off corresponds to the conducting state (high current)
    v_transition corresponds to the transition (steepest slope)

    Args:
        x (np.ndarray): Input x values
        y (np.ndarray): Input y values

    Returns:
        Tuple[int, int, int]:
            (transition_v_ind, conducting_v_ind, pinchoff_v_ind)
    """
    grad = np.gradient(y, x)
    second = np.gradient(grad, x)

    # Transition voltage: at maximum absolute slope (steepest point)
    transition_v_ind = int(np.argmax(np.abs(grad)))

    # Find the two inflection points (extrema of second derivative)
    imin_second = int(np.argmin(second))
    imax_second = int(np.argmax(second))

    if y[imin_second] < y[imax_second]:
        pinchoff_v_ind = imin_second
        conducting_v_ind = imax_second
    else:
        pinchoff_v_ind = imax_second
        conducting_v_ind = imin_second

    return transition_v_ind, conducting_v_ind, pinchoff_v_ind


def _compute_initial_params(v_norm: np.ndarray, i_norm: np.ndarray) -> np.ndarray:
    """Compute initial parameter estimates from normalized data.

    For inverted curves (decreasing with voltage), we use negative b values
    to represent the inversion while keeping amplitude positive.
    """
    i_range = max(np.ptp(i_norm), 1.0)
    v_range = max(np.ptp(v_norm), 1.0)
    v_center = (v_norm.min() + v_norm.max()) / 2.0

    slope_sign = 1.0 if i_norm[-1] >= i_norm[0] else -1.0

    return np.array(
        [
            i_range / 2.0,  # a: amplitude (always positive in normalized space)
            slope_sign * 4.0 / v_range,  # b: slope (negative for inverted curves)
            -slope_sign * 4.0 * v_center / v_range,  # c: offset
        ]
    )


def _compute_parameter_bounds(
    v_norm: np.ndarray, i_norm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute robust parameter bounds from normalized data."""
    i_range = max(np.ptp(i_norm), 1.0)
    v_range = max(np.ptp(v_norm), 1.0)
    v_min, v_max = v_norm.min(), v_norm.max()

    # Amplitude bounds
    a_bounds = (max(0.01 * i_range, 1e-8), max(2.0 * i_range, 1.0))

    # Slope bounds (allow negative values for inverted curves)
    max_abs_b = min(20.0 / v_range, 100.0)
    b_bounds = (-max_abs_b, max_abs_b)

    # Offset bounds with margin
    c_bounds = (
        -b_bounds[1] * v_max - DEFAULT_C_MARGIN,
        -b_bounds[0] * v_min + DEFAULT_C_MARGIN,
    )

    # Validate bounds
    bounds_list = [a_bounds, b_bounds, c_bounds]

    for i, (lower, upper) in enumerate(bounds_list):
        if lower >= upper:
            bounds_list[i] = DEFAULT_BOUNDS[i]

    return np.array([b[0] for b in bounds_list]), np.array([b[1] for b in bounds_list])


def _map_index_to_voltage(index: int, voltages: np.ndarray) -> float | None:
    """Map an array index to the corresponding voltage value.

    Args:
        index: Array index
        voltages: Array of voltage values

    Returns:
        Voltage at the given index, or None if index is out of bounds
    """
    return voltages[index] if index < len(voltages) else None


def fit_pinchoff_parameters(
    voltages: np.ndarray, currents: np.ndarray, sigma: float = 2.0
) -> PinchoffFitResult:
    """Fit the pinchoff parameters a, b, c of the pinchoff curve, and returns
    the pinchoff, transition, and conducting voltages.

    Args:
        voltages (np.ndarray): Input voltages
        currents (np.ndarray): Input currents
        sigma (float): Gaussian filter bandwidth for smoothing

    Returns:
        PinchoffFitResult containing fitted voltages and parameters
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is not installed. Install with: pip install cq-stanza[routines]"
        )
    filtered_current = gaussian_filter(currents, sigma=sigma)
    v_norm = normalize(voltages)
    i_norm = normalize(filtered_current)

    p0 = _compute_initial_params(v_norm, i_norm)
    bounds = _compute_parameter_bounds(v_norm, i_norm)

    popt, pcov = curve_fit(
        pinchoff_curve, v_norm, i_norm, p0=p0, bounds=bounds, maxfev=2000
    )

    i_fit = pinchoff_curve(v_norm, *popt)

    transition_v_ind, conducting_v_ind, pinchoff_v_ind = derivative_extrema_indices(
        v_norm, i_fit
    )

    return PinchoffFitResult(
        v_pinch_off=_map_index_to_voltage(pinchoff_v_ind, voltages),
        v_transition=_map_index_to_voltage(transition_v_ind, voltages),
        v_cut_off=_map_index_to_voltage(conducting_v_ind, voltages),
        popt=popt,
        pcov=pcov,
        v_min=voltages.min(),
        v_max=voltages.max(),
        i_min=filtered_current.min(),
        i_max=filtered_current.max(),
    )
