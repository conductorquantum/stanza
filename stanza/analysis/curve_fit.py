from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


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


def derivative_extrema_indices(
    x: np.ndarray, y: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Return the indices of:
        - minimum of the first derivative
        - maximum of the first derivative
        - minimum of the second derivative
        - maximum of the second derivative

    Args:
        x (np.ndarray): Input x values
        y (np.ndarray): Input y values

    Returns:
        Tuple[int, int, int, int]:
            (imin_grad, imax_grad, imin_second, imax_second)
    """
    grad = np.gradient(y, x)
    second = np.gradient(grad, x)

    imin_grad = int(np.argmin(grad))
    imax_grad = int(np.argmax(grad))
    imin_second = int(np.argmin(second))
    imax_second = int(np.argmax(second))

    return imin_grad, imax_grad, imin_second, imax_second


def normalize(a: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]. Constant arrays map to 0."""
    amin, amax = a.min(), a.max()
    return np.zeros_like(a, dtype=float) if amin == amax else (a - amin) / (amax - amin)


def _compute_initial_params(v_norm: np.ndarray, i_norm: np.ndarray) -> np.ndarray:
    """Compute initial parameter estimates from normalized data."""
    i_range = max(np.ptp(i_norm), 1.0)
    v_range = max(np.ptp(v_norm), 1.0)
    v_center = (v_norm.min() + v_norm.max()) / 2.0

    return np.array(
        [
            i_range / 2.0,  # a: amplitude
            4.0 / v_range,  # b: slope
            -4.0 * v_center / v_range,  # c: offset
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

    # Slope bounds
    b_bounds = (max(0.1 / v_range, 0.1), min(20.0 / v_range, 100.0))

    # Offset bounds with margin
    c_margin = 10.0
    c_bounds = (-b_bounds[1] * v_max - c_margin, -b_bounds[0] * v_min + c_margin)

    # Validate bounds
    bounds_list = [a_bounds, b_bounds, c_bounds]
    defaults = [(1e-8, 1.0), (0.1, 20.0), (-10.0, 10.0)]

    for i, (lower, upper) in enumerate(bounds_list):
        if lower >= upper:
            bounds_list[i] = defaults[i]

    return np.array([b[0] for b in bounds_list]), np.array([b[1] for b in bounds_list])


def fit_pinchoff_parameters(
    voltages: np.ndarray, currents: np.ndarray, _sigma: float = 0.01
) -> dict[str, Any]:
    """Fit the pinchoff parameters a, b, c of the pinchoff curve, and returns the pinchoff, transition, and conducting voltages.

    Args:
        voltages (np.ndarray): Input voltages
        currents (np.ndarray): Input currents
        _sigma (float): sigma parameter for the gaussian filter

    Returns:
        Dict[str, Any]:
            (vp, vt, vc, popt, pcov)
    """
    pinchoff_v = None
    transition_v = None
    conducting_v = None

    filtered_current = gaussian_filter(currents, sigma=_sigma)
    v_norm = normalize(voltages)
    i_norm = normalize(filtered_current)

    p0 = _compute_initial_params(v_norm, i_norm)
    bounds = _compute_parameter_bounds(v_norm, i_norm)

    popt, pcov = curve_fit(
        pinchoff_curve, v_norm, i_norm, p0=p0, bounds=bounds, maxfev=2000
    )

    i_fit = pinchoff_curve(v_norm, *popt)

    _, transition_v_ind, conducting_v_ind, pinchoff_v_ind = derivative_extrema_indices(
        v_norm, i_fit
    )

    pinchoff_v = voltages[pinchoff_v_ind] if pinchoff_v_ind < len(voltages) else None
    transition_v = (
        voltages[transition_v_ind] if transition_v_ind < len(voltages) else None
    )
    conducting_v = (
        voltages[conducting_v_ind] if conducting_v_ind < len(voltages) else None
    )

    return {
        "vp": pinchoff_v,
        "vt": transition_v,
        "vc": conducting_v,
        "popt": popt,
        "pcov": pcov,
    }
