from collections.abc import Callable
from logging import getLogger

import numpy as np

try:
    from scipy.optimize import curve_fit

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from dataclasses import dataclass

from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession

logger = getLogger(__name__)


@dataclass
class ModelFitResult:
    """
    Detailed fit results for a single peak model (Lorentzian, sech², or pseudo-Voigt).

    Stores fit parameters, quality metrics, physical parameters, and residual analysis
    for one model fit to a Coulomb blockade peak.
    """

    # Model identification
    model_name: str  # Model type: 'Lorentzian', 'sech2', or 'Voigt'

    # Fit parameters
    amplitude: float  # Peak amplitude above baseline
    center_idx: float  # Peak center position in window coordinates
    width: float  # Peak width parameter (HWHM for Lorentzian)
    offset: float  # Baseline offset

    # Quality metrics
    r_squared: float  # R-squared goodness of fit
    rmse: float  # Root mean squared error
    aicc: float  # Corrected Akaike Information Criterion (lower is better)

    # Physical parameters
    fwhm: float  # Full-width at half-maximum
    area: float  # Integrated peak area (numerical)

    # Residual analysis
    skew_resid: float  # Skewness of residuals (asymmetry indicator)

    # Voigt mixing parameter (optional, only for Voigt model)
    eta: float | None = None  # Voigt mixing parameter (0=Gaussian, 1=Lorentzian)


@dataclass
class FittedPeak:
    """
    Multi-model fit information for a single Coulomb blockade peak.

    Stores results from fitting three peak models (Lorentzian, sech², pseudo-Voigt),
    selects best by AICc, and provides comprehensive quality metrics for automated
    peak ranking and charge sensor operating point selection.
    """

    # Best model selection
    best_model: str  # Best-fit model name selected by minimum AICc: 'Lorentzian', 'sech2', or 'Voigt'

    # All three model fits (complete details for each)
    lorentzian_fit: ModelFitResult  # Lorentzian model fit results
    sech2_fit: ModelFitResult  # Hyperbolic secant squared model fit results
    voigt_fit: ModelFitResult  # Pseudo-Voigt model fit results

    # Sensitivity metrics (charge sensor performance at operating point)
    sensitivity: (
        float  # Maximum gradient at steepest slope - charge sensing sensitivity
    )
    sensitivity_voltage: float  # Voltage at maximum gradient (optimal operating point)

    # Position information (from best model)
    peak_idx: int  # Original peak index in aggregated trace
    peak_voltage: float  # Voltage at peak center (from best model)
    window_currents: np.ndarray  # Currents in the window
    window_voltages: np.ndarray  # Voltages in the window

    # Overall quality score for peak ranking
    quality_score: float | None = (
        None  # Peak quality score (higher is better): good fit, narrow, symmetric, steep gradient
    )
    sensitivity_score: float | None = (
        None  # Normalized sensitivity score (0-1), set after all peaks fitted
    )


def lorentzian(
    x: np.ndarray, amplitude: float, center: float, width: float, offset: float
) -> np.ndarray:
    """
    Lorentzian (Cauchy) distribution.

    Args:
        x: Independent variable (voltage)
        amplitude: Peak height above offset
        center: Peak center position
        width: Half-width at half-maximum (HWHM)
        offset: Baseline offset
    """
    return amplitude / (1 + ((x - center) / width) ** 2) + offset


def sech_squared(
    x: np.ndarray, amplitude: float, center: float, width: float, offset: float
) -> np.ndarray:
    """
    Hyperbolic secant squared distribution.

    This model is particularly suited for tunneling resonances and can better
    capture peaks with sharper central regions and broader wings than Lorentzian.

    Args:
        x: Independent variable (voltage)
        amplitude: Peak height above offset
        center: Peak center position
        width: Width parameter (controls peak sharpness)
        offset: Baseline offset
    """
    # Avoid division by zero
    if width == 0:
        width = np.finfo(float).eps

    u = (x - center) / width
    # Use numerically stable formula to avoid overflow
    # For large |u|, sech(u) ≈ 2*exp(-|u|), so sech²(u) ≈ 4*exp(-2|u|)
    # Clamp u to prevent overflow in cosh
    u_clamped = np.clip(u, -700, 700)  # cosh(700) is near float64 max
    # Compute sech first, then square to avoid overflow in cosh²
    sech = 1 / np.cosh(u_clamped)
    sech_sq = sech**2

    # For values beyond the clamp, use exponential approximation
    large_mask = np.abs(u) > 700
    if np.any(large_mask):
        sech_sq[large_mask] = 4 * np.exp(-2 * np.abs(u[large_mask]))

    return amplitude * sech_sq + offset


def pseudo_voigt(
    x: np.ndarray,
    amplitude: float,
    center: float,
    width: float,
    offset: float,
    eta: float,
) -> np.ndarray:
    """
    Pseudo-Voigt profile with fitted mixing parameter.

    Linear combination of Gaussian and Lorentzian profiles, allowing the fit
    to adapt to peak shapes ranging from pure Gaussian (eta=0) to pure Lorentzian (eta=1).
    Particularly useful for peaks with intermediate character or thermal broadening.

    Args:
        x: Independent variable (voltage)
        amplitude: Peak height above offset
        center: Peak center position
        width: Width parameter
        offset: Baseline offset
        eta: Mixing parameter (0=pure Gaussian, 1=pure Lorentzian)
    """
    # Gaussian component
    gaussian = np.exp(-np.log(2) * ((x - center) / width) ** 2)
    # Lorentzian component
    lorentzian_component = 1 / (1 + ((x - center) / width) ** 2)
    # Mix with fitted eta parameter
    result: np.ndarray = (
        amplitude * (eta * lorentzian_component + (1 - eta) * gaussian) + offset
    )
    return result


def calculate_rmse(residuals: np.ndarray) -> float:
    """
    Calculate root mean squared error.

    Args:
        residuals: Array of residuals (observed - predicted)

    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean(residuals**2)))


def calculate_aicc(n: int, k: int, residuals: np.ndarray) -> float:
    """
    Calculate corrected Akaike Information Criterion.

    Lower AICc values indicate better model fit with parsimony penalty.
    The correction term accounts for small sample sizes.

    Args:
        n: Number of data points
        k: Number of fitted parameters (4 for Lorentzian/sech², 5 for Voigt)
        residuals: Array of residuals (observed - predicted)

    Returns:
        AICc value (lower is better)
    """
    rss = np.sum(residuals**2)
    # Avoid log(0) and division by zero
    if rss <= 0 or n <= k + 1:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    # Small sample size correction
    correction = (2 * k * (k + 1)) / (n - k - 1)
    return float(aic + correction)


def calculate_skew_residuals(residuals: np.ndarray) -> float:
    """
    Calculate skewness of residuals.

    Measures asymmetry in the residuals. Values near 0 indicate symmetric
    residuals (good fit). Positive values indicate tail on right side,
    negative values indicate tail on left side.

    Args:
        residuals: Array of residuals (observed - predicted)

    Returns:
        Skewness value (0 = symmetric)
    """
    from scipy.stats import skew

    return float(skew(residuals))


def calculate_fwhm(
    model_func: Callable[..., np.ndarray],
    params: tuple[float, ...],
    model_name: str,
    x_range: np.ndarray,
) -> float:
    """
    Calculate full-width at half-maximum for a fitted peak.

    Uses model-specific analytical formulas where available, or numerical
    calculation for complex models.

    Args:
        model_func: The model function (lorentzian, sech_squared, pseudo_voigt)
        params: Fitted parameters (amplitude, center, width, offset, [eta])
        model_name: Model identifier ('Lorentzian', 'sech2', 'Voigt')
        x_range: X-axis range for numerical calculation if needed

    Returns:
        FWHM value in same units as x
    """
    amplitude, center, width, offset = params[:4]

    if model_name == "Lorentzian":
        # For Lorentzian, FWHM = 2 * HWHM
        return float(2 * width)

    elif model_name == "sech2":
        # For sech²: FWHM = 2 * width * arccosh(sqrt(2))
        return float(2 * width * 1.7627471740390859)  # arccosh(sqrt(2))

    elif model_name == "Voigt":
        # For pseudo-Voigt, calculate numerically
        # Find points where function equals amplitude/2 + offset
        half_max = amplitude / 2 + offset
        y_vals = model_func(x_range, *params)

        # Find indices where curve crosses half maximum
        above_half = y_vals > half_max
        if not np.any(above_half):
            return float(2 * width)  # Fallback to width approximation

        # Find first and last crossing points
        crossings = np.where(np.diff(above_half.astype(int)) != 0)[0]
        if len(crossings) >= 2:
            # Distance between first and last crossing
            return float(x_range[crossings[-1]] - x_range[crossings[0]])
        else:
            return float(2 * width)  # Fallback

    return float(2 * width)  # Default fallback


def calculate_area(fitted_curve: np.ndarray, x_spacing: float) -> float:
    """
    Calculate integrated peak area using trapezoidal rule.

    Args:
        fitted_curve: Array of fitted y values
        x_spacing: Spacing between x points

    Returns:
        Integrated area
    """
    return float(np.trapz(fitted_curve, dx=x_spacing))


def calculate_quality_score(
    r_squared: float, rmse: float, y_max: float, skew: float, sensitivity_score: float
) -> float:
    """
    Calculate overall peak quality score with gradient-based metric.

    Formula: Q = 0.7·R² - 0.05·(RMSE/y_max) - 0.05·|skew| + 0.2·sensitivity_score

    Higher scores indicate better peaks for charge sensing:
    - High R² (good fit)
    - Low normalized RMSE (small errors)
    - Low |skew| (symmetric residuals)
    - High sensitivity_score (steep gradient for charge sensing)

    Args:
        r_squared: Coefficient of determination (0-1)
        rmse: Root mean squared error
        y_max: Maximum y value for normalization
        skew: Skewness of residuals
        sensitivity_score: Normalized gradient metric (0-1, higher = steeper)

    Returns:
        Quality score (higher is better)
    """
    # Normalize RMSE by peak height
    normalized_rmse = rmse / y_max if y_max > 0 else 0

    # Weighted combination (R² dominant, gradient as positive contribution)
    quality = (
        0.7 * r_squared
        - 0.05 * normalized_rmse
        - 0.05 * abs(skew)
        + 0.2 * sensitivity_score  # Positive: steeper peaks score higher
    )

    return float(quality)


def fit_peak_multi_model(
    window_currents: np.ndarray,
    window_indices: np.ndarray,
    peak_idx_in_window: int,
    aggregated_voltages: np.ndarray,
    window_start_idx: int,
    window_end_idx: int,
    peak_idx_aggregated: int,
) -> FittedPeak:
    """
    Fit Lorentzian, sech², and pseudo-Voigt models to a peak, select best by AICc.

    Performs comprehensive multi-model fitting and metric calculation:
    1. Fits three peak models with unbounded optimization
    2. Calculates quality metrics (R², RMSE, AICc, skew, DW) for each
    3. Selects best model by minimum AICc
    4. Calculates physical parameters (FWHM, area)
    5. Calculates sensitivity (max gradient) from best model
    6. Computes overall quality score for peak ranking

    Args:
        window_currents: Current values in the fitting window
        window_indices: Index array for the fitting window (0 to N-1)
        peak_idx_in_window: Peak position within the window
        aggregated_voltages: Full voltage array (for voltage lookup)
        window_start_idx: Start index of window in aggregated trace
        window_end_idx: End index of window in aggregated trace
        peak_idx_aggregated: Peak index in aggregated trace

    Returns:
        FittedPeak object with all three model fits and best model selected
    """
    # Generate initial guesses (same for all models)
    window_min = np.min(window_currents)
    offset_guess = window_min
    amplitude_guess = window_currents[peak_idx_in_window] - offset_guess
    center_guess = float(peak_idx_in_window)
    width_guess = 10.0

    # Storage for model fits
    model_fits = {}

    # ===== FIT LORENTZIAN MODEL =====
    try:
        initial_guess_lorentzian = [
            amplitude_guess,
            center_guess,
            width_guess,
            offset_guess,
        ]
        popt_lorentz, _ = curve_fit(
            lorentzian,
            window_indices,
            window_currents,
            p0=initial_guess_lorentzian,
            maxfev=20000,
        )

        # Calculate fitted curve and residuals
        fitted_lorentz = lorentzian(window_indices, *popt_lorentz)
        residuals_lorentz = window_currents - fitted_lorentz

        # Calculate metrics
        ss_res = np.sum(residuals_lorentz**2)
        ss_tot = np.sum((window_currents - np.mean(window_currents)) ** 2)
        r2_lorentz = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse_lorentz = calculate_rmse(residuals_lorentz)
        aicc_lorentz = calculate_aicc(len(window_currents), 4, residuals_lorentz)
        skew_lorentz = calculate_skew_residuals(residuals_lorentz)

        # Calculate physical parameters
        fwhm_lorentz = calculate_fwhm(
            lorentzian, popt_lorentz, "Lorentzian", window_indices
        )
        x_spacing = 1.0  # Indices are unit-spaced
        area_lorentz = calculate_area(fitted_lorentz, x_spacing)

        model_fits["Lorentzian"] = ModelFitResult(
            model_name="Lorentzian",
            amplitude=float(popt_lorentz[0]),
            center_idx=float(popt_lorentz[1]),
            width=float(popt_lorentz[2]),
            offset=float(popt_lorentz[3]),
            eta=None,
            r_squared=float(r2_lorentz),
            rmse=float(rmse_lorentz),
            aicc=float(aicc_lorentz),
            fwhm=float(fwhm_lorentz),
            area=float(area_lorentz),
            skew_resid=float(skew_lorentz),
        )
    except Exception as e:
        logger.warning("Lorentzian fit failed: %s", e)
        # Create a dummy ModelFitResult with infinite AICc
        model_fits["Lorentzian"] = ModelFitResult(
            model_name="Lorentzian",
            amplitude=0.0,
            center_idx=float(peak_idx_in_window),
            width=1.0,
            offset=0.0,
            eta=None,
            r_squared=0.0,
            rmse=np.inf,
            aicc=np.inf,
            fwhm=0.0,
            area=0.0,
            skew_resid=0.0,
        )

    # ===== FIT SECH² MODEL =====
    try:
        initial_guess_sech2 = [amplitude_guess, center_guess, width_guess, offset_guess]
        popt_sech2, _ = curve_fit(
            sech_squared,
            window_indices,
            window_currents,
            p0=initial_guess_sech2,
            maxfev=20000,
        )

        # Calculate fitted curve and residuals
        fitted_sech2 = sech_squared(window_indices, *popt_sech2)
        residuals_sech2 = window_currents - fitted_sech2

        # Calculate metrics
        ss_res = np.sum(residuals_sech2**2)
        ss_tot = np.sum((window_currents - np.mean(window_currents)) ** 2)
        r2_sech2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse_sech2 = calculate_rmse(residuals_sech2)
        aicc_sech2 = calculate_aicc(len(window_currents), 4, residuals_sech2)
        skew_sech2 = calculate_skew_residuals(residuals_sech2)

        # Calculate physical parameters
        fwhm_sech2 = calculate_fwhm(sech_squared, popt_sech2, "sech2", window_indices)
        area_sech2 = calculate_area(fitted_sech2, x_spacing)

        model_fits["sech2"] = ModelFitResult(
            model_name="sech2",
            amplitude=float(popt_sech2[0]),
            center_idx=float(popt_sech2[1]),
            width=float(popt_sech2[2]),
            offset=float(popt_sech2[3]),
            eta=None,
            r_squared=float(r2_sech2),
            rmse=float(rmse_sech2),
            aicc=float(aicc_sech2),
            fwhm=float(fwhm_sech2),
            area=float(area_sech2),
            skew_resid=float(skew_sech2),
        )
    except Exception as e:
        logger.warning("sech² fit failed: %s", e)
        model_fits["sech2"] = ModelFitResult(
            model_name="sech2",
            amplitude=0.0,
            center_idx=float(peak_idx_in_window),
            width=1.0,
            offset=0.0,
            eta=None,
            r_squared=0.0,
            rmse=np.inf,
            aicc=np.inf,
            fwhm=0.0,
            area=0.0,
            skew_resid=0.0,
        )

    # ===== FIT PSEUDO-VOIGT MODEL =====
    try:
        eta_guess = 0.5  # Start with equal mix
        initial_guess_voigt = [
            amplitude_guess,
            center_guess,
            width_guess,
            offset_guess,
            eta_guess,
        ]
        # Constrain eta to [0, 1] for physical validity
        bounds = (
            [-np.inf, -np.inf, 0, -np.inf, 0],  # lower bounds
            [np.inf, np.inf, np.inf, np.inf, 1],  # upper bounds
        )
        popt_voigt, _ = curve_fit(
            pseudo_voigt,
            window_indices,
            window_currents,
            p0=initial_guess_voigt,
            bounds=bounds,
            maxfev=20000,
        )

        # Calculate fitted curve and residuals
        fitted_voigt = pseudo_voigt(window_indices, *popt_voigt)
        residuals_voigt = window_currents - fitted_voigt

        # Calculate metrics
        ss_res = np.sum(residuals_voigt**2)
        ss_tot = np.sum((window_currents - np.mean(window_currents)) ** 2)
        r2_voigt = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse_voigt = calculate_rmse(residuals_voigt)
        aicc_voigt = calculate_aicc(
            len(window_currents), 5, residuals_voigt
        )  # 5 params
        skew_voigt = calculate_skew_residuals(residuals_voigt)

        # Calculate physical parameters
        fwhm_voigt = calculate_fwhm(pseudo_voigt, popt_voigt, "Voigt", window_indices)
        area_voigt = calculate_area(fitted_voigt, x_spacing)

        model_fits["Voigt"] = ModelFitResult(
            model_name="Voigt",
            amplitude=float(popt_voigt[0]),
            center_idx=float(popt_voigt[1]),
            width=float(popt_voigt[2]),
            offset=float(popt_voigt[3]),
            eta=float(popt_voigt[4]),
            r_squared=float(r2_voigt),
            rmse=float(rmse_voigt),
            aicc=float(aicc_voigt),
            fwhm=float(fwhm_voigt),
            area=float(area_voigt),
            skew_resid=float(skew_voigt),
        )
    except Exception as e:
        logger.warning("Voigt fit failed: %s", e)
        model_fits["Voigt"] = ModelFitResult(
            model_name="Voigt",
            amplitude=0.0,
            center_idx=float(peak_idx_in_window),
            width=1.0,
            offset=0.0,
            eta=0.5,
            r_squared=0.0,
            rmse=np.inf,
            aicc=np.inf,
            fwhm=0.0,
            area=0.0,
            skew_resid=0.0,
        )

    # ===== SELECT BEST MODEL BY AICc =====
    best_model_key = min(model_fits.keys(), key=lambda k: model_fits[k].aicc)
    best_model_result = model_fits[best_model_key]

    logger.info(
        "Peak at idx %d: Best model = %s (AICc: L=%.2f, S=%.2f, V=%.2f)",
        peak_idx_aggregated,
        best_model_key,
        model_fits["Lorentzian"].aicc,
        model_fits["sech2"].aicc,
        model_fits["Voigt"].aicc,
    )

    # ===== CALCULATE SENSITIVITY (MAX GRADIENT) FROM ACTUAL DATA =====
    # Extract voltage values corresponding to the window
    window_voltages = aggregated_voltages[window_start_idx:window_end_idx]

    # Calculate gradient of actual measured current with respect to voltage
    # np.gradient handles the voltage spacing automatically for proper A/V units
    gradient = np.gradient(window_currents, window_voltages)

    # Find maximum positive gradient (steepest slope for charge sensing)
    max_gradient_idx_in_window = int(np.argmax(gradient))
    max_gradient = gradient[max_gradient_idx_in_window]

    # Convert to absolute index and voltage
    max_gradient_absolute_idx = window_start_idx + max_gradient_idx_in_window
    max_gradient_absolute_idx = min(
        max(0, max_gradient_absolute_idx), int(len(aggregated_voltages) - 1)
    )
    max_gradient_voltage = aggregated_voltages[max_gradient_absolute_idx]

    # Convert the fitted center index (which is continuous within the window) into
    # an accurate voltage using interpolation so we are not limited by the sweep grid.
    window_length = len(window_voltages)
    window_index_axis = np.arange(window_length, dtype=float)
    fitted_center_idx = float(
        np.clip(best_model_result.center_idx, 0, window_length - 1)
    )
    peak_voltage = float(
        np.interp(
            fitted_center_idx,
            window_index_axis,
            window_voltages,
        )
    )

    # ===== CREATE FITTEDPEAK OBJECT =====
    fitted_peak = FittedPeak(
        best_model=best_model_key,
        lorentzian_fit=model_fits["Lorentzian"],
        sech2_fit=model_fits["sech2"],
        voigt_fit=model_fits["Voigt"],
        quality_score=None,  # Will be calculated after sensitivity normalization
        sensitivity=float(max_gradient),
        sensitivity_voltage=float(max_gradient_voltage),
        sensitivity_score=None,  # Will be set later via normalization
        peak_idx=int(peak_idx_aggregated),
        peak_voltage=peak_voltage,
        window_currents=window_currents,
        window_voltages=window_voltages,
    )

    return fitted_peak


def fit_and_log_single_peak(
    peak_idx: int,
    peak_number: int,
    start_idx: int,
    end_idx: int,
    aggregated_currents: np.ndarray,
    aggregated_voltages: np.ndarray,
    analysis_session: LoggerSession | None,
) -> FittedPeak | None:
    """
    Fit a single peak and log its analysis results.

    Args:
        peak_idx: Index of peak in aggregated trace
        peak_number: Sequential number of peak (1-based, for logging)
        start_idx: Start index of window in aggregated trace
        end_idx: End index of window in aggregated trace
        aggregated_currents: Full aggregated current trace
        aggregated_voltages: Full aggregated voltage trace
        analysis_session: Logging session for saving analysis results

    Returns:
        FittedPeak object if successful, None otherwise
    """
    window_currents = aggregated_currents[start_idx:end_idx]
    window_indices = np.arange(len(window_currents))
    peak_idx_in_window = peak_idx - start_idx

    if peak_idx_in_window < 0 or peak_idx_in_window >= len(window_currents):
        return None

    try:
        fitted_peak = fit_peak_multi_model(
            window_currents=window_currents,
            window_indices=window_indices,
            peak_idx_in_window=peak_idx_in_window,
            aggregated_voltages=aggregated_voltages,
            window_start_idx=start_idx,
            window_end_idx=end_idx,
            peak_idx_aggregated=peak_idx,
        )

        if analysis_session:
            analysis_session.log_analysis(
                name=f"peak_{peak_number}_multi_model_fit",
                data={
                    "peak_number": peak_number,
                    "peak_idx": fitted_peak.peak_idx,
                    "best_model": fitted_peak.best_model,
                    "quality_score": fitted_peak.quality_score,
                    "sensitivity": fitted_peak.sensitivity,
                    "sensitivity_voltage": fitted_peak.sensitivity_voltage,
                    "peak_voltage": fitted_peak.peak_voltage,
                    "lorentzian": {
                        "amplitude": fitted_peak.lorentzian_fit.amplitude,
                        "center_idx": fitted_peak.lorentzian_fit.center_idx,
                        "width": fitted_peak.lorentzian_fit.width,
                        "offset": fitted_peak.lorentzian_fit.offset,
                        "r_squared": fitted_peak.lorentzian_fit.r_squared,
                        "rmse": fitted_peak.lorentzian_fit.rmse,
                        "aicc": fitted_peak.lorentzian_fit.aicc,
                        "fwhm": fitted_peak.lorentzian_fit.fwhm,
                        "area": fitted_peak.lorentzian_fit.area,
                        "skew_resid": fitted_peak.lorentzian_fit.skew_resid,
                    },
                    "sech2": {
                        "amplitude": fitted_peak.sech2_fit.amplitude,
                        "center_idx": fitted_peak.sech2_fit.center_idx,
                        "width": fitted_peak.sech2_fit.width,
                        "offset": fitted_peak.sech2_fit.offset,
                        "r_squared": fitted_peak.sech2_fit.r_squared,
                        "rmse": fitted_peak.sech2_fit.rmse,
                        "aicc": fitted_peak.sech2_fit.aicc,
                        "fwhm": fitted_peak.sech2_fit.fwhm,
                        "area": fitted_peak.sech2_fit.area,
                        "skew_resid": fitted_peak.sech2_fit.skew_resid,
                    },
                    "voigt": {
                        "amplitude": fitted_peak.voigt_fit.amplitude,
                        "center_idx": fitted_peak.voigt_fit.center_idx,
                        "width": fitted_peak.voigt_fit.width,
                        "offset": fitted_peak.voigt_fit.offset,
                        "eta": fitted_peak.voigt_fit.eta,
                        "r_squared": fitted_peak.voigt_fit.r_squared,
                        "rmse": fitted_peak.voigt_fit.rmse,
                        "aicc": fitted_peak.voigt_fit.aicc,
                        "fwhm": fitted_peak.voigt_fit.fwhm,
                        "area": fitted_peak.voigt_fit.area,
                        "skew_resid": fitted_peak.voigt_fit.skew_resid,
                    },
                },
            )

        return fitted_peak

    except Exception as e:
        logger.warning("Failed to fit peak at index %s: %s", peak_idx, e)
        return None


def calculate_peak_window_bounds(
    peak_idx: int,
    peak_index: int,
    peak_indices: list[int],
    trace_length: int,
    window_fraction: float = 0.8,
    default_window_half_width: int = 128,
) -> tuple[int, int] | None:
    """
    Calculate window boundaries for fitting a peak based on its position.

    Uses spacing between peaks to determine appropriate window size, handling
    edge cases for first, middle, and last peaks.

    Args:
        peak_idx: Index of the current peak in the aggregated trace
        peak_index: Position of peak in the sorted peak_indices list (0-based)
        peak_indices: Sorted list of all peak indices
        trace_length: Total length of the aggregated trace
        window_fraction: Fraction of inter-peak distance to use for window (default: 0.8)
        default_window_half_width: Default half-width in points when peaks are widely spaced (default: 128)

    Returns:
        Tuple of (start_idx, end_idx) for the window, or None if invalid
    """
    if len(peak_indices) == 1:
        start_idx = 0
        end_idx = trace_length
    elif peak_index == 0:
        distance_to_next = peak_indices[peak_index + 1] - peak_idx
        distance_to_start = peak_idx

        start_idx = (
            int(peak_idx - window_fraction * distance_to_start)
            if distance_to_start < default_window_half_width
            else peak_idx - default_window_half_width
        )
        end_idx = (
            int(peak_idx + window_fraction * distance_to_next)
            if distance_to_next < default_window_half_width
            else peak_idx + default_window_half_width
        )
    elif peak_index == len(peak_indices) - 1:
        distance_to_end = trace_length - peak_idx
        distance_to_previous = peak_idx - peak_indices[peak_index - 1]

        start_idx = (
            int(peak_idx - distance_to_previous * window_fraction)
            if distance_to_previous < default_window_half_width
            else peak_idx - default_window_half_width
        )
        end_idx = (
            int(peak_idx + distance_to_end * window_fraction)
            if distance_to_end < default_window_half_width
            else peak_idx + default_window_half_width
        )
    else:
        distance_to_next = peak_indices[peak_index + 1] - peak_idx
        distance_to_previous = peak_idx - peak_indices[peak_index - 1]

        start_idx = (
            int(peak_idx - window_fraction * distance_to_previous)
            if distance_to_previous < default_window_half_width
            else peak_idx - default_window_half_width
        )
        end_idx = (
            int(peak_idx + window_fraction * distance_to_next)
            if distance_to_next < default_window_half_width
            else peak_idx + default_window_half_width
        )

    start_idx = int(max(0, start_idx))
    end_idx = int(min(trace_length, end_idx))

    if start_idx >= peak_idx:
        start_idx = int(max(0, peak_idx - 5))
    if end_idx <= peak_idx:
        end_idx = int(min(trace_length, peak_idx + 5))

    if peak_idx < start_idx or peak_idx >= end_idx:
        return None

    return (start_idx, end_idx)


def normalize_sensitivity_scores(fitted_peaks: list[FittedPeak]) -> None:
    """
    Normalize sensitivity scores across all fitted peaks using min-max scaling.

    Args:
        fitted_peaks: List of FittedPeak objects to normalize
    """
    if not fitted_peaks:
        return

    sensitivities = [peak.sensitivity for peak in fitted_peaks]
    min_sens = min(sensitivities)
    max_sens = max(sensitivities)
    sens_range = max_sens - min_sens

    if sens_range > 0:
        for peak in fitted_peaks:
            peak.sensitivity_score = (peak.sensitivity - min_sens) / sens_range
    else:
        for peak in fitted_peaks:
            peak.sensitivity_score = 1.0


def calculate_quality_scores(fitted_peaks: list[FittedPeak]) -> None:
    """
    Calculate quality scores for all fitted peaks using normalized sensitivity scores.

    Args:
        fitted_peaks: List of FittedPeak objects (must have sensitivity_score set)
    """
    for peak in fitted_peaks:
        best_fit = getattr(peak, f"{peak.best_model.lower()}_fit")

        y_max = (
            float(np.max(np.abs(peak.window_currents)))
            if len(peak.window_currents) > 0
            else 1.0
        )

        if peak.sensitivity_score is None:
            raise RoutineError(
                f"Sensitivity score not set for peak at index {peak.peak_idx}. "
                "Ensure normalize_sensitivity_scores() is called first."
            )
        peak.quality_score = calculate_quality_score(
            r_squared=best_fit.r_squared,
            rmse=best_fit.rmse,
            y_max=y_max,
            skew=best_fit.skew_resid,
            sensitivity_score=peak.sensitivity_score,
        )


def analyze_find_first_peak_voltages(
    aggregated_currents: np.ndarray,
    aggregated_voltages: np.ndarray,
    peak_indices_aggregated: list[int],
    analysis_session: LoggerSession | None,
    window_fraction: float = 0.8,
    default_window_half_width: int = 128,
) -> list[FittedPeak]:
    """
    Analyze and fit multiple peaks in aggregated current trace.

    Args:
        aggregated_currents: Aggregated current trace data
        aggregated_voltages: Aggregated voltage trace data
        peak_indices_aggregated: List of peak indices in aggregated trace
        analysis_session: Logging session for saving analysis results
        window_fraction: Fraction of inter-peak distance to use for window (default: 0.8)
        default_window_half_width: Default half-width in points when peaks are widely spaced (default: 128)

    Returns:
        List of FittedPeak objects with quality metrics
    """
    fitted_peaks = []
    peak_indices_aggregated = sorted(set(peak_indices_aggregated))

    for i, peak_idx in enumerate(peak_indices_aggregated):
        window_bounds = calculate_peak_window_bounds(
            peak_idx=peak_idx,
            peak_index=i,
            peak_indices=peak_indices_aggregated,
            trace_length=len(aggregated_currents),
            window_fraction=window_fraction,
            default_window_half_width=default_window_half_width,
        )

        if window_bounds is None:
            continue

        start_idx, end_idx = window_bounds

        fitted_peak = fit_and_log_single_peak(
            peak_idx=peak_idx,
            peak_number=i + 1,
            start_idx=start_idx,
            end_idx=end_idx,
            aggregated_currents=aggregated_currents,
            aggregated_voltages=aggregated_voltages,
            analysis_session=analysis_session,
        )

        if fitted_peak is not None:
            fitted_peaks.append(fitted_peak)

    if analysis_session:
        analysis_session.log_analysis(
            name="peak_fitting_summary",
            data={
                "num_peaks_detected": len(peak_indices_aggregated),
                "num_peaks_fitted": len(fitted_peaks),
                "fitted_peak_indices": [int(p.peak_idx) for p in fitted_peaks],
            },
        )

    if fitted_peaks:
        normalize_sensitivity_scores(fitted_peaks)
        calculate_quality_scores(fitted_peaks)

    return fitted_peaks
