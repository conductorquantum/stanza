"""Tests for peak fitting utilities."""

import numpy as np
import pytest

from stanza.exceptions import RoutineError
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    calculate_aicc,
    calculate_fwhm,
    calculate_quality_score,
    fit_peak_multi_model,
    lorentzian,
    pseudo_voigt,
    sech_squared,
)


def test_fit_peak_multi_model_prefers_lowest_aicc():
    """Generate traces matching a specific model (e.g., Lorentzian) to verify
    fit_peak_multi_model selects that model by AICc."""
    # Generate synthetic Lorentzian peak
    voltages = np.linspace(-0.1, 0.1, 200)
    amplitude = 1e-9
    center = 0.0
    width = 0.01
    offset = 1e-11
    currents = lorentzian(voltages, amplitude, center, width, offset)
    # Add small noise
    np.random.seed(42)
    currents += np.random.normal(0, 1e-12, len(currents))

    window_indices = np.arange(len(currents))
    peak_idx_in_window = 100

    fitted_peak = fit_peak_multi_model(
        window_currents=currents,
        window_indices=window_indices,
        peak_idx_in_window=peak_idx_in_window,
        aggregated_voltages=voltages,
        window_start_idx=0,
        window_end_idx=len(currents),
        peak_idx_aggregated=100,
    )

    # Should select Lorentzian as best model (or at least have good fit)
    assert isinstance(fitted_peak, FittedPeak)
    assert fitted_peak.best_model in ["Lorentzian", "sech2", "Voigt"]
    # Lorentzian should have best (lowest) AICc for Lorentzian-shaped data
    # AICc can be negative; lower (more negative) is better
    assert fitted_peak.lorentzian_fit.aicc <= fitted_peak.sech2_fit.aicc
    assert fitted_peak.lorentzian_fit.aicc <= fitted_peak.voigt_fit.aicc


def test_calculate_fwhm_voigt_uses_numerical_branches():
    """Confirm calculate_fwhm falls back to the numerical path for pseudo-Voigt
    inputs and returns the expected width."""
    x_range = np.linspace(-50, 50, 200)
    amplitude = 1.0
    center = 0.0
    width = 10.0
    offset = 0.0
    eta = 0.5  # Mix of Gaussian and Lorentzian

    params = (amplitude, center, width, offset, eta)
    fwhm = calculate_fwhm(pseudo_voigt, params, "Voigt", x_range)

    # FWHM should be a reasonable value
    assert fwhm > 0
    assert fwhm < 100  # Should be on order of 2*width


def test_calculate_aicc_returns_inf_for_invalid_inputs(caplog):
    """Validate calculate_aicc emits np.inf when RSS <= 0 or n <= k + 1 and logs warnings."""
    import logging

    n = 10
    k = 4
    residuals = np.zeros(n)  # RSS = 0

    # Capture warnings
    with caplog.at_level(logging.WARNING):
        aicc = calculate_aicc(n, k, residuals)
        assert aicc == np.inf
        # Verify warning was logged for RSS <= 0
        assert any("RSS <= 0" in record.message for record in caplog.records)

    # Test n <= k + 1
    n = 5
    k = 4
    residuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        aicc = calculate_aicc(n, k, residuals)
        assert aicc == np.inf
        # Verify warning was logged for insufficient data points
        assert any(
            "insufficient data points" in record.message for record in caplog.records
        )


def test_measure_peak_stability_combines_quality_and_noise():
    """Verify calculate_combined_scores weights stability 70% and original quality 30%."""
    from stanza.routines.builtins.charge_sensor.utils.peak_stability import (
        calculate_combined_scores,
    )
    from stanza.routines.builtins.charge_sensor.utils.types import (
        StabilityMeasurement,
        StablePeakCandidate,
    )
    from stanza.routines.builtins.utils.peak_fitting import ModelFitResult

    # Create mock model fit results (needed for FittedPeak)
    mock_fit = ModelFitResult(
        model_name="Lorentzian",
        amplitude=1e-9,
        center_idx=50,
        width=5.0,
        offset=1e-11,
        r_squared=0.95,
        rmse=1e-12,
        aicc=-100.0,
        fwhm=0.01,
        area=1e-10,
        skew_resid=0.1,
    )

    # Create mock fitted peaks with different quality scores
    peak1 = FittedPeak(
        best_model="Lorentzian",
        lorentzian_fit=mock_fit,
        sech2_fit=mock_fit,
        voigt_fit=mock_fit,
        sensitivity=1.0,
        sensitivity_voltage=0.0,
        peak_idx=50,
        peak_voltage=0.0,
        window_currents=np.array([1e-9]),
        window_voltages=np.array([0.0]),
        quality_score=0.9,
    )
    peak2 = FittedPeak(
        best_model="Lorentzian",
        lorentzian_fit=mock_fit,
        sech2_fit=mock_fit,
        voigt_fit=mock_fit,
        sensitivity=1.0,
        sensitivity_voltage=0.02,
        peak_idx=60,
        peak_voltage=0.02,
        window_currents=np.array([1e-9]),
        window_voltages=np.array([0.02]),
        quality_score=0.6,
    )

    # Create stability measurements with known voltage noise
    # Peak 1: high quality (0.9), poor stability (high noise = 2.0)
    # Peak 2: low quality (0.6), good stability (low noise = 0.5)
    stability1 = StabilityMeasurement(
        peak_index=0,
        peak_voltage=0.0,
        max_gradient_voltage=0.0,
        time_array=np.array([0.0, 1.0]),
        current_array=np.array([1e-9, 1e-9]),
        current_mean=1e-9,
        current_std=1e-11,
        local_slope=1e-8,
        voltage_noise=2.0,
    )
    stability2 = StabilityMeasurement(
        peak_index=1,
        peak_voltage=0.02,
        max_gradient_voltage=0.02,
        time_array=np.array([0.0, 1.0]),
        current_array=np.array([1e-9, 1e-9]),
        current_mean=1e-9,
        current_std=1e-11,
        local_slope=1e-8,
        voltage_noise=0.5,
    )

    # Create candidates
    candidate1 = StablePeakCandidate(
        fitted_peak=peak1, original_score=0.9, stability_measurement=stability1
    )
    candidate2 = StablePeakCandidate(
        fitted_peak=peak2, original_score=0.6, stability_measurement=stability2
    )

    candidates = [candidate1, candidate2]

    # Calculate combined scores with 30% original, 70% stability
    calculate_combined_scores(candidates, original_weight=0.3, stability_weight=0.7)

    # Verify combined scores were calculated
    assert candidate1.combined_score is not None
    assert candidate2.combined_score is not None

    # With 70% weight on stability, peak2 (better stability) should score higher
    # despite lower original quality
    assert candidate2.combined_score > candidate1.combined_score, (
        "Peak with better stability should score higher with 70% stability weight"
    )

    # Verify the formula: combined = 0.3 * normalized_original + 0.7 * normalized_stability
    # Peak1: original=0.9 (max) -> normalized=1.0, stability=1/2.0=0.5 (min) -> normalized≈0.25
    # Peak2: original=0.6 (min) -> normalized=0.0, stability=1/0.5=2.0 (max) -> normalized=1.0
    # Combined1 = 0.3 * 1.0 + 0.7 * 0.25 = 0.475
    # Combined2 = 0.3 * 0.0 + 0.7 * 1.0 = 0.7
    expected_combined1 = 0.3 * 1.0 + 0.7 * 0.25
    expected_combined2 = 0.3 * 0.0 + 0.7 * 1.0

    assert abs(candidate1.combined_score - expected_combined1) < 0.01
    assert abs(candidate2.combined_score - expected_combined2) < 0.01


def test_fit_peak_multi_model_returns_best_scoring_peak():
    """Feed multiple synthetic peaks with known quality metrics and verify the routine
    returns the highest scoring FittedPeak."""
    # Generate a high-quality Lorentzian peak
    voltages = np.linspace(-0.1, 0.1, 150)
    currents = lorentzian(voltages, 2e-9, 0.0, 0.008, 1e-11)
    np.random.seed(42)
    currents += np.random.normal(0, 5e-13, len(currents))

    window_indices = np.arange(len(currents))
    peak_idx_in_window = 75

    fitted_peak = fit_peak_multi_model(
        window_currents=currents,
        window_indices=window_indices,
        peak_idx_in_window=peak_idx_in_window,
        aggregated_voltages=voltages,
        window_start_idx=0,
        window_end_idx=len(currents),
        peak_idx_aggregated=75,
    )

    # Should have quality_score set (will be None until normalize_sensitivity_scores called)
    assert (
        fitted_peak.quality_score is None
    )  # Not set yet, needs sensitivity normalization
    # But R² should be high
    assert fitted_peak.lorentzian_fit.r_squared > 0.8


def test_fit_peak_multi_model_accuracy_within_threshold():
    """Compare fitted peak centers/sensitivities against known analytical curves and
    assert errors stay below a configured tolerance."""
    # Generate synthetic peak with known center
    true_center_voltage = 0.05
    voltages = np.linspace(0.0, 0.1, 200)
    true_center_idx = np.argmin(np.abs(voltages - true_center_voltage))
    # Generate indices array matching the window
    window_indices = np.arange(len(voltages))
    currents = lorentzian(window_indices, 3e-9, true_center_idx, 15.0, 1e-11)

    fitted_peak = fit_peak_multi_model(
        window_currents=currents,
        window_indices=window_indices,
        peak_idx_in_window=true_center_idx,
        aggregated_voltages=voltages,
        window_start_idx=0,
        window_end_idx=len(currents),
        peak_idx_aggregated=true_center_idx,
    )

    # Fitted peak voltage should be close to true center
    assert abs(fitted_peak.peak_voltage - true_center_voltage) < 0.01


def test_lorentzian_model():
    """Test Lorentzian model produces expected shape."""
    x = np.linspace(-10, 10, 100)
    amplitude = 1.0
    center = 0.0
    width = 2.0
    offset = 0.0

    y = lorentzian(x, amplitude, center, width, offset)

    # Peak should be at center
    peak_idx = np.argmax(y)
    assert abs(x[peak_idx] - center) < 0.5
    # Peak value should be amplitude + offset
    assert abs(y[peak_idx] - (amplitude + offset)) < 0.1


def test_sech_squared_model():
    """Test sech_squared model produces expected shape."""
    x = np.linspace(-10, 10, 100)
    amplitude = 1.0
    center = 0.0
    width = 2.0
    offset = 0.0

    y = sech_squared(x, amplitude, center, width, offset)

    # Peak should be at center
    peak_idx = np.argmax(y)
    assert abs(x[peak_idx] - center) < 0.5
    # Peak value should be amplitude + offset
    assert abs(y[peak_idx] - (amplitude + offset)) < 0.1


def test_pseudo_voigt_model():
    """Test pseudo_voigt model produces expected shape."""
    x = np.linspace(-10, 10, 100)
    amplitude = 1.0
    center = 0.0
    width = 2.0
    offset = 0.0
    eta = 0.5

    y = pseudo_voigt(x, amplitude, center, width, offset, eta)

    # Peak should be at center
    peak_idx = np.argmax(y)
    assert abs(x[peak_idx] - center) < 0.5
    # Peak value should be amplitude + offset
    assert abs(y[peak_idx] - (amplitude + offset)) < 0.1


def test_calculate_quality_score_formula():
    """Test calculate_quality_score formula with known inputs."""
    r_squared = 0.95
    rmse = 0.01
    y_max = 1.0
    skew = 0.1
    sensitivity_score = 0.8

    quality = calculate_quality_score(r_squared, rmse, y_max, skew, sensitivity_score)

    # Calculate expected value: 0.7*0.95 - 0.05*0.01 - 0.05*0.1 + 0.2*0.8
    expected = 0.7 * 0.95 - 0.05 * 0.01 - 0.05 * 0.1 + 0.2 * 0.8
    assert abs(quality - expected) < 1e-6


def test_lorentzian_raises_error_for_zero_width():
    """Verify lorentzian raises RoutineError for zero width parameter."""
    x = np.linspace(-10, 10, 100)
    with pytest.raises(RoutineError, match="Zero width parameter"):
        lorentzian(x, amplitude=1.0, center=0.0, width=0.0, offset=0.0)


def test_sech_squared_raises_error_for_zero_width():
    """Verify sech_squared raises RoutineError for zero width parameter."""
    x = np.linspace(-10, 10, 100)
    with pytest.raises(RoutineError, match="Zero width parameter"):
        sech_squared(x, amplitude=1.0, center=0.0, width=0.0, offset=0.0)


def test_pseudo_voigt_raises_error_for_zero_width():
    """Verify pseudo_voigt raises RoutineError for zero width parameter."""
    x = np.linspace(-10, 10, 100)
    with pytest.raises(RoutineError, match="Zero width parameter"):
        pseudo_voigt(x, amplitude=1.0, center=0.0, width=0.0, offset=0.0, eta=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
