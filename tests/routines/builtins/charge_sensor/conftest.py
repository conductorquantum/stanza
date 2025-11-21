import numpy as np
import pytest

from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
    StabilityMeasurement,
)
from stanza.routines.builtins.utils.peak_fitting import FittedPeak, ModelFitResult


@pytest.fixture
def model_fit_result():
    """Baseline ModelFitResult for charge sensor unit tests."""
    return ModelFitResult(
        model_name="Lorentzian",
        amplitude=1.0,
        center_idx=50.0,
        width=5.0,
        offset=0.0,
        r_squared=0.95,
        rmse=0.01,
        aicc=10.0,
        fwhm=10.0,
        area=100.0,
        skew_resid=0.0,
        eta=0.5,
    )


@pytest.fixture
def fitted_peak_factory(model_fit_result):
    """Factory fixture for constructing FittedPeak instances with sensible defaults."""

    def _factory(
        *,
        best_model: str = "Lorentzian",
        sensitivity: float = 1e-6,
        sensitivity_voltage: float = 0.0,
        peak_idx: int = 0,
        peak_voltage: float = 0.0,
        quality_score: float | None = None,
        sensitivity_score: float | None = None,
        window_currents: np.ndarray | None = None,
        window_voltages: np.ndarray | None = None,
        lorentzian_fit: ModelFitResult | None = None,
        sech2_fit: ModelFitResult | None = None,
        voigt_fit: ModelFitResult | None = None,
    ) -> FittedPeak:
        return FittedPeak(
            best_model=best_model,
            lorentzian_fit=lorentzian_fit or model_fit_result,
            sech2_fit=sech2_fit or model_fit_result,
            voigt_fit=voigt_fit or model_fit_result,
            sensitivity=sensitivity,
            sensitivity_voltage=sensitivity_voltage,
            peak_idx=peak_idx,
            peak_voltage=peak_voltage,
            window_currents=window_currents
            if window_currents is not None
            else np.ones(10),
            window_voltages=window_voltages
            if window_voltages is not None
            else np.linspace(0, 0.1, 10),
            quality_score=quality_score,
            sensitivity_score=sensitivity_score,
        )

    return _factory


@pytest.fixture
def stability_measurement_factory():
    """Factory fixture for constructing StabilityMeasurement instances."""

    def _factory(
        *,
        peak_index: int = 0,
        peak_voltage: float = -0.7,
        max_gradient_voltage: float = -0.68,
        time_array: np.ndarray | None = None,
        current_array: np.ndarray | None = None,
        current_mean: float = 1e-9,
        current_std: float = 1e-11,
        local_slope: float = 1e-6,
        voltage_noise: float = 1e-5,
    ) -> StabilityMeasurement:
        time_array_np = (
            np.array(time_array, dtype=np.float64)
            if time_array is not None
            else np.array([0.0, 1.0], dtype=np.float64)
        )
        current_array_np = (
            np.array(current_array, dtype=np.float64)
            if current_array is not None
            else np.array([1e-9, 1e-9], dtype=np.float64)
        )

        return StabilityMeasurement(
            peak_index=peak_index,
            peak_voltage=peak_voltage,
            max_gradient_voltage=max_gradient_voltage,
            time_array=time_array_np,
            current_array=current_array_np,
            current_mean=current_mean,
            current_std=current_std,
            local_slope=local_slope,
            voltage_noise=voltage_noise,
        )

    return _factory
