import numpy as np
import pytest

from stanza.analysis.fitting import (
    DEFAULT_BOUNDS,
    PinchoffFitResult,
    _compute_initial_params,
    _compute_parameter_bounds,
    _map_index_to_voltage,
    derivative_extrema_indices,
    fit_pinchoff_parameters,
    pinchoff_curve,
)
from stanza.analysis.preprocessing import normalize


class TestPinchoffCurve:
    def test_basic_output(self):
        x = np.linspace(-10, 10, 100)
        y = pinchoff_curve(x, 1.0, 1.0, 1.0)
        assert np.all(y >= 0.0)
        assert np.all(y <= 2.0)

    def test_amplitude_scaling(self):
        x = np.array([0.0])
        y1 = pinchoff_curve(x, 1.0, 1.0, 0.0)
        y2 = pinchoff_curve(x, 2.0, 1.0, 0.0)
        assert y2[0] == pytest.approx(2 * y1[0])

    def test_offset_effect(self):
        x = np.array([0.0])
        y1 = pinchoff_curve(x, 1.0, 1.0, 0.0)
        y2 = pinchoff_curve(x, 1.0, 1.0, 2.0)
        assert y2[0] > y1[0]


class TestDerivativeExtremaIndices:
    def test_basic_ordering(self):
        x = np.linspace(-10, 10, 100)
        y = pinchoff_curve(x, 1.0, 1.0, 1.0)
        imin_grad, imax_grad, imin_second, imax_second = derivative_extrema_indices(
            x, y
        )
        assert imin_grad > imax_grad
        assert imin_second > imax_second

    def test_negative_amplitude(self):
        x = np.linspace(-10, 10, 100)
        y = pinchoff_curve(x, 1.0, -1.0, 1.0)
        imin_grad, imax_grad, imin_second, imax_second = derivative_extrema_indices(
            x, y
        )
        assert imin_grad > imax_grad
        assert imin_second < imax_second

    def test_returns_integers(self):
        x = np.linspace(-10, 10, 50)
        y = pinchoff_curve(x, 1.0, 1.0, 1.0)
        result = derivative_extrema_indices(x, y)
        assert all(isinstance(idx, int) for idx in result)


class TestComputeInitialParams:
    def test_normalized_data(self):
        v_norm = np.linspace(0, 1, 100)
        i_norm = np.linspace(0, 1, 100)
        params = _compute_initial_params(v_norm, i_norm)
        assert len(params) == 3
        assert params[0] > 0  # amplitude
        assert params[1] > 0  # slope

    def test_constant_arrays(self):
        v_norm = np.ones(100)
        i_norm = np.ones(100)
        params = _compute_initial_params(v_norm, i_norm)
        assert len(params) == 3
        assert params[0] == 0.5  # i_range/2 when ptp=1.0


class TestComputeParameterBounds:
    def test_basic_bounds(self):
        v_norm = np.linspace(0, 1, 100)
        i_norm = np.linspace(0, 1, 100)
        lower, upper = _compute_parameter_bounds(v_norm, i_norm)
        assert len(lower) == 3
        assert len(upper) == 3
        assert np.all(lower < upper)

    def test_invalid_bounds_fallback(self):
        v_norm = np.array([0.5, 0.5])
        i_norm = np.array([0.5, 0.5])
        lower, upper = _compute_parameter_bounds(v_norm, i_norm)
        assert np.all(lower < upper)

    def test_amplitude_bounds_positive(self):
        v_norm = np.linspace(0, 1, 100)
        i_norm = np.linspace(0, 1, 100)
        lower, upper = _compute_parameter_bounds(v_norm, i_norm)
        assert lower[0] > 0

    def test_extreme_values_trigger_defaults(self):
        v_norm = np.array([0.0, 1e6])
        i_norm = np.array([0.0, 1.0])
        lower, upper = _compute_parameter_bounds(v_norm, i_norm)
        assert np.all(lower < upper)
        assert lower[1] == DEFAULT_BOUNDS[1][0]
        assert upper[1] == DEFAULT_BOUNDS[1][1]


class TestMapIndexToVoltage:
    def test_valid_index(self):
        voltages = np.array([1.0, 2.0, 3.0])
        assert _map_index_to_voltage(0, voltages) == 1.0
        assert _map_index_to_voltage(1, voltages) == 2.0
        assert _map_index_to_voltage(2, voltages) == 3.0

    def test_out_of_bounds_index(self):
        voltages = np.array([1.0, 2.0, 3.0])
        assert _map_index_to_voltage(3, voltages) is None
        assert _map_index_to_voltage(10, voltages) is None


class TestFitPinchoffParameters:
    def test_basic_fit(self):
        voltages = np.linspace(-10, 10, 100)
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        result = fit_pinchoff_parameters(voltages, currents)
        assert isinstance(result, PinchoffFitResult)
        assert result.vp is not None
        assert result.vt is not None
        assert result.vc is not None
        assert result.popt is not None
        assert result.pcov is not None

    def test_fit_with_noise(self):
        voltages = np.linspace(-10, 10, 100)
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        noisy_currents = currents + np.random.normal(0, 0.01, len(currents))
        result = fit_pinchoff_parameters(voltages, noisy_currents)
        assert result.vp is not None

    def test_fit_parameter_shapes(self):
        voltages = np.linspace(-10, 10, 100)
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        result = fit_pinchoff_parameters(voltages, currents)
        assert len(result.popt) == 3
        assert result.pcov.shape == (3, 3)

    def test_custom_sigma(self):
        voltages = np.linspace(-10, 10, 100)
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        result = fit_pinchoff_parameters(voltages, currents, sigma=0.05)
        assert result.vp is not None

    def test_edge_case_out_of_bounds_index(self):
        voltages = np.linspace(-10, 10, 10)
        currents = pinchoff_curve(voltages, 1.0, 1.0, 1.0)
        result = fit_pinchoff_parameters(voltages, currents)
        assert hasattr(result, "vp")
        assert hasattr(result, "vt")
        assert hasattr(result, "vc")


class TestNormalize:
    def test_basic_normalization(self):
        a = np.array([0, 5, 10])
        result = normalize(a)
        assert result[0] == 0.0
        assert result[-1] == 1.0
        assert np.all((result >= 0) & (result <= 1))

    def test_constant_array(self):
        a = np.array([5, 5, 5])
        result = normalize(a)
        assert np.all(result == 0.0)

    def test_negative_values(self):
        a = np.array([-10, 0, 10])
        result = normalize(a)
        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_single_element(self):
        a = np.array([42])
        result = normalize(a)
        assert result[0] == 0.0

    def test_preserves_order(self):
        a = np.array([1, 3, 2, 5, 4])
        result = normalize(a)
        assert result[0] < result[1]
        assert result[1] > result[2]
        assert result[2] < result[3]

    def test_float_dtype(self):
        a = np.array([1, 2, 3])
        result = normalize(a)
        assert result.dtype == float
