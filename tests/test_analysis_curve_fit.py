import numpy as np

from stanza.analysis.curve_fit import (
    derivative_extrema_indices,
    fit_pinchoff_parameters,
    normalize,
    pinchoff_curve,
)


class TestPinchoffCurve:
    def test_pinchoff_curve_shape(self):
        """Test that pinchoff_curve returns expected shape and range."""
        x = np.linspace(-2, 2, 100)
        a, b, c = 1.0, 1.0, 0.0
        y = pinchoff_curve(x, a, b, c)

        assert y.shape == x.shape
        assert np.all(y >= 0)
        assert np.all(y <= 2 * a)

    def test_pinchoff_curve_symmetric_at_center(self):
        """Test symmetry around the inflection point."""
        x = np.linspace(-2, 2, 100)
        a, b, c = 1.0, 1.0, 0.0
        y = pinchoff_curve(x, a, b, c)

        center_idx = len(x) // 2
        assert np.isclose(y[center_idx], a, atol=0.05)

    def test_pinchoff_curve_amplitude_scaling(self):
        """Test that amplitude parameter scales the output correctly."""
        x = np.linspace(-2, 2, 100)
        b, c = 1.0, 0.0

        y1 = pinchoff_curve(x, a=1.0, b=b, c=c)
        y2 = pinchoff_curve(x, a=2.0, b=b, c=c)

        np.testing.assert_allclose(y2, 2 * y1, rtol=0.01)

    def test_pinchoff_curve_slope_effect(self):
        """Test that slope parameter affects transition steepness."""
        x = np.linspace(-2, 2, 100)
        a, c = 1.0, 0.0

        y_shallow = pinchoff_curve(x, a, b=0.5, c=c)
        y_steep = pinchoff_curve(x, a, b=2.0, c=c)

        grad_shallow = np.gradient(y_shallow)
        grad_steep = np.gradient(y_steep)

        assert np.max(grad_steep) > np.max(grad_shallow)


class TestDerivativeExtremaIndices:
    def test_monotonic_increasing(self):
        """Test with monotonically increasing data."""
        x = np.linspace(0, 10, 100)
        y = x

        imin_grad, imax_grad, imin_second, imax_second = derivative_extrema_indices(
            x, y
        )

        assert 0 <= imin_grad < len(x)
        assert 0 <= imax_grad < len(x)
        assert 0 <= imin_second < len(x)
        assert 0 <= imax_second < len(x)

    def test_quadratic_function(self):
        """Test with quadratic function."""
        x = np.linspace(-5, 5, 100)
        y = x**2

        imin_grad, imax_grad, imin_second, imax_second = derivative_extrema_indices(
            x, y
        )

        assert 0 <= imin_grad < len(x)
        assert 0 <= imax_grad < len(x)

    def test_sigmoid_like_curve(self):
        """Test with sigmoid-like curve similar to pinchoff."""
        x = np.linspace(-5, 5, 200)
        y = 1 / (1 + np.exp(-x))

        imin_grad, imax_grad, imin_second, imax_second = derivative_extrema_indices(
            x, y
        )

        assert np.isclose(x[imax_grad], 0, atol=0.2)


class TestNormalize:
    def test_normalize_range(self):
        """Test that normalize maps to [0, 1]."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize(a)

        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)

    def test_normalize_constant_array(self):
        """Test that constant arrays map to zeros."""
        a = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = normalize(a)

        np.testing.assert_array_equal(normalized, np.zeros_like(a))

    def test_normalize_negative_values(self):
        """Test normalize with negative values."""
        a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        normalized = normalize(a)

        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)
        assert np.isclose(normalized[2], 0.5)

    def test_normalize_preserves_order(self):
        """Test that normalize preserves relative ordering."""
        a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        normalized = normalize(a)

        assert normalized[2] > normalized[0]
        assert normalized[4] > normalized[2]
        assert np.isclose(normalized[1], normalized[3])


class TestFitPinchoffParameters:
    def test_fit_ideal_pinchoff_curve(self):
        """Test fitting to a known pinchoff curve."""
        x = np.linspace(-2, 2, 200)
        a_true, b_true, c_true = 1.5, 2.0, 0.5

        y_true = pinchoff_curve(normalize(x), a_true, b_true, c_true)
        voltages = x
        currents = y_true * 1e-9

        result = fit_pinchoff_parameters(voltages, currents)

        assert result["vp"] is not None
        assert result["vt"] is not None
        assert result["vc"] is not None
        assert result["popt"] is not None
        assert result["pcov"] is not None
        assert len(result["popt"]) == 3

    def test_fit_with_noise(self):
        """Test fitting with noisy data."""
        np.random.seed(42)
        x = np.linspace(-2, 2, 200)
        a_true, b_true, c_true = 1.0, 1.5, 0.0

        y_true = pinchoff_curve(normalize(x), a_true, b_true, c_true)
        noise = np.random.normal(0, 0.02, len(y_true))
        y_noisy = y_true + noise

        voltages = x
        currents = y_noisy * 1e-9

        result = fit_pinchoff_parameters(voltages, currents)

        assert result["vp"] is not None
        assert result["pcov"] is not None
        assert not np.any(np.isnan(result["popt"]))

    def test_fit_returns_dict_with_expected_keys(self):
        """Test that fit returns dictionary with all expected keys."""
        x = np.linspace(0, 2, 100)
        y = np.linspace(0, 1, 100)

        result = fit_pinchoff_parameters(x, y)

        expected_keys = {"vp", "vt", "vc", "popt", "pcov"}
        assert set(result.keys()) == expected_keys

    def test_fit_with_monotonic_data(self):
        """Test fitting with simple monotonic data."""
        x = np.linspace(-1, 3, 150)
        y = 0.5 * (1 + np.tanh(2 * (x - 1)))

        voltages = x
        currents = y * 1e-9

        result = fit_pinchoff_parameters(voltages, currents)

        assert result["vp"] is not None
        assert -1 <= result["vp"] <= 3

    def test_fit_with_flat_data(self):
        """Test that fit handles flat data gracefully."""
        x = np.linspace(0, 1, 100)
        y = np.ones_like(x) * 0.5

        voltages = x
        currents = y * 1e-9

        result = fit_pinchoff_parameters(voltages, currents)

        assert "vp" in result
        assert "popt" in result
        assert "pcov" in result

    def test_fit_covariance_matrix_shape(self):
        """Test that covariance matrix has correct shape."""
        x = np.linspace(-2, 2, 100)
        y = 1 / (1 + np.exp(-x))

        result = fit_pinchoff_parameters(x, y * 1e-9)

        assert result["pcov"].shape == (3, 3)

    def test_fit_parameter_bounds(self):
        """Test that fitted parameters are within reasonable bounds."""
        x = np.linspace(-2, 2, 200)
        y = 1.5 * (1 + np.tanh(2 * x))

        voltages = x
        currents = y * 1e-9

        result = fit_pinchoff_parameters(voltages, currents)

        popt = result["popt"]
        assert popt[0] > 0
        assert not np.any(np.isinf(popt))
        assert not np.any(np.isnan(popt))
