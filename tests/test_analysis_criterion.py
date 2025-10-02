import numpy as np

from stanza.analysis.criterion import pcov_fail_criterion


class TestPcovFailCriterion:
    def test_good_covariance_matrix(self):
        """Test that a reasonable covariance matrix passes the criterion."""
        pcov = np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])

        assert not pcov_fail_criterion(pcov)

    def test_large_diagonal_values_fail(self):
        """Test that large diagonal values trigger failure."""
        pcov = np.array([[1e12, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])

        assert pcov_fail_criterion(pcov)

    def test_zero_covariance_matrix_fails(self):
        """Test that zero covariance matrix triggers failure."""
        pcov = np.zeros((3, 3))

        assert pcov_fail_criterion(pcov)

    def test_near_zero_covariance_matrix_fails(self):
        """Test that near-zero covariance matrix triggers failure."""
        pcov = np.array([[1e-15, 0, 0], [0, 1e-15, 0], [0, 0, 1e-15]])

        assert pcov_fail_criterion(pcov)

    def test_custom_ubound_threshold(self):
        """Test with custom upper bound threshold."""
        pcov = np.array([[1e5, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])

        assert not pcov_fail_criterion(pcov, ubound=1e6)
        assert pcov_fail_criterion(pcov, ubound=1e4)

    def test_off_diagonal_values_dont_affect_criterion(self):
        """Test that off-diagonal covariance values don't trigger failure."""
        pcov = np.array([[1e-3, 0.5, 0.5], [0.5, 1e-3, 0.5], [0.5, 0.5, 1e-3]])

        assert not pcov_fail_criterion(pcov)

    def test_mixed_diagonal_values(self):
        """Test with mixed diagonal values."""
        pcov = np.array([[1e-3, 0, 0], [0, 1e11, 0], [0, 0, 1e-3]])

        assert pcov_fail_criterion(pcov)

    def test_returns_bool_type(self):
        """Test that the function returns a Python bool, not numpy bool."""
        pcov = np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])

        result = pcov_fail_criterion(pcov)

        assert isinstance(result, bool)
        assert not isinstance(result, np.bool_)

    def test_edge_case_at_boundary(self):
        """Test behavior at exact boundary value."""
        ubound = 1e10
        pcov = np.array([[ubound, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])

        assert not pcov_fail_criterion(pcov, ubound=ubound)

        pcov_above = np.array([[ubound * 1.1, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])
        assert pcov_fail_criterion(pcov_above, ubound=ubound)

    def test_identity_matrix_passes(self):
        """Test that identity matrix passes the criterion."""
        pcov = np.eye(3)

        assert not pcov_fail_criterion(pcov)

    def test_larger_matrix_sizes(self):
        """Test with different matrix sizes."""
        pcov_2x2 = np.array([[1e-3, 0], [0, 1e-3]])
        assert not pcov_fail_criterion(pcov_2x2)

        pcov_5x5 = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
        assert not pcov_fail_criterion(pcov_5x5)

        pcov_5x5_fail = np.diag([1e-3, 1e-3, 1e12, 1e-3, 1e-3])
        assert pcov_fail_criterion(pcov_5x5_fail)

    def test_negative_diagonal_values(self):
        """Test behavior with negative diagonal values (shouldn't occur but test robustness)."""
        pcov = np.array([[-1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]])

        result = pcov_fail_criterion(pcov)
        assert isinstance(result, bool)
