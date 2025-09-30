from unittest.mock import Mock, patch

import numpy as np
import pytest

from stanza.drivers.utils import demod2volts, wait_until_job_is_paused


@pytest.fixture
def mock_job():
    return Mock()


class TestWaitUntilJobIsPaused:
    def test_returns_immediately_when_job_paused(self, mock_job):
        mock_job.is_paused.return_value = True

        assert wait_until_job_is_paused(mock_job) is True
        mock_job.is_paused.assert_called_once()

    @patch("time.sleep")
    @patch("time.time")
    def test_waits_until_pause_then_returns(self, mock_time, mock_sleep, mock_job):
        mock_job.is_paused.side_effect = [False, False, False, True]
        mock_time.side_effect = [0, 0.1, 0.2, 0.3]

        assert wait_until_job_is_paused(mock_job, timeout=5) is True
        assert mock_job.is_paused.call_count == 4
        assert mock_sleep.call_count == 3

    @patch("time.sleep")
    @patch("time.time")
    def test_raises_timeout_error_when_strict(self, mock_time, mock_sleep, mock_job):
        mock_job.is_paused.return_value = False
        mock_time.side_effect = [0, 1.5, 2.5]

        with pytest.raises(TimeoutError, match="Timeout \\(2s\\) was reached"):
            wait_until_job_is_paused(mock_job, timeout=2, strict_timeout=True)

    @patch("time.sleep")
    @patch("time.time")
    def test_warns_on_timeout_when_not_strict(self, mock_time, mock_sleep, mock_job):
        mock_job.is_paused.return_value = False
        mock_time.side_effect = [0, 1.5, 2.5]

        with pytest.warns(UserWarning, match="Timeout \\(2s\\) was reached"):
            assert (
                wait_until_job_is_paused(mock_job, timeout=2, strict_timeout=False)
                is True
            )

    @patch("time.sleep")
    @patch("time.time")
    def test_custom_timeout_value(self, mock_time, mock_sleep, mock_job):
        mock_job.is_paused.return_value = False
        mock_time.side_effect = [0, 5, 15, 25]

        with pytest.raises(TimeoutError, match="Timeout \\(10s\\) was reached"):
            wait_until_job_is_paused(mock_job, timeout=10)


class TestDemod2volts:
    def test_single_value_without_single_demod(self):
        assert demod2volts(2.5, 1000, single_demod=False) == 4096 * 2.5 / 1000

    def test_single_value_with_single_demod(self):
        assert demod2volts(2.5, 1000, single_demod=True) == 2 * 4096 * 2.5 / 1000

    def test_numpy_array_without_single_demod(self):
        data = np.array([1.0, 2.0, 3.0])
        result = demod2volts(data, 500, single_demod=False)
        np.testing.assert_array_equal(result, 4096 * data / 500)

    def test_numpy_array_with_single_demod(self):
        data = np.array([1.0, 2.0, 3.0])
        result = demod2volts(data, 500, single_demod=True)
        np.testing.assert_array_equal(result, 2 * 4096 * data / 500)

    @pytest.mark.parametrize("duration", [100, 1000, 5000])
    def test_different_durations(self, duration):
        single = demod2volts(1.0, duration, single_demod=True)
        dual = demod2volts(1.0, duration, single_demod=False)
        assert single == 2 * dual == 2 * 4096 / duration

    def test_zero_data(self):
        assert demod2volts(0.0, 1000) == 0.0

    def test_negative_data(self):
        assert demod2volts(-1.5, 1000, single_demod=False) == 4096 * (-1.5) / 1000

    def test_float_duration(self):
        result = demod2volts(1.0, 1000.5)
        assert abs(result - 4096 / 1000.5) < 1e-10

    def test_preserves_array_type(self):
        data = np.array([1.0, 2.0], dtype=np.float32)
        result = demod2volts(data, 1000)
        assert isinstance(result, np.ndarray) and result.dtype == data.dtype
