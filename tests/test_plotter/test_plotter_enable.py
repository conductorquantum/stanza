"""Tests for plotter enable_live_plotting function."""

from unittest.mock import Mock, patch

import pytest

from stanza.plotter import enable_live_plotting


@pytest.fixture
def mock_data_logger():
    """Mock DataLogger instance."""
    return Mock()


def test_enable_server_backend_reuses_existing_port(mock_data_logger):
    """Test that server backend reuses existing server on same port."""
    import time

    backend1 = enable_live_plotting(mock_data_logger, backend="server", port=5006)
    assert backend1.port == 5006

    mock_data_logger2 = Mock()
    backend2 = enable_live_plotting(mock_data_logger2, backend="server", port=5006)

    assert backend2 is backend1
    assert backend2.port == 5006
    assert mock_data_logger2._bokeh_backend is backend2

    backend1.stop()
    time.sleep(0.1)


def test_enable_inline_backend(mock_data_logger):
    """Test enabling inline backend."""
    with patch("stanza.plotter.InlineBackend") as MockInlineBackend:
        mock_backend = Mock()
        MockInlineBackend.return_value = mock_backend

        backend = enable_live_plotting(mock_data_logger, backend="inline")

        mock_backend.start.assert_called_once()
        assert backend is mock_backend
        assert mock_data_logger._bokeh_backend is mock_backend


def test_enable_unknown_backend_raises(mock_data_logger):
    """Test that unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        enable_live_plotting(mock_data_logger, backend="invalid")
