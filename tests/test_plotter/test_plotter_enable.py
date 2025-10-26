"""Tests for plotter enable_live_plotting function."""

from unittest.mock import Mock, patch

import pytest

from stanza.plotter import enable_live_plotting


@pytest.fixture
def mock_data_logger():
    """Mock DataLogger instance."""
    return Mock()


def test_enable_server_backend_finds_free_port(mock_data_logger):
    """Test that server backend finds next port if requested port is busy."""
    with patch("stanza.plotter._find_free_port") as mock_find_port:
        mock_find_port.return_value = 5007

        backend = enable_live_plotting(mock_data_logger, backend="server", port=5006)

        assert backend.port == 5007
        assert mock_data_logger._bokeh_backend is backend


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


def test_find_free_port_returns_first_available():
    """Test that _find_free_port returns the first available port."""
    from stanza.plotter import _find_free_port

    port = _find_free_port(5006)

    assert isinstance(port, int)
    assert port >= 5006


def test_find_free_port_raises_when_none_available():
    """Test that _find_free_port raises error when no ports available."""
    from stanza.plotter import _find_free_port

    with patch("socket.socket") as MockSocket:
        mock_sock = Mock()
        mock_sock.bind.side_effect = OSError("Port in use")
        MockSocket.return_value.__enter__.return_value = mock_sock

        with pytest.raises(RuntimeError, match="Could not find free port"):
            _find_free_port(5006, max_attempts=3)
