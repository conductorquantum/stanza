"""Tests for NotebookBackend."""

from unittest.mock import Mock, patch

import pytest

from stanza.plotter.backends.notebook import NotebookBackend


@pytest.fixture
def mock_bokeh():
    """Mock bokeh imports."""
    with (
        patch("stanza.plotter.backends.notebook.output_notebook") as mock_output,
        patch("stanza.plotter.backends.notebook.show") as mock_show,
        patch("stanza.plotter.backends.notebook.push_notebook") as mock_push,
        patch("stanza.plotter.backends.notebook.figure") as mock_figure_func,
    ):
        mock_show.return_value = Mock()
        mock_figure_func.return_value = Mock()

        yield {
            "output_notebook": mock_output,
            "show": mock_show,
            "push_notebook": mock_push,
            "figure": mock_figure_func,
        }


def test_initialization_calls_output_notebook(mock_bokeh):
    """Test that init calls output_notebook."""
    backend = NotebookBackend()

    mock_bokeh["output_notebook"].assert_called_once()
    assert len(backend._figures) == 0
    assert len(backend._handles) == 0


def test_create_figure_returns_tuple(mock_bokeh):
    """Test that create_figure returns tuple with figure and needs_display flag."""
    backend = NotebookBackend()

    fig, needs_display = backend.create_figure(name="Test", x_label="X", y_label="Y")

    assert mock_bokeh["figure"].called
    assert needs_display is True
    assert "Test" in backend._figures


def test_display_figure_shows_figure(mock_bokeh):
    """Test that display_figure shows the figure."""
    backend = NotebookBackend()

    fig, needs_display = backend.create_figure(name="Test", x_label="X", y_label="Y")

    if needs_display:
        backend.display_figure(fig)

    mock_bokeh["show"].assert_called_once()
    call_kwargs = mock_bokeh["show"].call_args[1]
    assert call_kwargs.get("notebook_handle") is True
    assert "Test" in backend._handles


def test_push_updates_calls_push_notebook(mock_bokeh):
    """Test that push_updates calls push_notebook."""
    backend = NotebookBackend()

    fig, needs_display = backend.create_figure("Test", "X", "Y")
    if needs_display:
        backend.display_figure(fig)

    backend.push_updates()

    assert mock_bokeh["push_notebook"].called


def test_initialization_handles_output_notebook_failure():
    """Test that initialization handles output_notebook failures gracefully."""
    with patch("stanza.plotter.backends.notebook.output_notebook") as mock_output:
        mock_output.side_effect = RuntimeError("Not in notebook")

        backend = NotebookBackend()
        assert len(backend._figures) == 0


def test_display_figure_handles_show_failure(mock_bokeh):
    """Test that display_figure handles show() failures gracefully."""
    backend = NotebookBackend()
    fig, needs_display = backend.create_figure("Test", "X", "Y")

    mock_bokeh["show"].side_effect = RuntimeError("Display failed")

    backend.display_figure(fig)


def test_push_updates_handles_push_failure(mock_bokeh):
    """Test that push_updates handles push_notebook failures gracefully."""
    backend = NotebookBackend()
    fig, needs_display = backend.create_figure("Test", "X", "Y")
    backend.display_figure(fig)

    mock_bokeh["push_notebook"].side_effect = RuntimeError("Push failed")

    backend.push_updates()
