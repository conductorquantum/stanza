"""Tests for InlineBackend."""

from unittest.mock import Mock, patch

import pytest

from stanza.plotter.backends.inline import InlineBackend


@pytest.fixture
def mock_bokeh():
    """Mock bokeh and jupyter imports."""
    with (
        patch("stanza.plotter.backends.inline.output_notebook") as mock_output,
        patch("stanza.plotter.backends.inline.display") as mock_display,
        patch("stanza.plotter.backends.inline.figure") as mock_figure_func,
        patch("stanza.plotter.backends.inline.BokehModel") as mock_bokeh_model,
    ):
        mock_fig = Mock()
        mock_figure_func.return_value = mock_fig

        yield {
            "output_notebook": mock_output,
            "display": mock_display,
            "figure": mock_figure_func,
            "BokehModel": mock_bokeh_model,
        }


def test_initialization(mock_bokeh):
    """Test backend initialization."""
    backend = InlineBackend()

    assert len(backend._sources) == 0
    assert len(backend._figures) == 0
    assert len(backend._displayed) == 0


def test_start_calls_output_notebook(mock_bokeh):
    """Test that start calls output_notebook."""
    backend = InlineBackend()
    backend.start()

    mock_bokeh["output_notebook"].assert_called_once()


def test_stop_cleanup(mock_bokeh):
    """Test that stop can be called."""
    backend = InlineBackend()
    backend.stop()


def test_create_line_plot(mock_bokeh):
    """Test creating a line plot."""
    backend = InlineBackend()

    backend.create_figure("test_line", "Voltage", "Current", plot_type="line")

    assert "test_line" in backend._sources
    assert "test_line" in backend._figures
    assert backend._plot_specs["test_line"]["plot_type"] == "line"


def test_create_heatmap_plot(mock_bokeh):
    """Test creating a heatmap plot."""
    backend = InlineBackend()

    backend.create_figure(
        "test_heatmap",
        "X",
        "Y",
        plot_type="heatmap",
        z_label="Intensity",
        cell_size=(0.5, 0.5),
    )

    assert "test_heatmap" in backend._sources
    assert "test_heatmap" in backend._figures
    spec = backend._plot_specs["test_heatmap"]
    assert spec["plot_type"] == "heatmap"
    assert spec["dx"] == 0.5
    assert spec["dy"] == 0.5
    assert "mapper" in spec


def test_create_unknown_plot_type_raises(mock_bokeh):
    """Test that unknown plot type raises ValueError."""
    backend = InlineBackend()

    with pytest.raises(ValueError, match="Unknown plot type"):
        backend.create_figure("test", "X", "Y", plot_type="invalid")


def test_stream_data_line_plot_displays_on_first_call(mock_bokeh):
    """Test that line plot is displayed on first stream."""
    backend = InlineBackend()
    backend.create_figure("test", "X", "Y", plot_type="line")

    backend.stream_data("test", {"x": [1.0, 2.0], "y": [3.0, 4.0]})

    assert "test" in backend._displayed
    mock_bokeh["display"].assert_called_once()


def test_stream_data_heatmap_displays_on_first_call(mock_bokeh):
    """Test that heatmap is displayed on first stream."""
    backend = InlineBackend()
    backend.create_figure("test", "X", "Y", plot_type="heatmap", cell_size=(0.1, 0.1))

    backend.stream_data(
        "test", {"x": [1.0, 2.0], "y": [1.0, 2.0], "value": [5.0, 10.0]}
    )

    assert "test" in backend._displayed
    mock_bokeh["display"].assert_called_once()


def test_stream_data_heatmap_updates_color_mapper(mock_bokeh):
    """Test that streaming heatmap data updates color mapper range."""
    backend = InlineBackend()
    backend.create_figure("test", "X", "Y", plot_type="heatmap")

    backend.stream_data("test", {"x": [1.0], "y": [1.0], "value": [5.0]})
    spec = backend._plot_specs["test"]

    assert spec["value_min"] <= 5.0
    assert spec["value_max"] >= 5.0


def test_stream_data_with_rollover(mock_bokeh):
    """Test that rollover limits data size."""
    backend = InlineBackend()
    backend.create_figure("test", "X", "Y", plot_type="line")

    backend.stream_data(
        "test", {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}, rollover=2
    )

    source = backend._sources["test"]
    assert len(source.data["x"]) == 2
    assert source.data["x"] == [2.0, 3.0]


def test_stream_data_ignores_unknown_plot(mock_bokeh):
    """Test that streaming to non-existent plot is a no-op."""
    backend = InlineBackend()

    backend.stream_data("nonexistent", {"x": [1.0], "y": [2.0]})


def test_stream_data_accumulates_multiple_calls(mock_bokeh):
    """Test that multiple stream calls accumulate data."""
    backend = InlineBackend()
    backend.create_figure("test", "X", "Y", plot_type="line")

    backend.stream_data("test", {"x": [1.0], "y": [2.0]})
    backend.stream_data("test", {"x": [3.0], "y": [4.0]})

    source = backend._sources["test"]
    assert source.data["x"] == [1.0, 3.0]
    assert source.data["y"] == [2.0, 4.0]


def test_create_figure_idempotent(mock_bokeh):
    """Test that creating same figure twice doesn't duplicate."""
    backend = InlineBackend()

    backend.create_figure("test", "X", "Y")
    backend.create_figure("test", "X", "Y")

    assert len(backend._sources) == 1
