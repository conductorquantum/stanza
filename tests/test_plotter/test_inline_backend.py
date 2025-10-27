"""Tests for InlineBackend."""

from unittest.mock import Mock, patch

import pytest
from bokeh.models import ColumnDataSource, LinearColorMapper

from stanza.plotter.backends.inline import InlineBackend
from stanza.plotter.backends.utils import PlotSpec, PlotState


@pytest.fixture
def mock_bokeh():
    """Mock bokeh and jupyter imports."""
    with (
        patch("stanza.plotter.backends.inline.output_notebook") as mock_output,
        patch("stanza.plotter.backends.inline.display") as mock_display,
        patch("stanza.plotter.backends.inline.make_line_plot") as mock_make_line,
        patch("stanza.plotter.backends.inline.make_heatmap_plot") as mock_make_heatmap,
        patch("stanza.plotter.backends.inline.BokehModel") as mock_bokeh_model,
    ):
        mock_fig = Mock()
        mock_source = Mock(spec=ColumnDataSource)
        mock_source.data = {}
        mock_mapper = Mock(spec=LinearColorMapper)

        line_spec = PlotSpec(name="test", plot_type="line", x_label="X", y_label="Y")
        heatmap_spec = PlotSpec(
            name="test",
            plot_type="heatmap",
            x_label="X",
            y_label="Y",
            z_label="Value",
            cell_size=(0.5, 0.5),
            mapper=mock_mapper,
            dx=0.5,
            dy=0.5,
        )

        line_state = PlotState(source=mock_source, figure=mock_fig, spec=line_spec)
        heatmap_state = PlotState(
            source=mock_source, figure=mock_fig, spec=heatmap_spec
        )

        mock_make_line.return_value = line_state
        mock_make_heatmap.return_value = heatmap_state

        yield {
            "output_notebook": mock_output,
            "display": mock_display,
            "make_line_plot": mock_make_line,
            "make_heatmap_plot": mock_make_heatmap,
            "BokehModel": mock_bokeh_model,
        }


def test_initialization(mock_bokeh):
    """Test backend initialization."""
    backend = InlineBackend()

    assert len(backend._plots) == 0
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

    assert "test_line" in backend._plots
    assert backend._plots["test_line"].spec.plot_type == "line"


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

    assert "test_heatmap" in backend._plots
    plot = backend._plots["test_heatmap"]
    assert plot.spec.plot_type == "heatmap"
    assert plot.spec.dx == 0.5
    assert plot.spec.dy == 0.5
    assert plot.spec.mapper is not None


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
    plot = backend._plots["test"]

    assert plot.spec.value_min <= 5.0
    assert plot.spec.value_max >= 5.0


def test_stream_data_with_rollover(mock_bokeh):
    """Test that rollover limits data size."""
    backend = InlineBackend()
    backend.create_figure("test", "X", "Y", plot_type="line")

    backend.stream_data(
        "test", {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}, rollover=2
    )

    plot = backend._plots["test"]
    assert len(plot.source.data["x"]) == 2
    assert plot.source.data["x"] == [2.0, 3.0]


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

    plot = backend._plots["test"]
    assert plot.source.data["x"] == [1.0, 3.0]
    assert plot.source.data["y"] == [2.0, 4.0]


def test_create_figure_idempotent(mock_bokeh):
    """Test that creating same figure twice doesn't duplicate."""
    backend = InlineBackend()

    backend.create_figure("test", "X", "Y")
    backend.create_figure("test", "X", "Y")

    assert len(backend._plots) == 1
