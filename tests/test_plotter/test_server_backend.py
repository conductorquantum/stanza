"""Tests for ServerBackend."""

from unittest.mock import Mock

import pytest
from bokeh.models import ColumnDataSource

from stanza.plotter.backends.server import ServerBackend
from stanza.plotter.backends.utils import (
    PlotSpec,
    PlotState,
    prepare_heatmap_data,
)


@pytest.fixture
def backend():
    """Create ServerBackend instance."""
    return ServerBackend(port=5050, daemon=True)


def test_initialization(backend):
    """Test backend initialization."""
    assert backend.port == 5050
    assert backend.daemon is True
    assert backend._running is False
    assert len(backend._plots) == 0
    assert len(backend._plot_specs) == 0


def test_start_raises_if_not_running():
    """Test that create_figure raises if server not started."""
    backend = ServerBackend(port=5051)

    with pytest.raises(RuntimeError, match="Server not started"):
        backend.create_figure("test", "X", "Y")


def test_create_figure_registers_line_plot():
    """Test that create_figure registers a line plot spec."""
    backend = ServerBackend(port=5052)
    backend._running = True

    backend.create_figure("test_line", "Voltage", "Current", plot_type="line")

    assert "test_line" in backend._plot_specs
    spec = backend._plot_specs["test_line"]
    assert spec.x_label == "Voltage"
    assert spec.y_label == "Current"
    assert spec.plot_type == "line"


def test_create_figure_registers_heatmap():
    """Test that create_figure registers a heatmap spec."""
    backend = ServerBackend(port=5053)
    backend._running = True

    backend.create_figure(
        "test_heatmap",
        "X",
        "Y",
        plot_type="heatmap",
        z_label="Intensity",
        cell_size=(0.1, 0.2),
    )

    assert "test_heatmap" in backend._plot_specs
    spec = backend._plot_specs["test_heatmap"]
    assert spec.plot_type == "heatmap"
    assert spec.z_label == "Intensity"
    assert spec.cell_size == (0.1, 0.2)


def test_stream_data_buffers_before_browser_connects():
    """Test that data is buffered if browser hasn't connected yet."""
    backend = ServerBackend(port=5054)
    backend._running = True

    backend.create_figure("test", "X", "Y")
    backend.stream_data("test", {"x": [1.0, 2.0], "y": [3.0, 4.0]})

    assert "test" in backend._buffer
    assert backend._buffer["test"]["x"] == [1.0, 2.0]
    assert backend._buffer["test"]["y"] == [3.0, 4.0]


def test_stream_data_multiple_times_accumulates_buffer():
    """Test that multiple stream calls accumulate data in buffer."""
    backend = ServerBackend(port=5055)
    backend._running = True

    backend.create_figure("test", "X", "Y")
    backend.stream_data("test", {"x": [1.0], "y": [2.0]})
    backend.stream_data("test", {"x": [3.0], "y": [4.0]})

    assert backend._buffer["test"]["x"] == [1.0, 3.0]
    assert backend._buffer["test"]["y"] == [2.0, 4.0]


def test_stop_stops_running_server():
    """Test that stop() sets running to False."""
    backend = ServerBackend(port=5056)
    backend._running = True
    backend._server = Mock()
    backend._server.io_loop = Mock()

    backend.stop()

    assert backend._running is False
    backend._server.io_loop.stop.assert_called_once()


def test_create_line_plot_with_buffered_data():
    """Test _create_line_plot uses buffered data."""
    backend = ServerBackend(port=5057)
    backend._running = True
    backend._doc = Mock()
    backend._plot_specs["test"] = PlotSpec(
        name="test", x_label="X", y_label="Y", plot_type="line"
    )
    backend._buffer["test"] = {"x": [1.0, 2.0], "y": [3.0, 4.0]}

    backend._create_line_plot("test")

    assert "test" in backend._plots
    assert "test" not in backend._buffer
    backend._doc.add_root.assert_called_once()


def test_create_heatmap_plot_spec():
    """Test _create_heatmap_plot sets up color mapper."""
    backend = ServerBackend(port=5058)
    backend._running = True
    backend._doc = Mock()
    backend._plot_specs["test"] = PlotSpec(
        name="test",
        x_label="X",
        y_label="Y",
        plot_type="heatmap",
        z_label="Value",
        cell_size=(0.5, 0.5),
    )

    backend._create_heatmap_plot("test")

    assert "test" in backend._plots
    plot = backend._plots["test"]
    assert plot.spec.mapper is not None
    assert plot.spec.dx == 0.5
    assert plot.spec.dy == 0.5
    backend._doc.add_root.assert_called_once()


def test_create_plot_raises_on_unknown_type():
    """Test that _create_plot raises on unknown plot type."""
    backend = ServerBackend(port=5059)
    backend._plot_specs["test"] = PlotSpec(
        name="test", x_label="X", y_label="Y", plot_type="unknown"
    )

    with pytest.raises(ValueError, match="Unknown plot type"):
        backend._create_plot("test")


def test_prepare_heatmap_data_updates_color_range():
    """Test that prepare_heatmap_data calculates cell sizes and updates range."""
    backend = ServerBackend(port=5060)
    spec = PlotSpec(name="test", plot_type="heatmap", x_label="X", y_label="Y")
    source = ColumnDataSource(data={"x": [], "y": [], "value": []})
    plot_state = PlotState(source=source, figure=Mock(), spec=spec)
    backend._plots["test"] = plot_state

    data = {"x": [1.0, 2.0], "y": [1.0, 1.0], "value": [5.0, 10.0]}
    result = prepare_heatmap_data(data, plot_state.source.data, plot_state.spec)

    assert "width" in result
    assert "height" in result
    assert plot_state.spec.value_min <= 5.0
    assert plot_state.spec.value_max >= 10.0


def test_start_is_idempotent():
    """Test that calling start multiple times doesn't cause issues."""
    backend = ServerBackend(port=5061)
    backend._running = True

    backend.start()
    assert backend._running is True


def test_create_figure_is_idempotent():
    """Test that creating same figure twice doesn't duplicate."""
    backend = ServerBackend(port=5062)
    backend._running = True

    backend.create_figure("test", "X", "Y")
    backend.create_figure("test", "X", "Y")

    assert len(backend._plot_specs) == 1


def test_create_figure_with_doc_adds_callback():
    """Test that create_figure adds callback when doc is already connected."""
    backend = ServerBackend(port=5063)
    backend._running = True
    backend._doc = Mock()
    backend._doc.add_next_tick_callback = Mock()

    backend.create_figure("test", "X", "Y")

    backend._doc.add_next_tick_callback.assert_called_once()


def test_stream_data_with_doc_adds_callback():
    """Test that streaming data with connected doc adds callback."""
    backend = ServerBackend(port=5064)
    backend._running = True
    backend._doc = Mock()

    spec = PlotSpec(name="test", plot_type="line", x_label="X", y_label="Y")
    source = ColumnDataSource(data={"x": [], "y": []})
    plot_state = PlotState(source=source, figure=Mock(), spec=spec)
    backend._plots["test"] = plot_state

    backend.stream_data("test", {"x": [1.0], "y": [2.0]})

    backend._doc.add_next_tick_callback.assert_called_once()
