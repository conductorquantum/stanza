"""Tests for plotter utilities."""

from stanza.plotter.backends.utils import PlotSpec, prepare_heatmap_data


def test_prepare_heatmap_data_with_existing_data():
    """Test prepare_heatmap_data with existing data points."""
    data = {"x": [1.0, 2.0], "y": [1.0, 1.0], "value": [5.0, 10.0]}
    existing_data = {"x": [0.0], "y": [0.0], "value": [3.0]}
    spec = PlotSpec(name="test", plot_type="heatmap", x_label="X", y_label="Y")

    result = prepare_heatmap_data(data, existing_data, spec)

    assert "width" in result
    assert "height" in result
    assert len(result["width"]) == 2
    assert spec.value_min == 5.0
    assert spec.value_max == 10.0


def test_prepare_heatmap_data_empty_key():
    """Test prepare_heatmap_data when a key is missing from new data."""
    data = {"y": [1.0, 2.0], "value": [5.0, 10.0]}
    existing_data = {"x": [0.0], "y": [0.0], "value": [3.0]}
    spec = PlotSpec(name="test", plot_type="heatmap", x_label="X", y_label="Y", dy=0.5)

    result = prepare_heatmap_data(data, existing_data, spec)

    assert spec.dx == 0.1
    assert "width" in result
    assert "height" in result


def test_prepare_heatmap_data_with_preset_cell_size():
    """Test prepare_heatmap_data when cell sizes are already set."""
    data = {"x": [1.0, 2.0], "y": [1.0, 1.0], "value": [5.0, 10.0]}
    existing_data = {}
    spec = PlotSpec(
        name="test", plot_type="heatmap", x_label="X", y_label="Y", dx=0.5, dy=0.3
    )

    result = prepare_heatmap_data(data, existing_data, spec)

    assert spec.dx == 0.5
    assert spec.dy == 0.3
    assert result["width"] == [0.5, 0.5]
    assert result["height"] == [0.3, 0.3]


def test_prepare_heatmap_data_no_value_field():
    """Test prepare_heatmap_data when value field is missing."""
    data = {"x": [1.0, 2.0], "y": [1.0, 1.0]}
    existing_data = {}
    spec = PlotSpec(name="test", plot_type="heatmap", x_label="X", y_label="Y")

    prepare_heatmap_data(data, existing_data, spec)

    assert spec.value_min == float("inf")
    assert spec.value_max == float("-inf")
