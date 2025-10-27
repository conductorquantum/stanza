"""Shared utilities for plotting backends.

This module centralizes small, reusable helpers used by multiple
plotting backends to keep implementations minimal and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.models.annotations import ColorBar
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

if TYPE_CHECKING:
    pass

# Plot dimensions
LINE_PLOT_WIDTH = 800
LINE_PLOT_HEIGHT = 400
HEATMAP_WIDTH = 800
HEATMAP_HEIGHT = 600


@dataclass
class PlotSpec:
    """Plot specification with type-safe fields."""

    name: str
    plot_type: str  # "line" or "heatmap"
    x_label: str
    y_label: str
    z_label: str | None = None
    cell_size: tuple[float, float] | None = None

    # Heatmap-specific fields
    mapper: LinearColorMapper | None = None
    dx: float | None = None
    dy: float | None = None
    value_min: float = float("inf")
    value_max: float = float("-inf")


@dataclass
class PlotState:
    """Complete plot state: source, figure, and spec in one place.

    Single source of truth for a plot's data, visualization, and configuration.
    Ownership is clear: whoever has the PlotState owns the entire plot.
    """

    source: ColumnDataSource
    figure: Any  # bokeh Figure
    spec: PlotSpec


def prepare_heatmap_data(
    data: dict[str, Any],
    existing_data: dict[str, Any],
    spec: PlotSpec,
) -> dict[str, Any]:
    """Calculate rect sizes and update color range for heatmap data.

    Args:
        data: New data to add (x, y, value)
        existing_data: Existing data in the data source
        spec: Plot specification with dx, dy, value_min, value_max

    Returns:
        Updated data with width and height fields added
    """

    def calc_delta(key: str) -> float:
        """Calculate minimum delta from existing + new data."""
        if key not in data or len(data[key]) == 0:
            return 0.1
        existing_vals = list(existing_data.get(key, []))
        all_vals = existing_vals + data[key]
        if len(all_vals) > 1:
            unique = sorted(set(all_vals))
            if len(unique) > 1:
                return float(min(np.diff(unique)))
        return 0.1

    # Calculate cell sizes if not set
    if spec.dx is None:
        spec.dx = calc_delta("x")
    if spec.dy is None:
        spec.dy = calc_delta("y")

    # Add width/height to data
    n = len(data.get("value", []))
    data["width"] = [spec.dx] * n
    data["height"] = [spec.dy] * n

    # Update value range for color mapping
    if "value" in data:
        values = np.array(data["value"])
        spec.value_min = min(spec.value_min, float(values.min()))
        spec.value_max = max(spec.value_max, float(values.max()))

    return data


def make_line_plot(
    name: str,
    x_label: str,
    y_label: str,
) -> PlotState:
    """Create a 1D line plot.

    Returns complete PlotState with source, figure, and spec.
    """
    source = ColumnDataSource(data={"x": [], "y": []})
    fig = figure(title=name, width=LINE_PLOT_WIDTH, height=LINE_PLOT_HEIGHT)
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = y_label
    fig.line("x", "y", source=source, line_width=2, color="navy")

    spec = PlotSpec(
        name=name,
        plot_type="line",
        x_label=x_label,
        y_label=y_label,
    )
    return PlotState(source=source, figure=fig, spec=spec)


def make_heatmap_plot(
    name: str,
    x_label: str,
    y_label: str,
    z_label: str | None,
    cell_size: tuple[float, float] | None,
) -> PlotState:
    """Create a 2D heatmap plot.

    Returns complete PlotState with source, figure, and spec.
    """
    source = ColumnDataSource(
        data={"x": [], "y": [], "value": [], "width": [], "height": []}
    )

    mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)

    fig = figure(title=name, width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = y_label

    fig.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=source,
        fill_color={"field": "value", "transform": mapper},
        line_color=None,
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        width=8,
        location=(0, 0),
        title=z_label or "Value",
    )
    fig.add_layout(color_bar, "right")

    spec = PlotSpec(
        name=name,
        plot_type="heatmap",
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        cell_size=cell_size,
        mapper=mapper,
        dx=cell_size[0] if cell_size else None,
        dy=cell_size[1] if cell_size else None,
    )
    return PlotState(source=source, figure=fig, spec=spec)
