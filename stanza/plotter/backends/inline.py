"""Inline plotting backend for Jupyter notebooks.

Uses jupyter_bokeh extension for reliable live updates in all notebook environments.
Install: pip install jupyter_bokeh
"""

from __future__ import annotations

from typing import Any

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_notebook
from IPython.display import display
from jupyter_bokeh.widgets import BokehModel  # type: ignore[import-untyped]


class InlineBackend:
    """Display live-updating plots directly in notebook cells."""

    def __init__(self) -> None:
        self._sources: dict[str, ColumnDataSource] = {}
        self._figures: dict[str, figure] = {}
        self._displayed: set[str] = set()
        self._plot_specs: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        """Initialize Bokeh notebook output."""
        output_notebook()

    def stop(self) -> None:
        """Clean up resources."""
        pass

    def create_figure(
        self,
        name: str,
        x_label: str,
        y_label: str,
        plot_type: str = "line",
        z_label: str | None = None,
        cell_size: tuple[float, float] | None = None,
    ) -> None:
        """Create a new plot configuration."""
        if name in self._sources:
            return

        if plot_type == "line":
            self._create_line_plot(name, x_label, y_label)
        elif plot_type == "heatmap":
            self._create_heatmap_plot(name, x_label, y_label, z_label, cell_size)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def _create_line_plot(self, name: str, x_label: str, y_label: str) -> None:
        """Create 1D line plot."""
        source = ColumnDataSource(data={"x": [], "y": []})
        self._sources[name] = source

        fig = figure(title=name, width=800, height=400)
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label
        fig.line("x", "y", source=source, line_width=2, color="navy")
        self._figures[name] = fig

        self._plot_specs[name] = {"plot_type": "line"}

    def _create_heatmap_plot(
        self,
        name: str,
        x_label: str,
        y_label: str,
        z_label: str | None,
        cell_size: tuple[float, float] | None,
    ) -> None:
        """Create 2D heatmap plot."""
        from bokeh.models import LinearColorMapper
        from bokeh.models.annotations import ColorBar
        from bokeh.palettes import Viridis256

        source = ColumnDataSource(
            data={"x": [], "y": [], "value": [], "width": [], "height": []}
        )
        self._sources[name] = source

        # Create color mapper
        mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)

        fig = figure(
            title=name,
            width=800,
            height=600,
        )
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label

        # Add rect glyph
        fig.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=source,
            fill_color={"field": "value", "transform": mapper},
            line_color=None,
        )

        # Add color bar
        color_bar = ColorBar(
            color_mapper=mapper,
            width=8,
            location=(0, 0),
            title=z_label or "Value",
        )
        fig.add_layout(color_bar, "right")

        self._figures[name] = fig
        self._plot_specs[name] = {
            "plot_type": "heatmap",
            "mapper": mapper,
            "dx": cell_size[0] if cell_size else None,
            "dy": cell_size[1] if cell_size else None,
            "value_min": float("inf"),
            "value_max": float("-inf"),
        }

    def stream_data(
        self, name: str, new_data: dict[str, Any], rollover: int | None = None
    ) -> None:
        """Add data to plot and display/update it.

        On first call: displays the plot
        On subsequent calls: updates automatically via ColumnDataSource
        """
        if name not in self._sources:
            return

        source = self._sources[name]
        spec = self._plot_specs.get(name, {})

        if spec.get("plot_type") == "heatmap" and "value" in new_data:
            # Prepare heatmap data
            new_data = self._prepare_heatmap_data(name, new_data)

        # Merge with existing data
        merged_data = {}
        for key, new_vals in new_data.items():
            merged = list(source.data.get(key, [])) + new_vals
            merged_data[key] = (
                merged[-rollover:] if rollover and len(merged) > rollover else merged
            )

        source.data = merged_data

        # Update color mapper for heatmaps
        if spec.get("plot_type") == "heatmap" and "mapper" in spec:
            spec["mapper"].low = spec["value_min"]
            spec["mapper"].high = spec["value_max"]

        # Display on first stream
        if name not in self._displayed:
            display(BokehModel(self._figures[name]))
            self._displayed.add(name)

    def _prepare_heatmap_data(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate rect sizes and update color range for heatmap data."""
        import numpy as np

        spec = self._plot_specs[name]
        source = self._sources[name]

        def calc_delta(key: str) -> float:
            """Calculate minimum delta from existing + new data."""
            if key not in data or len(data[key]) == 0:
                return 0.1
            existing = list(source.data.get(key, []))
            all_vals = existing + data[key]
            if len(all_vals) > 1:
                unique = sorted(set(all_vals))
                if len(unique) > 1:
                    return float(min(np.diff(unique)))
            return 0.1

        if spec["dx"] is None:
            spec["dx"] = calc_delta("x")
        if spec["dy"] is None:
            spec["dy"] = calc_delta("y")

        n = len(data.get("value", []))
        data["width"] = [spec["dx"]] * n
        data["height"] = [spec["dy"]] * n

        if "value" in data:
            values = np.array(data["value"])
            spec["value_min"] = min(spec["value_min"], float(values.min()))
            spec["value_max"] = max(spec["value_max"], float(values.max()))

        return data
