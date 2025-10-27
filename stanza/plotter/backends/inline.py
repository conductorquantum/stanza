"""Inline plotting backend for Jupyter notebooks.

Uses jupyter_bokeh extension for reliable live updates in all notebook environments.
Install: pip install jupyter_bokeh
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bokeh.plotting import output_notebook
from IPython.display import display
from jupyter_bokeh.widgets import BokehModel  # type: ignore[import-untyped]

from stanza.plotter.backends.utils import (
    PlotState,
    make_heatmap_plot,
    make_line_plot,
    prepare_heatmap_data,
)

if TYPE_CHECKING:
    pass


class InlineBackend:
    """Display live-updating plots directly in notebook cells."""

    def __init__(self) -> None:
        self._plots: dict[str, PlotState] = {}
        self._displayed: set[str] = set()

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
        if name in self._plots:
            return

        if plot_type == "line":
            plot_state = make_line_plot(name, x_label, y_label)
        elif plot_type == "heatmap":
            plot_state = make_heatmap_plot(name, x_label, y_label, z_label, cell_size)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        self._plots[name] = plot_state

    def stream_data(
        self, name: str, new_data: dict[str, Any], rollover: int | None = None
    ) -> None:
        """Add data to plot and display/update it.

        First call displays the plot, subsequent calls update via ColumnDataSource.
        """
        plot = self._plots.get(name)
        if plot is None:
            return

        if plot.spec.plot_type == "heatmap" and "value" in new_data:
            new_data = prepare_heatmap_data(new_data, plot.source.data, plot.spec)

        # Merge with existing data and apply rollover
        merged_data = {}
        for key, new_vals in new_data.items():
            merged = list(plot.source.data.get(key, [])) + new_vals
            if rollover and len(merged) > rollover:
                merged = merged[-rollover:]
            merged_data[key] = merged

        plot.source.data = merged_data

        if plot.spec.plot_type == "heatmap" and plot.spec.mapper:
            plot.spec.mapper.low = plot.spec.value_min
            plot.spec.mapper.high = plot.spec.value_max

        if name not in self._displayed:
            display(BokehModel(plot.figure))
            self._displayed.add(name)
