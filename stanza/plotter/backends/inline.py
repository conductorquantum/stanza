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

    def start(self) -> None:
        """Initialize Bokeh notebook output."""
        output_notebook()

    def stop(self) -> None:
        """Clean up resources."""
        pass

    def create_figure(
        self, name: str, x_label: str, y_label: str
    ) -> tuple[ColumnDataSource, bool]:
        """Create a new plot configuration.

        Returns:
            (source, is_new) tuple
        """
        if name in self._sources:
            return self._sources[name], False

        source = ColumnDataSource(data={"x": [], "y": []})
        self._sources[name] = source

        fig = figure(title=name, width=800, height=400)
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label
        fig.line("x", "y", source=source, line_width=2, color="navy")
        self._figures[name] = fig

        return source, True

    def display_figure(self, _source: Any) -> None:
        """Not needed - display happens on first data stream."""
        pass

    def stream_data(self, name: str, new_data: dict[str, Any], rollover: int) -> None:
        """Add data to plot and display/update it.

        On first call: displays the plot
        On subsequent calls: updates automatically via ColumnDataSource
        """
        if name not in self._sources:
            return

        source = self._sources[name]

        x = list(source.data["x"]) + new_data["x"]
        y = list(source.data["y"]) + new_data["y"]

        if rollover and len(x) > rollover:
            x = x[-rollover:]
            y = y[-rollover:]

        source.data = {"x": x, "y": y}

        if name not in self._displayed:
            display(BokehModel(self._figures[name]))
            self._displayed.add(name)

    def push_updates(self) -> None:
        """Not needed - updates are automatic."""
        pass
