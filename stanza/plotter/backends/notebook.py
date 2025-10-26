"""Bokeh backend for Jupyter notebooks using push_notebook."""

import logging
from typing import TYPE_CHECKING, Any

from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure

if TYPE_CHECKING:
    from bokeh.plotting._figure import figure as Figure

logger = logging.getLogger(__name__)


class NotebookBackend:
    """Bokeh backend for Jupyter notebooks using push_notebook."""

    def __init__(self) -> None:
        try:
            output_notebook(hide_banner=True)
        except Exception as e:
            logger.warning(
                f"Failed to initialize notebook output. "
                f"Make sure you're running in a Jupyter notebook. Error: {e}"
            )

        self._figures: dict[str, Figure] = {}
        self._handles: dict[str, Any] = {}

    def create_figure(
        self, name: str, x_label: str, y_label: str
    ) -> tuple["Figure", bool]:
        """Create new figure but don't display until renderers are added."""
        fig = figure(
            title=name,
            width=800,
            height=400,
            sizing_mode="stretch_width",
        )
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label

        self._figures[name] = fig
        return fig, True

    def display_figure(self, fig: Any) -> None:
        """Display the figure after renderers have been added."""
        try:
            for name, figure_obj in self._figures.items():
                if figure_obj is fig and name not in self._handles:
                    handle = show(figure_obj, notebook_handle=True)
                    self._handles[name] = handle
                    break
        except Exception as e:
            logger.error(f"Failed to display figure: {e}")

    def push_updates(self) -> None:
        """Push updates to notebook display."""
        for handle in self._handles.values():
            try:
                push_notebook(handle=handle)
            except Exception as e:
                logger.error(f"Failed to push updates to notebook: {e}")
