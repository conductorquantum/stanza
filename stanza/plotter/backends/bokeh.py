"""Bokeh backend protocol."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BokehBackend(Protocol):
    """Protocol for bokeh rendering backends."""

    def create_figure(self, name: str, x_label: str, y_label: str) -> tuple[Any, bool]:
        """Create a new bokeh figure.

        Returns:
            Tuple of (figure, needs_display) where needs_display indicates
            if display_figure should be called after adding renderers.
        """
        ...

    def display_figure(self, fig: Any) -> None:
        """Display the figure after renderers have been added."""
        ...

    def push_updates(self) -> None:
        """Push all pending updates to display."""
        ...
