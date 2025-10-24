"""Data writer for live plotting with Bokeh."""

from __future__ import annotations

from typing import Any

from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData
from stanza.logger.writers.base import AbstractDataWriter


class BokehLiveWriter(AbstractDataWriter):
    """Stream sweep data to Bokeh plots."""

    def __init__(self, backend: Any, max_points: int = 1000) -> None:
        """
        Args:
            backend: ServerBackend or InlineBackend instance
            max_points: Maximum points per plot (older data rolls off)
        """
        self.backend = backend
        self.max_points = max_points
        self._plots: set[str] = set()
        self._figures: dict[str, Any] = {}
        self._sources: dict[str, Any] = {}
        self._initialized: bool = False

    def initialize_session(self, metadata: SessionMetadata) -> None:
        """Start of new session."""
        self._initialized = True

    def write_measurement(self, data: MeasurementData) -> None:
        """Write single measurement (not used for plotting)."""
        pass

    def write_sweep(self, data: SweepData) -> None:
        """Stream sweep data to plot."""
        if data.name not in self._plots:
            fig = self.backend.create_figure(
                name=data.name,
                x_label=data.x_label or "X",
                y_label=data.y_label or "Y",
            )
            self._plots.add(data.name)
            if isinstance(fig, tuple):
                source, _ = fig
                self._sources[data.name] = source
                self._figures[data.name] = None
            else:
                self._figures[data.name] = fig
                self._sources[data.name] = None

        # Stream data to plot
        self.backend.stream_data(
            name=data.name,
            new_data={"x": list(data.x_data), "y": list(data.y_data)},
            rollover=self.max_points,
        )

    def flush(self) -> None:
        """Flush any pending updates."""
        self.backend.push_updates()

    def finalize_session(self, metadata: SessionMetadata | None = None) -> None:
        """End of session."""
        self.flush()


__all__ = ["BokehLiveWriter"]
