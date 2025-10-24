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
            max_points: Maximum points per plot (older data rolls off for 1D)
        """
        self.backend = backend
        self.max_points = max_points
        self._plots: dict[str, dict[str, Any]] = {}  # Store plot metadata
        self._initialized: bool = False

    def initialize_session(self, metadata: SessionMetadata) -> None:
        """Start of new session."""
        self._initialized = True

    def write_measurement(self, data: MeasurementData) -> None:
        """Write single measurement (not used for plotting)."""
        pass

    def write_sweep(self, data: SweepData) -> None:
        """Stream sweep data to plot."""
        dim = data.metadata.get("_dim") or (
            1 if data.x_data.ndim == 1 else data.x_data.shape[1]
        )

        if dim not in (1, 2):
            raise ValueError(f"Only 1D and 2D supported, got {dim}D")

        if data.name not in self._plots:
            (self._create_1d_plot if dim == 1 else self._create_2d_plot)(data)

        (self._stream_1d if dim == 1 else self._stream_2d)(data)

    def _create_1d_plot(self, data: SweepData) -> None:
        """Create 1D line plot."""
        self.backend.create_figure(
            name=data.name,
            x_label=data.x_label if isinstance(data.x_label, str) else "X",
            y_label=data.y_label or "Y",
            plot_type="line",
        )
        self._plots[data.name] = {"dim": 1}

    def _create_2d_plot(self, data: SweepData) -> None:
        """Create 2D heatmap using rect glyph."""
        labels = data.x_label if isinstance(data.x_label, list) else ["X", "Y"]
        x_label = labels[0] if len(labels) > 0 else "X"
        y_label = labels[1] if len(labels) > 1 else "Y"

        self.backend.create_figure(
            name=data.name,
            x_label=x_label,
            y_label=y_label,
            z_label=data.y_label or "Value",
            plot_type="heatmap",
            cell_size=data.metadata.get("cell_size"),
        )
        self._plots[data.name] = {"dim": 2}

    def _stream_1d(self, data: SweepData) -> None:
        """Stream 1D data to line plot."""
        self.backend.stream_data(
            name=data.name,
            new_data={"x": list(data.x_data), "y": list(data.y_data)},
            rollover=self.max_points,
        )

    def _stream_2d(self, data: SweepData) -> None:
        """Stream 2D data to heatmap."""
        self.backend.stream_data(
            name=data.name,
            new_data={
                "x": data.x_data[:, 0].tolist(),
                "y": data.x_data[:, 1].tolist(),
                "value": data.y_data.tolist(),
            },
            rollover=None,
        )

    def flush(self) -> None:
        """Flush any pending updates."""
        if hasattr(self.backend, "push_updates"):
            self.backend.push_updates()

    def finalize_session(self, metadata: SessionMetadata | None = None) -> None:
        """End of session."""
        self.flush()


__all__ = ["BokehLiveWriter"]
