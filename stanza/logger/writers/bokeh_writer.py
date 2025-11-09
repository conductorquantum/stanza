"""Data writer for live plotting with Bokeh."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData
from stanza.logger.writers.base import AbstractDataWriter

logger = logging.getLogger(__name__)


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
        self._name_counters: dict[str, int] = {}  # Track counters for duplicate names
        self._initialized: bool = False

    def initialize_session(self, metadata: SessionMetadata) -> None:
        """Start of new session."""
        self._initialized = True

    def write_measurement(self, data: MeasurementData) -> None:
        """Write single measurement (not used for plotting)."""
        pass

    def write_sweep(self, data: SweepData) -> None:
        """Stream sweep data to plot."""
        # Handle multi-dimensional x_data by using only changing gates
        x_data = data.x_data
        if x_data.ndim == 2 and x_data.shape[1] > 0:
            # Find which gates are actually changing (have variance)
            gate_variances = np.var(x_data, axis=0)
            changing_gates_mask = gate_variances > 1e-6
            num_changing = np.sum(changing_gates_mask)
            
            if num_changing == 0:
                # All gates constant (edge case), use first gate for 1D plot
                x_data = x_data[:, 0]
            elif num_changing == 1:
                # Single changing gate: use it for 1D plot
                changing_indices = np.where(changing_gates_mask)[0]
                x_data = x_data[:, changing_indices[0]]
            elif num_changing == 2:
                # Two changing gates: use both for 2D plot
                changing_indices = np.where(changing_gates_mask)[0]
                x_data = x_data[:, changing_indices]
            else:
                # More than 2 changing gates: average them for 1D plot
                x_data = np.mean(x_data[:, changing_gates_mask], axis=1)
        
        # Determine dimension after reduction
        dim = data.metadata.get("_dim") or (
            1 if x_data.ndim == 1 else x_data.shape[1] if x_data.ndim == 2 else 1
        )

        if dim not in (1, 2):
            raise ValueError(f"Only 1D and 2D supported, got {dim}D")

        # Generate unique plot name if duplicate exists
        plot_name = self._get_unique_plot_name(data.name)
        
        # Create modified SweepData with reduced x_data and unique name
        reduced_data = SweepData(
            name=plot_name,
            x_data=x_data,
            y_data=data.y_data,
            x_label=data.x_label,
            y_label=data.y_label,
            metadata={**data.metadata, "_dim": dim},
            timestamp=data.timestamp,
            session_id=data.session_id,
            routine_name=data.routine_name,
        )

        if not self._plot_name_exists(plot_name):
            (self._create_1d_plot if dim == 1 else self._create_2d_plot)(reduced_data)

        (self._stream_1d if dim == 1 else self._stream_2d)(reduced_data)

    def _plot_name_exists(self, name: str) -> bool:
        """Check if a plot name exists in either our registry or the backend's registry.
        
        Args:
            name: Plot name to check
            
        Returns:
            True if plot exists, False otherwise
        """
        # Check our own registry
        if name in self._plots:
            return True
        
        # Check backend's registry (both backends have _plots or _plot_specs)
        backend_plots = getattr(self.backend, "_plots", {})
        backend_specs = getattr(self.backend, "_plot_specs", {})
        
        return name in backend_plots or name in backend_specs
    
    def _get_unique_plot_name(self, base_name: str) -> str:
        """Generate a unique plot name by appending a counter if name already exists.
        
        Scans all existing plots to find the highest counter, ensuring no duplicates
        even if the counter variable gets out of sync.
        
        Args:
            base_name: Original plot name
            
        Returns:
            Unique plot name (base_name if first occurrence, base_name_N for duplicates)
        """
        # Check if base name exists (in our registry or backend's)
        if not self._plot_name_exists(base_name):
            # First occurrence, use base name
            self._name_counters[base_name] = 0
            return base_name
        
        # Name exists, scan all existing plots to find the highest counter
        # This ensures we don't create duplicates even if counter gets out of sync
        max_counter = -1
        
        # Check our registry
        for plot_name in self._plots.keys():
            if plot_name == base_name:
                max_counter = max(max_counter, 0)
            elif plot_name.startswith(f"{base_name}_"):
                try:
                    counter = int(plot_name.split("_", 1)[1])
                    max_counter = max(max_counter, counter)
                except (ValueError, IndexError):
                    pass
        
        # Check backend registries
        backend_plots = getattr(self.backend, "_plots", {})
        backend_specs = getattr(self.backend, "_plot_specs", {})
        
        for plot_name in list(backend_plots.keys()) + list(backend_specs.keys()):
            if plot_name == base_name:
                max_counter = max(max_counter, 0)
            elif plot_name.startswith(f"{base_name}_"):
                try:
                    counter = int(plot_name.split("_", 1)[1])
                    max_counter = max(max_counter, counter)
                except (ValueError, IndexError):
                    pass
        
        # Start from max_counter + 1 and find first available name
        counter = max_counter + 1
        while True:
            candidate_name = f"{base_name}_{counter}"
            if not self._plot_name_exists(candidate_name):
                # Update our counter tracking
                self._name_counters[base_name] = counter
                return candidate_name
            counter += 1

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
            try:
                self.backend.push_updates()
            except AttributeError as e:
                # Handle Bokeh version compatibility issues with DocumentCallbackManager
                # Bokeh updates are asynchronous and don't require synchronous flushing
                if "_change_callbacks" in str(e) or "DocumentCallbackManager" in str(e):
                    logger.debug(
                        "Bokeh flush skipped due to version compatibility: %s", e
                    )
                else:
                    raise

    def finalize_session(self, metadata: SessionMetadata | None = None) -> None:
        """End of session."""
        self.flush()


__all__ = ["BokehLiveWriter"]
