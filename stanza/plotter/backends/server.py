"""Server-based plotting backend for live visualization in browser.

Runs a Bokeh server in a background thread, streams data updates via WebSocket.
Works from any environment: scripts, notebooks, or interactive sessions.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server


class ServerBackend:
    """Live plotting in browser via Bokeh server.

    Can run as either:
    - Embedded server (daemon=True): Runs in background thread, dies with parent process
    - Persistent server (daemon=False): Runs in main thread, persists until explicitly stopped
    """

    def __init__(self, port: int = 5006, daemon: bool = True) -> None:
        self.port = port
        self.daemon = daemon
        self._sources: dict[str, ColumnDataSource] = {}
        self._server: Server | None = None
        self._doc: Any = None
        self._running = False

        # Plot configurations and data buffered before browser connects
        self._plot_specs: dict[str, dict[str, Any]] = {}
        self._buffer: dict[str, dict[str, list[Any]]] = {}

    def start(self, block: bool = False) -> None:
        """Start Bokeh server.

        Args:
            block: If True, blocks until server is stopped (for persistent mode).
                   If False, runs in background thread (for embedded mode).
        """
        if self._running:
            return

        def make_document(doc: Any) -> None:
            """Initialize document when browser connects."""
            self._doc = doc

            for name in self._plot_specs.keys():
                self._create_plot(name)

        def run_server() -> None:
            """Server thread: create event loop and start server."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            app = Application(FunctionHandler(make_document))
            self._server = Server(
                {"/": app},
                port=self.port,
                allow_websocket_origin=[f"localhost:{self.port}"],
            )

            self._server.start()
            self._server.io_loop.start()

        if block or not self.daemon:
            # Run in main thread (blocks)
            self._running = True
            run_server()
        else:
            # Run in background daemon thread
            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()
            self._running = True
            time.sleep(1.0)  # Give server time to start

    def stop(self) -> None:
        """Stop the Bokeh server."""
        if self._server and self._running:
            self._server.io_loop.stop()
            self._running = False

    def create_figure(
        self,
        name: str,
        x_label: str,
        y_label: str,
        plot_type: str = "line",
        z_label: str | None = None,
        cell_size: tuple[float, float] | None = None,
    ) -> None:
        """Register a new plot. Created when browser connects."""
        if not self._running:
            raise RuntimeError("Server not started")

        if name in self._plot_specs:
            return

        self._plot_specs[name] = {
            "x_label": x_label,
            "y_label": y_label,
            "plot_type": plot_type,
            "z_label": z_label,
            "cell_size": cell_size,
        }

        # If browser already connected, create plot immediately
        if self._doc is not None and name not in self._sources:
            self._doc.add_next_tick_callback(lambda: self._create_plot(name))

    def _create_plot(self, name: str) -> None:
        """Create plot based on spec."""
        spec = self._plot_specs[name]

        if spec["plot_type"] == "line":
            self._create_line_plot(name, spec)
        elif spec["plot_type"] == "heatmap":
            self._create_heatmap_plot(name, spec)
        else:
            raise ValueError(f"Unknown plot type: {spec['plot_type']}")

    def _create_line_plot(self, name: str, spec: dict[str, Any]) -> None:
        """Create 1D line plot."""
        data = self._buffer.pop(name, {"x": [], "y": []})
        source = ColumnDataSource(data=data)
        self._sources[name] = source

        plot = figure(title=name, width=800, height=400)
        plot.xaxis.axis_label = spec["x_label"]
        plot.yaxis.axis_label = spec["y_label"]
        plot.line("x", "y", source=source, line_width=2, color="navy")

        if self._doc:
            self._doc.add_root(plot)

    def _create_heatmap_plot(self, name: str, spec: dict[str, Any]) -> None:
        """Create 2D heatmap with rect glyph and linear color mapping."""
        from bokeh.models import LinearColorMapper
        from bokeh.models.annotations import ColorBar
        from bokeh.palettes import Viridis256

        data = self._buffer.pop(
            name, {"x": [], "y": [], "value": [], "width": [], "height": []}
        )
        source = ColumnDataSource(data=data)
        self._sources[name] = source

        # Create color mapper
        mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)

        plot = figure(
            title=name,
            width=800,
            height=600,
        )
        plot.xaxis.axis_label = spec["x_label"]
        plot.yaxis.axis_label = spec["y_label"]

        # Add rect glyph with color mapping
        plot.rect(
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
            title=spec.get("z_label", "Value"),
        )
        plot.add_layout(color_bar, "right")

        # Store plot state for dynamic updates
        cell_size = spec.get("cell_size", (None, None))
        self._plot_specs[name].update(
            {
                "mapper": mapper,
                "dx": cell_size[0] if cell_size else None,
                "dy": cell_size[1] if cell_size else None,
                "value_min": float("inf"),
                "value_max": float("-inf"),
            }
        )

        if self._doc:
            self._doc.add_root(plot)

    def stream_data(
        self, name: str, new_data: dict[str, Any], rollover: int | None = None
    ) -> None:
        """Add data to plot. Buffers if browser not yet connected."""
        # Buffer data if plot not yet created
        if name not in self._sources:
            if name not in self._buffer:
                self._buffer[name] = {k: [] for k in new_data.keys()}
            for key, values in new_data.items():
                self._buffer[name].setdefault(key, []).extend(values)
            return

        # For heatmap, calculate rect sizes and update color range
        spec = self._plot_specs.get(name, {})
        if spec.get("plot_type") == "heatmap" and "value" in new_data:
            new_data = self._prepare_heatmap_data(name, new_data)

        # Stream to existing plot (thread-safe via callback)
        if self._doc:

            def do_stream() -> None:
                self._sources[name].stream(new_data, rollover=rollover)

                # Update color mapper for heatmaps
                if spec.get("plot_type") == "heatmap" and "mapper" in spec:
                    spec["mapper"].low = spec["value_min"]
                    spec["mapper"].high = spec["value_max"]

            self._doc.add_next_tick_callback(do_stream)

    def _prepare_heatmap_data(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate rect sizes and update color range for heatmap data."""
        import numpy as np

        spec = self._plot_specs[name]

        def calc_delta(key: str) -> float:
            """Calculate minimum delta from existing + new data."""
            if key not in data or len(data[key]) == 0:
                return 0.1
            existing = list(self._sources[name].data.get(key, []))
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
            spec["value_min"] = float(min(spec["value_min"], float(values.min())))
            spec["value_max"] = float(max(spec["value_max"], float(values.max())))

        return data
