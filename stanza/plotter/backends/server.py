"""Server-based plotting backend for live visualization in browser.

Runs a Bokeh server in a background thread, streams data updates via WebSocket.
Works from any environment: scripts, notebooks, or interactive sessions.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.server.server import Server

from stanza.plotter.backends.utils import (
    PlotSpec,
    PlotState,
    make_heatmap_plot,
    make_line_plot,
    prepare_heatmap_data,
)

logger = logging.getLogger(__name__)


class ServerBackend:
    """Live plotting in browser via Bokeh server.

    Runs embedded server in background thread. Server dies when parent process exits.
    """

    def __init__(self, port: int = 5006, daemon: bool = True) -> None:
        self.port = port
        self.daemon = daemon
        self._server: Server | None = None
        self._doc: Any = None
        self._running = False

        # Plot configurations registered before browser connects
        self._plot_specs: dict[str, PlotSpec] = {}
        # Complete plot states after browser connects and plots are created
        self._plots: dict[str, PlotState] = {}
        # Data buffered before browser connects
        self._buffer: dict[str, dict[str, list[Any]]] = {}

    def start(self, block: bool = False) -> None:
        """Start Bokeh server.

        Args:
            block: If True, blocks until server is stopped (for persistent mode).
                   If False, runs in background thread (for embedded mode).
        """
        if self._running:
            return

        logger.info(f"Instantiating new Bokeh server on port {self.port}")

        def make_document(doc: Any) -> None:
            """Initialize document when browser connects."""
            self._doc = doc

            for name in self._plot_specs:
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
        """Stop the Bokeh server and remove from active servers registry."""
        if self._server and self._running:
            self._server.unlisten()
            self._server.stop()
            self._server.io_loop.stop()
            self._running = False

            from stanza.plotter import _active_servers

            if self.port in _active_servers and _active_servers[self.port] is self:
                del _active_servers[self.port]

    def reset(self) -> None:
        """Reset the document, clearing all plots and data sources.

        This allows reusing the same server instance across multiple sessions.
        """
        if not self._running:
            return

        self._plots.clear()
        self._plot_specs.clear()
        self._buffer.clear()

        # If browser is connected, clear the document too
        if self._doc is not None:

            def do_clear() -> None:
                if self._doc is not None:
                    self._doc.clear()

            self._doc.add_next_tick_callback(do_clear)

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

        # Store partial spec; will be replaced with full spec when plot is created
        self._plot_specs[name] = PlotSpec(
            name=name,
            plot_type=plot_type,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            cell_size=cell_size,
        )

        # If browser already connected, create plot immediately
        if self._doc is not None and name not in self._plots:
            self._doc.add_next_tick_callback(lambda: self._create_plot(name))

    def _create_plot(self, name: str) -> None:
        """Create plot based on spec."""
        spec = self._plot_specs[name]

        if spec.plot_type == "line":
            self._create_line_plot(name)
        elif spec.plot_type == "heatmap":
            self._create_heatmap_plot(name)
        else:
            raise ValueError(f"Unknown plot type: {spec.plot_type}")

    def _create_line_plot(self, name: str) -> None:
        """Create 1D line plot."""
        spec = self._plot_specs[name]
        data = self._buffer.pop(name, {"x": [], "y": []})
        plot_state = make_line_plot(name, spec.x_label, spec.y_label)
        if data["x"] or data["y"]:
            plot_state.source.data = data
        self._plots[name] = plot_state
        if self._doc:
            self._doc.add_root(plot_state.figure)

    def _create_heatmap_plot(self, name: str) -> None:
        """Create 2D heatmap with rect glyph and linear color mapping."""
        spec = self._plot_specs[name]
        data = self._buffer.pop(
            name, {"x": [], "y": [], "value": [], "width": [], "height": []}
        )
        plot_state = make_heatmap_plot(
            name, spec.x_label, spec.y_label, spec.z_label, spec.cell_size
        )
        if any(data.get(k) for k in ("x", "y", "value", "width", "height")):
            plot_state.source.data = data
        self._plots[name] = plot_state
        if self._doc:
            self._doc.add_root(plot_state.figure)

    def stream_data(
        self, name: str, new_data: dict[str, Any], rollover: int | None = None
    ) -> None:
        """Add data to plot. Buffers if browser not yet connected."""
        # Try to get created plot first
        plot = self._plots.get(name)

        if plot is not None:
            # Plot exists, stream to it
            if plot.spec.plot_type == "heatmap" and "value" in new_data:
                new_data = prepare_heatmap_data(new_data, plot.source.data, plot.spec)

            # Stream to existing plot (thread-safe via callback)
            if self._doc:

                def do_stream() -> None:
                    plot.source.stream(new_data, rollover=rollover)

                    # Update color mapper for heatmaps
                    if plot.spec.plot_type == "heatmap" and plot.spec.mapper:
                        plot.spec.mapper.low = plot.spec.value_min
                        plot.spec.mapper.high = plot.spec.value_max

                self._doc.add_next_tick_callback(do_stream)
        else:
            # Plot not yet created, buffer the data
            spec = self._plot_specs.get(name)
            if spec is None:
                return

            if spec.plot_type == "heatmap" and "value" in new_data:
                # Initialize heatmap spec fields if buffering data before plot creation
                if spec.dx is None and spec.cell_size:
                    spec.dx = spec.cell_size[0]
                if spec.dy is None and spec.cell_size:
                    spec.dy = spec.cell_size[1]
                new_data = prepare_heatmap_data(new_data, {}, spec)

            if name not in self._buffer:
                self._buffer[name] = {k: [] for k in new_data.keys()}
            for key, values in new_data.items():
                self._buffer[name].setdefault(key, []).extend(values)
