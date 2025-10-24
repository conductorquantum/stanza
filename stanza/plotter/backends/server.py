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
    """Live plotting in browser via Bokeh server."""

    def __init__(self, port: int = 5006) -> None:
        self.port = port
        self._sources: dict[str, ColumnDataSource] = {}
        self._server: Server | None = None
        self._doc: Any = None
        self._running = False

        # Plot configurations and data buffered before browser connects
        self._plot_specs: dict[str, dict[str, str]] = {}
        self._buffer: dict[str, dict[str, list[Any]]] = {}

    def start(self) -> None:
        """Start Bokeh server in background thread."""
        if self._running:
            return

        def make_document(doc: Any) -> None:
            """Initialize document when browser connects."""
            self._doc = doc

            for name, spec in self._plot_specs.items():
                data = self._buffer.get(name, {"x": [], "y": []})

                source = ColumnDataSource(data=data)
                self._sources[name] = source

                plot = figure(title=name, width=800, height=400)
                plot.xaxis.axis_label = spec["x_label"]
                plot.yaxis.axis_label = spec["y_label"]
                plot.line("x", "y", source=source, line_width=2, color="navy")

                doc.add_root(plot)

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
        self, name: str, x_label: str, y_label: str
    ) -> tuple[ColumnDataSource, bool]:
        """Register a new plot. Created when browser connects."""
        if not self._running:
            raise RuntimeError("Server not started")

        if name in self._plot_specs:
            return ColumnDataSource(data={"x": [], "y": []}), True

        self._plot_specs[name] = {"x_label": x_label, "y_label": y_label}

        # If browser already connected, create plot immediately
        if self._doc is not None and name not in self._sources:

            def create() -> None:
                data = self._buffer.pop(name, {"x": [], "y": []})
                source = ColumnDataSource(data=data)
                self._sources[name] = source

                spec = self._plot_specs[name]
                plot = figure(title=name, width=800, height=400)
                plot.xaxis.axis_label = spec["x_label"]
                plot.yaxis.axis_label = spec["y_label"]
                plot.line("x", "y", source=source, line_width=2, color="navy")

                if self._doc:
                    self._doc.add_root(plot)

            self._doc.add_next_tick_callback(create)

        return ColumnDataSource(data={"x": [], "y": []}), True

    def stream_data(self, name: str, new_data: dict[str, Any], rollover: int) -> None:
        """Add data to plot. Buffers if browser not yet connected."""
        # Buffer data if plot not yet created
        if name not in self._sources:
            if name not in self._buffer:
                self._buffer[name] = {"x": [], "y": []}
            self._buffer[name]["x"].extend(new_data["x"])
            self._buffer[name]["y"].extend(new_data["y"])
            return

        # Stream to existing plot (thread-safe via callback)
        if self._doc:

            def do_stream() -> None:
                self._sources[name].stream(new_data, rollover=rollover)

            self._doc.add_next_tick_callback(do_stream)
