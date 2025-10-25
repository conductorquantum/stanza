"""Live plotting for data logging.

Two backends available:
- server: Plots in browser window (works everywhere)
- inline: Plots in notebook cells (requires jupyter_bokeh)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from stanza.plotter.backends.inline import InlineBackend
from stanza.plotter.backends.server import ServerBackend

if TYPE_CHECKING:
    from stanza.logger.data_logger import DataLogger

logger = logging.getLogger(__name__)


def enable_live_plotting(
    data_logger: DataLogger,
    backend: Literal["server", "inline"] = "server",
    port: int = 5006,
) -> ServerBackend | InlineBackend:
    """Enable live plotting for a data logger.

    Args:
        data_logger: DataLogger instance
        backend: "server" (browser) or "inline" (notebook)
        port: Server port (server backend only)

    Returns:
        Backend instance

    Example (server):
        >>> backend = enable_live_plotting(logger, backend="server", port=5006)
        >>> # Open http://localhost:5006 in browser

    Example (inline):
        >>> backend = enable_live_plotting(logger, backend="inline")
        >>> # Plots appear in notebook cells
    """
    bokeh_backend: ServerBackend | InlineBackend
    if backend == "server":
        from stanza.plotter.backends.server import ServerBackend

        bokeh_backend = ServerBackend(port=port)
        bokeh_backend.start()
        logger.info(f"Bokeh Server started: http://localhost:{port}")

    elif backend == "inline":
        from stanza.plotter.backends.inline import InlineBackend

        bokeh_backend = InlineBackend()
        bokeh_backend.start()
        logger.info("Bokeh Inline plotting enabled")

    else:
        raise ValueError(f"Unknown backend: {backend}")

    data_logger._bokeh_backend = bokeh_backend
    return bokeh_backend


__all__ = ["enable_live_plotting"]
