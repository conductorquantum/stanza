"""Live plotting for data logging.

Two backends available:
- server: Plots in browser window (works everywhere)
- inline: Plots in notebook cells (requires jupyter_bokeh)
"""

from __future__ import annotations

import logging
import socket
from typing import TYPE_CHECKING, Literal

from stanza.plotter.backends.inline import InlineBackend
from stanza.plotter.backends.server import ServerBackend

if TYPE_CHECKING:
    from stanza.logger.data_logger import DataLogger

logger = logging.getLogger(__name__)


def _find_free_port(start_port: int, max_attempts: int = 10) -> int:
    """Find a free port starting from start_port.

    Args:
        start_port: Port to start checking from
        max_attempts: Maximum number of ports to try

    Returns:
        First available port number

    Raises:
        RuntimeError: If no free port found within max_attempts
    """
    for offset in range(max_attempts):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"Could not find free port in range {start_port}-{start_port + max_attempts - 1}"
    )


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
        # Find free port if requested port is taken
        actual_port = _find_free_port(port)
        if actual_port != port:
            logger.warning(
                f"Port {port} already in use, using port {actual_port} instead"
            )

        bokeh_backend = ServerBackend(port=actual_port)
        bokeh_backend.start()
        logger.info(f"Bokeh Server started: http://localhost:{actual_port}")

    elif backend == "inline":
        bokeh_backend = InlineBackend()
        bokeh_backend.start()
        logger.info("Bokeh Inline plotting enabled")

    else:
        raise ValueError(f"Unknown backend: {backend}")

    data_logger._bokeh_backend = bokeh_backend
    return bokeh_backend


__all__ = ["enable_live_plotting"]
