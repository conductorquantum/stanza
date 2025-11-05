"""Jupyter server management for Stanza (optional dependency)."""

from typing import Any, NoReturn

try:
    from stanza.jupyter import logs as log_stream
    from stanza.jupyter.core import kill_kernel, list_sessions, start, status, stop
    from stanza.jupyter.utils import format_size

    __all__ = [
        "start",
        "stop",
        "status",
        "list_sessions",
        "kill_kernel",
        "log_stream",
        "format_size",
    ]

except ImportError:

    def _jupyter_not_available(*_args: Any, **_kwargs: Any) -> NoReturn:
        """Raise ImportError when Jupyter dependencies are not installed."""
        raise ImportError(
            "Jupyter dependencies not installed. "
            "Install with: pip install cq-stanza[notebook]"
        )

    start = stop = status = list_sessions = kill_kernel = _jupyter_not_available
    log_stream = format_size = None  # type: ignore[assignment]
