"""Automatic logging of Jupyter notebook cell I/O to disk.

Hooks stdout, stderr, and exceptions. Writes to notebook_name.log if the notebook
path can be discovered via the Jupyter sessions API, otherwise jupyter_session.log.
"""

from __future__ import annotations

import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from jupyter_server import serverapp
from jupyter_server.utils import url_path_join


def _get_ipython() -> Any | None:
    """Get the current IPython instance if running in IPython.

    Returns:
        IPython instance if available, None otherwise
    """
    try:
        from IPython import get_ipython as _gi

        return _gi()
    except Exception:
        return None


def _in_ipykernel(ip: Any) -> bool:
    """Check if the IPython instance is running in a Jupyter kernel.

    Args:
        ip: IPython instance to check

    Returns:
        True if running in a Jupyter kernel, False otherwise
    """
    try:
        return bool(ip) and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


def _kernel_id(ip: Any) -> str | None:
    """Extract the kernel ID from the IPython kernel connection file.

    Args:
        ip: IPython instance containing kernel configuration

    Returns:
        Kernel ID string if found, None otherwise
    """
    try:
        conn = ip.config["IPKernelApp"]["connection_file"]
        m = re.search(r"kernel-(.+)\.json", conn)
        return m.group(1) if m else None
    except Exception:
        return None


def _resolve_notebook_path(kernel_id: str) -> Path | None:
    """Find the notebook path by querying running Jupyter servers for the kernel ID.

    Iterates through all running Jupyter servers and queries their sessions API
    to match the kernel ID with an active notebook session.

    Args:
        kernel_id: Kernel ID to search for

    Returns:
        Absolute path to the notebook file if found, None otherwise
    """
    for server in serverapp.list_running_servers():
        try:
            url = url_path_join(server["url"], "api/sessions")
            token = server.get("token", "")
            r = requests.get(url, params={"token": token}, timeout=1)
            r.raise_for_status()
            for session in r.json():
                kid = session.get("kernel", {}).get("id")
                if kid == kernel_id:
                    path = session.get("notebook", {}).get("path") or session.get(
                        "path"
                    )
                    if path:
                        return Path(server["root_dir"]) / path  # type: ignore[no-any-return]
        except requests.RequestException:
            continue
        except Exception:
            continue
    return None


def _append(path: Path, text: str) -> None:
    """Append text to the log file, silently ignoring write failures.

    Args:
        path: Path to the log file
        text: Text content to append
    """
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass


class _Tee:
    """Tee stream wrapper that duplicates writes to a log file.

    Acts as a transparent proxy for a stream (like sys.stdout or sys.stderr)
    while also writing all output to a log file.
    """

    def __init__(self, original: Any, log_path: Path):
        """Initialize tee with the original stream and log file path.

        Args:
            original: Original stream to wrap (e.g., sys.stdout)
            log_path: Path to the log file for duplicated output
        """
        self._orig = original
        self._path = log_path

    def write(self, text: str) -> int:
        """Write text to both the original stream and the log file.

        Args:
            text: Text content to write

        Returns:
            Number of characters written to the original stream
        """
        written: int = self._orig.write(text)
        _append(self._path, text)
        return written

    def flush(self) -> None:
        """Flush the original stream, ignoring errors."""
        try:
            self._orig.flush()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the original stream.

        Args:
            name: Attribute name to access

        Returns:
            Value of the attribute from the original stream
        """
        return getattr(self._orig, name)


def _install_logging(ip: Any, log_file: Path) -> None:
    """Install logging hooks for stdout, stderr, and IPython exceptions.

    Writes a header to the log file, wraps sys.stdout and sys.stderr with
    _Tee instances, and hooks IPython's showtraceback method to log exception
    tracebacks.

    Args:
        ip: IPython instance to install hooks on
        log_file: Path to the log file for output
    """
    header = (
        f"\n{'=' * 60}\n"
        f"Kernel started: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Log file: {log_file}\n"
        f"{'=' * 60}\n"
    )
    _append(log_file, header)

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    original_show = getattr(ip, "showtraceback", None)

    def _showtraceback(*args: Any, **kwargs: Any) -> None:
        """Custom showtraceback that logs exceptions to the log file."""
        if callable(original_show):
            try:
                original_show(*args, **kwargs)
            except Exception:
                pass
        etype, evalue, etb = sys.exc_info()
        if etype is not None:
            _append(log_file, "".join(traceback.format_exception(etype, evalue, etb)))

    if callable(original_show):
        ip.showtraceback = _showtraceback

    print(f"📝 Auto-logging enabled: {log_file.name}")


def main() -> None:
    """Entry point: set up auto-logging for the current Jupyter kernel session.

    Discovers the notebook path from the running Jupyter servers, creates a
    log file, and installs logging hooks. Falls back to jupyter_session.log
    if the notebook path cannot be determined.
    """
    ip = _get_ipython()
    if not _in_ipykernel(ip):
        return

    kid = _kernel_id(ip)
    nb_path = _resolve_notebook_path(kid) if kid else None

    log_path = (
        nb_path.with_suffix(".log")
        if nb_path and nb_path.exists()
        else Path.cwd() / "jupyter_session.log"
    )

    _install_logging(ip, log_path)


main()
