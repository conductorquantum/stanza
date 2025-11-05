import calendar
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import requests
from jupyter_core.paths import jupyter_runtime_dir

from stanza.jupyter.utils import tail_log


@dataclass
class ServerState:
    """Persistent Jupyter server state stored on disk.

    Attributes:
        pid: Process ID of the Jupyter server
        url: Full URL with token for accessing JupyterLab
        started_at: ISO 8601 timestamp when server was started
        root_dir: Absolute path to the notebook root directory
    """

    pid: int
    url: str
    started_at: str
    root_dir: str


@dataclass
class ServerStatus:
    """Runtime Jupyter server status with calculated uptime.

    Attributes:
        pid: Process ID of the running Jupyter server
        url: Full URL with token for accessing JupyterLab
        uptime_seconds: Number of seconds the server has been running
        root_dir: Absolute path to the notebook root directory
    """

    pid: int
    url: str
    uptime_seconds: float
    root_dir: str


@dataclass
class RuntimeInfo:
    """Jupyter runtime information parsed from jpserver-{pid}.json.

    Attributes:
        url: JupyterLab URL with authentication token
        token: Authentication token for API requests
        port: Port number the server is listening on
        runtime_file: Path to the Jupyter runtime JSON file
    """

    url: str
    token: str
    port: int
    runtime_file: str


@dataclass
class SessionInfo:
    """Active Jupyter notebook session with log file metadata.

    Attributes:
        notebook_path: Absolute path to the notebook file
        log_path: Absolute path to the notebook's log file
        size_bytes: Size of the log file in bytes
        line_count: Number of lines in the log file
    """

    notebook_path: str
    log_path: str
    size_bytes: int
    line_count: int


@dataclass
class Config:
    """Configuration for Jupyter server management.

    Attributes:
        state_dir: Directory for storing server state and logs
        log_max_size: Maximum log file size in bytes before truncation
    """

    state_dir: Path = Path(".stanza/jupyter")
    log_max_size: int = 1024 * 1024

    @property
    def state_file(self) -> Path:
        """Path to the server state JSON file."""
        return self.state_dir / "state.json"

    @property
    def lock_file(self) -> Path:
        """Path to the file lock for preventing concurrent operations."""
        return self.state_dir / ".lock"

    @property
    def stdout_log(self) -> Path:
        """Path to the Jupyter server stdout log."""
        return self.state_dir / "stdout.log"

    @property
    def stderr_log(self) -> Path:
        """Path to the Jupyter server stderr log."""
        return self.state_dir / "stderr.log"

    @property
    def ipython_dir(self) -> Path:
        """Path to stanza's managed IPython configuration directory."""
        return self.state_dir / "ipython"

    @property
    def ipython_startup_dir(self) -> Path:
        """Path to IPython startup scripts directory for auto-logging."""
        return self.ipython_dir / "profile_default" / "startup"


_config = Config()


def _read_state() -> ServerState | None:
    """Read the Jupyter server state from disk.

    Returns:
        ServerState if state file exists and is valid, None otherwise
    """
    if not _config.state_file.exists():
        return None

    try:
        data = json.loads(_config.state_file.read_text())
        return ServerState(
            pid=data["pid"],
            url=data.get("url", ""),
            started_at=data.get("started_at", ""),
            root_dir=data.get("root_dir", ""),
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return None


def _write_state(state: ServerState) -> None:
    """Write the Jupyter server state to disk atomically with secure permissions.

    Uses a temporary file and atomic rename to prevent corruption. Sets file
    permissions to 0o600 (owner read/write only) for security.

    Args:
        state: ServerState to persist to disk
    """
    _config.state_dir.mkdir(parents=True, exist_ok=True)

    tmp = _config.state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(_config.state_file)

    try:
        os.chmod(_config.state_file, 0o600)
    except (OSError, NotImplementedError):
        pass


def _clear_state() -> None:
    """Delete the server state file and log files, ignoring errors."""
    for path in [_config.state_file, _config.stdout_log, _config.stderr_log]:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def _is_alive(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Args:
        pid: Process ID to check

    Returns:
        True if process is running, False otherwise
    """
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class _FileLock:
    """File-based lock for preventing concurrent Jupyter server operations.

    Uses fcntl.flock for process-level locking to ensure only one stanza
    process can modify the Jupyter server state at a time.
    """

    def __init__(self, lock_file: Path, timeout: float = 5.0):
        """Initialize lock with the lock file path and acquisition timeout.

        Args:
            lock_file: Path to the lock file
            timeout: Maximum seconds to wait for lock acquisition
        """
        self.lock_file = lock_file
        self.timeout = timeout
        self.fd: Any = None

    def __enter__(self) -> "_FileLock":
        """Acquire exclusive lock, blocking until available or timeout.

        Returns:
            Self for context manager usage

        Raises:
            RuntimeError: If lock cannot be acquired within timeout period
        """
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.lock_file, "w")

        start = time.time()
        while True:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() - start > self.timeout:
                    self.fd.close()
                    raise RuntimeError(
                        f"Lock timeout after {self.timeout}s. Another stanza process "
                        f"is accessing the Jupyter server. Try again in a moment."
                    ) from None
                time.sleep(0.1)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release lock and close file descriptor."""
        if self.fd:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self.fd.close()


def _open_logs(tail_bytes: int = 50000) -> tuple[TextIO, TextIO]:
    """Open stdout and stderr log files, truncating if they exceed max size.

    If a log file exceeds log_max_size, it is truncated to keep only the last
    tail_bytes with a marker indicating truncation occurred.

    Args:
        tail_bytes: Number of bytes to preserve when truncating logs

    Returns:
        Tuple of (stdout_file, stderr_file) opened in append mode
    """
    _config.state_dir.mkdir(parents=True, exist_ok=True)

    for log_file in [_config.stdout_log, _config.stderr_log]:
        if log_file.exists() and log_file.stat().st_size > _config.log_max_size:
            with open(log_file, "rb") as f:
                file_size = f.seek(0, os.SEEK_END)
                seek_pos = max(0, file_size - tail_bytes)
                f.seek(seek_pos, os.SEEK_SET)
                tail = f.read()
            log_file.write_bytes(b"[...truncated...]\n" + tail)

    stdout = open(_config.stdout_log, "a")
    stderr = open(_config.stderr_log, "a")

    return stdout, stderr


def _discover_runtime(
    pid: int, timeout: float = 10.0, poll_interval: float = 0.2
) -> RuntimeInfo:
    """Discover Jupyter runtime information by polling for jpserver-{pid}.json.

    Waits for Jupyter to write its runtime file containing the server URL, token,
    and port. Constructs a JupyterLab URL with authentication token.

    Args:
        pid: Process ID of the Jupyter server
        timeout: Maximum seconds to wait for runtime file
        poll_interval: Seconds between file existence checks

    Returns:
        RuntimeInfo containing server URL, token, port, and runtime file path

    Raises:
        RuntimeError: If runtime file is not found within timeout period
    """
    runtime_dir = Path(jupyter_runtime_dir())
    runtime_file = runtime_dir / f"jpserver-{pid}.json"

    start = time.time()
    while time.time() - start < timeout:
        if runtime_file.exists():
            try:
                runtime = json.loads(runtime_file.read_text())
                url = runtime["url"]
                token = runtime.get("token", "")
                port = runtime.get("port", 8888)

                base_url = url.split("?")[0].rstrip("/")
                lab_url = f"{base_url}/lab"
                if token:
                    lab_url = f"{lab_url}?token={token}"

                return RuntimeInfo(
                    url=lab_url,
                    token=token,
                    port=port,
                    runtime_file=str(runtime_file),
                )
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        time.sleep(poll_interval)

    raise RuntimeError(
        f"Jupyter runtime file not found after {timeout}s. "
        f"Server may have failed to start. Check {_config.stderr_log}"
    )


def _shutdown(
    state: ServerState, timeout: float = 5.0, poll_interval: float = 0.2
) -> None:
    """Gracefully shutdown Jupyter server via REST API.

    Sends a shutdown request to the Jupyter server using the authentication token,
    then polls until the process terminates or timeout is reached.

    Args:
        state: Server state containing URL and PID
        timeout: Maximum seconds to wait for shutdown
        poll_interval: Seconds between process aliveness checks
    """
    try:
        token = state.url.split("token=")[1] if "token=" in state.url else ""
        if token:
            url_base = state.url.split("?")[0]
            requests.post(
                f"{url_base}api/shutdown",
                headers={"Authorization": f"token {token}"},
                timeout=2,
            )

            max_polls = int(timeout / poll_interval)
            for _ in range(max_polls):
                if not _is_alive(state.pid):
                    _clear_state()
                    return
                time.sleep(poll_interval)

    except (requests.RequestException, KeyError, IndexError):
        pass


def _kill(pid: int, timeout: float = 5.0, poll_interval: float = 0.2) -> None:
    """Terminate Jupyter server process with SIGTERM.

    Sends SIGTERM signal and waits for process to exit gracefully.

    Args:
        pid: Process ID to terminate
        timeout: Maximum seconds to wait for termination
        poll_interval: Seconds between process aliveness checks
    """
    try:
        os.kill(pid, signal.SIGTERM)
        max_polls = int(timeout / poll_interval)
        for _ in range(max_polls):
            if not _is_alive(pid):
                _clear_state()
                return
            time.sleep(poll_interval)
    except (OSError, ProcessLookupError):
        pass


def _force_kill(pid: int) -> None:
    """Force kill Jupyter server process with SIGKILL.

    Args:
        pid: Process ID to force kill
    """
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
    except (OSError, ProcessLookupError):
        pass


def _api_request(state: ServerState, endpoint: str) -> Any:
    """Make authenticated request to Jupyter API.

    Args:
        state: Server state containing URL with token
        endpoint: API endpoint path (e.g., "api/sessions")

    Returns:
        JSON response from the API

    Raises:
        requests.HTTPError: If the request fails
    """
    token = state.url.split("token=")[1] if "token=" in state.url else ""
    url_base = state.url.split("?")[0].replace("/lab", "")
    headers = {"Authorization": f"token {token}"} if token else {}

    response = requests.get(f"{url_base}/{endpoint}", headers=headers, timeout=2)
    response.raise_for_status()
    return response.json()


def _setup_ipython_startup() -> None:
    """Copy auto-logging startup script to stanza's IPython configuration directory.

    Raises:
        RuntimeError: If the startup script cannot be found
    """
    _config.ipython_startup_dir.mkdir(parents=True, exist_ok=True)

    source_script = Path(__file__).parent / "startup.py"
    dest_script = _config.ipython_startup_dir / "00-auto-logging.py"

    if source_script.exists():
        shutil.copy2(source_script, dest_script)
    else:
        raise RuntimeError(f"Auto-logging script not found at {source_script}")


def start(
    notebook_dir: Path, port: int = 8888, startup_wait: float = 0.5
) -> dict[str, Any]:
    """Start a Jupyter server in the background with auto-logging enabled.

    Launches jupyter_server as a detached process, discovers its runtime information,
    and saves the state to disk. Configures IPYTHONDIR to enable automatic cell
    output logging to notebook_name.log files.

    Args:
        notebook_dir: Root directory for notebooks
        port: Port number for the server (default: 8888)
        startup_wait: Seconds to wait before checking if server started successfully

    Returns:
        Dictionary with keys: pid, url, started_at, root_dir

    Raises:
        RuntimeError: If server is already running or fails to start
    """
    state = _read_state()
    if state and _is_alive(state.pid):
        raise RuntimeError(
            f"Jupyter server already running (PID {state.pid}). "
            f"Use 'stanza jupyter stop' first."
        )

    _clear_state()
    _setup_ipython_startup()

    with _FileLock(_config.lock_file):
        stdout_file, stderr_file = _open_logs()
        try:
            cmd = [
                sys.executable,
                "-m",
                "jupyter_server",
                "--no-browser",
                "--ServerApp.ip=127.0.0.1",
                f"--ServerApp.port={port}",
                f"--ServerApp.notebook_dir={notebook_dir.absolute()}",
            ]

            env = os.environ.copy()
            env["IPYTHONDIR"] = str(_config.ipython_dir.absolute())

            def preexec_fn() -> None:
                os.setsid()
                signal.signal(signal.SIGHUP, signal.SIG_IGN)

            proc = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                preexec_fn=preexec_fn,
            )

            pid = proc.pid

            time.sleep(startup_wait)
            if not _is_alive(pid):
                stderr_tail = tail_log(_config.stderr_log, lines=10)
                raise RuntimeError(
                    f"Jupyter server failed to start. Last 10 lines of stderr:\n"
                    f"{stderr_tail}"
                )

            runtime = _discover_runtime(pid)
            state = ServerState(
                pid=pid,
                url=runtime.url,
                started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                root_dir=str(notebook_dir.absolute()),
            )
            _write_state(state)
            return asdict(state)

        finally:
            stdout_file.close()
            stderr_file.close()


def stop() -> None:
    """Stop the Jupyter server using escalating shutdown strategies.

    Attempts graceful shutdown via REST API, then SIGTERM, then SIGKILL if necessary.
    Cleans up state files when complete. Safe to call even if no server is running.
    """
    state = _read_state()
    if not state:
        return

    if not _is_alive(state.pid):
        _clear_state()
        return

    try:
        _shutdown(state)
    except (requests.RequestException, KeyError, IndexError):
        pass

    if _is_alive(state.pid):
        _kill(state.pid)

    if _is_alive(state.pid):
        _force_kill(state.pid)

    _clear_state()


def status() -> dict[str, Any] | None:
    """Get the current Jupyter server status including uptime.

    Returns:
        Dictionary with keys: pid, url, uptime_seconds, root_dir.
        None if no server is running or state is stale.
    """
    state = _read_state()
    if not state:
        return None

    if not _is_alive(state.pid):
        _clear_state()
        return None

    try:
        started_at = time.strptime(state.started_at, "%Y-%m-%dT%H:%M:%SZ")
        started_timestamp = calendar.timegm(started_at)
        uptime_seconds = time.time() - started_timestamp
    except (KeyError, ValueError):
        uptime_seconds = 0.0

    return asdict(
        ServerStatus(
            pid=state.pid,
            url=state.url,
            uptime_seconds=uptime_seconds,
            root_dir=state.root_dir,
        )
    )


def list_sessions() -> list[dict[str, Any]]:
    """List active notebook sessions with log metadata.

    Returns:
        List of dictionaries, each containing: notebook_path, log_path,
        size_bytes, line_count. Empty list if no server is running.
    """
    state = _read_state()
    if not state or not _is_alive(state.pid):
        return []

    try:
        sessions = _api_request(state, "api/sessions")

        results = []
        for session in sessions:
            notebook_path = session.get("notebook", {}).get("path") or session.get(
                "path"
            )
            if not notebook_path:
                continue

            full_notebook_path = Path(state.root_dir) / notebook_path
            log_path = full_notebook_path.with_suffix(".log")

            size_bytes = 0
            line_count = 0
            if log_path.exists():
                try:
                    size_bytes = log_path.stat().st_size
                    with open(log_path, "rb") as f:
                        line_count = sum(1 for _ in f)
                except OSError:
                    pass

            results.append(
                asdict(
                    SessionInfo(
                        notebook_path=str(full_notebook_path),
                        log_path=str(log_path),
                        size_bytes=size_bytes,
                        line_count=line_count,
                    )
                )
            )

        return results

    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError):
        return []


def kill_kernel(notebook_name: str) -> None:
    """Kill kernel for notebook via Jupyter API.

    Args:
        notebook_name: Name of the notebook (case-insensitive substring match)

    Raises:
        RuntimeError: If no server is running, notebook not found, or API call fails
    """
    state = _read_state()
    if not state or not _is_alive(state.pid):
        raise RuntimeError("No Jupyter server running")

    try:
        sessions = _api_request(state, "api/sessions")

        for session in sessions:
            notebook_path = session.get("notebook", {}).get("path") or session.get(
                "path"
            )
            if not notebook_path:
                continue

            full_notebook_path = Path(state.root_dir) / notebook_path

            if notebook_name.lower() in full_notebook_path.name.lower():
                kernel_id = session.get("kernel", {}).get("id")
                if not kernel_id:
                    raise RuntimeError(f"No kernel ID for {notebook_path}")

                token = state.url.split("token=")[1] if "token=" in state.url else ""
                url_base = state.url.split("?")[0].replace("/lab", "")
                headers = {"Authorization": f"token {token}"} if token else {}

                response = requests.delete(
                    f"{url_base}/api/kernels/{kernel_id}", headers=headers, timeout=2
                )
                response.raise_for_status()
                return

        raise RuntimeError(f"No session for '{notebook_name}'")

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to kill kernel: {e}") from e
