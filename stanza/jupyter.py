import calendar
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import requests
from jupyter_core.paths import jupyter_runtime_dir


@dataclass
class ServerState:
    """The jupyter server state."""

    pid: int
    url: str
    started_at: str
    root_dir: str


@dataclass
class ServerStatus:
    """The jupyter server status."""

    pid: int
    url: str
    uptime_seconds: float
    root_dir: str


@dataclass
class RuntimeInfo:
    """The jupyter runtime info."""

    url: str
    token: str
    port: int
    runtime_file: str


@dataclass
class Config:
    """The jupyter server config."""

    state_dir: Path = Path(".stanza/jupyter")
    log_max_size: int = 1024 * 1024

    @property
    def state_file(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / ".lock"

    @property
    def stdout_log(self) -> Path:
        return self.state_dir / "stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.state_dir / "stderr.log"


_config = Config()


def _read_state() -> ServerState | None:
    """Read the jupyter server state."""
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
    """Write the jupyter server state."""
    _config.state_dir.mkdir(parents=True, exist_ok=True)

    tmp = _config.state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(_config.state_file)

    try:
        os.chmod(_config.state_file, 0o600)
    except (OSError, NotImplementedError):
        pass


def _clear_state() -> None:
    """Clear the jupyter server state."""
    for path in [_config.state_file, _config.stdout_log, _config.stderr_log]:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def _is_alive(pid: int) -> bool:
    """Check if the jupyter server process is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class _FileLock:
    """File lock interface for thread-safe operations."""

    def __init__(self, lock_file: Path, timeout: float = 5.0):
        self.lock_file = lock_file
        self.timeout = timeout
        self.fd: Any = None

    def __enter__(self) -> "_FileLock":
        """Acquire the lock."""
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
        """Release the lock."""
        if self.fd:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self.fd.close()


def _open_logs(tail_bytes: int = 50000) -> tuple[TextIO, TextIO]:
    """Open the logs."""
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


def _tail_log(log_file: Path, lines: int = 20) -> str:
    """Tail the log file."""
    if not log_file.exists():
        return ""

    try:
        with open(log_file, "rb") as f:
            file_size = f.seek(0, os.SEEK_END)
            if file_size == 0:
                return ""

            # Read last 4KB or whole file, whichever is smaller
            chunk_size = min(4096, file_size)
            f.seek(-chunk_size, os.SEEK_END)
            tail_bytes = f.read()

        tail_text = tail_bytes.decode("utf-8", errors="replace")
        return "\n".join(tail_text.splitlines()[-lines:])
    except (OSError, UnicodeDecodeError):
        return ""


def _discover_runtime(
    pid: int, timeout: float = 10.0, poll_interval: float = 0.2
) -> RuntimeInfo:
    """Discover the jupyter runtime."""
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
    """Shutdown the jupyter server."""
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
    """Kill the jupyter server."""
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
    """Force kill the jupyter server."""
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
    except (OSError, ProcessLookupError):
        pass


def start(
    notebook_dir: Path, port: int = 8888, startup_wait: float = 0.5
) -> dict[str, Any]:
    """Start the jupyter server."""
    state = _read_state()
    if state and _is_alive(state.pid):
        raise RuntimeError(
            f"Jupyter server already running (PID {state.pid}). "
            f"Use 'stanza jupyter stop' first."
        )

    _clear_state()
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

            def preexec_fn() -> None:
                os.setsid()
                signal.signal(signal.SIGHUP, signal.SIG_IGN)

            proc = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                preexec_fn=preexec_fn,
            )

            pid = proc.pid

            time.sleep(startup_wait)
            if not _is_alive(pid):
                stderr_tail = _tail_log(_config.stderr_log, lines=10)
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
    """Stop the jupyter server."""
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
