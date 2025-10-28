"""Minimal Jupyter server lifecycle management."""

import calendar
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from jupyter_core.paths import jupyter_runtime_dir


@dataclass
class ServerState:
    pid: int
    url: str
    started_at: str
    root_dir: str


@dataclass
class ServerStatus:
    pid: int
    url: str
    uptime_seconds: float
    root_dir: str


@dataclass
class RuntimeInfo:
    url: str
    token: str
    port: int
    runtime_file: str


@dataclass
class Config:
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

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def _read_state() -> ServerState | None:
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
    _config.state_dir.mkdir(parents=True, exist_ok=True)

    tmp = _config.state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(_config.state_file)

    try:
        os.chmod(_config.state_file, 0o600)
    except (OSError, NotImplementedError):
        pass


def _clear_state() -> None:
    for path in [_config.state_file, _config.stdout_log, _config.stderr_log]:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class _FileLock:
    def __init__(self, lock_file: Path, timeout: float = 5.0):
        self.lock_file = lock_file
        self.timeout = timeout
        self.fd: Any = None

    def __enter__(self) -> "_FileLock":
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.lock_file, "w")

        start = time.time()
        while True:
            try:
                if sys.platform == "win32":
                    msvcrt.locking(self.fd.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() - start > self.timeout:
                    self.fd.close()
                    raise RuntimeError(
                        "Could not acquire lock (another process may be starting). "
                        "Wait a moment and try again."
                    ) from None
                time.sleep(0.1)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.fd:
            try:
                if sys.platform == "win32":
                    msvcrt.locking(self.fd.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self.fd.close()


def _open_logs() -> tuple[Any, Any]:
    _config.state_dir.mkdir(parents=True, exist_ok=True)

    for log_file in [_config.stdout_log, _config.stderr_log]:
        if log_file.exists() and log_file.stat().st_size > _config.log_max_size:
            content = log_file.read_bytes()
            log_file.write_bytes(b"[...truncated...]\n" + content[-50000:])

    stdout = open(_config.stdout_log, "a")
    stderr = open(_config.stderr_log, "a")

    return stdout, stderr


def _tail_log(log_file: Path, lines: int = 20) -> str:
    if not log_file.exists():
        return ""

    try:
        content = log_file.read_text()
        return "\n".join(content.splitlines()[-lines:])
    except (OSError, UnicodeDecodeError):
        return ""


def _discover_runtime(pid: int, timeout: float = 10.0) -> RuntimeInfo:
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

        time.sleep(0.2)

    raise RuntimeError(
        f"Jupyter runtime file not found after {timeout}s. "
        f"Server may have failed to start. Check {_config.stderr_log}"
    )


def start(notebook_dir: Path) -> dict[str, Any]:
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
                "--ServerApp.port=8888",
                f"--ServerApp.root_dir={notebook_dir.absolute()}",
                "--MappingKernelManager.buffer_offline_messages=True",
                "--ZMQChannelsWebsocketConnection.iopub_msg_rate_limit=10000",
                "--ZMQChannelsWebsocketConnection.iopub_data_rate_limit=10000000",
            ]

            if sys.platform == "win32":
                import msvcrt  # noqa: F401

                creationflags = (
                    subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                )
                proc = subprocess.Popen(
                    cmd,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    creationflags=creationflags,
                )
            else:

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

            time.sleep(0.5)
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
    state = _read_state()
    if not state:
        return

    if not _is_alive(state.pid):
        _clear_state()
        return

    try:
        token = state.url.split("token=")[1] if "token=" in state.url else ""
        if token:
            url_base = state.url.split("?")[0]
            requests.post(
                f"{url_base}api/shutdown",
                headers={"Authorization": f"token {token}"},
                timeout=2,
            )

            for _ in range(25):
                if not _is_alive(state.pid):
                    _clear_state()
                    return
                time.sleep(0.2)
    except (requests.RequestException, KeyError, IndexError):
        pass

    if _is_alive(state.pid):
        try:
            os.kill(state.pid, signal.SIGTERM)

            for _ in range(25):
                if not _is_alive(state.pid):
                    _clear_state()
                    return
                time.sleep(0.2)
        except (OSError, ProcessLookupError):
            pass

    if _is_alive(state.pid):
        try:
            os.kill(state.pid, signal.SIGKILL)
            time.sleep(0.5)
        except (OSError, ProcessLookupError):
            pass

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
