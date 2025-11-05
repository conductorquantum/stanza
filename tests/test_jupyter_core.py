"""Tests for stanza/jupyter/core.py."""

import json
import os
import signal
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from stanza.jupyter.core import (
    Config,
    RuntimeInfo,
    ServerState,
    ServerStatus,
    SessionInfo,
    _api_request,
    _clear_state,
    _discover_runtime,
    _FileLock,
    _force_kill,
    _is_alive,
    _kill,
    _open_logs,
    _read_state,
    _setup_ipython_startup,
    _shutdown,
    _write_state,
    kill_kernel,
    list_sessions,
    start,
    status,
    stop,
)


class TestDataclasses:
    """Test suite for dataclass models."""

    def test_server_state_creation(self):
        """Test ServerState dataclass creation."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        assert state.pid == 1234
        assert state.url == "http://localhost:8888"

    def test_server_status_creation(self):
        """Test ServerStatus dataclass creation."""
        status_obj = ServerStatus(
            pid=1234,
            url="http://localhost:8888",
            uptime_seconds=100.5,
            root_dir="/test",
        )
        assert status_obj.uptime_seconds == 100.5

    def test_runtime_info_creation(self):
        """Test RuntimeInfo dataclass creation."""
        runtime = RuntimeInfo(
            url="http://localhost:8888/lab?token=abc",
            token="abc",
            port=8888,
            runtime_file="/tmp/jpserver-1234.json",
        )
        assert runtime.token == "abc"
        assert runtime.port == 8888

    def test_session_info_creation(self):
        """Test SessionInfo dataclass creation."""
        session = SessionInfo(
            notebook_path="/path/to/notebook.ipynb",
            log_path="/path/to/notebook.log",
            size_bytes=1024,
            line_count=50,
        )
        assert session.size_bytes == 1024
        assert session.line_count == 50


class TestConfig:
    """Test suite for Config class."""

    def test_default_state_dir(self):
        """Test Config default state directory."""
        config = Config()
        assert config.state_dir == Path(".stanza/jupyter")

    def test_state_file_property(self):
        """Test Config state_file property."""
        config = Config()
        assert config.state_file == Path(".stanza/jupyter/state.json")

    def test_lock_file_property(self):
        """Test Config lock_file property."""
        config = Config()
        assert config.lock_file == Path(".stanza/jupyter/.lock")

    def test_stdout_log_property(self):
        """Test Config stdout_log property."""
        config = Config()
        assert config.stdout_log == Path(".stanza/jupyter/stdout.log")

    def test_stderr_log_property(self):
        """Test Config stderr_log property."""
        config = Config()
        assert config.stderr_log == Path(".stanza/jupyter/stderr.log")

    def test_ipython_dir_property(self):
        """Test Config ipython_dir property."""
        config = Config()
        assert config.ipython_dir == Path(".stanza/jupyter/ipython")

    def test_ipython_startup_dir_property(self):
        """Test Config ipython_startup_dir property."""
        config = Config()
        assert config.ipython_startup_dir == Path(
            ".stanza/jupyter/ipython/profile_default/startup"
        )


class TestReadState:
    """Test suite for _read_state function."""

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        """Test _read_state returns None when state file doesn't exist."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        assert _read_state() is None

    def test_reads_valid_state_file(self, tmp_path, monkeypatch):
        """Test _read_state reads valid state file."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        state_data = {
            "pid": 1234,
            "url": "http://localhost:8888",
            "started_at": "2025-01-01T00:00:00Z",
            "root_dir": "/test",
        }
        config.state_file.write_text(json.dumps(state_data))
        state = _read_state()
        assert state.pid == 1234
        assert state.url == "http://localhost:8888"

    def test_returns_none_on_corrupt_json(self, tmp_path, monkeypatch):
        """Test _read_state returns None with corrupt JSON."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        config.state_file.write_text("{invalid json")
        assert _read_state() is None

    def test_handles_missing_optional_keys(self, tmp_path, monkeypatch):
        """Test _read_state handles missing optional keys with defaults."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        config.state_file.write_text(json.dumps({"pid": 1234}))
        state = _read_state()
        assert state.pid == 1234
        assert state.url == ""

    def test_handles_oserror(self, tmp_path, monkeypatch):
        """Test _read_state handles OSError gracefully."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        config.state_file.write_text("test")
        os.chmod(config.state_file, 0o000)
        try:
            assert _read_state() is None
        finally:
            os.chmod(config.state_file, 0o600)


class TestWriteState:
    """Test suite for _write_state function."""

    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        """Test _write_state creates state directory if missing."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        _write_state(state)
        assert config.state_dir.exists()

    def test_writes_state_file_with_correct_json(self, tmp_path, monkeypatch):
        """Test _write_state creates state file with correct JSON."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        _write_state(state)
        data = json.loads(config.state_file.read_text())
        assert data["pid"] == 1234
        assert data["url"] == "http://localhost:8888"

    def test_sets_file_permissions_to_600(self, tmp_path, monkeypatch):
        """Test _write_state sets file permissions to 0o600."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        _write_state(state)
        stat = os.stat(config.state_file)
        assert stat.st_mode & 0o777 == 0o600

    def test_handles_permission_errors_gracefully(self, tmp_path, monkeypatch):
        """Test _write_state handles permission errors gracefully."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with patch("os.chmod", side_effect=OSError):
            _write_state(state)
        assert config.state_file.exists()


class TestClearState:
    """Test suite for _clear_state function."""

    def test_removes_state_file(self, tmp_path, monkeypatch):
        """Test _clear_state removes state file."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        config.state_file.write_text("test")
        _clear_state()
        assert not config.state_file.exists()

    def test_removes_log_files(self, tmp_path, monkeypatch):
        """Test _clear_state removes log files."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        config.stdout_log.write_text("stdout")
        config.stderr_log.write_text("stderr")
        _clear_state()
        assert not config.stdout_log.exists()
        assert not config.stderr_log.exists()

    def test_ignores_missing_files(self, tmp_path, monkeypatch):
        """Test _clear_state ignores missing files."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        _clear_state()

    def test_ignores_oserror_during_deletion(self, tmp_path, monkeypatch):
        """Test _clear_state ignores OSError during deletion."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        config.state_file.write_text("test")
        with patch("pathlib.Path.unlink", side_effect=OSError):
            _clear_state()


class TestIsAlive:
    """Test suite for _is_alive function."""

    def test_returns_true_for_running_process(self):
        """Test _is_alive returns True for running process."""
        assert _is_alive(os.getpid()) is True

    def test_returns_false_for_nonexistent_pid(self):
        """Test _is_alive returns False for non-existent PID."""
        assert _is_alive(999999) is False

    def test_returns_false_on_process_lookup_error(self):
        """Test _is_alive returns False on ProcessLookupError."""
        with patch("os.kill", side_effect=ProcessLookupError):
            assert _is_alive(1234) is False


class TestFileLock:
    """Test suite for _FileLock class."""

    def test_acquires_lock_successfully(self, tmp_path):
        """Test _FileLock acquires lock successfully."""
        lock_file = tmp_path / ".lock"
        with _FileLock(lock_file):
            assert lock_file.exists()

    def test_creates_directory_if_missing(self, tmp_path):
        """Test _FileLock creates directory if missing."""
        lock_file = tmp_path / "subdir" / ".lock"
        with _FileLock(lock_file):
            assert lock_file.parent.exists()

    def test_timeout_raises_runtime_error(self, tmp_path):
        """Test _FileLock timeout raises RuntimeError."""
        lock_file = tmp_path / ".lock"
        with _FileLock(lock_file, timeout=5.0):
            mock_time = Mock(side_effect=[0, 0.05, 0.15])
            with (
                patch("stanza.jupyter.core.time.time", mock_time),
                patch("stanza.jupyter.core.time.sleep"),
                pytest.raises(RuntimeError, match="Lock timeout"),
            ):
                with _FileLock(lock_file, timeout=0.1):
                    pass

    def test_releases_lock_on_exit(self, tmp_path):
        """Test _FileLock releases lock on exit."""
        lock_file = tmp_path / ".lock"
        with _FileLock(lock_file):
            pass
        with _FileLock(lock_file):
            pass

    def test_handles_oserror_during_unlock(self, tmp_path):
        """Test _FileLock handles OSError during unlock."""
        lock_file = tmp_path / ".lock"
        lock = _FileLock(lock_file)
        lock.__enter__()
        with patch("fcntl.flock", side_effect=OSError):
            lock.__exit__(None, None, None)


class TestOpenLogs:
    """Test suite for _open_logs function."""

    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        """Test _open_logs creates directory if missing."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        stdout, stderr = _open_logs()
        try:
            assert config.state_dir.exists()
        finally:
            stdout.close()
            stderr.close()

    def test_opens_files_in_append_mode(self, tmp_path, monkeypatch):
        """Test _open_logs opens files in append mode."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        config.stdout_log.write_text("existing")
        stdout, stderr = _open_logs()
        try:
            stdout.write("new")
            stdout.close()
            stderr.close()
            assert config.stdout_log.read_text() == "existingnew"
        finally:
            if not stdout.closed:
                stdout.close()
            if not stderr.closed:
                stderr.close()

    def test_truncates_large_log_files(self, tmp_path, monkeypatch):
        """Test _open_logs truncates large log files."""
        config = Config(state_dir=tmp_path / "jupyter", log_max_size=100)
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        large_content = "x" * 200
        config.stdout_log.write_text(large_content)
        stdout, stderr = _open_logs(tail_bytes=50)
        try:
            content = config.stdout_log.read_text()
            assert content.startswith("[...truncated...]")
            assert len(content) < len(large_content)
        finally:
            stdout.close()
            stderr.close()

    def test_preserves_tail_when_truncating(self, tmp_path, monkeypatch):
        """Test _open_logs preserves tail when truncating."""
        config = Config(state_dir=tmp_path / "jupyter", log_max_size=100)
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        content = "start" + "x" * 200 + "end"
        config.stdout_log.write_text(content)
        stdout, stderr = _open_logs(tail_bytes=10)
        try:
            result = config.stdout_log.read_text()
            assert "end" in result
        finally:
            stdout.close()
            stderr.close()


class TestDiscoverRuntime:
    """Test suite for _discover_runtime function."""

    def test_discovers_runtime_file_and_returns_info(self, tmp_path):
        """Test _discover_runtime finds runtime file and returns RuntimeInfo."""
        runtime_file = tmp_path / "jpserver-1234.json"
        runtime_data = {
            "url": "http://127.0.0.1:8888/",
            "token": "abc123",
            "port": 8888,
        }
        runtime_file.write_text(json.dumps(runtime_data))
        with patch(
            "stanza.jupyter.core.jupyter_runtime_dir", return_value=str(tmp_path)
        ):
            runtime = _discover_runtime(1234, timeout=2.0, poll_interval=0.1)
        assert runtime.token == "abc123"
        assert runtime.port == 8888

    def test_constructs_jupyterlab_url_with_token(self, tmp_path):
        """Test _discover_runtime constructs correct JupyterLab URL with token."""
        runtime_file = tmp_path / "jpserver-1234.json"
        runtime_data = {
            "url": "http://127.0.0.1:8888/?token=abc123",
            "token": "abc123",
            "port": 8888,
        }
        runtime_file.write_text(json.dumps(runtime_data))
        with patch(
            "stanza.jupyter.core.jupyter_runtime_dir", return_value=str(tmp_path)
        ):
            runtime = _discover_runtime(1234, timeout=2.0, poll_interval=0.1)
        assert runtime.url == "http://127.0.0.1:8888/lab?token=abc123"

    def test_handles_missing_token(self, tmp_path):
        """Test _discover_runtime handles missing token."""
        runtime_file = tmp_path / "jpserver-1234.json"
        runtime_data = {"url": "http://127.0.0.1:8888/", "port": 8888}
        runtime_file.write_text(json.dumps(runtime_data))
        with patch(
            "stanza.jupyter.core.jupyter_runtime_dir", return_value=str(tmp_path)
        ):
            runtime = _discover_runtime(1234, timeout=2.0, poll_interval=0.1)
        assert runtime.token == ""

    def test_raises_runtime_error_on_timeout(self, tmp_path):
        """Test _discover_runtime raises RuntimeError on timeout."""
        mock_time = Mock(side_effect=[0, 0.05, 0.15])
        with (
            patch(
                "stanza.jupyter.core.jupyter_runtime_dir", return_value=str(tmp_path)
            ),
            patch("stanza.jupyter.core.time.time", mock_time),
            patch("stanza.jupyter.core.time.sleep"),
            pytest.raises(RuntimeError, match="Jupyter runtime file not found"),
        ):
            _discover_runtime(1234, timeout=0.1, poll_interval=0.05)

    def test_handles_corrupt_json(self, tmp_path):
        """Test _discover_runtime handles corrupt JSON."""
        runtime_file = tmp_path / "jpserver-1234.json"
        runtime_file.write_text("{invalid json")
        mock_time = Mock(side_effect=[0, 0.1, 0.25])
        with (
            patch(
                "stanza.jupyter.core.jupyter_runtime_dir", return_value=str(tmp_path)
            ),
            patch("stanza.jupyter.core.time.time", mock_time),
            patch("stanza.jupyter.core.time.sleep"),
            pytest.raises(RuntimeError, match="not found"),
        ):
            _discover_runtime(1234, timeout=0.2, poll_interval=0.1)


class TestShutdown:
    """Test suite for _shutdown function."""

    def test_sends_rest_api_shutdown_request(self):
        """Test _shutdown sends REST API shutdown request."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("requests.post") as mock_post,
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state"),
        ):
            _shutdown(state)
            mock_post.assert_called_once()
            assert "api/shutdown" in mock_post.call_args[0][0]

    def test_waits_for_process_to_terminate(self):
        """Test _shutdown waits for process to terminate."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("requests.post"),
            patch("stanza.jupyter.core._is_alive", side_effect=[True, True, False]),
            patch("stanza.jupyter.core._clear_state") as mock_clear,
        ):
            _shutdown(state, timeout=0.5, poll_interval=0.1)
            mock_clear.assert_called_once()

    def test_handles_missing_token(self):
        """Test _shutdown handles missing token."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888/lab",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state"),
        ):
            _shutdown(state)

    def test_handles_request_exception(self):
        """Test _shutdown handles RequestException."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("requests.post", side_effect=requests.RequestException),
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state"),
        ):
            _shutdown(state)


class TestKillAndForceKill:
    """Test suite for _kill and _force_kill functions."""

    def test_kill_sends_sigterm(self):
        """Test _kill sends SIGTERM."""
        with (
            patch("os.kill") as mock_kill,
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state"),
        ):
            _kill(1234)
            mock_kill.assert_called_once_with(1234, signal.SIGTERM)

    def test_kill_waits_for_termination(self):
        """Test _kill waits for process termination."""
        with (
            patch("os.kill"),
            patch("stanza.jupyter.core._is_alive", side_effect=[True, False]),
            patch("stanza.jupyter.core._clear_state") as mock_clear,
        ):
            _kill(1234, timeout=0.5, poll_interval=0.1)
            mock_clear.assert_called_once()

    def test_kill_handles_oserror(self):
        """Test _kill handles OSError."""
        with patch("os.kill", side_effect=OSError):
            _kill(1234)

    def test_force_kill_sends_sigkill(self):
        """Test _force_kill sends SIGKILL."""
        with patch("os.kill") as mock_kill:
            _force_kill(1234)
            mock_kill.assert_called_once_with(1234, signal.SIGKILL)

    def test_force_kill_handles_oserror(self):
        """Test _force_kill handles OSError."""
        with patch("os.kill", side_effect=OSError):
            _force_kill(1234)


class TestApiRequest:
    """Test suite for _api_request function."""

    def test_makes_authenticated_request(self):
        """Test _api_request makes authenticated request."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"data": "test"}
            result = _api_request(state, "api/sessions")
            assert result == {"data": "test"}
            mock_get.assert_called_once()
            assert "Authorization" in mock_get.call_args[1]["headers"]

    def test_handles_missing_token(self):
        """Test _api_request handles missing token."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888/lab",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {}
            _api_request(state, "api/sessions")
            assert "Authorization" not in mock_get.call_args[1]["headers"]

    def test_raises_on_http_error(self):
        """Test _api_request raises on HTTP error."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with patch("requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = requests.HTTPError
            with pytest.raises(requests.HTTPError):
                _api_request(state, "api/sessions")


class TestSetupIpythonStartup:
    """Test suite for _setup_ipython_startup function."""

    def test_creates_directory(self, tmp_path, monkeypatch):
        """Test _setup_ipython_startup creates directory."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_script = source_dir / "startup.py"
        source_script.write_text("# startup script")
        with patch("stanza.jupyter.core.Path") as mock_path:
            mock_path.return_value.parent = source_dir
            _setup_ipython_startup()
        assert config.ipython_startup_dir.exists()

    def test_copies_startup_script(self, tmp_path, monkeypatch):
        """Test _setup_ipython_startup copies startup script."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_script = source_dir / "startup.py"
        source_script.write_text("# startup script")
        with patch("stanza.jupyter.core.Path") as mock_path:
            mock_path.return_value.parent = source_dir
            _setup_ipython_startup()
        dest_script = config.ipython_startup_dir / "00-auto-logging.py"
        assert dest_script.exists()

    def test_raises_if_source_script_missing(self, tmp_path, monkeypatch):
        """Test _setup_ipython_startup raises if source script missing."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        with patch("stanza.jupyter.core.Path") as mock_path:
            mock_path.return_value.parent = tmp_path
            with pytest.raises(RuntimeError, match="Auto-logging script not found"):
                _setup_ipython_startup()


class TestStart:
    """Test suite for start function."""

    def test_raises_if_server_already_running(self, tmp_path, monkeypatch):
        """Test start raises if server already running."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        config.state_dir.mkdir(parents=True)
        config.state_file.write_text(
            json.dumps(
                {"pid": os.getpid(), "url": "", "started_at": "", "root_dir": ""}
            )
        )
        with pytest.raises(RuntimeError, match="already running"):
            start(tmp_path)

    def test_launches_jupyter_server_process(self, tmp_path, monkeypatch):
        """Test start launches jupyter_server process."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        mock_proc = Mock(pid=1234)
        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch("stanza.jupyter.core._discover_runtime") as mock_discover,
            patch("stanza.jupyter.core._setup_ipython_startup"),
        ):
            mock_discover.return_value = RuntimeInfo(
                url="http://localhost:8888/lab?token=abc",
                token="abc",
                port=8888,
                runtime_file="/tmp/jpserver-1234.json",
            )
            result = start(tmp_path, port=8888)
            mock_popen.assert_called_once()
            assert result["pid"] == 1234

    def test_sets_ipythondir_environment_variable(self, tmp_path, monkeypatch):
        """Test start sets IPYTHONDIR environment variable."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        mock_proc = Mock(pid=1234)
        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch("stanza.jupyter.core._discover_runtime") as mock_discover,
            patch("stanza.jupyter.core._setup_ipython_startup"),
        ):
            mock_discover.return_value = RuntimeInfo(
                url="http://localhost:8888/lab?token=abc",
                token="abc",
                port=8888,
                runtime_file="/tmp/jpserver-1234.json",
            )
            start(tmp_path, port=8888)
            env = mock_popen.call_args[1]["env"]
            assert "IPYTHONDIR" in env

    def test_raises_if_process_fails_during_startup(self, tmp_path, monkeypatch):
        """Test start raises if process fails during startup."""
        config = Config(state_dir=tmp_path / "jupyter")
        monkeypatch.setattr("stanza.jupyter.core._config", config)
        mock_proc = Mock(pid=1234)
        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._setup_ipython_startup"),
            patch("stanza.jupyter.core.tail_log", return_value="error logs"),
        ):
            with pytest.raises(RuntimeError, match="failed to start"):
                start(tmp_path)


class TestStop:
    """Test suite for stop function."""

    def test_returns_if_no_state_file(self):
        """Test stop returns if no state file exists."""
        with patch("stanza.jupyter.core._read_state", return_value=None):
            stop()

    def test_clears_state_if_process_dead(self):
        """Test stop clears state if process dead."""
        state = ServerState(
            pid=999999,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state") as mock_clear,
        ):
            stop()
            mock_clear.assert_called()

    def test_attempts_graceful_shutdown(self):
        """Test stop attempts graceful shutdown."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", side_effect=[True, False, False]),
            patch("stanza.jupyter.core._shutdown") as mock_shutdown,
            patch("stanza.jupyter.core._clear_state"),
        ):
            stop()
            mock_shutdown.assert_called_once()

    def test_falls_back_to_sigterm(self):
        """Test stop falls back to SIGTERM."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", side_effect=[True, True, False]),
            patch("stanza.jupyter.core._shutdown"),
            patch("stanza.jupyter.core._kill") as mock_kill,
            patch("stanza.jupyter.core._clear_state"),
        ):
            stop()
            mock_kill.assert_called_once()

    def test_falls_back_to_sigkill(self):
        """Test stop falls back to SIGKILL."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch(
                "stanza.jupyter.core._is_alive", side_effect=[True, True, True, False]
            ),
            patch("stanza.jupyter.core._shutdown"),
            patch("stanza.jupyter.core._kill"),
            patch("stanza.jupyter.core._force_kill") as mock_force_kill,
            patch("stanza.jupyter.core._clear_state"),
        ):
            stop()
            mock_force_kill.assert_called_once()

    def test_handles_shutdown_request_exception(self):
        """Test stop handles RequestException from _shutdown."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch(
                "stanza.jupyter.core._shutdown",
                side_effect=requests.RequestException("Connection failed"),
            ),
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state") as mock_clear,
        ):
            stop()
            mock_clear.assert_called_once()

    def test_handles_shutdown_key_error(self):
        """Test stop handles KeyError from _shutdown."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._shutdown", side_effect=KeyError("missing key")),
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state") as mock_clear,
        ):
            stop()
            mock_clear.assert_called_once()

    def test_handles_shutdown_index_error(self):
        """Test stop handles IndexError from _shutdown."""
        state = ServerState(
            pid=1234,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._shutdown", side_effect=IndexError("bad index")),
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state") as mock_clear,
        ):
            stop()
            mock_clear.assert_called_once()


class TestStatus:
    """Test suite for status function."""

    def test_returns_none_with_no_state_file(self):
        """Test status returns None with no state file."""
        with patch("stanza.jupyter.core._read_state", return_value=None):
            assert status() is None

    def test_returns_none_with_dead_process(self):
        """Test status returns None with dead process."""
        state = ServerState(
            pid=999999,
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=False),
            patch("stanza.jupyter.core._clear_state"),
        ):
            assert status() is None

    def test_returns_server_status_with_uptime(self):
        """Test status returns ServerStatus dict with uptime."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888",
            started_at="2025-01-01T00:00:00Z",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
        ):
            result = status()
            assert result is not None
            assert "uptime_seconds" in result
            assert result["pid"] == os.getpid()

    def test_handles_invalid_timestamp(self):
        """Test status handles invalid started_at timestamp."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888",
            started_at="invalid",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
        ):
            result = status()
            assert result is not None
            assert result["uptime_seconds"] == 0.0


class TestListSessions:
    """Test suite for list_sessions function."""

    def test_returns_empty_list_with_no_server(self):
        """Test list_sessions returns empty list with no server."""
        with patch("stanza.jupyter.core._read_state", return_value=None):
            assert list_sessions() == []

    def test_queries_jupyter_api(self, tmp_path):
        """Test list_sessions queries Jupyter API."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir=str(tmp_path),
        )
        api_response = [
            {
                "notebook": {"path": "test.ipynb"},
                "kernel": {"id": "kernel-123"},
            }
        ]
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch("stanza.jupyter.core._api_request", return_value=api_response),
        ):
            result = list_sessions()
            assert len(result) == 1
            assert "test.ipynb" in result[0]["notebook_path"]

    def test_reads_log_file_metadata(self, tmp_path):
        """Test list_sessions reads log file metadata."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir=str(tmp_path),
        )
        notebook = tmp_path / "test.ipynb"
        notebook.write_text("{}")
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\nline3")
        api_response = [{"notebook": {"path": "test.ipynb"}}]
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch("stanza.jupyter.core._api_request", return_value=api_response),
        ):
            result = list_sessions()
            assert result[0]["line_count"] == 3
            assert result[0]["size_bytes"] > 0

    def test_handles_missing_log_files(self, tmp_path):
        """Test list_sessions handles missing log files."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir=str(tmp_path),
        )
        api_response = [{"notebook": {"path": "test.ipynb"}}]
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch("stanza.jupyter.core._api_request", return_value=api_response),
        ):
            result = list_sessions()
            assert result[0]["size_bytes"] == 0
            assert result[0]["line_count"] == 0

    def test_handles_request_exception(self):
        """Test list_sessions handles RequestException."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch(
                "stanza.jupyter.core._api_request",
                side_effect=requests.RequestException,
            ),
        ):
            assert list_sessions() == []


class TestKillKernel:
    """Test suite for kill_kernel function."""

    def test_raises_if_no_server_running(self):
        """Test kill_kernel raises if no server running."""
        with patch("stanza.jupyter.core._read_state", return_value=None):
            with pytest.raises(RuntimeError, match="No Jupyter server running"):
                kill_kernel("test")

    def test_finds_session_by_notebook_name(self, tmp_path):
        """Test kill_kernel finds session by notebook name."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir=str(tmp_path),
        )
        api_response = [
            {
                "notebook": {"path": "test_notebook.ipynb"},
                "kernel": {"id": "kernel-123"},
            }
        ]
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch("stanza.jupyter.core._api_request", return_value=api_response),
            patch("requests.delete") as mock_delete,
        ):
            mock_delete.return_value.raise_for_status = Mock()
            kill_kernel("test")
            mock_delete.assert_called_once()
            assert "kernel-123" in mock_delete.call_args[0][0]

    def test_raises_if_notebook_not_found(self):
        """Test kill_kernel raises if notebook not found."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir="/test",
        )
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch("stanza.jupyter.core._api_request", return_value=[]),
        ):
            with pytest.raises(RuntimeError, match="No session"):
                kill_kernel("nonexistent")

    def test_handles_request_exception(self, tmp_path):
        """Test kill_kernel handles RequestException."""
        state = ServerState(
            pid=os.getpid(),
            url="http://localhost:8888/lab?token=abc",
            started_at="2025-01-01",
            root_dir=str(tmp_path),
        )
        api_response = [
            {"notebook": {"path": "test.ipynb"}, "kernel": {"id": "kernel-123"}}
        ]
        with (
            patch("stanza.jupyter.core._read_state", return_value=state),
            patch("stanza.jupyter.core._is_alive", return_value=True),
            patch("stanza.jupyter.core._api_request", return_value=api_response),
            patch("requests.delete", side_effect=requests.RequestException("error")),
        ):
            with pytest.raises(RuntimeError, match="Failed to kill kernel"):
                kill_kernel("test")
