"""Tests for stanza/jupyter/startup.py."""

import sys
from unittest.mock import Mock, PropertyMock, patch

from stanza.jupyter.startup import (
    _append,
    _get_ipython,
    _in_ipykernel,
    _install_logging,
    _kernel_id,
    _resolve_notebook_path,
    _Tee,
    main,
)


class TestGetIpython:
    """Test suite for _get_ipython function."""

    def test_returns_ipython_instance_when_available(self):
        """Test _get_ipython returns IPython instance when available."""
        mock_ip = Mock()
        mock_get_ipython = Mock(return_value=mock_ip)
        with patch.dict("sys.modules", {"IPython": Mock(get_ipython=mock_get_ipython)}):
            result = _get_ipython()
            assert result is mock_ip

    def test_returns_none_when_ipython_not_available(self):
        """Test _get_ipython returns None when IPython not available."""
        with patch.dict("sys.modules", {"IPython": None}):
            result = _get_ipython()
            assert result is None

    def test_handles_general_exception(self):
        """Test _get_ipython handles general exceptions."""
        mock_get_ipython = Mock(side_effect=RuntimeError)
        with patch.dict("sys.modules", {"IPython": Mock(get_ipython=mock_get_ipython)}):
            result = _get_ipython()
            assert result is None


class TestInIpykernel:
    """Test suite for _in_ipykernel function."""

    def test_returns_true_in_jupyter_kernel(self):
        """Test _in_ipykernel returns True in Jupyter kernel."""
        mock_ip = Mock()
        mock_ip.config = {"IPKernelApp": {}}
        assert _in_ipykernel(mock_ip) is True

    def test_returns_false_outside_kernel(self):
        """Test _in_ipykernel returns False outside kernel."""
        mock_ip = Mock()
        mock_ip.config = {}
        assert _in_ipykernel(mock_ip) is False

    def test_returns_false_with_none(self):
        """Test _in_ipykernel returns False with None."""
        assert _in_ipykernel(None) is False

    def test_returns_false_without_config(self):
        """Test _in_ipykernel returns False without config attribute."""
        mock_ip = Mock(spec=[])
        assert _in_ipykernel(mock_ip) is False

    def test_handles_exception(self):
        """Test _in_ipykernel handles exceptions."""
        mock_ip = Mock()
        type(mock_ip).config = PropertyMock(side_effect=RuntimeError)
        assert _in_ipykernel(mock_ip) is False


class TestKernelId:
    """Test suite for _kernel_id function."""

    def test_extracts_kernel_id_from_connection_file(self):
        """Test _kernel_id extracts kernel ID from connection file."""
        mock_ip = Mock()
        mock_ip.config = {
            "IPKernelApp": {"connection_file": "/path/kernel-abc123.json"}
        }
        result = _kernel_id(mock_ip)
        assert result == "abc123"

    def test_returns_none_if_pattern_doesnt_match(self):
        """Test _kernel_id returns None if pattern doesn't match."""
        mock_ip = Mock()
        mock_ip.config = {"IPKernelApp": {"connection_file": "/path/invalid.json"}}
        result = _kernel_id(mock_ip)
        assert result is None

    def test_returns_none_without_config(self):
        """Test _kernel_id returns None without config."""
        mock_ip = Mock(spec=[])
        result = _kernel_id(mock_ip)
        assert result is None

    def test_handles_exception(self):
        """Test _kernel_id handles exceptions."""
        mock_ip = Mock()
        type(mock_ip).config = PropertyMock(side_effect=RuntimeError)
        result = _kernel_id(mock_ip)
        assert result is None


class TestResolveNotebookPath:
    """Test suite for _resolve_notebook_path function."""

    def test_finds_notebook_by_kernel_id(self, tmp_path):
        """Test _resolve_notebook_path finds notebook by kernel ID."""
        mock_server = {
            "url": "http://localhost:8888/",
            "token": "abc123",
            "root_dir": str(tmp_path),
        }
        mock_sessions = [
            {
                "kernel": {"id": "kernel-123"},
                "notebook": {"path": "test.ipynb"},
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = mock_sessions
        with (
            patch(
                "stanza.jupyter.startup.serverapp.list_running_servers",
                return_value=[mock_server],
            ),
            patch("stanza.jupyter.startup.requests.get", return_value=mock_response),
        ):
            result = _resolve_notebook_path("kernel-123")
            assert result == tmp_path / "test.ipynb"

    def test_returns_none_if_kernel_not_found(self):
        """Test _resolve_notebook_path returns None if kernel not found."""
        mock_server = {
            "url": "http://localhost:8888/",
            "token": "abc123",
            "root_dir": "/tmp",
        }
        mock_sessions = [
            {
                "kernel": {"id": "other-kernel"},
                "notebook": {"path": "test.ipynb"},
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = mock_sessions
        with (
            patch(
                "stanza.jupyter.startup.serverapp.list_running_servers",
                return_value=[mock_server],
            ),
            patch("stanza.jupyter.startup.requests.get", return_value=mock_response),
        ):
            result = _resolve_notebook_path("kernel-123")
            assert result is None

    def test_handles_request_exception(self):
        """Test _resolve_notebook_path handles RequestException."""
        import requests

        mock_server = {
            "url": "http://localhost:8888/",
            "token": "abc123",
            "root_dir": "/tmp",
        }
        with (
            patch(
                "stanza.jupyter.startup.serverapp.list_running_servers",
                return_value=[mock_server],
            ),
            patch(
                "stanza.jupyter.startup.requests.get",
                side_effect=requests.RequestException("Connection failed"),
            ),
        ):
            result = _resolve_notebook_path("kernel-123")
            assert result is None

    def test_handles_path_key_for_sessions(self, tmp_path):
        """Test _resolve_notebook_path handles 'path' key in session."""
        mock_server = {
            "url": "http://localhost:8888/",
            "token": "abc123",
            "root_dir": str(tmp_path),
        }
        mock_sessions = [
            {
                "kernel": {"id": "kernel-123"},
                "path": "test.ipynb",
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = mock_sessions
        with (
            patch(
                "stanza.jupyter.startup.serverapp.list_running_servers",
                return_value=[mock_server],
            ),
            patch("stanza.jupyter.startup.requests.get", return_value=mock_response),
        ):
            result = _resolve_notebook_path("kernel-123")
            assert result == tmp_path / "test.ipynb"

    def test_handles_general_exception(self):
        """Test _resolve_notebook_path handles general Exception."""
        mock_server = {
            "url": "http://localhost:8888/",
            "token": "abc123",
            "root_dir": "/tmp",
        }
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        with (
            patch(
                "stanza.jupyter.startup.serverapp.list_running_servers",
                return_value=[mock_server],
            ),
            patch("stanza.jupyter.startup.requests.get", return_value=mock_response),
        ):
            result = _resolve_notebook_path("kernel-123")
            assert result is None


class TestAppend:
    """Test suite for _append function."""

    def test_appends_text_to_file(self, tmp_path):
        """Test _append writes text to file."""
        log_file = tmp_path / "test.log"
        _append(log_file, "test content\n")
        assert log_file.read_text() == "test content\n"

    def test_appends_multiple_times(self, tmp_path):
        """Test _append appends on multiple calls."""
        log_file = tmp_path / "test.log"
        _append(log_file, "line 1\n")
        _append(log_file, "line 2\n")
        assert log_file.read_text() == "line 1\nline 2\n"

    def test_handles_oserror_silently(self, tmp_path):
        """Test _append handles OSError silently."""
        log_file = tmp_path / "nonexistent" / "test.log"
        _append(log_file, "test")


class TestTee:
    """Test suite for _Tee class."""

    def test_writes_to_both_streams(self, tmp_path):
        """Test _Tee writes to both original stream and log."""
        log_file = tmp_path / "test.log"
        original = Mock()
        original.write.return_value = 5
        tee = _Tee(original, log_file)
        result = tee.write("test")
        assert result == 5
        original.write.assert_called_once_with("test")
        assert log_file.read_text() == "test"

    def test_flush_calls_original_flush(self, tmp_path):
        """Test _Tee flush calls original flush."""
        log_file = tmp_path / "test.log"
        original = Mock()
        tee = _Tee(original, log_file)
        tee.flush()
        original.flush.assert_called_once()

    def test_flush_handles_exception(self, tmp_path):
        """Test _Tee flush handles exceptions."""
        log_file = tmp_path / "test.log"
        original = Mock()
        original.flush.side_effect = RuntimeError
        tee = _Tee(original, log_file)
        tee.flush()

    def test_getattr_delegates_to_original(self, tmp_path):
        """Test _Tee __getattr__ delegates to original."""
        log_file = tmp_path / "test.log"
        original = Mock()
        original.custom_attr = "test_value"
        tee = _Tee(original, log_file)
        assert tee.custom_attr == "test_value"


class TestInstallLogging:
    """Test suite for _install_logging function."""

    def test_writes_header_to_log(self, tmp_path, capsys):
        """Test _install_logging writes header to log."""
        log_file = tmp_path / "test.log"
        mock_ip = Mock()
        mock_ip.showtraceback = Mock()
        _install_logging(mock_ip, log_file)
        content = log_file.read_text()
        assert "Kernel started:" in content
        assert "Log file:" in content
        captured = capsys.readouterr()
        assert "Auto-logging enabled" in captured.out

    def test_wraps_stdout_and_stderr(self, tmp_path):
        """Test _install_logging wraps stdout and stderr."""
        log_file = tmp_path / "test.log"
        mock_ip = Mock()
        mock_ip.showtraceback = Mock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            _install_logging(mock_ip, log_file)
            assert isinstance(sys.stdout, _Tee)
            assert isinstance(sys.stderr, _Tee)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_hooks_showtraceback(self, tmp_path):
        """Test _install_logging hooks IPython showtraceback."""
        log_file = tmp_path / "test.log"
        mock_ip = Mock()
        original_show = Mock()
        mock_ip.showtraceback = original_show
        _install_logging(mock_ip, log_file)
        assert mock_ip.showtraceback != original_show

    def test_showtraceback_logs_exceptions(self, tmp_path):
        """Test _install_logging showtraceback logs exceptions."""
        log_file = tmp_path / "test.log"
        mock_ip = Mock()
        mock_ip.showtraceback = Mock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            _install_logging(mock_ip, log_file)
            try:
                raise ValueError("test error")
            except ValueError:
                mock_ip.showtraceback()
            content = log_file.read_text()
            assert "ValueError" in content
            assert "test error" in content
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_handles_non_callable_showtraceback(self, tmp_path):
        """Test _install_logging handles non-callable showtraceback."""
        log_file = tmp_path / "test.log"
        mock_ip = Mock()
        mock_ip.showtraceback = "not callable"
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            _install_logging(mock_ip, log_file)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_showtraceback_handles_original_show_exception(self, tmp_path):
        """Test showtraceback wrapper handles exception from original show."""
        log_file = tmp_path / "test.log"
        mock_ip = Mock()
        mock_show = Mock(side_effect=RuntimeError("Show failed"))
        mock_ip.showtraceback = mock_show
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            _install_logging(mock_ip, log_file)
            try:
                raise ValueError("test error")
            except ValueError:
                mock_ip.showtraceback()
            content = log_file.read_text()
            assert "ValueError" in content
            assert "test error" in content
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


class TestMain:
    """Test suite for main function."""

    def test_does_nothing_outside_ipython_kernel(self):
        """Test main does nothing outside IPython kernel."""
        with patch("stanza.jupyter.startup._get_ipython", return_value=None):
            main()

    def test_does_nothing_in_regular_ipython(self):
        """Test main does nothing in regular IPython."""
        mock_ip = Mock()
        mock_ip.config = {}
        with patch("stanza.jupyter.startup._get_ipython", return_value=mock_ip):
            main()

    def test_installs_logging_with_notebook_path(self, tmp_path, monkeypatch):
        """Test main installs logging with discovered notebook path."""
        monkeypatch.chdir(tmp_path)
        notebook = tmp_path / "test.ipynb"
        notebook.write_text("{}")
        log_file = tmp_path / "test.log"
        mock_ip = Mock()
        mock_ip.config = {"IPKernelApp": {"connection_file": "/path/kernel-123.json"}}
        mock_ip.showtraceback = Mock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with (
                patch("stanza.jupyter.startup._get_ipython", return_value=mock_ip),
                patch(
                    "stanza.jupyter.startup._resolve_notebook_path",
                    return_value=notebook,
                ),
            ):
                main()
            assert log_file.exists()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_falls_back_to_jupyter_session_log(self, tmp_path, monkeypatch):
        """Test main falls back to jupyter_session.log."""
        monkeypatch.chdir(tmp_path)
        mock_ip = Mock()
        mock_ip.config = {"IPKernelApp": {"connection_file": "/path/kernel-123.json"}}
        mock_ip.showtraceback = Mock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with (
                patch("stanza.jupyter.startup._get_ipython", return_value=mock_ip),
                patch(
                    "stanza.jupyter.startup._resolve_notebook_path", return_value=None
                ),
            ):
                main()
            log_file = tmp_path / "jupyter_session.log"
            assert log_file.exists()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_handles_nonexistent_notebook(self, tmp_path, monkeypatch):
        """Test main handles nonexistent notebook path."""
        monkeypatch.chdir(tmp_path)
        notebook = tmp_path / "nonexistent.ipynb"
        mock_ip = Mock()
        mock_ip.config = {"IPKernelApp": {"connection_file": "/path/kernel-123.json"}}
        mock_ip.showtraceback = Mock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with (
                patch("stanza.jupyter.startup._get_ipython", return_value=mock_ip),
                patch(
                    "stanza.jupyter.startup._resolve_notebook_path",
                    return_value=notebook,
                ),
            ):
                main()
            log_file = tmp_path / "jupyter_session.log"
            assert log_file.exists()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_handles_missing_kernel_id(self, tmp_path, monkeypatch):
        """Test main handles missing kernel ID."""
        monkeypatch.chdir(tmp_path)
        mock_ip = Mock()
        mock_ip.config = {"IPKernelApp": {"connection_file": "/invalid/path.json"}}
        mock_ip.showtraceback = Mock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with patch("stanza.jupyter.startup._get_ipython", return_value=mock_ip):
                main()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
