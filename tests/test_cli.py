"""Tests for Stanza CLI (stanza/cli.py)."""

import json
import re
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from stanza.cli import cli
from stanza.context import StanzaSession


class TestCLI:
    """Test suite for Stanza CLI commands."""

    def test_cli_help(self):
        """Test that CLI help command works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Stanza" in result.output
        assert "Build tune up sequences" in result.output
        assert "init" in result.output
        assert "status" in result.output

    def test_cli_version(self):
        """Test that CLI version command works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "(Stanza)" in result.output
        # Check for valid semantic version format (e.g., 0.1.0, 1.2.3)
        assert re.search(r"\d+\.\d+\.\d+", result.output) is not None


class TestInitCommand:
    """Test suite for 'stanza init' command."""

    def test_init_creates_session_with_default_name(self):
        """Test that init command creates session directory with default name."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert "✓ Created session directory" in result.output
            assert "_untitled" in result.output
            assert "Session initialized successfully" in result.output

            sessions = list(Path.cwd().glob("*_untitled"))
            assert len(sessions) == 1
            assert sessions[0].exists()
            assert (sessions[0] / ".stanza" / "config.json").exists()

            notebooks = list(sessions[0].glob("*_untitled_notebook.ipynb"))
            assert len(notebooks) == 1
            assert notebooks[0].exists()

    def test_init_creates_session_with_custom_name(self):
        """Test that init command accepts custom name parameter."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "--name", "my_experiment"])

            assert result.exit_code == 0
            assert "_my_experiment" in result.output

            sessions = list(Path.cwd().glob("*_my_experiment"))
            assert len(sessions) == 1
            assert sessions[0].exists()

            notebooks = list(sessions[0].glob("*_my_experiment.ipynb"))
            assert len(notebooks) == 1
            assert notebooks[0].exists()

    def test_init_creates_session_with_custom_path(self):
        """Test that init command accepts custom path parameter."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            custom_path = Path.cwd() / "custom_location"
            custom_path.mkdir()

            result = runner.invoke(cli, ["init", "--path", str(custom_path)])

            assert result.exit_code == 0
            assert "✓ Created session directory" in result.output

            sessions = list(custom_path.glob("*_untitled"))
            assert len(sessions) == 1

    def test_init_sets_active_session(self):
        """Test that init command sets the active session."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0

            active_session = StanzaSession.get_active_session()
            assert active_session is not None
            assert active_session.exists()

    def test_init_fails_gracefully_on_error(self):
        """Test that init command handles errors gracefully."""
        runner = CliRunner()

        result = runner.invoke(cli, ["init", "--path", "/nonexistent/path/xyz"])

        assert result.exit_code != 0
        assert "Error" in result.output or "does not exist" in result.output

    def test_init_handles_file_exists_error(self):
        """Test that init handles FileExistsError gracefully."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            runner.invoke(cli, ["init", "--name", "test"])

            with patch("stanza.cli.StanzaSession.create_session_directory") as mock:
                mock.side_effect = FileExistsError("Directory exists")

                result = runner.invoke(cli, ["init", "--name", "test"])

                assert result.exit_code != 0
                assert "already exists" in result.output

    def test_init_with_different_names_creates_multiple_directories(self):
        """Test that init with different names creates multiple directories."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result1 = runner.invoke(cli, ["init", "--name", "exp1"])
            assert result1.exit_code == 0

            result2 = runner.invoke(cli, ["init", "--name", "exp2"])
            assert result2.exit_code == 0

            exp1_sessions = list(Path.cwd().glob("*_exp1"))
            exp2_sessions = list(Path.cwd().glob("*_exp2"))
            assert len(exp1_sessions) == 1
            assert len(exp2_sessions) == 1

    def test_init_creates_valid_jupyter_notebook(self):
        """Test that init creates a valid Jupyter notebook with proper structure."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init", "--name", "test_notebook"])
            assert result.exit_code == 0

            sessions = list(Path.cwd().glob("*_test_notebook"))
            assert len(sessions) == 1

            notebooks = list(sessions[0].glob("*_test_notebook.ipynb"))
            assert len(notebooks) == 1

            with open(notebooks[0]) as f:
                notebook_data = json.load(f)

            assert "cells" in notebook_data
            assert "metadata" in notebook_data
            assert "nbformat" in notebook_data
            assert notebook_data["nbformat"] == 4

            cells = notebook_data["cells"]
            assert len(cells) >= 2

            assert cells[0]["cell_type"] == "markdown"
            assert "Test Notebook" in "".join(cells[0]["source"])

            assert cells[1]["cell_type"] == "code"
            source = "".join(cells[1]["source"])
            assert "from stanza.routines import RoutineRunner" in source
            assert "from stanza.utils import load_device_config" in source


class TestStatusCommand:
    """Test suite for 'stanza status' command."""

    def test_status_shows_no_session_when_not_initialized(self):
        """Test that status command shows helpful message when no session exists."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "No active session" in result.output
            assert "stanza init" in result.output

    def test_status_shows_active_session_info(self):
        """Test that status command displays active session information."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            init_result = runner.invoke(cli, ["init", "--name", "test"])
            assert init_result.exit_code == 0

            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "Active session:" in result.output
            assert "_test" in result.output
            assert "Location:" in result.output
            assert "Created:" in result.output

    def test_status_shows_creation_timestamp(self):
        """Test that status command displays creation timestamp."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            init_result = runner.invoke(cli, ["init"])
            assert init_result.exit_code == 0

            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "Created:" in result.output
            assert "202" in result.output

    def test_status_handles_deleted_session_directory(self):
        """Test that status handles case where session directory was deleted."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            init_result = runner.invoke(cli, ["init"])
            assert init_result.exit_code == 0

            active_session = StanzaSession.get_active_session()
            import shutil

            shutil.rmtree(active_session)

            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "No active session" in result.output

    def test_status_handles_missing_metadata(self):
        """Test that status handles case where metadata file is missing."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            init_result = runner.invoke(cli, ["init"])
            assert init_result.exit_code == 0

            active_session = StanzaSession.get_active_session()
            metadata_file = active_session / ".stanza" / "config.json"
            metadata_file.unlink()

            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "Active session:" in result.output
            assert "Location:" in result.output
            assert "Created:" not in result.output


class TestLivePlotCommand:
    """Test suite for 'stanza live-plot' commands."""

    def test_live_plot_enable_server_backend(self):
        """Test enabling live plotting with server backend."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["live-plot", "enable", "--backend", "server", "--port", "5010"]
            )

            assert result.exit_code == 0
            assert "Live plotting enabled" in result.output
            assert "server" in result.output
            assert "5010" in result.output

            config_file = Path.cwd() / ".stanza" / "live_plot_config.json"
            assert config_file.exists()

            config = json.loads(config_file.read_text())
            assert config["enabled"] is True
            assert config["backend"] == "server"
            assert config["port"] == 5010

    def test_live_plot_enable_inline_backend(self):
        """Test enabling live plotting with inline backend."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["live-plot", "enable", "--backend", "inline"])

            assert result.exit_code == 0
            assert "Live plotting enabled" in result.output
            assert "inline" in result.output

            config_file = Path.cwd() / ".stanza" / "live_plot_config.json"
            config = json.loads(config_file.read_text())
            assert config["backend"] == "inline"

    def test_live_plot_disable(self):
        """Test disabling live plotting."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            runner.invoke(cli, ["live-plot", "enable"])

            result = runner.invoke(cli, ["live-plot", "disable"])

            assert result.exit_code == 0
            assert "disabled" in result.output

            config_file = Path.cwd() / ".stanza" / "live_plot_config.json"
            config = json.loads(config_file.read_text())
            assert config["enabled"] is False

    def test_live_plot_status_when_disabled(self):
        """Test live plot status when not configured."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["live-plot", "status"])

            assert result.exit_code == 0
            assert "disabled" in result.output

    def test_live_plot_status_when_enabled(self):
        """Test live plot status shows configuration."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            runner.invoke(
                cli, ["live-plot", "enable", "--backend", "server", "--port", "6000"]
            )

            result = runner.invoke(cli, ["live-plot", "status"])

            assert result.exit_code == 0
            assert "enabled" in result.output
            assert "server" in result.output
            assert "6000" in result.output

    def test_live_plot_status_after_disable(self):
        """Test live plot status after disabling."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            runner.invoke(cli, ["live-plot", "enable"])
            runner.invoke(cli, ["live-plot", "disable"])

            result = runner.invoke(cli, ["live-plot", "status"])

            assert result.exit_code == 0
            assert "disabled" in result.output


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_init_then_status_workflow(self):
        """Test complete workflow of init followed by status."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            init_result = runner.invoke(cli, ["init", "--name", "workflow_test"])
            assert init_result.exit_code == 0
            assert "_workflow_test" in init_result.output

            status_result = runner.invoke(cli, ["status"])
            assert status_result.exit_code == 0
            assert "_workflow_test" in status_result.output
            assert "Active session:" in status_result.output

    def test_multiple_init_commands_update_active_session(self):
        """Test that running init multiple times updates the active session."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result1 = runner.invoke(cli, ["init", "--name", "first"])
            assert result1.exit_code == 0

            result2 = runner.invoke(cli, ["init", "--name", "second"])
            assert result2.exit_code == 0

            status_result = runner.invoke(cli, ["status"])
            assert status_result.exit_code == 0
            assert "_second" in status_result.output
            assert "_first" not in status_result.output


class TestJupyterStartCommand:
    """Test suite for 'stanza jupyter start' command."""

    def test_starts_server_with_default_params(self, tmp_path):
        """Test starts server in current directory with default port."""
        runner = CliRunner()
        mock_state = {
            "pid": 12345,
            "url": "http://localhost:8888/?token=abc",
            "root_dir": str(tmp_path),
            "port": 8888,
        }
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("stanza.cli.jupyter.start", return_value=mock_state),
        ):
            result = runner.invoke(cli, ["jupyter", "start"])
            assert result.exit_code == 0
            assert "Jupyter server started successfully" in result.output
            assert "PID: 12345" in result.output
            assert "http://localhost:8888" in result.output

    def test_starts_server_with_custom_port(self, tmp_path):
        """Test starts server with custom port parameter."""
        runner = CliRunner()
        mock_state = {
            "pid": 12345,
            "url": "http://localhost:9999/?token=abc",
            "root_dir": str(tmp_path),
            "port": 9999,
        }
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("stanza.cli.jupyter.start", return_value=mock_state) as mock_start,
        ):
            result = runner.invoke(cli, ["jupyter", "start", "--port", "9999"])
            assert result.exit_code == 0
            mock_start.assert_called_once()
            assert mock_start.call_args[1]["port"] == 9999

    def test_starts_server_with_custom_directory(self, tmp_path):
        """Test starts server in custom directory."""
        runner = CliRunner()
        custom_dir = tmp_path / "notebooks"
        custom_dir.mkdir()
        mock_state = {
            "pid": 12345,
            "url": "http://localhost:8888/?token=abc",
            "root_dir": str(custom_dir),
            "port": 8888,
        }
        with patch("stanza.cli.jupyter.start", return_value=mock_state) as mock_start:
            result = runner.invoke(cli, ["jupyter", "start", str(custom_dir)])
            assert result.exit_code == 0
            mock_start.assert_called_once()

    def test_handles_runtime_error(self, tmp_path):
        """Test handles RuntimeError from jupyter.start."""
        runner = CliRunner()
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("stanza.cli.jupyter.start", side_effect=RuntimeError("Port in use")),
        ):
            result = runner.invoke(cli, ["jupyter", "start"])
            assert result.exit_code == 1
            assert "Error: Port in use" in result.output

    def test_handles_unexpected_error(self, tmp_path):
        """Test handles unexpected Exception from jupyter.start."""
        runner = CliRunner()
        with (
            runner.isolated_filesystem(temp_dir=tmp_path),
            patch("stanza.cli.jupyter.start", side_effect=ValueError("Unexpected")),
        ):
            result = runner.invoke(cli, ["jupyter", "start"])
            assert result.exit_code == 1
            assert "Unexpected error" in result.output


class TestJupyterStopCommand:
    """Test suite for 'stanza jupyter stop' command."""

    def test_stops_running_server(self):
        """Test stops running server successfully."""
        runner = CliRunner()
        with patch("stanza.cli.jupyter.stop") as mock_stop:
            result = runner.invoke(cli, ["jupyter", "stop"])
            assert result.exit_code == 0
            assert "Jupyter server stopped successfully" in result.output
            mock_stop.assert_called_once()

    def test_handles_stop_error(self):
        """Test handles Exception from jupyter.stop."""
        runner = CliRunner()
        with patch("stanza.cli.jupyter.stop", side_effect=RuntimeError("No server")):
            result = runner.invoke(cli, ["jupyter", "stop"])
            assert result.exit_code == 1
            assert "Error: No server" in result.output


class TestJupyterStatusCommand:
    """Test suite for 'stanza jupyter status' command."""

    def test_shows_running_server_status(self):
        """Test displays server status when running."""
        runner = CliRunner()
        mock_state = {
            "pid": 12345,
            "url": "http://localhost:8888/?token=abc",
            "root_dir": "/path/to/notebooks",
            "uptime_seconds": 7265,
        }
        with patch("stanza.cli.jupyter.status", return_value=mock_state):
            result = runner.invoke(cli, ["jupyter", "status"])
            assert result.exit_code == 0
            assert "Jupyter server is running" in result.output
            assert "PID: 12345" in result.output
            assert "http://localhost:8888" in result.output
            assert "2h 1m" in result.output
            assert "/path/to/notebooks" in result.output

    def test_shows_no_server_running(self):
        """Test displays message when no server running."""
        runner = CliRunner()
        with patch("stanza.cli.jupyter.status", return_value=None):
            result = runner.invoke(cli, ["jupyter", "status"])
            assert result.exit_code == 0
            assert "No Jupyter server is currently running" in result.output
            assert "stanza jupyter start" in result.output

    def test_handles_status_error(self):
        """Test handles Exception from jupyter.status."""
        runner = CliRunner()
        with patch("stanza.cli.jupyter.status", side_effect=RuntimeError("Error")):
            result = runner.invoke(cli, ["jupyter", "status"])
            assert result.exit_code == 1
            assert "Error: Error" in result.output


class TestJupyterOpenCommand:
    """Test suite for 'stanza jupyter open' command."""

    def test_opens_browser_with_url(self):
        """Test opens webbrowser with server URL."""
        runner = CliRunner()
        mock_state = {
            "pid": 12345,
            "url": "http://localhost:8888/?token=abc",
            "root_dir": "/path/to/notebooks",
        }
        with (
            patch("stanza.cli.jupyter.status", return_value=mock_state),
            patch("stanza.cli.webbrowser.open") as mock_open,
        ):
            result = runner.invoke(cli, ["jupyter", "open"])
            assert result.exit_code == 0
            assert "✓ Opened http://localhost:8888" in result.output
            mock_open.assert_called_once_with("http://localhost:8888/?token=abc")

    def test_auto_starts_when_no_server(self):
        """Test auto-starts server when none is running."""
        runner = CliRunner()
        mock_state = {
            "pid": 12345,
            "url": "http://localhost:8888/?token=abc",
            "root_dir": "/path/to/notebooks",
        }
        with (
            patch("stanza.cli.jupyter.status", return_value=None),
            patch("stanza.cli.jupyter.start", return_value=mock_state) as mock_start,
            patch("stanza.cli.webbrowser.open") as mock_open,
        ):
            result = runner.invoke(cli, ["jupyter", "open"])
            assert result.exit_code == 0
            assert "No Jupyter server running. Starting one..." in result.output
            assert "✓ Jupyter server started successfully" in result.output
            mock_start.assert_called_once()
            mock_open.assert_called_once_with("http://localhost:8888/?token=abc")

    def test_handles_auto_start_failure(self):
        """Test handles failure when auto-starting server."""
        runner = CliRunner()
        with (
            patch("stanza.cli.jupyter.status", return_value=None),
            patch(
                "stanza.cli.jupyter.start",
                side_effect=RuntimeError("Server already running"),
            ),
        ):
            result = runner.invoke(cli, ["jupyter", "open"])
            assert result.exit_code == 1
            assert "No Jupyter server running. Starting one..." in result.output
            assert "✗ Error: Server already running" in result.output

    def test_handles_open_error(self):
        """Test handles Exception from webbrowser.open."""
        runner = CliRunner()
        mock_state = {"url": "http://localhost:8888"}
        with (
            patch("stanza.cli.jupyter.status", return_value=mock_state),
            patch("stanza.cli.webbrowser.open", side_effect=RuntimeError("No browser")),
        ):
            result = runner.invoke(cli, ["jupyter", "open"])
            assert result.exit_code == 1


class TestJupyterListCommand:
    """Test suite for 'stanza jupyter list' command."""

    def test_lists_active_sessions(self):
        """Test displays active session notebook names."""
        runner = CliRunner()
        mock_sessions = [
            {"notebook_path": "/path/to/notebook1.ipynb"},
            {"notebook_path": "/path/to/notebook2.ipynb"},
        ]
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=mock_sessions),
        ):
            result = runner.invoke(cli, ["jupyter", "list"])
            assert result.exit_code == 0
            assert "notebook1.ipynb" in result.output
            assert "notebook2.ipynb" in result.output

    def test_shows_no_sessions_message(self):
        """Test displays message when no sessions active."""
        runner = CliRunner()
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=[]),
        ):
            result = runner.invoke(cli, ["jupyter", "list"])
            assert result.exit_code == 0
            assert "No active sessions" in result.output

    def test_requires_running_server(self):
        """Test aborts when no server running."""
        runner = CliRunner()
        with patch("stanza.cli.jupyter.status", return_value=None):
            result = runner.invoke(cli, ["jupyter", "list"])
            assert result.exit_code == 1
            assert "No Jupyter server running" in result.output


class TestJupyterLogsCommand:
    """Test suite for 'stanza jupyter logs' command."""

    def test_lists_all_logs_without_notebook_arg(self):
        """Test shows all log files with metadata."""
        runner = CliRunner()
        mock_sessions = [
            {
                "notebook_path": "/path/to/notebook1.ipynb",
                "log_path": "/path/to/notebook1.log",
                "size_bytes": 2048,
                "line_count": 100,
            },
            {
                "notebook_path": "/path/to/notebook2.ipynb",
                "log_path": "/path/to/notebook2.log",
                "size_bytes": 4096,
                "line_count": 200,
            },
        ]
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=mock_sessions),
        ):
            result = runner.invoke(cli, ["jupyter", "logs"])
            assert result.exit_code == 0
            assert "notebook1.ipynb" in result.output
            assert "notebook2.ipynb" in result.output
            assert "100 lines" in result.output
            assert "200 lines" in result.output

    def test_tails_specific_notebook_log(self):
        """Test follows log for specific notebook."""
        runner = CliRunner()
        mock_session = {
            "notebook_path": "/path/to/test.ipynb",
            "log_path": "/path/to/test.log",
        }
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=[mock_session]),
            patch("stanza.cli.log_stream.follow") as mock_follow,
        ):
            result = runner.invoke(cli, ["jupyter", "logs", "test"])
            assert result.exit_code == 0
            assert "Tailing test.ipynb" in result.output
            mock_follow.assert_called_once()

    def test_shows_no_sessions_message(self):
        """Test displays message when no sessions active."""
        runner = CliRunner()
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=[]),
        ):
            result = runner.invoke(cli, ["jupyter", "logs"])
            assert result.exit_code == 0
            assert "No active sessions" in result.output

    def test_requires_running_server(self):
        """Test aborts when no server running."""
        runner = CliRunner()
        with patch("stanza.cli.jupyter.status", return_value=None):
            result = runner.invoke(cli, ["jupyter", "logs"])
            assert result.exit_code == 1
            assert "No Jupyter server running" in result.output

    def test_handles_nonexistent_notebook(self):
        """Test aborts when notebook not found."""
        runner = CliRunner()
        mock_session = {"notebook_path": "/path/to/other.ipynb"}
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=[mock_session]),
        ):
            result = runner.invoke(cli, ["jupyter", "logs", "nonexistent"])
            assert result.exit_code == 1
            assert "No notebook matching 'nonexistent'" in result.output

    def test_handles_ambiguous_notebook_name(self):
        """Test aborts when multiple notebooks match."""
        runner = CliRunner()
        mock_sessions = [
            {"notebook_path": "/path/to/test1.ipynb"},
            {"notebook_path": "/path/to/test2.ipynb"},
        ]
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=mock_sessions),
        ):
            result = runner.invoke(cli, ["jupyter", "logs", "test"])
            assert result.exit_code == 1
            assert "Multiple notebooks match 'test'" in result.output


class TestJupyterAttachCommand:
    """Test suite for 'stanza jupyter attach' command."""

    def test_attaches_to_notebook(self):
        """Test attaches to log with kill callback."""
        runner = CliRunner()
        mock_session = {
            "notebook_path": "/path/to/test.ipynb",
            "log_path": "/path/to/test.log",
        }
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=[mock_session]),
            patch("stanza.cli.log_stream.attach") as mock_attach,
        ):
            result = runner.invoke(cli, ["jupyter", "attach", "test"])
            assert result.exit_code == 0
            assert "Attached to test.ipynb" in result.output
            mock_attach.assert_called_once()

    def test_requires_running_server(self):
        """Test aborts when no server running."""
        runner = CliRunner()
        with patch("stanza.cli.jupyter.status", return_value=None):
            result = runner.invoke(cli, ["jupyter", "attach", "test"])
            assert result.exit_code == 1
            assert "No Jupyter server running" in result.output

    def test_handles_nonexistent_notebook(self):
        """Test aborts when notebook not found."""
        runner = CliRunner()
        mock_session = {"notebook_path": "/path/to/other.ipynb"}
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=[mock_session]),
        ):
            result = runner.invoke(cli, ["jupyter", "attach", "nonexistent"])
            assert result.exit_code == 1
            assert "No notebook matching 'nonexistent'" in result.output

    def test_handles_ambiguous_notebook_name(self):
        """Test aborts when multiple notebooks match."""
        runner = CliRunner()
        mock_sessions = [
            {"notebook_path": "/path/to/test1.ipynb"},
            {"notebook_path": "/path/to/test2.ipynb"},
        ]
        with (
            patch("stanza.cli.jupyter.status", return_value={"pid": 123}),
            patch("stanza.cli.jupyter.list_sessions", return_value=mock_sessions),
        ):
            result = runner.invoke(cli, ["jupyter", "attach", "test"])
            assert result.exit_code == 1
            assert "Multiple notebooks match 'test'" in result.output
