"""Tests for Stanza CLI (stanza/cli.py)."""

import json
import re
from pathlib import Path

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
            assert (
                "Created:" not in result.output or result.output.count("Created:") == 0
            )


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


# ============================================================================
# Phase 8: Jupyter CLI Tests
# ============================================================================


class TestJupyterCommands:
    """Test suite for 'stanza jupyter' commands."""

    def test_jupyter_help(self):
        """Test that jupyter help command works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["jupyter", "--help"])

        assert result.exit_code == 0
        assert "Manage Jupyter notebook server" in result.output
        assert "start" in result.output
        assert "stop" in result.output
        assert "status" in result.output

    def test_jupyter_status_no_server(self):
        """Test jupyter status when no server is running."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["jupyter", "status"])

            assert result.exit_code == 0
            assert "No Jupyter server is currently running" in result.output
            assert "stanza jupyter start" in result.output

    def test_jupyter_start_success(self, tmp_path, monkeypatch):
        """Test jupyter start command with mocked jupyter.start()."""
        runner = CliRunner()

        # Mock the jupyter.start function
        def mock_start(notebook_dir):
            return {
                "pid": 12345,
                "url": "http://localhost:8888?token=test",
                "started_at": "2024-01-01T00:00:00Z",
                "root_dir": str(notebook_dir),
            }

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "start", mock_start)

        with runner.isolated_filesystem():
            notebook_dir = Path.cwd() / "notebooks"
            notebook_dir.mkdir()

            result = runner.invoke(cli, ["jupyter", "start", str(notebook_dir)])

            assert result.exit_code == 0
            assert "✓ Jupyter server started successfully" in result.output
            assert "PID: 12345" in result.output
            assert "http://localhost:8888" in result.output
            assert "survive terminal closure" in result.output

    def test_jupyter_start_default_directory(self, tmp_path, monkeypatch):
        """Test jupyter start uses current directory by default."""
        runner = CliRunner()

        called_with = None

        def mock_start(notebook_dir):
            nonlocal called_with
            called_with = notebook_dir
            return {
                "pid": 12345,
                "url": "http://localhost:8888",
                "started_at": "2024-01-01T00:00:00Z",
                "root_dir": str(notebook_dir),
            }

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "start", mock_start)

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["jupyter", "start"])

            assert result.exit_code == 0
            assert called_with is not None
            assert called_with.resolve() == Path.cwd().resolve()

    def test_jupyter_start_already_running(self, monkeypatch):
        """Test jupyter start when server already running."""
        runner = CliRunner()

        def mock_start(notebook_dir):
            raise RuntimeError("Jupyter server already running (PID 12345)")

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "start", mock_start)

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["jupyter", "start"])

            assert result.exit_code == 1
            assert "Error" in result.output
            assert "already running" in result.output

    def test_jupyter_stop_success(self, monkeypatch):
        """Test jupyter stop command."""
        runner = CliRunner()

        stopped = False

        def mock_stop():
            nonlocal stopped
            stopped = True

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "stop", mock_stop)

        result = runner.invoke(cli, ["jupyter", "stop"])

        assert result.exit_code == 0
        assert "✓ Jupyter server stopped successfully" in result.output
        assert stopped is True

    def test_jupyter_status_running(self, monkeypatch):
        """Test jupyter status when server is running."""
        runner = CliRunner()

        def mock_status():
            return {
                "pid": 12345,
                "url": "http://localhost:8888?token=test",
                "uptime_seconds": 3665,  # 1h 1m 5s
                "root_dir": "/tmp/notebooks",
            }

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "status", mock_status)

        result = runner.invoke(cli, ["jupyter", "status"])

        assert result.exit_code == 0
        assert "Jupyter server is running" in result.output
        assert "PID: 12345" in result.output
        assert "http://localhost:8888" in result.output
        assert "Uptime: 1h 1m" in result.output
        assert "/tmp/notebooks" in result.output

    def test_jupyter_status_uptime_formatting(self, monkeypatch):
        """Test that status formats uptime correctly."""
        runner = CliRunner()

        def mock_status():
            return {
                "pid": 12345,
                "url": "http://localhost:8888",
                "uptime_seconds": 7320,  # 2h 2m
                "root_dir": "/tmp",
            }

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "status", mock_status)

        result = runner.invoke(cli, ["jupyter", "status"])

        assert result.exit_code == 0
        assert "Uptime: 2h 2m" in result.output

    def test_jupyter_open_success(self, monkeypatch):
        """Test jupyter open command."""
        runner = CliRunner()

        def mock_status():
            return {
                "pid": 12345,
                "url": "http://localhost:8888?token=test123",
                "uptime_seconds": 100,
                "root_dir": "/tmp",
            }

        # Mock webbrowser.open
        opened_url = None

        def mock_webbrowser_open(url):
            nonlocal opened_url
            opened_url = url

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "status", mock_status)
        monkeypatch.setattr("webbrowser.open", mock_webbrowser_open)

        result = runner.invoke(cli, ["jupyter", "open"])

        assert result.exit_code == 0
        assert "✓ Opened" in result.output
        assert "http://localhost:8888?token=test123" in result.output
        assert opened_url == "http://localhost:8888?token=test123"

    def test_jupyter_open_not_running(self, monkeypatch):
        """Test jupyter open when server not running."""
        runner = CliRunner()

        def mock_status():
            return None

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "status", mock_status)

        result = runner.invoke(cli, ["jupyter", "open"])

        assert result.exit_code == 1
        assert "Error: No Jupyter server is currently running" in result.output
        assert "stanza jupyter start" in result.output


class TestJupyterCLIIntegration:
    """Integration tests for Jupyter CLI workflow."""

    def test_jupyter_full_workflow(self, monkeypatch):
        """Test complete workflow: start -> status -> stop."""
        runner = CliRunner()

        # Mock state
        server_state = None

        def mock_start(notebook_dir):
            nonlocal server_state
            server_state = {
                "pid": 12345,
                "url": "http://localhost:8888?token=test",
                "started_at": "2024-01-01T00:00:00Z",
                "root_dir": str(notebook_dir),
            }
            return server_state.copy()

        def mock_status():
            if server_state is None:
                return None
            return {**server_state, "uptime_seconds": 100}

        def mock_stop():
            nonlocal server_state
            server_state = None

        import stanza.cli

        monkeypatch.setattr(stanza.cli.jupyter, "start", mock_start)
        monkeypatch.setattr(stanza.cli.jupyter, "status", mock_status)
        monkeypatch.setattr(stanza.cli.jupyter, "stop", mock_stop)

        with runner.isolated_filesystem():
            # Check status when not running
            result1 = runner.invoke(cli, ["jupyter", "status"])
            assert result1.exit_code == 0
            assert "No Jupyter server" in result1.output

            # Start server
            result2 = runner.invoke(cli, ["jupyter", "start"])
            assert result2.exit_code == 0
            assert "started successfully" in result2.output

            # Check status when running
            result3 = runner.invoke(cli, ["jupyter", "status"])
            assert result3.exit_code == 0
            assert "Jupyter server is running" in result3.output
            assert "PID: 12345" in result3.output

            # Stop server
            result4 = runner.invoke(cli, ["jupyter", "stop"])
            assert result4.exit_code == 0
            assert "stopped successfully" in result4.output

            # Check status after stop
            result5 = runner.invoke(cli, ["jupyter", "status"])
            assert result5.exit_code == 0
            assert "No Jupyter server" in result5.output
