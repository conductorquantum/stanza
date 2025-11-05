"""Tests for stanza/jupyter/logs.py."""

import signal
import sys
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from stanza.jupyter.logs import _print_tail, _stream_log, _wait_for_log, attach, follow


class TestPrintTail:
    """Test suite for _print_tail function."""

    def test_prints_nothing_for_empty_log(self, tmp_path, capsys):
        """Test _print_tail prints nothing for empty log file."""
        log_file = tmp_path / "empty.log"
        log_file.write_text("")
        _print_tail(log_file, lines=10)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prints_tail_with_proper_line_endings(self, tmp_path, capsys):
        """Test _print_tail uses \\r\\n line endings."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\nline3")
        _print_tail(log_file, lines=10)
        captured = capsys.readouterr()
        assert "line1\r\n" in captured.out
        assert "line2\r\n" in captured.out
        assert "line3\r\n" in captured.out

    def test_cleans_carriage_returns_in_tail(self, tmp_path, capsys):
        """Test _print_tail cleans carriage return artifacts."""
        log_file = tmp_path / "test.log"
        log_file.write_text("old\rnew\nline2")
        _print_tail(log_file, lines=10)
        captured = capsys.readouterr()
        assert "old" not in captured.out
        assert "new" in captured.out

    def test_respects_line_limit(self, tmp_path, capsys):
        """Test _print_tail respects line limit parameter."""
        log_file = tmp_path / "test.log"
        lines = "\n".join([f"line{i}" for i in range(20)])
        log_file.write_text(lines)
        _print_tail(log_file, lines=5)
        captured = capsys.readouterr()
        assert "line15" in captured.out
        assert "line19" in captured.out
        assert "line10" not in captured.out

    def test_handles_nonexistent_file_gracefully(self, tmp_path, capsys):
        """Test _print_tail handles non-existent file gracefully."""
        log_file = tmp_path / "nonexistent.log"
        _print_tail(log_file, lines=10)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestWaitForLog:
    """Test suite for _wait_for_log function."""

    def test_returns_immediately_if_file_exists(self, tmp_path):
        """Test _wait_for_log returns immediately if file exists."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        _wait_for_log(log_file, timeout=1.0)

    def test_waits_until_file_appears(self, tmp_path):
        """Test _wait_for_log waits and returns when file appears."""
        log_file = tmp_path / "test.log"
        call_count = [0]

        def create_after_delay(*args):
            call_count[0] += 1
            return call_count[0] > 2

        with (
            patch.object(Path, "exists", side_effect=create_after_delay),
            patch("stanza.jupyter.logs.time.sleep"),
        ):
            log_file.write_text("test")
            _wait_for_log(log_file, timeout=2.0)

    def test_exits_on_timeout(self, tmp_path):
        """Test _wait_for_log exits with status 1 on timeout."""
        log_file = tmp_path / "nonexistent.log"
        mock_time = Mock(side_effect=[0, 0.05, 0.15])
        with (
            patch("stanza.jupyter.logs.time.time", mock_time),
            patch("stanza.jupyter.logs.time.sleep"),
            pytest.raises(SystemExit, match="1"),
        ):
            _wait_for_log(log_file, timeout=0.1)


class TestStreamLog:
    """Test suite for _stream_log function."""

    def test_reads_and_prints_new_lines(self, tmp_path, capsys):
        """Test _stream_log reads and prints new lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test content\n")
        mock_file = mock_open(read_data="test line\n")()
        _stream_log(mock_file, 0.1, log_file)
        captured = capsys.readouterr()
        assert "test line" in captured.out

    def test_uses_crlf_line_endings(self, tmp_path, capsys):
        """Test _stream_log outputs \\r\\n line endings."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test\n")
        mock_file = mock_open(read_data="test line\n")()
        _stream_log(mock_file, 0.1, log_file)
        captured = capsys.readouterr()
        assert "test line\r\n" in captured.out

    def test_strips_carriage_returns_from_input(self, tmp_path, capsys):
        """Test _stream_log strips \\r artifacts from progress bars."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test\n")
        mock_file = mock_open(read_data="old_text\rnew_text\n")()
        _stream_log(mock_file, 0.1, log_file)
        captured = capsys.readouterr()
        assert "old_text" not in captured.out
        assert "new_text" in captured.out

    def test_handles_multiple_carriage_returns(self, tmp_path, capsys):
        """Test _stream_log keeps only final text after multiple \\r."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test\n")
        mock_file = mock_open(read_data="first\rsecond\rthird\n")()
        _stream_log(mock_file, 0.1, log_file)
        captured = capsys.readouterr()
        assert "first" not in captured.out
        assert "second" not in captured.out
        assert "third" in captured.out

    def test_skips_empty_lines(self, tmp_path, capsys):
        """Test _stream_log skips empty lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test\n")
        mock_file = mock_open(read_data="\n")()
        _stream_log(mock_file, 0.1, log_file)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_skips_whitespace_only_lines(self, tmp_path, capsys):
        """Test _stream_log skips whitespace-only lines."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test\n")
        mock_file = mock_open(read_data="   \n")()
        _stream_log(mock_file, 0.1, log_file)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_sleeps_when_no_new_data(self, tmp_path):
        """Test _stream_log sleeps when no new data."""
        log_file = tmp_path / "test.log"
        log_file.write_text("")
        mock_file = mock_open(read_data="")()
        mock_file.readline.return_value = ""
        with patch("stanza.jupyter.logs.time.sleep") as mock_sleep:
            _stream_log(mock_file, 0.5, log_file)
            mock_sleep.assert_called_once_with(0.5)

    def test_exits_if_log_file_deleted(self, tmp_path):
        """Test _stream_log exits if log file deleted."""
        log_file = tmp_path / "test.log"
        mock_file = mock_open(read_data="")()
        mock_file.readline.return_value = ""
        with pytest.raises(SystemExit, match="1"):
            _stream_log(mock_file, 0.1, log_file)


class TestFollow:
    """Test suite for follow function."""

    def test_waits_for_log_file(self, tmp_path):
        """Test follow waits for log file."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\n")
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        with (
            patch("stanza.jupyter.logs._wait_for_log") as mock_wait,
            patch("stanza.jupyter.logs.tail_log", return_value=""),
            patch("builtins.open", return_value=mock_file),
            patch("stanza.jupyter.logs._stream_log", side_effect=[None, SystemExit(0)]),
        ):
            with pytest.raises(SystemExit):
                follow(log_file, lines=5)
            mock_wait.assert_called_once()

    def test_prints_initial_tail(self, tmp_path, capsys):
        """Test follow prints initial tail."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\n")
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        with (
            patch("stanza.jupyter.logs._wait_for_log"),
            patch("stanza.jupyter.logs.tail_log", return_value="initial content"),
            patch("builtins.open", return_value=mock_file),
            patch("stanza.jupyter.logs._stream_log", side_effect=SystemExit(0)),
        ):
            with pytest.raises(SystemExit):
                follow(log_file, lines=10)
            captured = capsys.readouterr()
            assert "initial content" in captured.out

    def test_handles_sigint_gracefully(self, tmp_path):
        """Test follow handles SIGINT gracefully."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        with (
            patch("stanza.jupyter.logs._wait_for_log"),
            patch("stanza.jupyter.logs.tail_log", return_value=""),
            patch("builtins.open", return_value=mock_file),
            patch("stanza.jupyter.logs._stream_log", side_effect=SystemExit(0)),
        ):
            with pytest.raises(SystemExit, match="0"):
                follow(log_file)

    def test_sigint_handler_prints_message(self, tmp_path, capsys):
        """Test SIGINT handler prints detach message."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        handler_ref = []

        def capture_handler(*args, **kwargs):
            if args[0] == signal.SIGINT:
                handler_ref.append(args[1])

        with (
            patch("stanza.jupyter.logs._wait_for_log"),
            patch("stanza.jupyter.logs.tail_log", return_value=""),
            patch("builtins.open", return_value=mock_file),
            patch("signal.signal", side_effect=capture_handler),
            patch("stanza.jupyter.logs._stream_log", side_effect=SystemExit(0)),
        ):
            with pytest.raises(SystemExit):
                follow(log_file)
            if handler_ref:
                with pytest.raises(SystemExit, match="0"):
                    handler_ref[0](signal.SIGINT, None)
                captured = capsys.readouterr()
                assert "Detached from" in captured.err
                assert "test.log" in captured.err


class TestAttach:
    """Test suite for attach function."""

    def test_waits_for_log_file(self, tmp_path):
        """Test attach waits for log file."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        kill_callback = Mock()
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        with (
            patch("stanza.jupyter.logs._wait_for_log") as mock_wait,
            patch("stanza.jupyter.logs.tail_log", return_value=""),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setraw"),
            patch.object(sys.stdin, "fileno", return_value=0),
            patch("builtins.open", return_value=mock_file),
            patch("select.select", side_effect=SystemExit(0)),
        ):
            with pytest.raises(SystemExit):
                attach(log_file, kill_callback, lines=5)
            mock_wait.assert_called_once()

    def test_prints_initial_tail(self, tmp_path, capsys):
        """Test attach prints initial tail."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        kill_callback = Mock()
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        with (
            patch("stanza.jupyter.logs._wait_for_log"),
            patch("stanza.jupyter.logs.tail_log", return_value="initial tail"),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch.object(sys.stdin, "fileno", return_value=0),
            patch("tty.setraw"),
            patch("builtins.open", return_value=mock_file),
            patch("select.select", side_effect=SystemExit(0)),
        ):
            with pytest.raises(SystemExit):
                attach(log_file, kill_callback)
            captured = capsys.readouterr()
            assert "initial tail" in captured.out

    def test_ctrl_c_kills_kernel(self, tmp_path):
        """Test attach handles Ctrl+C to kill kernel."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        kill_callback = Mock()
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.readline.return_value = ""
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        stdin_reads = ["\x03"]

        def mock_select(*args):
            if stdin_reads:
                return [[sys.stdin], [], []]
            return [[], [], []]

        def mock_read(n):
            if stdin_reads:
                key = stdin_reads.pop(0)
                return key
            raise SystemExit(0)

        with (
            patch("stanza.jupyter.logs._wait_for_log"),
            patch("stanza.jupyter.logs.tail_log", return_value=""),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setraw"),
            patch.object(sys.stdin, "fileno", return_value=0),
            patch("builtins.open", return_value=mock_file),
            patch("select.select", side_effect=mock_select),
            patch.object(sys.stdin, "read", side_effect=mock_read),
        ):
            with pytest.raises(SystemExit, match="0"):
                attach(log_file, kill_callback, poll_interval=0.01)
            kill_callback.assert_called_once()

    def test_esc_exits_without_kill(self, tmp_path):
        """Test attach handles ESC to exit without killing."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        kill_callback = Mock()
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.readline.return_value = ""
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        stdin_reads = ["\x1b", "\x1b"]

        def mock_select(*args):
            if stdin_reads:
                return [[sys.stdin], [], []]
            return [[], [], []]

        def mock_read(n):
            if stdin_reads:
                return stdin_reads.pop(0)
            raise SystemExit(0)

        with (
            patch("stanza.jupyter.logs._wait_for_log"),
            patch("stanza.jupyter.logs.tail_log", return_value=""),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setraw"),
            patch("builtins.open", return_value=mock_file),
            patch("select.select", side_effect=mock_select),
            patch.object(sys.stdin, "fileno", return_value=0),
            patch.object(sys.stdin, "read", side_effect=mock_read),
        ):
            with pytest.raises(SystemExit, match="0"):
                attach(log_file, kill_callback, poll_interval=0.01)
            kill_callback.assert_not_called()

    def test_restores_terminal_settings_on_exit(self, tmp_path):
        """Test attach restores terminal settings on exit."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        kill_callback = Mock()
        old_settings = ["original", "settings"]
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        with (
            patch("stanza.jupyter.logs._wait_for_log"),
            patch("stanza.jupyter.logs.tail_log", return_value=""),
            patch("termios.tcgetattr", return_value=old_settings),
            patch("termios.tcsetattr") as mock_set,
            patch("tty.setraw"),
            patch.object(sys.stdin, "fileno", return_value=0),
            patch("builtins.open", return_value=mock_file),
            patch("select.select", side_effect=SystemExit(0)),
        ):
            with pytest.raises(SystemExit):
                attach(log_file, kill_callback)
            mock_set.assert_called()

    def test_kill_callback_exception_handling(self, tmp_path, capsys):
        """Test attach handles exceptions during kill_callback."""
        log_file = tmp_path / "test.log"
        log_file.write_text("test")
        kill_callback = Mock(side_effect=RuntimeError("Kill failed"))
        mock_file = Mock()
        mock_file.seek = Mock()
        mock_file.readline.return_value = ""
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)
        stdin_reads = ["\x03"]

        def mock_select(*args):
            if stdin_reads:
                return [[sys.stdin], [], []]
            return [[], [], []]

        def mock_read(n):
            if stdin_reads:
                key = stdin_reads.pop(0)
                return key
            raise SystemExit(0)

        with (
            patch("stanza.jupyter.logs._wait_for_log"),
            patch("stanza.jupyter.logs.tail_log", return_value=""),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setraw"),
            patch.object(sys.stdin, "fileno", return_value=0),
            patch("builtins.open", return_value=mock_file),
            patch("select.select", side_effect=mock_select),
            patch.object(sys.stdin, "read", side_effect=mock_read),
        ):
            with pytest.raises(SystemExit, match="0"):
                attach(log_file, kill_callback, poll_interval=0.01)
            captured = capsys.readouterr()
            assert "Error" in captured.err
