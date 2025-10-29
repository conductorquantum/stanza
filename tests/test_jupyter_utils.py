"""Tests for stanza/jupyter/utils.py."""

from unittest.mock import patch

from stanza.jupyter.utils import format_size, tail_log


class TestTailLog:
    """Test suite for tail_log function."""

    def test_nonexistent_file_returns_empty_string(self, tmp_path):
        """Test tail_log returns empty string for non-existent file."""
        log_file = tmp_path / "nonexistent.log"
        assert tail_log(log_file) == ""

    def test_empty_file_returns_empty_string(self, tmp_path):
        """Test tail_log returns empty string for empty file."""
        log_file = tmp_path / "empty.log"
        log_file.write_text("")
        assert tail_log(log_file) == ""

    def test_reads_last_n_lines(self, tmp_path):
        """Test tail_log reads last N lines from file."""
        log_file = tmp_path / "test.log"
        content = "\n".join([f"line {i}" for i in range(1, 21)])
        log_file.write_text(content)
        result = tail_log(log_file, lines=5)
        assert result == "\n".join([f"line {i}" for i in range(16, 21)])

    def test_reads_all_lines_if_fewer_than_limit(self, tmp_path):
        """Test tail_log reads all lines if file has fewer than N lines."""
        log_file = tmp_path / "short.log"
        log_file.write_text("line 1\nline 2\nline 3")
        result = tail_log(log_file, lines=10)
        assert result == "line 1\nline 2\nline 3"

    def test_limits_read_to_4kb(self, tmp_path):
        """Test tail_log only reads last 4KB of large files."""
        log_file = tmp_path / "large.log"
        large_content = "x" * 10000
        log_file.write_text(large_content)
        result = tail_log(log_file, lines=10)
        assert len(result) <= 4096

    def test_handles_unicode_decode_errors(self, tmp_path):
        """Test tail_log handles invalid UTF-8 gracefully."""
        log_file = tmp_path / "invalid.log"
        log_file.write_bytes(b"\x80\x81\x82\x83")
        result = tail_log(log_file)
        assert "\ufffd" in result

    def test_single_line_file(self, tmp_path):
        """Test tail_log with single line file."""
        log_file = tmp_path / "single.log"
        log_file.write_text("single line")
        result = tail_log(log_file, lines=5)
        assert result == "single line"

    def test_handles_oserror_during_read(self, tmp_path):
        """Test tail_log handles OSError during file read."""
        log_file = tmp_path / "test.log"
        log_file.write_text("content")
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            result = tail_log(log_file)
            assert result == ""


class TestFormatSize:
    """Test suite for format_size function."""

    def test_formats_bytes(self):
        """Test format_size formats bytes correctly."""
        assert format_size(0) == "0 B"
        assert format_size(1) == "1 B"
        assert format_size(512) == "512 B"
        assert format_size(1023) == "1023 B"

    def test_formats_kilobytes(self):
        """Test format_size formats KB correctly."""
        assert format_size(1024) == "1.0 KB"
        assert format_size(2048) == "2.0 KB"
        assert format_size(1536) == "1.5 KB"

    def test_formats_megabytes(self):
        """Test format_size formats MB correctly."""
        assert format_size(1024 * 1024) == "1.0 MB"
        assert format_size(1024 * 1024 * 2) == "2.0 MB"
        assert format_size(1024 * 1024 + 512 * 1024) == "1.5 MB"

    def test_rounds_to_one_decimal(self):
        """Test format_size rounds to one decimal place."""
        assert format_size(1075) == "1.0 KB"
        assert format_size(1126) == "1.1 KB"
        assert format_size(1024 * 1024 + 102 * 1024) == "1.1 MB"
