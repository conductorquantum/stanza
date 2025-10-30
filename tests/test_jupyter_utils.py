"""Tests for stanza/jupyter/utils.py."""

from unittest.mock import patch

from stanza.jupyter.utils import clean_carriage_returns, format_size, tail_log


class TestCleanCarriageReturns:
    """Test suite for clean_carriage_returns function."""

    def test_empty_string_returns_empty(self):
        """Test clean_carriage_returns returns empty string for empty input."""
        assert clean_carriage_returns("") == ""

    def test_no_carriage_returns_unchanged(self):
        """Test clean_carriage_returns leaves text without \\r unchanged."""
        text = "line1\nline2\nline3"
        assert clean_carriage_returns(text) == text

    def test_single_carriage_return_keeps_last_segment(self):
        """Test clean_carriage_returns keeps only text after \\r on single line."""
        text = "old_text\rnew_text"
        assert clean_carriage_returns(text) == "new_text"

    def test_multiple_carriage_returns_keeps_final_segment(self):
        """Test clean_carriage_returns keeps only final segment with multiple \\r."""
        text = "first\rsecond\rthird"
        assert clean_carriage_returns(text) == "third"

    def test_progress_bar_artifact_cleaned(self):
        """Test clean_carriage_returns removes progress bar artifacts."""
        text = "old_text\r                              new_text"
        assert clean_carriage_returns(text) == "new_text"

    def test_multiline_with_carriage_returns(self):
        """Test clean_carriage_returns handles multiple lines with \\r."""
        text = "line1\nold\rnew\nline3"
        expected = "line1\nnew\nline3"
        assert clean_carriage_returns(text) == expected

    def test_empty_lines_removed(self):
        """Test clean_carriage_returns removes empty lines."""
        text = "line1\n\nline3"
        assert clean_carriage_returns(text) == "line1\nline3"

    def test_whitespace_only_lines_removed(self):
        """Test clean_carriage_returns removes whitespace-only lines."""
        text = "line1\n   \nline3"
        assert clean_carriage_returns(text) == "line1\nline3"

    def test_carriage_return_at_line_end(self):
        """Test clean_carriage_returns handles \\r at end of line."""
        text = "text1\r\ntext2"
        assert clean_carriage_returns(text) == "text2"

    def test_real_world_log_example(self):
        """Test clean_carriage_returns with real world log scenario."""
        text = (
            "stanza.logger - INFO - Message 1\n"
            "Progress: 0%\rProgress: 50%\rProgress: 100%\n"
            "stanza.logger - INFO - Message 2"
        )
        expected = (
            "stanza.logger - INFO - Message 1\n"
            "Progress: 100%\n"
            "stanza.logger - INFO - Message 2"
        )
        assert clean_carriage_returns(text) == expected


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
