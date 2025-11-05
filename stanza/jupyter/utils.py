"""Utilities for Jupyter notebook management."""

import os
from pathlib import Path


def clean_carriage_returns(text: str) -> str:
    """Remove carriage return artifacts from logged output.

    Terminal progress bars use \\r to overwrite previous text. When logged to a file,
    this creates lines like "old_text\\r                              new_text".
    We keep only the text after the last \\r on each line, which is what would be
    visible on a real terminal.

    Args:
        text: Raw text containing carriage return artifacts

    Returns:
        Cleaned text with only the final visible content from each line
    """
    if not text:
        return ""

    # Split on \n only (not \r) to preserve \r within lines
    lines = []
    for line in text.split("\n"):
        # Handle \r within line - keep only text after last \r
        if "\r" in line:
            line = line.split("\r")[-1]
        # Strip leading/trailing whitespace and skip empty lines
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def tail_log(log_file: Path, lines: int = 10) -> str:
    """Read last N lines from log file.

    Only reads last 4KB to avoid loading large files into memory.
    Splits on \\n only to preserve \\r for carriage return processing.

    Args:
        log_file: Path to the log file to read
        lines: Number of lines to return from the end of the file

    Returns:
        Last N lines from the log file, or empty string if file doesn't exist
    """
    if not log_file.exists():
        return ""

    try:
        with open(log_file, "rb") as f:
            file_size = f.seek(0, os.SEEK_END)
            if file_size == 0:
                return ""

            chunk_size = min(4096, file_size)
            f.seek(-chunk_size, os.SEEK_END)
            tail_bytes = f.read()

        tail_text = tail_bytes.decode("utf-8", errors="replace")
        # Split on \n only (not \r) to preserve \r for carriage return processing
        return "\n".join(tail_text.split("\n")[-lines:])
    except (OSError, UnicodeDecodeError):
        return ""


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable string (e.g., "1.5 KB", "2.3 MB")
    """
    kb = size_bytes / 1024
    if kb < 1:
        return f"{size_bytes} B"
    if kb < 1024:
        return f"{kb:.1f} KB"
    return f"{kb / 1024:.1f} MB"
