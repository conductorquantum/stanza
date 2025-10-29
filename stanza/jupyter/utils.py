"""Utilities for Jupyter notebook management."""

import os
from pathlib import Path


def tail_log(log_file: Path, lines: int = 10) -> str:
    """Read last N lines from log file.

    Only reads last 4KB to avoid loading large files into memory.
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
        return "\n".join(tail_text.splitlines()[-lines:])
    except (OSError, UnicodeDecodeError):
        return ""


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    kb = size_bytes / 1024
    if kb < 1:
        return f"{size_bytes} B"
    if kb < 1024:
        return f"{kb:.1f} KB"
    return f"{kb / 1024:.1f} MB"
