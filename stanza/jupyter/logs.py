"""Log streaming for Jupyter notebooks and servers."""

import os
import select
import signal
import sys
import termios
import time
import tty
from collections.abc import Callable
from pathlib import Path

from stanza.jupyter.utils import clean_carriage_returns, tail_log


def _wait_for_log(log_file: Path, timeout: float = 30.0) -> None:
    """Wait for log file to exist or timeout.

    Polls for the log file's existence, displaying a waiting message to stderr.
    Exits with status 1 if the file doesn't appear within the timeout period.

    Args:
        log_file: Path to the log file to wait for
        timeout: Maximum seconds to wait before exiting
    """
    if log_file.exists():
        return

    sys.stderr.write(f"Waiting for {log_file.name}...\r\n")
    sys.stderr.flush()
    start = time.time()
    while not log_file.exists():
        if time.time() - start > timeout:
            sys.stderr.write(f"Timeout after {timeout}s\r\n")
            sys.stderr.flush()
            sys.exit(1)
        time.sleep(0.1)


def _print_tail(log_file: Path, lines: int) -> None:
    """Print last N lines from log file with proper terminal alignment.

    Args:
        log_file: Path to the log file to read
        lines: Number of lines to print from the end of the file
    """
    tail = clean_carriage_returns(tail_log(log_file, lines))
    if tail:
        sys.stdout.write(tail.replace("\n", "\r\n") + "\r\n")
        sys.stdout.flush()


def _stream_log(f: object, poll_interval: float, log_file: Path) -> None:
    """Read and print new log lines with proper terminal alignment.

    Strips carriage return artifacts from progress bars and formats output
    for proper terminal display. Exits if the log file is deleted.

    Args:
        f: Open file handle positioned at the current read position
        poll_interval: Seconds to sleep when no new data is available
        log_file: Path to the log file (for existence checking)
    """
    line = f.readline()  # type: ignore[attr-defined]
    if line:
        # Strip \r artifacts from progress bars - keep only final visible text
        if "\r" in line:
            line = line.split("\r")[-1]
        line = line.rstrip()

        if line:
            # Use \r\n instead of \n to reset cursor to column 0
            sys.stdout.write(line + "\r\n")
            sys.stdout.flush()
    elif not log_file.exists():
        sys.stderr.write("\r\nLog deleted\r\n")
        sys.stderr.flush()
        sys.exit(1)
    else:
        time.sleep(poll_interval)


def follow(log_file: Path, lines: int = 10, poll_interval: float = 0.1) -> None:
    """Stream log file until Ctrl+C.

    Displays the last N lines of the log file, then continuously streams new
    lines as they are written. Exits gracefully on Ctrl+C.

    Args:
        log_file: Path to the log file to follow
        lines: Number of initial lines to display from the end of the file
        poll_interval: Seconds to wait between checks for new content
    """
    _wait_for_log(log_file)

    def sigint_handler(_sig: int, _frame: object) -> None:
        """Handle Ctrl+C by displaying detach message and exiting."""
        sys.stderr.write(f"\r\nDetached from {log_file.name}\r\n")
        sys.stderr.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    _print_tail(log_file, lines)

    with open(log_file, encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            _stream_log(f, poll_interval, log_file)


def attach(
    log_file: Path,
    kill_callback: Callable[[], None],
    lines: int = 10,
    poll_interval: float = 0.1,
) -> None:
    """Stream log file with active kernel control.

    Displays the last N lines of the log file, then continuously streams new
    lines. Ctrl+C kills the kernel via the callback. ESC exits without killing
    (press twice for safety).

    Args:
        log_file: Path to the log file to follow
        kill_callback: Function to call when Ctrl+C is pressed
        lines: Number of initial lines to display from the end of the file
        poll_interval: Seconds to wait between checks for new content
    """
    _wait_for_log(log_file)
    _print_tail(log_file, lines)

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        esc_pressed = False

        with open(log_file, encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)

            while True:
                ready, _, _ = select.select([sys.stdin, f], [], [], poll_interval)

                if sys.stdin in ready:
                    key = sys.stdin.read(1)
                    if key == "\x03":  # Ctrl+C
                        sys.stderr.write("\r\033[K\r\n\r\nKilling kernel...\r\n")
                        sys.stderr.flush()
                        try:
                            kill_callback()
                            sys.stderr.write("Kernel killed\r\n")
                            sys.stderr.flush()
                        except Exception as e:
                            sys.stderr.write(f"Error: {e}\r\n")
                            sys.stderr.flush()
                        sys.exit(0)
                    elif key == "\x1b":  # ESC
                        if esc_pressed:
                            sys.stderr.write("\r\033[KExited\r\n")
                            sys.stderr.flush()
                            sys.exit(0)
                        esc_pressed = True
                        sys.stderr.write("\r\033[K\r\nPress ESC again to exit\r\n")
                        sys.stderr.flush()
                    else:
                        esc_pressed = False

                if f in ready or not ready:
                    _stream_log(f, poll_interval, log_file)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
