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
    """Wait for log file to exist or timeout."""
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
    """Print last N lines from log file with proper terminal alignment."""
    tail = clean_carriage_returns(tail_log(log_file, lines))
    if tail:
        sys.stdout.write(tail.replace("\n", "\r\n") + "\r\n")
        sys.stdout.flush()


def _stream_log(f: object, poll_interval: float, log_file: Path) -> None:
    """Read and print new log lines with proper terminal alignment."""
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
    """Stream log file until Ctrl+C."""
    _wait_for_log(log_file)

    def sigint_handler(_sig: int, _frame: object) -> None:
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
    """Stream log file. Ctrl+C kills kernel, ESC exits without killing."""
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
