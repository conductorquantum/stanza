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

from stanza.jupyter.utils import tail_log


def _wait_for_log(log_file: Path, timeout: float = 30.0) -> None:
    """Wait for log file to exist or timeout."""
    if log_file.exists():
        return

    print(f"Waiting for {log_file.name}...", file=sys.stderr)
    start = time.time()
    while not log_file.exists():
        if time.time() - start > timeout:
            print(f"Timeout after {timeout}s", file=sys.stderr)
            sys.exit(1)
        time.sleep(0.1)


def _stream_log(f: object, poll_interval: float, log_file: Path) -> None:
    """Read and print new log lines."""
    line = f.readline()  # type: ignore[attr-defined]
    if line:
        print(line, end="")
        sys.stdout.flush()
    elif not log_file.exists():
        print("\nLog deleted", file=sys.stderr)
        sys.exit(1)
    else:
        time.sleep(poll_interval)


def follow(log_file: Path, lines: int = 10, poll_interval: float = 0.1) -> None:
    """Stream log file until Ctrl+C."""
    _wait_for_log(log_file)

    def sigint_handler(_sig: int, _frame: object) -> None:
        print(f"\nDetached from {log_file.name}", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    tail = tail_log(log_file, lines)
    if tail:
        print(tail)
        sys.stdout.flush()

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

    tail = tail_log(log_file, lines)
    if tail:
        print(tail)
        sys.stdout.flush()

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
                    if key == "\x03":
                        print("\n\nKilling kernel...", file=sys.stderr)
                        try:
                            kill_callback()
                            print("Kernel killed", file=sys.stderr)
                        except Exception as e:
                            print(f"Error: {e}", file=sys.stderr)
                        sys.exit(0)
                    elif key == "\x1b":
                        if esc_pressed:
                            print("\nExited", file=sys.stderr)
                            sys.exit(0)
                        esc_pressed = True
                        print("\nPress ESC again to exit", file=sys.stderr)
                    else:
                        esc_pressed = False

                if f in ready or not ready:
                    _stream_log(f, poll_interval, log_file)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
