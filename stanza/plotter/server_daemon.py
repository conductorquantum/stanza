#!/usr/bin/env python3
"""Persistent Bokeh server daemon for live plotting."""

import argparse
import signal
import sys

from stanza.plotter.backends.server import ServerBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5006)
    args = parser.parse_args()

    server = ServerBackend(port=args.port, daemon=False)

    def shutdown(_sig: int, _frame: object) -> None:
        print("\nShutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Starting server: http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    server.start(block=True)


if __name__ == "__main__":
    main()
