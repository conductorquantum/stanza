"""Stanza CLI - Command-line interface for Stanza experiment framework."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import click

from stanza import __version__
from stanza.context import StanzaSession


@click.group()
@click.version_option(version=__version__, message="%(version)s (Stanza)")
def cli() -> None:
    """Stanza - Build tune up sequences for quantum computers fast.

    Easy to code. Easy to run.
    """
    pass


@cli.command()
@click.option(
    "--path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Base directory for session (default: current directory)",
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="Name suffix for directory (default: 'data')",
)
def init(path: Path | None, name: str | None) -> None:
    """Initialize a new timestamped experiment session directory.

    Creates a directory like: 20251020100010_data/

    All experiment data from routines will be logged inside this directory.

    Examples:

        stanza init

        stanza init --name my_experiment

        stanza init --path /data/experiments
    """
    try:
        session_dir = StanzaSession.create_session_directory(
            base_path=path,
            name=name,
        )

        StanzaSession.set_active_session(session_dir)

        click.echo(f"✓ Created session directory: {session_dir}")
        click.echo(f"  Active session set to: {session_dir.name}")
        click.echo()
        click.echo("Session initialized successfully!")
        click.echo("All experiment data will be logged to this directory.")

    except FileExistsError as e:
        click.echo("✗ Error: Session directory already exists", err=True)
        raise click.Abort() from e
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e


@cli.command()
def status() -> None:
    """Show current active session information."""
    active_session = StanzaSession.get_active_session()

    if active_session is None:
        click.echo("No active session")
        click.echo()
        click.echo("Initialize a session with: stanza init")
        return

    metadata = StanzaSession.get_session_metadata(active_session)

    click.echo(f"Active session: {active_session.name}")
    click.echo(f"  Location: {active_session}")

    if metadata:
        from datetime import datetime

        created = datetime.fromtimestamp(metadata["created_at"])
        click.echo(f"  Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")


def _get_config_paths() -> tuple[Path, Path, Path]:
    """Get paths for live plot config, PID, and log files."""
    config_dir = Path.cwd() / StanzaSession.CONFIG_DIR
    config_dir.mkdir(exist_ok=True)
    return (
        config_dir / "live_plot_config.json",
        config_dir / "live_plot_server.pid",
        config_dir / "live_plot_server.log",
    )


def _read_config() -> dict[str, Any]:
    """Read live plot config, return empty dict if not found."""
    config_file, _, _ = _get_config_paths()
    if not config_file.exists():
        return {}
    with open(config_file) as f:
        return cast(dict[str, Any], json.load(f))


def _write_config(config: dict[str, Any]) -> None:
    """Write live plot config."""
    config_file, _, _ = _get_config_paths()
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def _is_process_running(pid: int) -> bool:
    """Check if process with given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@cli.group()
def live_plot() -> None:
    """Manage live plotting."""
    pass


@live_plot.command(name="enable")
@click.option("--backend", type=click.Choice(["server", "inline"]), default="server")
@click.option("--port", type=int, default=5006)
def enable_live_plot(backend: str, port: int) -> None:
    """Enable live plotting. Starts persistent server for 'server' backend."""
    _, pid_file, log_file = _get_config_paths()

    # Check if server already running
    if pid_file.exists():
        with open(pid_file) as f:
            pid = int(f.read().strip())
        if _is_process_running(pid):
            click.echo(f"✗ Server already running (PID {pid})")
            click.echo("  Run 'stanza live-plot disable' first")
            raise click.Abort()
        pid_file.unlink()

    _write_config({"enabled": True, "backend": backend, "port": port})

    if backend == "inline":
        click.echo("✓ Live plotting enabled (inline mode)")
        return

    # Start persistent server
    server_script = Path(__file__).parent / "plotter" / "server_daemon.py"
    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(server_script), "--port", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_file.write_text(str(proc.pid))
    time.sleep(2)

    if not _is_process_running(proc.pid):
        click.echo(f"✗ Server failed to start. Check logs: {log_file}")
        pid_file.unlink(missing_ok=True)
        raise click.Abort()

    click.echo(f"✓ Server started (PID {proc.pid})")
    click.echo(f"  http://localhost:{port}")
    click.echo(f"  Logs: {log_file}")


@live_plot.command(name="disable")
def disable_live_plot() -> None:
    """Disable live plotting and stop server."""
    _, pid_file, _ = _get_config_paths()

    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"✓ Stopped server (PID {pid})")
        except OSError:
            click.echo(f"Server (PID {pid}) not found")
        pid_file.unlink()

    _write_config({"enabled": False})
    click.echo("✓ Live plotting disabled")


@live_plot.command(name="status")
def live_plot_status() -> None:
    """Show live plotting status."""
    config = _read_config()
    _, pid_file, _ = _get_config_paths()

    if not config.get("enabled"):
        click.echo("Live plotting: disabled")
        return

    backend = config.get("backend", "server")
    click.echo(f"Live plotting: enabled ({backend})")

    if backend != "server":
        return

    port = config.get("port", 5006)
    click.echo(f"  Port: {port}")

    if not pid_file.exists():
        click.echo("  Server: not started")
        return

    pid = int(pid_file.read_text().strip())
    if _is_process_running(pid):
        click.echo(f"  Server: running (PID {pid})")
        click.echo(f"  http://localhost:{port}")
    else:
        click.echo("  Server: not running")
        pid_file.unlink()


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
