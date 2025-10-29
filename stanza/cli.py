"""Stanza CLI - Command-line interface for Stanza experiment framework."""

from __future__ import annotations

import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import click

from stanza import __version__, jupyter
from stanza.context import StanzaSession
from stanza.jupyter import logs as log_stream
from stanza.jupyter.utils import format_size


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
        created = datetime.fromtimestamp(metadata["created_at"])
        click.echo(f"  Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")


def _get_config_file() -> Path:
    """Get path to live plot config file."""
    config_dir = Path.cwd() / StanzaSession.CONFIG_DIR
    config_dir.mkdir(exist_ok=True)
    return config_dir / "live_plot_config.json"


def _read_config() -> dict[str, Any]:
    """Read live plot config, return empty dict if not found."""
    config_file = _get_config_file()
    if not config_file.exists():
        return {}
    return cast(dict[str, Any], json.loads(config_file.read_text()))


def _write_config(config: dict[str, Any]) -> None:
    """Write live plot config."""
    config_file = _get_config_file()
    config_file.write_text(json.dumps(config, indent=2) + "\n")


@cli.group()
def live_plot() -> None:
    """Manage live plotting."""
    pass


@live_plot.command(name="enable")
@click.option("--backend", type=click.Choice(["server", "inline"]), default="server")
@click.option("--port", type=int, default=5006)
def enable_live_plot(backend: str, port: int) -> None:
    """Enable live plotting configuration."""
    _write_config({"enabled": True, "backend": backend, "port": port})

    click.echo(f"✓ Live plotting enabled ({backend} backend)")
    if backend == "server":
        click.echo(f"  Port: {port}")
        click.echo(f"  DataLogger will auto-start server on port {port}")
        click.echo(
            f"  Open http://localhost:{port} in browser when running experiments"
        )


@live_plot.command(name="disable")
def disable_live_plot() -> None:
    """Disable live plotting configuration."""
    _write_config({"enabled": False})
    click.echo("✓ Live plotting disabled")


@live_plot.command(name="status")
def live_plot_status() -> None:
    """Show live plotting configuration."""
    config = _read_config()

    if not config.get("enabled"):
        click.echo("Live plotting: disabled")
        return

    backend = config.get("backend", "server")
    port = config.get("port", 5006)

    click.echo(f"Live plotting: enabled ({backend} backend)")
    if backend == "server":
        click.echo(f"  Port: {port}")


@cli.group(name="jupyter")
def jupyter_cli() -> None:
    """Manage Jupyter notebook server."""
    pass


@jupyter_cli.command(name="start")
@click.argument(
    "notebook_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=".",
)
@click.option(
    "--port",
    type=int,
    default=8888,
    help="Port for Jupyter server (default: 8888)",
)
def jupyter_start(notebook_dir: Path, port: int) -> None:
    """Start Jupyter notebook server in background.

    Examples:
        stanza jupyter start
        stanza jupyter start /path/to/notebooks --port 8889
    """
    try:
        notebook_dir = notebook_dir.resolve()
        click.echo(f"Starting Jupyter server in {notebook_dir}...")

        state = jupyter.start(notebook_dir, port=port)

        click.echo("✓ Jupyter server started successfully")
        click.echo(f"  PID: {state['pid']}")
        click.echo(f"  URL: {state['url']}")
        click.echo(f"  Root: {state['root_dir']}")
        click.echo()
        click.echo("Server is running in background and will survive terminal closure.")
        click.echo("Use 'stanza jupyter stop' to shut down the server.")

    except RuntimeError as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e
    except Exception as e:
        click.echo(f"✗ Unexpected error: {e}", err=True)
        raise click.Abort() from e


@jupyter_cli.command(name="stop")
def jupyter_stop() -> None:
    """Stop Jupyter notebook server gracefully.
    Uses escalating shutdown: REST API -> SIGTERM -> SIGKILL
    Examples:
        stanza jupyter stop
    """
    try:
        click.echo("Stopping Jupyter server...")
        jupyter.stop()
        click.echo("✓ Jupyter server stopped successfully")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e


@jupyter_cli.command(name="status")
def jupyter_status() -> None:
    """Show Jupyter server status.
    Displays PID, URL, uptime, and root directory if server is running.
    Examples:
        stanza jupyter status
    """
    try:
        state = jupyter.status()

        if state is None:
            click.echo("No Jupyter server is currently running")
            click.echo()
            click.echo("Start a server with: stanza jupyter start")
            return

        uptime_hours = state["uptime_seconds"] / 3600
        uptime_mins = (state["uptime_seconds"] % 3600) / 60

        click.echo("Jupyter server is running")
        click.echo(f"  PID: {state['pid']}")
        click.echo(f"  URL: {state['url']}")
        click.echo(f"  Uptime: {int(uptime_hours)}h {int(uptime_mins)}m")
        click.echo(f"  Root: {state['root_dir']}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e


@jupyter_cli.command(name="open")
def jupyter_open() -> None:
    """Open Jupyter in browser.
    Opens the Jupyter URL with authentication token in your default browser.
    Examples:
        stanza jupyter open
    """
    try:
        state = jupyter.status()

        if state is None:
            click.echo("✗ Error: No Jupyter server is currently running", err=True)
            click.echo()
            click.echo("Start a server with: stanza jupyter start")
            raise click.Abort()

        webbrowser.open(state["url"])
        click.echo(f"✓ Opened {state['url']}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e


def _require_server() -> None:
    """Check if server is running, abort if not."""
    if jupyter.status() is None:
        click.echo("✗ No Jupyter server running", err=True)
        click.echo("Start with: stanza jupyter start")
        raise click.Abort()


def _find_notebook(name: str) -> dict[str, Any]:
    """Find notebook session by name, abort if not found or ambiguous."""
    sessions = jupyter.list_sessions()
    matches = [
        s for s in sessions if name.lower() in Path(s["notebook_path"]).name.lower()
    ]

    if not matches:
        click.echo(f"✗ No notebook matching '{name}'", err=True)
        if sessions:
            click.echo("Active notebooks:")
            for s in sessions:
                click.echo(f"  - {Path(s['notebook_path']).name}")
        raise click.Abort()

    if len(matches) > 1:
        click.echo(f"✗ Multiple notebooks match '{name}':", err=True)
        for s in matches:
            click.echo(f"  - {Path(s['notebook_path']).name}")
        raise click.Abort()

    return matches[0]


@jupyter_cli.command(name="list")
def jupyter_list() -> None:
    """List active notebook sessions."""
    _require_server()
    sessions = jupyter.list_sessions()

    if not sessions:
        click.echo("No active sessions")
        return

    for s in sessions:
        click.echo(Path(s["notebook_path"]).name)


@jupyter_cli.command(name="logs")
@click.argument("notebook", required=False)
@click.option("-n", "--lines", type=int, default=10, help="Lines to show initially")
def jupyter_logs(notebook: str | None, lines: int) -> None:
    """List logs or tail a notebook's log file."""
    _require_server()

    if notebook is None:
        sessions = jupyter.list_sessions()
        if not sessions:
            click.echo("No active sessions")
            return

        for s in sessions:
            nb = Path(s["notebook_path"]).name
            log = Path(s["log_path"]).name
            size = format_size(s["size_bytes"])
            click.echo(f"{nb} → {log}  ({s['line_count']} lines, {size})")
        return

    session = _find_notebook(notebook)
    log_path = Path(session["log_path"])
    click.echo(f"Tailing {Path(session['notebook_path']).name} (Ctrl+C to detach)")
    log_stream.follow(log_path, lines=lines)


@jupyter_cli.command(name="attach")
@click.argument("notebook", required=True)
@click.option("-n", "--lines", type=int, default=10, help="Lines to show initially")
def jupyter_attach(notebook: str, lines: int) -> None:
    """Attach to notebook with active control (Ctrl+C kills kernel)."""
    _require_server()
    session = _find_notebook(notebook)
    log_path = Path(session["log_path"])
    notebook_name = Path(session["notebook_path"]).name

    click.echo(f"Attached to {notebook_name} (Ctrl+C kills kernel, ESC exits)")
    log_stream.attach(log_path, lambda: jupyter.kill_kernel(notebook_name), lines=lines)


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
