"""Stanza CLI - Command-line interface for Stanza experiment framework."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import click

from stanza import __version__, jupyter
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
def jupyter_start(notebook_dir: Path) -> None:
    """Start Jupyter notebook server in background.

    Args:
        notebook_dir: Directory to serve notebooks from (default: current directory)

    Examples:

        stanza jupyter start

        stanza jupyter start /path/to/notebooks
    """
    try:
        notebook_dir = notebook_dir.resolve()
        click.echo(f"Starting Jupyter server in {notebook_dir}...")

        state = jupyter.start(notebook_dir)

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

        # Open browser with full URL including token
        webbrowser.open(state["url"])
        click.echo(f"✓ Opened {state['url']}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
