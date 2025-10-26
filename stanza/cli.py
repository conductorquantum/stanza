"""Stanza CLI - Command-line interface for Stanza experiment framework."""

from __future__ import annotations

import json
from datetime import datetime
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


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
