"""Stanza CLI - Command-line interface for Stanza experiment framework."""

from __future__ import annotations

from pathlib import Path

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


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
