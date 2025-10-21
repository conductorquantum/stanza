"""Stanza session management for Stanza CLI.

This module manages the top-level Stanza session (the timestamped directory).
This is distinct from LoggerSession which handles per-routine data logging.

Architecture:
    StanzaSession (this module):
        - Creates timestamped directories like: 20251020100010_untitled/
        - Manages session-level metadata and active session tracking
        - Scope: Global/session-wide, persists across routine runs

    LoggerSession (stanza/logger/session.py):
        - Handles per-routine data collection and buffered writing
        - Scope: Per-routine execution, created/destroyed per run
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class StanzaSession:
    """Manages the active Stanza session directory.

    This class handles the top-level timestamped session directory
    (e.g., 20251020100010_untitled/) and tracks which session is currently active.
    """

    CONFIG_DIR = ".stanza"
    CONFIG_FILE = "config.json"

    @staticmethod
    def create_session_directory(
        base_path: Path | str | None = None,
        name: str | None = None,
    ) -> Path:
        """Create a new timestamped Stanza session directory.

        Args:
            base_path: Base directory for session (default: current directory)
            name: Optional suffix for directory name (default: "untitled")

        Returns:
            Path to created session directory

        Example:
            >>> StanzaSession.create_session_directory()
            PosixPath('20251020100010_untitled')

            >>> StanzaSession.create_session_directory(name="experiment")
            PosixPath('20251020100010_experiment')
        """
        if base_path is None:
            base_path = Path.cwd()
        else:
            base_path = Path(base_path)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        suffix = name if name else "untitled"
        dir_name = f"{timestamp}_{suffix}"
        session_dir = base_path / dir_name

        session_dir.mkdir(parents=True, exist_ok=False)
        config_dir = session_dir / StanzaSession.CONFIG_DIR
        config_dir.mkdir(exist_ok=True)

        metadata = {
            "created_at": time.time(),
            "timestamp": timestamp,
            "directory": str(session_dir),
            "name": suffix,
        }

        config_file = config_dir / StanzaSession.CONFIG_FILE
        with open(config_file, "w") as f:
            json.dump(metadata, f, indent=2)

        notebook_suffix = "untitled_notebook" if name is None else suffix
        notebook_name = f"{timestamp}_{notebook_suffix}.ipynb"
        notebook_path = session_dir / notebook_name
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {' '.join(word.capitalize() for word in notebook_suffix.split('_'))}\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "from stanza.utils import load_device_config\n",
                        "from stanza.routines import RoutineRunner",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        with open(notebook_path, "w") as f:
            json.dump(notebook_content, f, indent=1)

        return session_dir

    @staticmethod
    def get_active_session() -> Path | None:
        """Get the active Stanza session directory from working directory config.

        Returns:
            Path to active session directory, or None if not set

        Example:
            >>> StanzaSession.get_active_session()
            PosixPath('20251020100010_untitled')
        """
        config_file = Path.cwd() / StanzaSession.CONFIG_DIR / "active_session.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file) as f:
                data = json.load(f)
                session_path = Path(data["session_directory"])

                if session_path.exists():
                    return session_path

        except (json.JSONDecodeError, KeyError, OSError):
            pass

        return None

    @staticmethod
    def set_active_session(session_dir: Path | str) -> None:
        """Set the active Stanza session directory.

        Args:
            session_dir: Path to session directory

        Example:
            >>> StanzaSession.set_active_session("20251020100010_untitled")
        """
        session_dir = Path(session_dir)

        if not session_dir.exists():
            raise ValueError(f"Session directory does not exist: {session_dir}")

        config_dir = Path.cwd() / StanzaSession.CONFIG_DIR
        config_dir.mkdir(exist_ok=True)

        config_file = config_dir / "active_session.json"
        data = {
            "session_directory": str(session_dir.resolve()),
            "set_at": time.time(),
        }

        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def get_session_metadata(session_dir: Path | str) -> dict[str, Any] | None:
        """Get metadata for a Stanza session directory.

        Args:
            session_dir: Path to session directory

        Returns:
            Metadata dictionary or None if not found
        """
        session_dir = Path(session_dir)
        config_file = session_dir / StanzaSession.CONFIG_DIR / StanzaSession.CONFIG_FILE

        if not config_file.exists():
            return None

        try:
            with open(config_file) as f:
                data: dict[str, Any] = json.load(f)
                return data
        except (json.JSONDecodeError, OSError):
            return None
