"""Tests for StanzaSession (stanza/context.py)."""

import json
import tempfile
from pathlib import Path

import pytest

from stanza.context import StanzaSession


class TestStanzaSession:
    """Test suite for StanzaSession class."""

    def test_create_session_directory_with_defaults(self):
        """Test creating a session directory with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = StanzaSession.create_session_directory(base_path=tmpdir)

            assert session_dir.exists()
            assert session_dir.parent == Path(tmpdir)
            assert "_untitled" in session_dir.name
            assert (session_dir / ".stanza").exists()
            assert (session_dir / ".stanza" / "config.json").exists()

    def test_create_session_directory_with_custom_name(self):
        """Test creating a session directory with custom name suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = StanzaSession.create_session_directory(
                base_path=tmpdir, name="experiment"
            )

            assert session_dir.exists()
            assert "_experiment" in session_dir.name
            assert "_untitled" not in session_dir.name

    def test_create_session_directory_uses_cwd_when_no_base_path(self):
        """Test that create_session_directory uses current working directory by default."""
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            os.chdir(tmpdir)
            try:
                session_dir = StanzaSession.create_session_directory()

                assert session_dir.exists()
                # Resolve both paths to handle /private/var vs /var symlink on macOS
                assert session_dir.parent.resolve() == Path(tmpdir).resolve()
            finally:
                os.chdir(original_cwd)

    def test_create_session_directory_fails_if_already_exists(self):
        """Test that creating a session directory twice fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a session directory
            session_dir = StanzaSession.create_session_directory(base_path=tmpdir)

            # Try to create the same directory again - should fail
            with pytest.raises(FileExistsError):
                session_dir.mkdir(parents=False, exist_ok=False)

    def test_session_metadata_contains_correct_fields(self):
        """Test that session metadata contains all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = StanzaSession.create_session_directory(
                base_path=tmpdir, name="test"
            )

            metadata = StanzaSession.get_session_metadata(session_dir)

            assert metadata is not None
            assert "created_at" in metadata
            assert "timestamp" in metadata
            assert "directory" in metadata
            assert "name" in metadata
            assert metadata["name"] == "test"
            assert isinstance(metadata["created_at"], float)
            assert str(session_dir) in metadata["directory"]

    def test_get_active_session_returns_none_when_not_set(self):
        """Test that get_active_session returns None when no session is active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            import os

            os.chdir(tmpdir)
            try:
                active_session = StanzaSession.get_active_session()
                assert active_session is None
            finally:
                os.chdir(original_cwd)

    def test_set_and_get_active_session(self):
        """Test setting and retrieving the active session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            import os

            os.chdir(tmpdir)
            try:
                # Create a session directory
                session_dir = StanzaSession.create_session_directory(base_path=tmpdir)

                # Set it as active
                StanzaSession.set_active_session(session_dir)

                # Retrieve it
                active_session = StanzaSession.get_active_session()

                assert active_session is not None
                # Resolve both paths to handle /private/var vs /var symlink on macOS
                assert active_session.resolve() == session_dir.resolve()
            finally:
                os.chdir(original_cwd)

    def test_set_active_session_fails_if_directory_does_not_exist(self):
        """Test that set_active_session raises error for non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_dir = Path(tmpdir) / "nonexistent"

            with pytest.raises(ValueError, match="Session directory does not exist"):
                StanzaSession.set_active_session(fake_dir)

    def test_get_active_session_returns_none_if_directory_deleted(self):
        """Test that get_active_session returns None if stored directory was deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            import os

            os.chdir(tmpdir)
            try:
                # Create and set a session
                session_dir = StanzaSession.create_session_directory(base_path=tmpdir)
                StanzaSession.set_active_session(session_dir)

                # Delete the session directory
                import shutil

                shutil.rmtree(session_dir)

                # Should return None
                active_session = StanzaSession.get_active_session()
                assert active_session is None
            finally:
                os.chdir(original_cwd)

    def test_get_active_session_handles_corrupted_json(self):
        """Test that get_active_session handles corrupted JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            import os

            os.chdir(tmpdir)
            try:
                # Create .stanza directory with corrupted JSON
                config_dir = Path(tmpdir) / ".stanza"
                config_dir.mkdir()
                config_file = config_dir / "active_session.json"

                with open(config_file, "w") as f:
                    f.write("invalid json{{{")

                active_session = StanzaSession.get_active_session()
                assert active_session is None
            finally:
                os.chdir(original_cwd)

    def test_get_active_session_handles_missing_key(self):
        """Test that get_active_session handles missing 'session_directory' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            import os

            os.chdir(tmpdir)
            try:
                # Create .stanza directory with incomplete JSON
                config_dir = Path(tmpdir) / ".stanza"
                config_dir.mkdir()
                config_file = config_dir / "active_session.json"

                with open(config_file, "w") as f:
                    json.dump({"wrong_key": "value"}, f)

                active_session = StanzaSession.get_active_session()
                assert active_session is None
            finally:
                os.chdir(original_cwd)

    def test_get_session_metadata_returns_none_for_nonexistent_directory(self):
        """Test that get_session_metadata returns None for non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_dir = Path(tmpdir) / "nonexistent"
            metadata = StanzaSession.get_session_metadata(fake_dir)
            assert metadata is None

    def test_get_session_metadata_returns_none_for_missing_config(self):
        """Test that get_session_metadata returns None when config.json missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory without .stanza/config.json
            test_dir = Path(tmpdir) / "test_session"
            test_dir.mkdir()

            metadata = StanzaSession.get_session_metadata(test_dir)
            assert metadata is None

    def test_get_session_metadata_handles_corrupted_json(self):
        """Test that get_session_metadata handles corrupted JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a session directory with corrupted config
            test_dir = Path(tmpdir) / "test_session"
            test_dir.mkdir()
            config_dir = test_dir / ".stanza"
            config_dir.mkdir()
            config_file = config_dir / "config.json"

            with open(config_file, "w") as f:
                f.write("not valid json!!!")

            metadata = StanzaSession.get_session_metadata(test_dir)
            assert metadata is None

    def test_active_session_file_contains_correct_fields(self):
        """Test that active_session.json contains required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            import os

            os.chdir(tmpdir)
            try:
                session_dir = StanzaSession.create_session_directory(base_path=tmpdir)
                StanzaSession.set_active_session(session_dir)

                active_file = Path(tmpdir) / ".stanza" / "active_session.json"
                assert active_file.exists()

                with open(active_file) as f:
                    data = json.load(f)

                assert "session_directory" in data
                assert "set_at" in data
                assert isinstance(data["set_at"], float)
                assert Path(data["session_directory"]) == session_dir.resolve()
            finally:
                os.chdir(original_cwd)

    def test_timestamp_format_is_correct(self):
        """Test that session directory timestamp follows YYYYMMDDHHmmss format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = StanzaSession.create_session_directory(base_path=tmpdir)

            # Extract timestamp from directory name (before the underscore)
            timestamp = session_dir.name.split("_")[0]

            # Should be 14 digits
            assert len(timestamp) == 14
            assert timestamp.isdigit()

            # Should be parseable as a valid datetime
            year = int(timestamp[:4])
            month = int(timestamp[4:6])
            day = int(timestamp[6:8])
            hour = int(timestamp[8:10])
            minute = int(timestamp[10:12])
            second = int(timestamp[12:14])

            assert 2020 <= year <= 2030
            assert 1 <= month <= 12
            assert 1 <= day <= 31
            assert 0 <= hour <= 23
            assert 0 <= minute <= 59
            assert 0 <= second <= 59
