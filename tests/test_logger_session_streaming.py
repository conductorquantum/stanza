"""Tests for streaming sweep functionality in LoggerSession.

This module tests the SweepContext API for streaming sweeps with live plotting,
including context manager behavior, data validation, persistence, and integration
with various writers.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from stanza.exceptions import LoggerSessionError
from stanza.logger.datatypes import SessionMetadata
from stanza.logger.session import LoggerSession
from stanza.logger.writers.bokeh_writer import BokehLiveWriter
from stanza.logger.writers.hdf5_writer import HDF5Writer
from stanza.logger.writers.jsonl_writer import JSONLWriter


@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def session_metadata():
    return SessionMetadata(
        session_id="test_session",
        start_time=100.0,
        user="test_user",
    )


@pytest.fixture
def logger_session(tmpdir_path, session_metadata):
    writer = JSONLWriter(tmpdir_path)
    return LoggerSession(
        metadata=session_metadata,
        writer_pool={"jsonl": writer},
        writer_refs=["jsonl"],
        base_dir=tmpdir_path,
    )


class TestSweepContext:
    """Unit tests for SweepContext (streaming sweep handle)."""

    def test_accumulates_data_across_multiple_appends(self, logger_session):
        """Test that multiple append calls accumulate data correctly."""
        logger_session.initialize()

        with logger_session.sweep("signal", "Time", "Amplitude") as sweep:
            sweep.append([0.0, 1.0], [0.5, 0.6])
            sweep.append([2.0], [0.7])
            sweep.append([3.0, 4.0, 5.0], [0.8, 0.9, 1.0])

        assert not sweep._active
        assert len(sweep._x_data) == 6
        assert len(sweep._y_data) == 6

        logger_session.finalize()

    def test_rejects_empty_arrays(self, logger_session):
        """Test that append rejects empty x or y arrays."""
        logger_session.initialize()
        sweep = logger_session.sweep("test", "X", "Y")

        with pytest.raises(ValueError, match="cannot be empty"):
            sweep.append([], [1.0])

        with pytest.raises(ValueError, match="cannot be empty"):
            sweep.append([1.0], [])

        sweep.cancel()
        logger_session.finalize()

    def test_rejects_length_mismatch(self, logger_session):
        """Test that append rejects x and y arrays with different lengths for 1D sweeps."""
        logger_session.initialize()
        sweep = logger_session.sweep("test", "X", "Y")

        # First establish it's a 1D sweep by appending matching lengths
        sweep.append([1.0], [2.0])

        # Now try mismatched lengths - should fail
        with pytest.raises(ValueError, match="Length mismatch"):
            sweep.append([1.0, 2.0], [3.0])

        sweep.cancel()
        logger_session.finalize()

    def test_enforces_1d_arrays_only(self, logger_session):
        """Test that multidimensional numpy arrays are rejected."""
        logger_session.initialize()
        sweep = logger_session.sweep("test", "X", "Y")

        with pytest.raises(ValueError, match="must be 1D arrays"):
            sweep.append(np.array([[1, 2], [3, 4]]), [1.0, 2.0])

        with pytest.raises(ValueError, match="must be 1D arrays"):
            sweep.append([1.0, 2.0], np.array([[1, 2], [3, 4]]))

        sweep.cancel()
        logger_session.finalize()

    def test_cancel_prevents_persistence(self, tmpdir_path, logger_session):
        """Test that cancel() prevents data from being written to file."""
        logger_session.initialize()

        sweep = logger_session.sweep("test", "X", "Y")
        sweep.append([1.0, 2.0], [3.0, 4.0])
        sweep.cancel()

        assert not sweep._active
        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert not sweep_file.exists() or sweep_file.stat().st_size == 0

    def test_end_writes_to_file(self, tmpdir_path, logger_session):
        """Test that end() writes accumulated data to file."""
        logger_session.initialize()

        sweep = logger_session.sweep("test_sweep", "X", "Y")
        sweep.append([1.0, 2.0], [3.0, 4.0])
        sweep.end()

        assert not sweep._active
        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert sweep_file.exists()
        assert sweep_file.stat().st_size > 0

    def test_context_exit_normal_calls_end(self, tmpdir_path, logger_session):
        """Test that normal context exit calls end()."""
        logger_session.initialize()

        with logger_session.sweep("signal", "Time", "Amplitude") as sweep:
            sweep.append([1.0], [2.0])

        assert not sweep._active
        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert sweep_file.exists()

    def test_context_exit_with_exception_calls_cancel(
        self, tmpdir_path, logger_session
    ):
        """Test that exception in context calls cancel() and doesn't persist."""
        logger_session.initialize()

        with pytest.raises(RuntimeError):
            with logger_session.sweep("signal", "Time", "Amplitude") as sweep:
                sweep.append([1.0], [2.0])
                raise RuntimeError("Test exception")

        assert not sweep._active
        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert not sweep_file.exists() or sweep_file.stat().st_size == 0

    def test_append_after_end_raises_error(self, logger_session):
        """Test that append after end() raises error."""
        logger_session.initialize()

        sweep = logger_session.sweep("test", "X", "Y")
        sweep.append([1.0], [2.0])
        sweep.end()

        with pytest.raises(ValueError, match="no longer active"):
            sweep.append([3.0], [4.0])

        logger_session.finalize()

    def test_append_after_cancel_raises_error(self, logger_session):
        """Test that append after cancel() raises error."""
        logger_session.initialize()

        sweep = logger_session.sweep("test", "X", "Y")
        sweep.append([1.0], [2.0])
        sweep.cancel()

        with pytest.raises(ValueError, match="no longer active"):
            sweep.append([3.0], [4.0])

        logger_session.finalize()

    def test_end_with_no_data_does_nothing(self, logger_session, caplog):
        """Test that end() on empty sweep logs debug and doesn't write."""
        logger_session.initialize()

        with caplog.at_level(logging.DEBUG):
            sweep = logger_session.sweep("empty", "X", "Y")
            sweep.end()

        assert any(
            "empty, skipping write" in record.message for record in caplog.records
        )
        logger_session.finalize()

    def test_double_end_is_safe(self, logger_session):
        """Test that calling end() multiple times is safe (idempotent)."""
        logger_session.initialize()

        sweep = logger_session.sweep("test", "X", "Y")
        sweep.append([1.0], [2.0])
        sweep.end()
        sweep.end()
        sweep.end()

        logger_session.finalize()

    def test_converts_lists_to_numpy_arrays(self, logger_session):
        """Test that append accepts and converts lists to numpy arrays."""
        logger_session.initialize()

        sweep = logger_session.sweep("test", "X", "Y")
        sweep.append([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

        assert len(sweep._x_data) == 3
        assert len(sweep._y_data) == 3

        sweep.end()
        logger_session.finalize()

    def test_metadata_is_copied_defensively(self, logger_session):
        """Test that metadata is copied to prevent external mutation."""
        logger_session.initialize()

        original_metadata = {"key": "value"}
        sweep = logger_session.sweep("test", "X", "Y", metadata=original_metadata)

        original_metadata["key"] = "changed"
        original_metadata["new_key"] = "new_value"

        assert sweep.metadata == {"key": "value"}

        sweep.cancel()
        logger_session.finalize()


class TestStreamingSweepIntegration:
    """Integration tests for streaming sweep functionality with session."""

    def test_session_sweep_creates_context(self, logger_session):
        """Test that session.sweep() creates and registers a SweepContext."""
        logger_session.initialize()

        sweep = logger_session.sweep("signal", "Time", "Amplitude")

        assert sweep.name == "signal"
        assert sweep.x_label == "Time"
        assert sweep.y_label == "Amplitude"
        assert "signal" in logger_session._active_sweeps

        sweep.cancel()
        logger_session.finalize()

    def test_sweep_creates_multiple_contexts(self, logger_session):
        """Test that sweep() can create multiple independent contexts."""
        logger_session.initialize()

        sweep1 = logger_session.sweep("test1", "X", "Y")
        sweep2 = logger_session.sweep("test2", "X", "Y")

        assert type(sweep1) is type(sweep2)
        assert "test1" in logger_session._active_sweeps
        assert "test2" in logger_session._active_sweeps

        sweep1.cancel()
        sweep2.cancel()
        logger_session.finalize()

    def test_rejects_duplicate_active_sweep_name(self, logger_session):
        """Test that starting a sweep with an active name raises error."""
        logger_session.initialize()

        sweep1 = logger_session.sweep("duplicate", "X", "Y")

        with pytest.raises(LoggerSessionError, match="already active"):
            logger_session.sweep("duplicate", "X", "Y")

        sweep1.end()
        sweep2 = logger_session.sweep("duplicate", "X", "Y")
        sweep2.cancel()

        logger_session.finalize()

    def test_sweep_before_initialize_raises_error(self, logger_session):
        """Test that creating sweep before initialization raises error."""
        with pytest.raises(LoggerSessionError, match="not initialized"):
            logger_session.sweep("test", "X", "Y")

    def test_live_streaming_calls_bokeh_writer(self, tmpdir_path, session_metadata):
        """Test that append() streams to BokehLiveWriter on each call."""
        mock_bokeh = MagicMock(spec=BokehLiveWriter)
        jsonl_writer = JSONLWriter(tmpdir_path)

        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"bokeh": mock_bokeh, "jsonl": jsonl_writer},
            writer_refs=["bokeh", "jsonl"],
            base_dir=tmpdir_path,
        )

        session.initialize()

        with session.sweep("signal", "Time", "Amplitude") as sweep:
            sweep.append([1.0], [2.0])
            sweep.append([3.0], [4.0])

        assert mock_bokeh.write_sweep.call_count == 2

        session.finalize()

    def test_persistence_skips_bokeh_writer(self, tmpdir_path, session_metadata):
        """Test that end() writes to file writers but skips BokehLiveWriter."""
        mock_bokeh = MagicMock(spec=BokehLiveWriter)
        jsonl_writer = JSONLWriter(tmpdir_path)

        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"bokeh": mock_bokeh, "jsonl": jsonl_writer},
            writer_refs=["bokeh", "jsonl"],
            base_dir=tmpdir_path,
        )

        session.initialize()

        with session.sweep("signal", "Time", "Amplitude") as sweep:
            sweep.append([1.0, 2.0], [3.0, 4.0])

        assert mock_bokeh.write_sweep.call_count == 1

        session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert sweep_file.exists()
        assert sweep_file.stat().st_size > 0

    def test_writes_exactly_one_record_to_file(self, tmpdir_path, logger_session):
        """Test that streaming sweep writes exactly one record to file."""
        logger_session.initialize()

        with logger_session.sweep("multi_append", "X", "Y") as sweep:
            for i in range(10):
                sweep.append([float(i)], [float(i * 2)])

        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        with open(sweep_file) as f:
            lines = f.readlines()

        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["name"] == "multi_append"
        assert len(record["x_data"]) == 10
        assert len(record["y_data"]) == 10

    def test_finalize_auto_ends_active_sweeps(
        self, tmpdir_path, logger_session, caplog
    ):
        """Test that finalize() automatically ends active sweeps with warning."""
        logger_session.initialize()

        sweep1 = logger_session.sweep("active1", "X", "Y")
        sweep1.append([1.0], [2.0])

        sweep2 = logger_session.sweep("active2", "X", "Y")
        sweep2.append([3.0], [4.0])

        with caplog.at_level(logging.WARNING):
            logger_session.finalize()

        assert any("Auto-completing" in record.message for record in caplog.records)
        assert any("active1" in record.message for record in caplog.records)

        assert not sweep1._active
        assert not sweep2._active

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert sweep_file.exists()

    def test_coexistence_with_log_sweep(self, tmpdir_path, logger_session):
        """Test that streaming sweeps and log_sweep() can coexist without duplicates."""
        logger_session.initialize()

        logger_session.log_sweep(
            "complete_sweep", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], "X", "Y"
        )

        with logger_session.sweep("streaming_sweep", "X", "Y") as sweep:
            sweep.append([7.0, 8.0], [9.0, 10.0])

        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        with open(sweep_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

        records = [json.loads(line) for line in lines]
        names = {r["name"] for r in records}
        assert names == {"complete_sweep", "streaming_sweep"}

    def test_streaming_sweep_with_hdf5_writer(self, tmpdir_path, session_metadata):
        """Test that streaming sweeps work with HDF5Writer."""
        hdf5_writer = HDF5Writer(tmpdir_path)

        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"hdf5": hdf5_writer},
            writer_refs=["hdf5"],
            base_dir=tmpdir_path,
        )

        session.initialize()

        with session.sweep("hdf5_sweep", "Voltage", "Current") as sweep:
            sweep.append([0.0, 0.1], [0.0, 1e-6])
            sweep.append([0.2, 0.3], [2e-6, 3e-6])

        session.finalize()

        hdf5_file = tmpdir_path / f"{session_metadata.session_id}.h5"
        assert hdf5_file.exists()

        with h5py.File(hdf5_file, "r") as f:
            assert "sweeps" in f
            assert "hdf5_sweep" in f["sweeps"]

    def test_multiple_writers_receive_complete_sweep(
        self, tmpdir_path, session_metadata
    ):
        """Test that all file writers receive the complete sweep on end()."""
        jsonl_dir = tmpdir_path / "jsonl"
        hdf5_dir = tmpdir_path / "hdf5"
        jsonl_dir.mkdir()
        hdf5_dir.mkdir()

        jsonl_writer = JSONLWriter(jsonl_dir)
        hdf5_writer = HDF5Writer(hdf5_dir)

        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"jsonl": jsonl_writer, "hdf5": hdf5_writer},
            writer_refs=["jsonl", "hdf5"],
            base_dir=tmpdir_path,
        )

        session.initialize()

        with session.sweep("multi_writer", "X", "Y") as sweep:
            sweep.append([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

        session.finalize()

        jsonl_file = jsonl_dir / "sweep.jsonl"
        assert jsonl_file.exists()

        hdf5_file = hdf5_dir / f"{session_metadata.session_id}.h5"
        assert hdf5_file.exists()

        with h5py.File(hdf5_file, "r") as f:
            assert "multi_writer" in f["sweeps"]

    def test_streaming_sweep_preserves_metadata(self, tmpdir_path, logger_session):
        """Test that user metadata is preserved in the persisted sweep."""
        logger_session.initialize()

        custom_metadata = {"temperature": 300, "device": "QD1", "notes": "test run"}

        with logger_session.sweep(
            "meta_sweep", "Voltage", "Current", metadata=custom_metadata
        ) as sweep:
            sweep.append([1.0], [2.0])

        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        with open(sweep_file) as f:
            record = json.loads(f.readline())

        assert record["metadata"] == custom_metadata

    def test_routine_name_attached_to_streaming_sweep(
        self, tmpdir_path, session_metadata
    ):
        """Test that routine name is attached to streaming sweep."""
        session_metadata.routine_name = "test_routine"
        jsonl_writer = JSONLWriter(tmpdir_path)

        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"jsonl": jsonl_writer},
            writer_refs=["jsonl"],
            base_dir=tmpdir_path,
        )

        session.initialize()

        with session.sweep("routine_sweep", "X", "Y") as sweep:
            sweep.append([1.0], [2.0])

        session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        with open(sweep_file) as f:
            record = json.loads(f.readline())

        assert record["routine_name"] == "test_routine"

    def test_live_writer_error_doesnt_stop_persistence(
        self, tmpdir_path, session_metadata, caplog
    ):
        """Test that live writer errors don't prevent file persistence."""
        mock_bokeh = MagicMock(spec=BokehLiveWriter)
        mock_bokeh.write_sweep.side_effect = RuntimeError("Bokeh plot failed")

        jsonl_writer = JSONLWriter(tmpdir_path)

        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"bokeh": mock_bokeh, "jsonl": jsonl_writer},
            writer_refs=["bokeh", "jsonl"],
            base_dir=tmpdir_path,
        )

        session.initialize()

        with caplog.at_level(logging.WARNING):
            with session.sweep("resilient", "X", "Y") as sweep:
                sweep.append([1.0], [2.0])

        assert any(
            "Live plot stream failed" in record.message for record in caplog.records
        )

        session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert sweep_file.exists()

    def test_file_writer_error_during_persistence_raises(
        self, tmpdir_path, session_metadata
    ):
        """Test that file writer errors during persistence raise LoggerSessionError."""

        class FailingWriter(JSONLWriter):
            def write_sweep(self, sweep):
                raise RuntimeError("File write failed")

        writer = FailingWriter(tmpdir_path)

        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"jsonl": writer},
            writer_refs=["jsonl"],
            base_dir=tmpdir_path,
        )

        session.initialize()

        with pytest.raises(LoggerSessionError, match="Sweep write failed"):
            with session.sweep("failing", "X", "Y") as sweep:
                sweep.append([1.0], [2.0])

        session._active = False  # Force inactive to allow cleanup

    def test_no_live_writers_doesnt_error(self, tmpdir_path, logger_session):
        """Test that streaming works when no live writers are configured."""
        logger_session.initialize()

        with logger_session.sweep("no_live", "X", "Y") as sweep:
            sweep.append([1.0, 2.0], [3.0, 4.0])

        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert sweep_file.exists()

    def test_session_repr_includes_active_sweeps(self, logger_session):
        """Test that session repr includes count of active sweeps."""
        logger_session.initialize()

        repr_before = repr(logger_session)
        assert "active_sweeps=0" in repr_before

        sweep1 = logger_session.sweep("s1", "X", "Y")
        sweep2 = logger_session.sweep("s2", "X", "Y")

        repr_during = repr(logger_session)
        assert "active_sweeps=2" in repr_during

        sweep1.end()
        sweep2.end()

        repr_after = repr(logger_session)
        assert "active_sweeps=0" in repr_after

        logger_session.finalize()
