import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from stanza.exceptions import LoggerSessionError
from stanza.logger.datatypes import SessionMetadata
from stanza.logger.session import LoggerSession
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


class TestLoggerSession:
    def test_initializes_and_finalizes_session(self, logger_session):
        logger_session.initialize()
        assert logger_session._active is True

        logger_session.finalize()
        assert logger_session._active is False

    def test_logs_measurement_and_flushes_to_writer(self, tmpdir_path, logger_session):
        logger_session.initialize()
        logger_session.log_measurement(
            name="voltage",
            data={"value": 1.5},
            metadata={"device": "test"},
        )
        logger_session.finalize()

        measurement_file = tmpdir_path / "measurement.jsonl"
        assert measurement_file.exists()

    def test_logs_sweep_data(self, tmpdir_path, logger_session):
        logger_session.initialize()
        logger_session.log_sweep(
            name="iv_sweep",
            x_data=np.array([0.0, 1.0, 2.0]),
            y_data=np.array([0.0, 1e-6, 2e-6]),
            x_label="Voltage",
            y_label="Current",
        )
        logger_session.finalize()

        sweep_file = tmpdir_path / "sweep.jsonl"
        assert sweep_file.exists()

    def test_logs_analysis_to_separate_file(self, tmpdir_path, logger_session):
        logger_session.initialize()
        logger_session.log_analysis(
            name="fit_result",
            data={"slope": 1.5, "intercept": 0.1},
            metadata={"algorithm": "linear_fit"},
        )
        logger_session.finalize()

        analysis_file = tmpdir_path / "analysis.jsonl"
        assert analysis_file.exists()

    def test_auto_flushes_when_buffer_full(self, tmpdir_path, session_metadata):
        writer = JSONLWriter(tmpdir_path)
        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"jsonl": writer},
            writer_refs=["jsonl"],
            base_dir=tmpdir_path,
            buffer_size=3,
        )

        session.initialize()

        for i in range(5):
            session.log_measurement(name=f"m{i}", data={"value": i})

        measurement_file = tmpdir_path / "measurement.jsonl"
        assert measurement_file.exists()

        with open(measurement_file) as f:
            lines = f.readlines()
            assert len(lines) >= 3

        session.finalize()

    def test_updates_session_parameters(self, logger_session):
        logger_session.metadata.parameters = {"initial": "value"}
        logger_session.initialize()
        logger_session.log_parameters({"new_param": 42, "another": "test"})

        assert logger_session.metadata.parameters["initial"] == "value"
        assert logger_session.metadata.parameters["new_param"] == 42
        assert logger_session.metadata.parameters["another"] == "test"

        logger_session.finalize()

    def test_initializes_parameters_when_none(self, logger_session):
        logger_session.metadata.parameters = None
        logger_session.initialize()
        logger_session.log_parameters({"param": "value"})

        assert logger_session.metadata.parameters == {"param": "value"}
        logger_session.finalize()

    def test_rejects_operations_before_initialization(self, logger_session):
        with pytest.raises(LoggerSessionError, match="not initialized"):
            logger_session.log_measurement("test", {"value": 1})

        with pytest.raises(LoggerSessionError, match="not initialized"):
            logger_session.log_sweep("test", np.array([1.0]), np.array([2.0]), "X", "Y")

        with pytest.raises(LoggerSessionError, match="not initialized"):
            logger_session.log_parameters({"param": "value"})

    def test_rejects_double_initialization(self, logger_session):
        logger_session.initialize()

        with pytest.raises(LoggerSessionError, match="already initialized"):
            logger_session.initialize()

        logger_session.finalize()

    def test_context_manager_auto_initializes_and_finalizes(self, logger_session):
        with logger_session:
            assert logger_session._active is True
            logger_session.log_measurement("test", {"value": 1})

        assert logger_session._active is False

    def test_validates_empty_measurement_name(self, logger_session):
        logger_session.initialize()

        with pytest.raises(LoggerSessionError, match="cannot be empty"):
            logger_session.log_measurement("", {"value": 1})

        with pytest.raises(LoggerSessionError, match="cannot be empty"):
            logger_session.log_measurement("  ", {"value": 1})

        logger_session.finalize()

    def test_validates_empty_sweep_data(self, logger_session):
        logger_session.initialize()

        with pytest.raises(LoggerSessionError, match="cannot be empty"):
            logger_session.log_sweep("test", np.array([]), np.array([]), "X", "Y")

        logger_session.finalize()

    def test_rejects_nonexistent_base_directory(self, session_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "nonexistent"
            writer = JSONLWriter(tmpdir)

            with pytest.raises(LoggerSessionError, match="does not exist"):
                LoggerSession(
                    metadata=session_metadata,
                    writer_pool={"jsonl": writer},
                    writer_refs=["jsonl"],
                    base_dir=base_dir,
                )

    def test_writes_to_multiple_writers(self, tmpdir_path, session_metadata):
        writer1_dir = tmpdir_path / "writer1"
        writer2_dir = tmpdir_path / "writer2"
        writer1_dir.mkdir()
        writer2_dir.mkdir()

        writer1 = JSONLWriter(writer1_dir)
        writer2 = JSONLWriter(writer2_dir)

        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"jsonl1": writer1, "jsonl2": writer2},
            writer_refs=["jsonl1", "jsonl2"],
            base_dir=tmpdir_path,
        )

        session.initialize()
        session.log_measurement("test", {"value": 1})
        session.finalize()

        assert (writer1_dir / "measurement.jsonl").exists()
        assert (writer2_dir / "measurement.jsonl").exists()

    def test_initialize_session_error_propagates(self, tmpdir_path, session_metadata):
        class FailingWriter(JSONLWriter):
            def initialize_session(self, session):
                raise RuntimeError("Simulated writer failure")

        writer = FailingWriter(tmpdir_path)
        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"failing": writer},
            writer_refs=["failing"],
            base_dir=tmpdir_path,
        )

        with pytest.raises(LoggerSessionError, match="Failed to initialize"):
            session.initialize()

        assert session._active is False

    def test_finalize_session_error_sets_inactive(self, tmpdir_path, session_metadata):
        class FailingWriter(JSONLWriter):
            def finalize_session(self, session=None):
                raise RuntimeError("Simulated finalization failure")

        writer = FailingWriter(tmpdir_path)
        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"failing": writer},
            writer_refs=["failing"],
            base_dir=tmpdir_path,
        )

        session._writer_pool = {"failing": writer}
        session._active = True

        with pytest.raises(LoggerSessionError, match="Failed to finalize"):
            session.finalize()

        assert session._active is False

    def test_flush_handles_write_errors(self, tmpdir_path, session_metadata):
        class FailingWriter(JSONLWriter):
            def write_measurement(self, data):
                raise RuntimeError("Write failure")

            def flush(self):
                raise RuntimeError("Flush failure")

        writer = FailingWriter(tmpdir_path)
        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"failing": writer},
            writer_refs=["failing"],
            base_dir=tmpdir_path,
            buffer_size=10,
        )

        session._writer_pool = {"failing": writer}
        session._active = True

        session.log_measurement("test", {"value": 1})

        # Flush should raise an exception when writers fail
        with pytest.raises(LoggerSessionError, match="Flush failed"):
            session.flush()

        # Buffer should NOT be cleared when flush fails
        assert len(session._buffer) == 1

    def test_finalize_when_not_active_raises_error(self, logger_session):
        with pytest.raises(LoggerSessionError, match="not initialized"):
            logger_session.finalize()

    def test_context_manager_with_exception(self, tmpdir_path, session_metadata):
        class FailingWriter(JSONLWriter):
            def finalize_session(self, session=None):
                raise RuntimeError("Finalization failure")

        writer = FailingWriter(tmpdir_path)
        session = LoggerSession(
            metadata=session_metadata,
            writer_pool={"failing": writer},
            writer_refs=["failing"],
            base_dir=tmpdir_path,
        )

        with pytest.raises(LoggerSessionError):
            with session:
                pass

    def test_validates_empty_measurement_data(self, logger_session):
        logger_session.initialize()

        with pytest.raises(LoggerSessionError, match="cannot be empty"):
            logger_session.log_measurement("test", {})

        logger_session.finalize()

    def test_validates_empty_parameters(self, logger_session):
        logger_session.initialize()

        with pytest.raises(LoggerSessionError, match="cannot be empty"):
            logger_session.log_parameters({})

        logger_session.finalize()

    def test_routine_name_property_returns_empty_when_none(self, tmpdir_path):
        metadata = SessionMetadata(
            session_id="test_session",
            start_time=100.0,
            user="test_user",
            routine_name=None,
        )
        writer = JSONLWriter(tmpdir_path)
        session = LoggerSession(
            metadata=metadata,
            writer_pool={"jsonl": writer},
            writer_refs=["jsonl"],
            base_dir=tmpdir_path,
        )

        assert session.routine_name == ""


def test_time_based_auto_flush_triggers_on_measurement(tmpdir_path, session_metadata):
    """Test that auto-flush triggers based on elapsed time during log_measurement."""
    writer = JSONLWriter(tmpdir_path)
    session = LoggerSession(
        metadata=session_metadata,
        writer_pool={"jsonl": writer},
        writer_refs=["jsonl"],
        base_dir=tmpdir_path,
        auto_flush_interval=30.0,
        buffer_size=100,  # Large buffer to prevent size-based flush
    )

    with patch("time.time") as mock_time:
        # Set initial time
        mock_time.return_value = 1000.0
        session.initialize()

        # Log first measurement at t=1000
        session.log_measurement("m1", {"value": 1})
        assert len(session._buffer) == 1

        # Advance time past auto-flush interval (30s + 1s)
        mock_time.return_value = 1031.0

        # Log second measurement - should trigger time-based flush of both items
        session.log_measurement("m2", {"value": 2})

        # Buffer should be empty after time-based flush
        assert len(session._buffer) == 0

        # Verify data was written to file
        measurement_file = tmpdir_path / "measurement.jsonl"
        assert measurement_file.exists()

        session.finalize()


def test_time_based_auto_flush_triggers_on_sweep(tmpdir_path, session_metadata):
    """Test that auto-flush triggers based on elapsed time during log_sweep."""
    writer = JSONLWriter(tmpdir_path)
    session = LoggerSession(
        metadata=session_metadata,
        writer_pool={"jsonl": writer},
        writer_refs=["jsonl"],
        base_dir=tmpdir_path,
        auto_flush_interval=30.0,
        buffer_size=100,  # Large buffer to prevent size-based flush
    )

    with patch("time.time") as mock_time:
        # Set initial time
        mock_time.return_value = 1000.0
        session.initialize()

        # Log first sweep at t=1000
        session.log_sweep("s1", [1.0, 2.0], [3.0, 4.0], "X", "Y")
        assert len(session._buffer) == 1

        # Advance time past auto-flush interval (30s + 1s)
        mock_time.return_value = 1031.0

        # Log second sweep - should trigger time-based flush of both items
        session.log_sweep("s2", [5.0, 6.0], [7.0, 8.0], "X", "Y")

        # Buffer should be empty after time-based flush
        assert len(session._buffer) == 0

        # Verify data was written to file
        sweep_file = tmpdir_path / "sweep.jsonl"
        assert sweep_file.exists()

        session.finalize()


def test_no_auto_flush_within_interval(tmpdir_path, session_metadata):
    """Test that multiple measurements within interval don't trigger auto-flush."""
    writer = JSONLWriter(tmpdir_path)
    session = LoggerSession(
        metadata=session_metadata,
        writer_pool={"jsonl": writer},
        writer_refs=["jsonl"],
        base_dir=tmpdir_path,
        auto_flush_interval=30.0,
        buffer_size=100,  # Large buffer to prevent size-based flush
    )

    with patch("time.time") as mock_time:
        # Set initial time
        mock_time.return_value = 1000.0
        session.initialize()

        # Log multiple measurements with small time increments (within interval)
        for i in range(5):
            mock_time.return_value = 1000.0 + i * 2.0  # Increment by 2s each time
            session.log_measurement(f"m{i}", {"value": i})

        # All measurements should still be in buffer (total elapsed: 8s < 30s)
        assert len(session._buffer) == 5

        session.finalize()


def test_buffer_size_warning(tmpdir_path, session_metadata, caplog):
    """Test that buffer size warning is logged when buffer grows too large."""
    import logging

    # Use a failing writer so flush() fails and buffer grows
    class FailingWriter(JSONLWriter):
        def write_measurement(self, data):
            raise RuntimeError("Simulated failure")

    writer = FailingWriter(tmpdir_path)
    session = LoggerSession(
        metadata=session_metadata,
        writer_pool={"jsonl": writer},
        writer_refs=["jsonl"],
        base_dir=tmpdir_path,
        buffer_size=10,  # Small buffer, warning at 100
        auto_flush_interval=None,  # Disable auto-flush
    )

    session.initialize()

    # Add items - flush will fail every 10 items, so buffer will grow
    with caplog.at_level(logging.WARNING):
        for i in range(101):
            try:
                session.log_measurement(f"m{i}", {"value": i})
            except LoggerSessionError:
                # Flush will fail, that's expected
                pass

        # Check that warning was logged
        assert any(
            "Buffer size critical" in record.message for record in caplog.records
        )

    # Verify warning flag was set
    assert session._buffer_size_warned is True

    session._active = False  # Prevent finalize from trying to flush


def test_auto_flush_disabled_when_none(tmpdir_path, session_metadata):
    """Test that auto-flush is disabled when interval is None."""
    writer = JSONLWriter(tmpdir_path)
    session = LoggerSession(
        metadata=session_metadata,
        writer_pool={"jsonl": writer},
        writer_refs=["jsonl"],
        base_dir=tmpdir_path,
        auto_flush_interval=None,  # Disabled
        buffer_size=10,
    )

    with patch("time.time") as mock_time:
        # Set initial time
        mock_time.return_value = 1000.0
        session.initialize()

        # Log multiple measurements
        for i in range(5):
            mock_time.return_value = 1000.0 + i * 10.0
            session.log_measurement(f"m{i}", {"value": i})

        # Advance time significantly
        mock_time.return_value = 2000.0

        # Log another measurement
        session.log_measurement("m5", {"value": 5})

        # All measurements should still be in buffer (no time-based flush)
        assert len(session._buffer) == 6

        session.finalize()


def test_last_flush_time_updated_on_flush(tmpdir_path, session_metadata):
    """Test that _last_flush_time is updated after successful flush."""
    writer = JSONLWriter(tmpdir_path)
    session = LoggerSession(
        metadata=session_metadata,
        writer_pool={"jsonl": writer},
        writer_refs=["jsonl"],
        base_dir=tmpdir_path,
        auto_flush_interval=30.0,
        buffer_size=100,
    )

    with patch("time.time") as mock_time:
        # Set initial time
        mock_time.return_value = 1000.0
        session.initialize()
        initial_flush_time = session._last_flush_time

        # Advance time and log a measurement
        mock_time.return_value = 1010.0
        session.log_measurement("test", {"value": 1})

        # Advance time and manually flush
        mock_time.return_value = 1020.0
        session.flush()

        # _last_flush_time should be updated to 1020.0
        assert session._last_flush_time == 1020.0
        assert session._last_flush_time > initial_flush_time

        session.finalize()
