import tempfile
from pathlib import Path

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
        session.flush()

        assert len(session._buffer) == 0

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
