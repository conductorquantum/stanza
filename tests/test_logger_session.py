import tempfile
from pathlib import Path

import numpy as np
import pytest

from stanza.exceptions import LoggerSessionError
from stanza.logger.datatypes import SessionMetadata
from stanza.logger.session import LoggerSession
from stanza.logger.writers.jsonl_writer import JSONLWriter


class TestLoggerSession:
    def test_initializes_and_finalizes_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()
            assert session._active is True

            session.finalize()
            assert session._active is False

    def test_logs_measurement_and_flushes_to_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()
            session.log_measurement(
                name="voltage",
                data={"value": 1.5},
                metadata={"device": "test"},
            )

            session.finalize()

            measurement_file = base_dir / "measurement.jsonl"
            assert measurement_file.exists()

    def test_logs_sweep_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()
            session.log_sweep(
                name="iv_sweep",
                x_data=np.array([0.0, 1.0, 2.0]),
                y_data=np.array([0.0, 1e-6, 2e-6]),
                x_label="Voltage",
                y_label="Current",
            )

            session.finalize()

            sweep_file = base_dir / "sweep.jsonl"
            assert sweep_file.exists()

    def test_logs_analysis_to_separate_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()
            session.log_analysis(
                name="fit_result",
                data={"slope": 1.5, "intercept": 0.1},
                metadata={"algorithm": "linear_fit"},
            )

            session.finalize()

            analysis_file = base_dir / "analysis.jsonl"
            assert analysis_file.exists()

    def test_auto_flushes_when_buffer_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
                buffer_size=3,
            )

            session.initialize()

            for i in range(5):
                session.log_measurement(
                    name=f"m{i}",
                    data={"value": i},
                )

            measurement_file = base_dir / "measurement.jsonl"
            assert measurement_file.exists()

            with open(measurement_file) as f:
                lines = f.readlines()
                assert len(lines) >= 3

            session.finalize()

    def test_updates_session_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
                parameters={"initial": "value"},
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()
            session.log_parameters({"new_param": 42, "another": "test"})

            assert session.metadata.parameters["initial"] == "value"
            assert session.metadata.parameters["new_param"] == 42
            assert session.metadata.parameters["another"] == "test"

            session.finalize()

    def test_initializes_parameters_when_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
                parameters=None,
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()
            session.log_parameters({"param": "value"})

            assert session.metadata.parameters == {"param": "value"}

            session.finalize()

    def test_rejects_operations_before_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            with pytest.raises(LoggerSessionError, match="not initialized"):
                session.log_measurement("test", {"value": 1})

            with pytest.raises(LoggerSessionError, match="not initialized"):
                session.log_sweep(
                    "test",
                    np.array([1.0]),
                    np.array([2.0]),
                    "X",
                    "Y",
                )

            with pytest.raises(LoggerSessionError, match="not initialized"):
                session.log_parameters({"param": "value"})

    def test_rejects_double_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()

            with pytest.raises(LoggerSessionError, match="already initialized"):
                session.initialize()

            session.finalize()

    def test_context_manager_auto_initializes_and_finalizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            with session:
                assert session._active is True
                session.log_measurement("test", {"value": 1})

            assert session._active is False

    def test_validates_empty_measurement_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()

            with pytest.raises(LoggerSessionError, match="cannot be empty"):
                session.log_measurement("", {"value": 1})

            with pytest.raises(LoggerSessionError, match="cannot be empty"):
                session.log_measurement("  ", {"value": 1})

            session.finalize()

    def test_validates_empty_sweep_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()

            with pytest.raises(LoggerSessionError, match="cannot be empty"):
                session.log_sweep(
                    "test",
                    np.array([]),
                    np.array([]),
                    "X",
                    "Y",
                )

            session.finalize()

    def test_rejects_nonexistent_base_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "nonexistent"
            writer = JSONLWriter(tmpdir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            with pytest.raises(LoggerSessionError, match="does not exist"):
                LoggerSession(
                    metadata=metadata,
                    writer_pool={"jsonl": writer},
                    writer_refs=["jsonl"],
                    base_dir=base_dir,
                )

    def test_writes_to_multiple_writers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer1_dir = base_dir / "writer1"
            writer2_dir = base_dir / "writer2"
            writer1_dir.mkdir()
            writer2_dir.mkdir()

            writer1 = JSONLWriter(writer1_dir)
            writer2 = JSONLWriter(writer2_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl1": writer1, "jsonl2": writer2},
                writer_refs=["jsonl1", "jsonl2"],
                base_dir=base_dir,
            )

            session.initialize()
            session.log_measurement("test", {"value": 1})
            session.finalize()

            assert (writer1_dir / "measurement.jsonl").exists()
            assert (writer2_dir / "measurement.jsonl").exists()

    def test_initialize_session_error_propagates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            class FailingWriter(JSONLWriter):
                def initialize_session(self, session):
                    raise RuntimeError("Simulated writer failure")

            writer = FailingWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"failing": writer},
                writer_refs=["failing"],
                base_dir=base_dir,
            )

            with pytest.raises(LoggerSessionError, match="Failed to initialize"):
                session.initialize()

            assert session._active is False

    def test_finalize_session_error_sets_inactive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            class FailingWriter(JSONLWriter):
                def finalize_session(self, session=None):
                    raise RuntimeError("Simulated finalization failure")

            writer = FailingWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"failing": writer},
                writer_refs=["failing"],
                base_dir=base_dir,
            )

            session._writer_pool = {"failing": writer}
            session._active = True

            with pytest.raises(LoggerSessionError, match="Failed to finalize"):
                session.finalize()

            assert session._active is False

    def test_flush_handles_write_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            class FailingWriter(JSONLWriter):
                def write_measurement(self, data):
                    raise RuntimeError("Write failure")

                def flush(self):
                    raise RuntimeError("Flush failure")

            writer = FailingWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"failing": writer},
                writer_refs=["failing"],
                base_dir=base_dir,
                buffer_size=10,
            )

            session._writer_pool = {"failing": writer}
            session._active = True

            session.log_measurement("test", {"value": 1})

            session.flush()

            assert len(session._buffer) == 0

    def test_finalize_when_not_active_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            with pytest.raises(LoggerSessionError, match="not initialized"):
                session.finalize()

    def test_context_manager_with_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            class FailingWriter(JSONLWriter):
                def finalize_session(self, session=None):
                    raise RuntimeError("Finalization failure")

            writer = FailingWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"failing": writer},
                writer_refs=["failing"],
                base_dir=base_dir,
            )

            with pytest.raises(LoggerSessionError):
                with session:
                    pass

    def test_validates_empty_measurement_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()

            with pytest.raises(LoggerSessionError, match="cannot be empty"):
                session.log_measurement("test", {})

            session.finalize()

    def test_validates_empty_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            session.initialize()

            with pytest.raises(LoggerSessionError, match="cannot be empty"):
                session.log_parameters({})

            session.finalize()

    def test_routine_name_property_returns_empty_when_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = JSONLWriter(base_dir)
            metadata = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
                routine_name=None,
            )

            session = LoggerSession(
                metadata=metadata,
                writer_pool={"jsonl": writer},
                writer_refs=["jsonl"],
                base_dir=base_dir,
            )

            assert session.routine_name == ""
