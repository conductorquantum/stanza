import json
import tempfile

import numpy as np
import pytest

from stanza.exceptions import LoggingError
from stanza.logger.data_logger import DataLogger


class TestDataLogger:
    def test_creates_data_logger_with_default_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            assert logger.routine_name == "test_routine"
            assert logger.base_directory.exists()
            assert logger._formats == ["jsonl"]

    def test_creates_data_logger_with_custom_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
                formats=["jsonl"],
            )

            assert logger._formats == ["jsonl"]

    def test_slugifies_routine_name_for_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="Test Routine With Spaces!",
                base_dir=tmpdir,
            )

            assert "Test_Routine_With_Spaces_" in str(logger.base_directory)

    def test_creates_and_closes_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            session = logger.create_session()
            assert session is not None
            assert logger.current_session == session

            logger.close_session(session.session_id)
            assert logger.current_session is None

    def test_auto_generates_session_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            session1 = logger.create_session()
            logger.close_session(session1.session_id)

            session2 = logger.create_session()
            logger.close_session(session2.session_id)

            assert session1.session_id != session2.session_id

    def test_creates_session_with_custom_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            session = logger.create_session(session_id="custom_session_id")
            assert session.session_id == "custom_session_id"

            logger.close_session(session.session_id)

    def test_auto_flushes_before_closing_current_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            session1 = logger.create_session(session_id="session1")
            session1.log_measurement("test", {"value": 1})

            session2 = logger.create_session(session_id="session2")

            measurement_file = logger.base_directory / "session1" / "measurement.jsonl"
            assert measurement_file.exists()

            logger.close_session(session2.session_id)

    def test_close_session_flushes_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
                buffer_size=100,  # Large buffer to prevent auto-flush
            )

            session = logger.create_session(session_id="test_session")

            # Log measurements without triggering buffer flush
            session.log_measurement("m1", {"value": 1})
            session.log_measurement("m2", {"value": 2})
            session.log_measurement("m3", {"value": 3})

            # Verify buffer has data
            assert len(session._buffer) > 0

            # Close session - should flush buffer
            logger.close_session("test_session")

            # Verify data was written to file
            measurement_file = (
                logger.base_directory / "test_session" / "measurement.jsonl"
            )
            assert measurement_file.exists()

            with open(measurement_file) as f:
                lines = f.readlines()
                assert len(lines) == 3
                data = json.loads(lines[0])
                assert data["name"] == "m1"

    def test_end_to_end_measurement_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="characterization",
                base_dir=tmpdir,
            )

            session = logger.create_session(session_id="test_session")

            session.log_measurement(
                name="gate_voltage",
                data={"voltage": 1.5, "current": 1e-6},
                metadata={"gate": "P1"},
            )

            session.log_measurement(
                name="drain_voltage",
                data={"voltage": 0.5, "current": 5e-7},
                metadata={"drain": "D1"},
            )

            logger.close_session(session.session_id)

            measurement_file = (
                logger.base_directory / "test_session" / "measurement.jsonl"
            )
            assert measurement_file.exists()

            with open(measurement_file) as f:
                lines = f.readlines()
                assert len(lines) == 2

                data1 = json.loads(lines[0])
                assert data1["name"] == "gate_voltage"
                assert data1["data"]["voltage"] == 1.5

                data2 = json.loads(lines[1])
                assert data2["name"] == "drain_voltage"

    def test_end_to_end_sweep_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="iv_characterization",
                base_dir=tmpdir,
            )

            session = logger.create_session(session_id="test_session")

            voltages = np.linspace(0, 1, 50)
            currents = voltages * 1e-6 + np.random.randn(50) * 1e-9

            session.log_sweep(
                name="iv_sweep",
                x_data=voltages,
                y_data=currents,
                x_label="Voltage (V)",
                y_label="Current (A)",
                metadata={"device": "test_device"},
            )

            logger.close_session(session.session_id)

            sweep_file = logger.base_directory / "test_session" / "sweep.jsonl"
            assert sweep_file.exists()

            with open(sweep_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["name"] == "iv_sweep"
                assert len(data["x_data"]) == 50
                assert len(data["y_data"]) == 50

    def test_end_to_end_analysis_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="analysis",
                base_dir=tmpdir,
            )

            session = logger.create_session(session_id="test_session")

            session.log_analysis(
                name="linear_fit",
                data={"slope": 1.23, "intercept": 0.45, "r_squared": 0.98},
                metadata={"algorithm": "least_squares"},
            )

            logger.close_session(session.session_id)

            analysis_file = logger.base_directory / "test_session" / "analysis.jsonl"
            assert analysis_file.exists()

            with open(analysis_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["name"] == "linear_fit"
                assert data["data"]["slope"] == 1.23
                assert data["metadata"]["data_type"] == "analysis"

    def test_context_manager_closes_all_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataLogger(routine_name="test", base_dir=tmpdir) as logger:
                session = logger.create_session(session_id="test_session")
                session.log_measurement("test", {"value": 1})

            assert logger.current_session is None

            measurement_file = (
                logger.base_directory / "test_session" / "measurement.jsonl"
            )
            assert measurement_file.exists()

    def test_rejects_duplicate_session_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            logger.create_session(session_id="duplicate")

            with pytest.raises(LoggingError, match="already exists"):
                logger.create_session(session_id="duplicate")

            logger.close_session("duplicate")

    def test_rejects_invalid_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid format"):
                DataLogger(
                    routine_name="test_routine",
                    base_dir=tmpdir,
                    formats=["invalid_format"],
                )

    def test_rejects_empty_routine_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Routine name is required"):
                DataLogger(
                    routine_name="",
                    base_dir=tmpdir,
                )

            with pytest.raises(ValueError, match="Routine name is required"):
                DataLogger(
                    routine_name="   ",
                    base_dir=tmpdir,
                )

    def test_get_session_returns_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            session = logger.create_session(session_id="test_session")
            retrieved = logger.get_session("test_session")

            assert retrieved == session

            logger.close_session("test_session")

    def test_get_session_returns_none_for_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            assert logger.get_session("nonexistent") is None

    def test_active_sessions_property_returns_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            assert len(logger.active_sessions) == 0

            logger.create_session(session_id="session1")
            assert len(logger.active_sessions) == 1

            logger.close_session("session1")
            assert len(logger.active_sessions) == 0

    def test_close_nonexistent_session_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            with pytest.raises(LoggingError, match="does not exist"):
                logger.close_session("nonexistent")

    def test_custom_routine_directory_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
                routine_dir_name="custom_dir",
            )

            assert logger.base_directory.name == "custom_dir"

    def test_compression_settings_passed_to_writers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
                compression="gzip",
                compression_level=9,
            )

            session = logger.create_session(session_id="test_session")
            session.log_measurement("test", {"value": 1})
            logger.close_session("test_session")

            compressed_file = (
                logger.base_directory / "test_session" / "measurement.jsonl.gz"
            )
            assert compressed_file.exists()

    def test_buffer_size_setting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
                buffer_size=2,
            )

            session = logger.create_session(session_id="test_session")

            session.log_measurement("m1", {"value": 1})
            session.log_measurement("m2", {"value": 2})

            measurement_file = (
                logger.base_directory / "test_session" / "measurement.jsonl"
            )
            assert measurement_file.exists()

            logger.close_session("test_session")

    def test_mixed_data_types_in_single_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="mixed_test",
                base_dir=tmpdir,
            )

            session = logger.create_session(session_id="test_session")

            session.log_measurement("measurement1", {"value": 1.5})

            session.log_sweep(
                name="sweep1",
                x_data=np.array([0.0, 1.0]),
                y_data=np.array([0.0, 1.0]),
                x_label="X",
                y_label="Y",
            )

            session.log_analysis("analysis1", {"result": 42})

            logger.close_session("test_session")

            assert (
                logger.base_directory / "test_session" / "measurement.jsonl"
            ).exists()
            assert (logger.base_directory / "test_session" / "sweep.jsonl").exists()
            assert (logger.base_directory / "test_session" / "analysis.jsonl").exists()

    def test_session_parameters_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            session = logger.create_session(session_id="test_session")

            session.log_parameters({"temperature": 4.2, "magnetic_field": 1.5})
            session.log_parameters({"sample_id": "S001"})

            logger.close_session("test_session")

            metadata_file = (
                logger.base_directory / "test_session" / "session_metadata.json"
            )
            with open(metadata_file) as f:
                metadata = json.load(f)
                assert metadata["parameters"]["temperature"] == 4.2
                assert metadata["parameters"]["magnetic_field"] == 1.5
                assert metadata["parameters"]["sample_id"] == "S001"

    def test_close_session_handles_finalization_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            session = logger.create_session(session_id="test_session")

            session._writer_pool = {
                "jsonl": type(
                    "FailingWriter",
                    (),
                    {
                        "finalize_session": lambda s, m=None: (_ for _ in ()).throw(
                            RuntimeError("Fail")
                        )
                    },
                )()
            }

            logger.close_session("test_session")

            assert logger.get_session("test_session") is None

    def test_close_all_sessions_handles_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DataLogger(
                routine_name="test_routine",
                base_dir=tmpdir,
            )

            session1 = logger.create_session(session_id="session1")
            logger.create_session(session_id="session2")

            session1._writer_pool = {
                "jsonl": type(
                    "FailingWriter",
                    (),
                    {
                        "finalize_session": lambda s, m=None: (_ for _ in ()).throw(
                            RuntimeError("Fail")
                        )
                    },
                )()
            }

            logger.close_all_sessions()

            assert len(logger.active_sessions) == 0
