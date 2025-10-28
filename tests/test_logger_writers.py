import gzip
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from stanza.exceptions import WriterError
from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData
from stanza.logger.writers.jsonl_writer import JSONLWriter

try:
    import h5py

    from stanza.logger.writers.hdf5_writer import HDF5Writer

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestHDF5Writer:
    def test_creates_hdf5_file_and_writes_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
                routine_name="test_routine",
            )

            writer.initialize_session(session)
            assert writer._session_file.exists()
            assert writer._h5_file is not None

            writer.finalize_session()
            assert writer._h5_file is None

    def test_writes_measurement_data_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            measurement = MeasurementData(
                name="test_measurement",
                data={"voltage": 1.5, "current": 1e-6},
                metadata={"device": "test"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_measurement(measurement)
            writer.flush()

            with h5py.File(writer._session_file, "r") as f:
                assert "measurements" in f
                assert "test_measurement" in f["measurements"]

            writer.finalize_session()

    def test_writes_sweep_data_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            sweep = SweepData(
                name="voltage_sweep",
                x_data=np.array([0.0, 1.0, 2.0]),
                y_data=np.array([0.0, 1e-6, 2e-6]),
                x_label="Voltage (V)",
                y_label="Current (A)",
                metadata={"gate": "P1"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_sweep(sweep)
            writer.flush()

            with h5py.File(writer._session_file, "r") as f:
                assert "sweeps" in f
                assert "voltage_sweep" in f["sweeps"]
                assert "data" in f["sweeps/voltage_sweep"]

            writer.finalize_session()

    def test_hdf5_writer_uniquifies_sweep_names(self):
        """Test that HDF5 writer adds suffixes to duplicate sweep names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            # Write first sweep with name "signal"
            sweep1 = SweepData(
                name="signal",
                x_data=np.array([0.0, 1.0]),
                y_data=np.array([0.0, 1.0]),
                x_label="X",
                y_label="Y",
                metadata={},
                timestamp=100.0,
                session_id="test_session",
            )
            writer.write_sweep(sweep1)

            # Write second sweep with same name "signal"
            sweep2 = SweepData(
                name="signal",
                x_data=np.array([2.0, 3.0]),
                y_data=np.array([2.0, 3.0]),
                x_label="X",
                y_label="Y",
                metadata={},
                timestamp=101.0,
                session_id="test_session",
            )
            writer.write_sweep(sweep2)

            # Write third sweep with same name "signal"
            sweep3 = SweepData(
                name="signal",
                x_data=np.array([4.0, 5.0]),
                y_data=np.array([4.0, 5.0]),
                x_label="X",
                y_label="Y",
                metadata={},
                timestamp=102.0,
                session_id="test_session",
            )
            writer.write_sweep(sweep3)

            writer.flush()

            # Verify all three sweeps are in the file with unique names
            with h5py.File(writer._session_file, "r") as f:
                assert "sweeps" in f
                assert "signal" in f["sweeps"]
                assert "signal_0" in f["sweeps"]
                assert "signal_1" in f["sweeps"]

                # Verify data integrity
                np.testing.assert_array_equal(
                    f["sweeps/signal/data/Y"][:], np.array([0.0, 1.0])
                )
                np.testing.assert_array_equal(
                    f["sweeps/signal_0/data/Y"][:], np.array([2.0, 3.0])
                )
                np.testing.assert_array_equal(
                    f["sweeps/signal_1/data/Y"][:], np.array([4.0, 5.0])
                )

            writer.finalize_session()

    def test_raises_error_when_writing_without_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)

            measurement = MeasurementData(
                name="test",
                data={},
                metadata={},
                timestamp=0.0,
                session_id="s1",
            )

            with pytest.raises(WriterError, match="No active session"):
                writer.write_measurement(measurement)

    def test_flush_requires_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)

            with pytest.raises(WriterError, match="No active session"):
                writer.flush()

    def test_creates_directory_if_not_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "subdir" / "nested"
            _writer = HDF5Writer(new_dir)
            assert new_dir.exists()

    def test_compression_settings_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir, compression="gzip", compression_level=4)
            assert writer.compression == "gzip"
            assert writer.compression_level == 4

    def test_measurement_with_array_data_and_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir, compression="gzip", compression_level=4)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            measurement = MeasurementData(
                name="array_measurement",
                data={"voltages": np.array([1.0, 2.0, 3.0]), "scalar": 5.0},
                metadata={"device": "test"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_measurement(measurement)
            writer.finalize_session()

    def test_sweep_with_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir, compression="gzip", compression_level=4)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            sweep = SweepData(
                name="compressed_sweep",
                x_data=np.linspace(0, 10, 100),
                y_data=np.random.rand(100),
                x_label="X",
                y_label="Y",
                metadata={"test": "value"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_sweep(sweep)
            writer.finalize_session()

    def test_finalize_session_without_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            with pytest.raises(WriterError, match="No active session"):
                writer.finalize_session()

    def test_write_sweep_without_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            sweep = SweepData(
                name="test",
                x_data=np.array([1.0]),
                y_data=np.array([2.0]),
                x_label="X",
                y_label="Y",
                metadata={},
                timestamp=0.0,
                session_id="s1",
            )
            with pytest.raises(WriterError, match="No active session"):
                writer.write_sweep(sweep)

    def test_initialize_session_error_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test",
                start_time=100.0,
                user="user",
            )
            writer.initialize_session(session)

            if writer._h5_file:
                writer._h5_file.close()
                writer._h5_file = None

            with pytest.raises(WriterError, match="No active session"):
                writer.flush()

    def test_finalize_session_error_on_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test",
                start_time=100.0,
                user="user",
            )
            writer.initialize_session(session)

            writer._h5_file.close()

            writer.finalize_session()

    def test_initialize_session_error_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test",
                start_time=100.0,
                user="user",
            )

            mock_file = patch("h5py.File")
            with mock_file as mock_h5:
                mock_h5_instance = mock_h5.return_value
                mock_h5_instance.create_group.side_effect = RuntimeError("Mock error")

                with pytest.raises(WriterError, match="Failed to initialize"):
                    writer.initialize_session(session)

            assert writer._session_file is None
            assert writer._h5_file is None

    def test_finalize_session_error_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test",
                start_time=100.0,
                user="user",
            )
            writer.initialize_session(session)

            with patch.object(
                writer._h5_file, "close", side_effect=RuntimeError("Mock")
            ):
                with pytest.raises(WriterError, match="Failed to finalize"):
                    writer.finalize_session()

    def test_finalize_session_updates_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
                parameters={"temp": 4.2},
                end_time=200.0,
            )
            writer.initialize_session(session)

            updated_session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
                parameters={"temp": 4.2, "field": 1.5},
                end_time=250.0,
            )

            writer.finalize_session(updated_session)

            with h5py.File(Path(tmpdir) / "test_session.h5", "r") as f:
                assert f["metadata"].attrs["end_time"] == 250.0

    def test_write_measurement_error_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HDF5Writer(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            class UnserializableObject:
                pass

            measurement = MeasurementData(
                name="test_measurement",
                data={"bad": UnserializableObject()},
                metadata={},
                timestamp=100.0,
                session_id="test_session",
            )

            with pytest.raises(WriterError, match="Failed to write measurement"):
                writer.write_measurement(measurement)

            writer.finalize_session()


class TestJSONLWriter:
    def test_creates_jsonl_file_and_writes_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
                routine_name="test_routine",
            )

            writer.initialize_session(session)
            assert writer._session_dir is not None
            assert writer._session_dir.exists()

            metadata_file = writer._session_dir / "session_metadata.json"
            assert metadata_file.exists()

            with open(metadata_file) as f:
                data = json.load(f)
                assert data["session_id"] == "test_session"
                assert data["user"] == "test_user"

            writer.finalize_session()
            assert writer._session_dir is None

    def test_writes_measurement_data_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            measurement = MeasurementData(
                name="test_measurement",
                data={"voltage": 1.5, "current": 1e-6},
                metadata={"device": "test"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_measurement(measurement)
            writer.flush()

            measurement_file = writer._session_dir / "measurement.jsonl"
            assert measurement_file.exists()

            with open(measurement_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["name"] == "test_measurement"
                assert data["data"]["voltage"] == 1.5

            writer.finalize_session()

    def test_writes_sweep_data_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            sweep = SweepData(
                name="voltage_sweep",
                x_data=np.array([0.0, 1.0, 2.0]),
                y_data=np.array([0.0, 1e-6, 2e-6]),
                x_label="Voltage (V)",
                y_label="Current (A)",
                metadata={"gate": "P1"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_sweep(sweep)
            writer.flush()

            sweep_file = writer._session_dir / "sweep.jsonl"
            assert sweep_file.exists()

            with open(sweep_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["name"] == "voltage_sweep"
                assert data["x_label"] == "Voltage (V)"

            writer.finalize_session()

    def test_raises_error_when_writing_without_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)

            measurement = MeasurementData(
                name="test",
                data={},
                metadata={},
                timestamp=0.0,
                session_id="s1",
            )

            with pytest.raises(WriterError, match="No active session"):
                writer.write_measurement(measurement)

    def test_creates_directory_if_not_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "subdir" / "nested"
            _writer = JSONLWriter(new_dir)
            assert new_dir.exists()

    def test_compression_settings_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir, compression="gzip", compression_level=4)
            assert writer.compression == "gzip"
            assert writer.compression_level == 4

    def test_measurement_with_array_data_and_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir, compression="gzip", compression_level=4)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            measurement = MeasurementData(
                name="array_measurement",
                data={"voltages": np.array([1.0, 2.0, 3.0]), "scalar": 5.0},
                metadata={"device": "test"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_measurement(measurement)

            measurement_file = writer._session_dir / "measurement.jsonl.gz"
            assert measurement_file.exists()

            with gzip.open(measurement_file, "rt") as f:
                line = f.readline()
                data = json.loads(line)
                assert data["name"] == "array_measurement"
                assert data["data"]["voltages"] == [1.0, 2.0, 3.0]

            writer.finalize_session()

    def test_sweep_with_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir, compression="gzip", compression_level=4)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            sweep = SweepData(
                name="compressed_sweep",
                x_data=np.linspace(0, 10, 100),
                y_data=np.random.rand(100),
                x_label="X",
                y_label="Y",
                metadata={"test": "value"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_sweep(sweep)

            sweep_file = writer._session_dir / "sweep.jsonl.gz"
            assert sweep_file.exists()

            with gzip.open(sweep_file, "rt") as f:
                line = f.readline()
                data = json.loads(line)
                assert data["name"] == "compressed_sweep"

            writer.finalize_session()

    def test_finalize_session_without_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            with pytest.raises(WriterError, match="No active session"):
                writer.finalize_session()

    def test_write_sweep_without_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            sweep = SweepData(
                name="test",
                x_data=np.array([1.0]),
                y_data=np.array([2.0]),
                x_label="X",
                y_label="Y",
                metadata={},
                timestamp=0.0,
                session_id="s1",
            )
            with pytest.raises(WriterError, match="No active session"):
                writer.write_sweep(sweep)

    def test_analysis_measurement_written_to_separate_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            analysis_measurement = MeasurementData(
                name="analysis_result",
                data={"result": 42.0},
                metadata={"data_type": "analysis"},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_measurement(analysis_measurement)

            analysis_file = writer._session_dir / "analysis.jsonl"
            assert analysis_file.exists()

            with open(analysis_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["name"] == "analysis_result"
                assert data["metadata"]["data_type"] == "analysis"

            writer.finalize_session()

    def test_multiple_measurements_appended_to_same_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            for i in range(3):
                measurement = MeasurementData(
                    name=f"measurement_{i}",
                    data={"value": i},
                    metadata={},
                    timestamp=100.0 + i,
                    session_id="test_session",
                )
                writer.write_measurement(measurement)

            measurement_file = writer._session_dir / "measurement.jsonl"
            with open(measurement_file) as f:
                lines = f.readlines()
                assert len(lines) == 3
                for i, line in enumerate(lines):
                    data = json.loads(line)
                    assert data["name"] == f"measurement_{i}"

            writer.finalize_session()

    def test_initialize_session_twice_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            with pytest.raises(WriterError, match="Session already initialized"):
                writer.initialize_session(session)

    def test_encoder_handles_numpy_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            measurement = MeasurementData(
                name="numpy_types",
                data={
                    "np_array": np.array([1, 2, 3]),
                    "np_int": np.int64(42),
                    "np_float": np.float64(3.14),
                    "np_bool": np.bool_(True),
                    "regular_int": 10,
                },
                metadata={},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_measurement(measurement)

            measurement_file = writer._session_dir / "measurement.jsonl"
            with open(measurement_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert data["data"]["np_array"] == [1, 2, 3]
                assert data["data"]["np_int"] == 42
                assert data["data"]["np_float"] == 3.14
                assert data["data"]["np_bool"] is True

            writer.finalize_session()

    def test_initialize_session_error_on_file_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )

            with patch("builtins.open", side_effect=OSError("Mocked error")):
                with pytest.raises(
                    WriterError, match="Error creating session directory"
                ):
                    writer.initialize_session(session)

            assert writer._session_dir is None

    def test_write_measurement_error_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            class UnserializableObject:
                pass

            measurement = MeasurementData(
                name="bad_measurement",
                data={"bad": UnserializableObject()},
                metadata={},
                timestamp=100.0,
                session_id="test_session",
            )

            with pytest.raises(WriterError, match="Error writing measurement"):
                writer.write_measurement(measurement)

            writer.finalize_session()

    def test_write_sweep_error_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            class UnserializableObject:
                pass

            sweep = SweepData(
                name="bad_sweep",
                x_data=np.array([1.0]),
                y_data=np.array([2.0]),
                x_label="X",
                y_label="Y",
                metadata={"bad": UnserializableObject()},
                timestamp=100.0,
                session_id="test_session",
            )

            with pytest.raises(WriterError, match="Error writing sweep"):
                writer.write_sweep(sweep)

            writer.finalize_session()

    def test_finalize_session_error_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            writer._file_handles["bad"] = "not_a_path"

            writer.finalize_session()

    def test_encoder_handles_all_numpy_float_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            measurement = MeasurementData(
                name="float_types",
                data={
                    "float32": np.float32(1.5),
                    "float64": np.float64(2.5),
                },
                metadata={},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_measurement(measurement)

            measurement_file = writer._session_dir / "measurement.jsonl"
            with open(measurement_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert abs(data["data"]["float32"] - 1.5) < 0.0001
                assert abs(data["data"]["float64"] - 2.5) < 0.0001

            writer.finalize_session()

    def test_encoder_handles_timestamp_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            dt = datetime(2024, 1, 1, 12, 0, 0)

            measurement = MeasurementData(
                name="timestamp_test",
                data={"time": dt},
                metadata={},
                timestamp=100.0,
                session_id="test_session",
            )

            writer.write_measurement(measurement)

            measurement_file = writer._session_dir / "measurement.jsonl"
            with open(measurement_file) as f:
                line = f.readline()
                data = json.loads(line)
                assert isinstance(data["data"]["time"], (int, float))

            writer.finalize_session()

    def test_finalize_session_exception_during_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLWriter(tmpdir)
            session = SessionMetadata(
                session_id="test_session",
                start_time=100.0,
                user="test_user",
            )
            writer.initialize_session(session)

            with patch("stanza.logger.writers.jsonl_writer.logger") as mock_logger:
                mock_logger.info.side_effect = RuntimeError("Simulated error")

                with pytest.raises(WriterError, match="Error finalizing session"):
                    writer.finalize_session()
