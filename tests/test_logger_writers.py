import tempfile
from pathlib import Path

import numpy as np
import pytest

from stanza.exceptions import WriterError
from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData

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
