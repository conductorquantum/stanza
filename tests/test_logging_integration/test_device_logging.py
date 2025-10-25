import json
import tempfile
from pathlib import Path

import pytest

from stanza.logger.data_logger import DataLogger


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestSweep1DLogging:
    def test_returns_correct_measurement_lengths(self, device, temp_dir):
        """Test that 1D sweep returns correct number of voltage and current measurements."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        voltages, currents = device.sweep_1d("gate1", [0.0, 1.0], "contact1", session)

        assert len(voltages) == 2
        assert len(currents) == 2

        logger.close_session(session.session_id)

    def test_buffers_sweep_data_correctly(self, device, temp_dir):
        """Test that 1D sweep data is correctly written using streaming API."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_1d("gate1", [0.0, 1.0], "contact1", session)

        # With streaming API, data is written directly to file, not buffered
        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        assert jsonl_file.exists()

        # Verify sweep data was written correctly
        with open(jsonl_file) as f:
            sweep_data = json.loads(f.readline())
            assert len(sweep_data["x_data"]) == 2
            assert len(sweep_data["y_data"]) == 2

        logger.close_session(session.session_id)

    def test_writes_sweep_to_jsonl_file(self, device, temp_dir):
        """Test that 1D sweep data is written to JSONL file on session close."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_1d("gate1", [0.0, 1.0], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        assert jsonl_file.exists()

    def test_jsonl_file_contains_correct_data(self, device, temp_dir):
        """Test that 1D sweep JSONL file contains correctly formatted data."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_1d("gate1", [0.0, 1.0], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        with open(jsonl_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert len(data["x_data"]) == 2
            assert len(data["y_data"]) == 2
            assert data["name"] == "gate1 sweep"
            assert data["x_label"] == "Voltage"
            assert data["y_label"] == "Current"


class TestSweepAllLogging:
    def test_returns_correct_measurement_lengths(self, device, temp_dir):
        """Test that sweep_all returns correct number of voltage and current measurements."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        voltages, currents = device.sweep_all([0.0, 1.0], "contact1", session)

        assert len(voltages) == 2
        assert len(currents) == 2

        logger.close_session(session.session_id)

    def test_buffers_sweep_data_correctly(self, device, temp_dir):
        """Test that sweep_all data is correctly written using streaming API."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_all([0.0, 1.0], "contact1", session)

        # With streaming API, data is written directly to file, not buffered
        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        assert jsonl_file.exists()

        # Verify sweep data was written correctly
        with open(jsonl_file) as f:
            sweep_data = json.loads(f.readline())
            assert len(sweep_data["x_data"]) == 2
            assert len(sweep_data["y_data"]) == 2

        logger.close_session(session.session_id)

    def test_writes_sweep_to_jsonl_file(self, device, temp_dir):
        """Test that sweep_all data is written to JSONL file on session close."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_all([0.0, 1.0], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        assert jsonl_file.exists()

    def test_jsonl_file_contains_correct_data(self, device, temp_dir):
        """Test that sweep_all JSONL file contains correctly formatted data."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_all([0.0, 1.0], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        with open(jsonl_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert len(data["x_data"]) == 2
            assert len(data["y_data"]) == 2
            assert data["name"] == "all gates sweep"


class TestSweepNDLogging:
    def test_returns_correct_measurement_lengths(self, device, temp_dir):
        """Test that ND sweep returns correct number of voltage and current measurements."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        voltages, currents = device.sweep_nd(
            ["gate1"], [[0.0], [1.0]], "contact1", session
        )

        assert len(voltages) == 2
        assert len(currents) == 2

        logger.close_session(session.session_id)

    def test_buffers_sweep_data_correctly(self, device, temp_dir):
        """Test that ND sweep data is correctly buffered in session."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_nd(["gate1"], [[0.0], [1.0]], "contact1", session)

        assert len(session._buffer) == 1
        sweep_data = session._buffer[0]
        assert len(sweep_data.x_data) == 2
        assert len(sweep_data.y_data) == 2

        logger.close_session(session.session_id)

    def test_writes_sweep_to_jsonl_file(self, device, temp_dir):
        """Test that ND sweep data is written to JSONL file on session close."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_nd(["gate1"], [[0.0], [1.0]], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        assert jsonl_file.exists()

    def test_jsonl_file_contains_correct_data(self, device, temp_dir):
        """Test that ND sweep JSONL file contains correctly formatted data."""
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_nd(["gate1"], [[0.0], [1.0]], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        with open(jsonl_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert len(data["x_data"]) == 2
            assert len(data["y_data"]) == 2
            assert data["name"] == "n gates sweep"
