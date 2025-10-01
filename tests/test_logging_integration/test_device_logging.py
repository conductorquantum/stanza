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
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        voltages, currents = device.sweep_1d("gate1", [0.0, 1.0], "contact1", session)

        assert len(voltages) == 2
        assert len(currents) == 2

        logger.close_session(session.session_id)

    def test_buffers_sweep_data_correctly(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_1d("gate1", [0.0, 1.0], "contact1", session)

        assert len(session._buffer) == 1
        sweep_data = session._buffer[0]
        assert len(sweep_data.x_data) == 2
        assert len(sweep_data.y_data) == 2

        logger.close_session(session.session_id)

    def test_writes_sweep_to_jsonl_file(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_1d("gate1", [0.0, 1.0], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        assert jsonl_file.exists()

    def test_jsonl_file_contains_correct_data(self, device, temp_dir):
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


class TestSweep2DLogging:
    def test_raises_length_mismatch_error(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        with pytest.raises(
            ValueError, match="x_data and y_data must have the same length"
        ):
            device.sweep_2d("gate1", [0.0, 1.0], "gate1", [0.5], "contact1", session)

        logger.close_session(session.session_id)

    def test_does_not_buffer_data_on_error(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        with pytest.raises(ValueError):
            device.sweep_2d("gate1", [0.0, 1.0], "gate1", [0.5], "contact1", session)

        assert len(session._buffer) == 0

        logger.close_session(session.session_id)


class TestSweepAllLogging:
    def test_returns_correct_measurement_lengths(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        voltages, currents = device.sweep_all([0.0, 1.0], "contact1", session)

        assert len(voltages) == 2
        assert len(currents) == 2

        logger.close_session(session.session_id)

    def test_buffers_sweep_data_correctly(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_all([0.0, 1.0], "contact1", session)

        assert len(session._buffer) == 1
        sweep_data = session._buffer[0]
        assert len(sweep_data.x_data) == 2
        assert len(sweep_data.y_data) == 2

        logger.close_session(session.session_id)

    def test_writes_sweep_to_jsonl_file(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_all([0.0, 1.0], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        assert jsonl_file.exists()

    def test_jsonl_file_contains_correct_data(self, device, temp_dir):
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
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        voltages, currents = device.sweep_nd(
            ["gate1"], [[0.0], [1.0]], "contact1", session
        )

        assert len(voltages) == 2
        assert len(currents) == 2

        logger.close_session(session.session_id)

    def test_buffers_sweep_data_correctly(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_nd(["gate1"], [[0.0], [1.0]], "contact1", session)

        assert len(session._buffer) == 1
        sweep_data = session._buffer[0]
        assert len(sweep_data.x_data) == 2
        assert len(sweep_data.y_data) == 2

        logger.close_session(session.session_id)

    def test_writes_sweep_to_jsonl_file(self, device, temp_dir):
        logger = DataLogger("test_routine", temp_dir)
        session = logger.create_session()

        device.sweep_nd(["gate1"], [[0.0], [1.0]], "contact1", session)
        logger.close_session(session.session_id)

        jsonl_file = temp_dir / "test_routine" / session.session_id / "sweep.jsonl"
        assert jsonl_file.exists()

    def test_jsonl_file_contains_correct_data(self, device, temp_dir):
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
