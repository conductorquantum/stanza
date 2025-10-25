import numpy as np
import pytest

from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData


class TestMeasurementData:
    def test_stores_and_serializes_measurement(self):
        data = MeasurementData(
            name="gate_voltage",
            data={"voltage": 1.5, "current": 1e-6},
            metadata={"device": "test_device"},
            timestamp=1234567890.0,
            session_id="session_001",
            routine_name="characterization",
        )

        serialized = data.to_dict()
        assert serialized["name"] == "gate_voltage"
        assert serialized["data"]["voltage"] == 1.5
        assert serialized["routine_name"] == "characterization"


class TestSweepData:
    def test_validates_matching_array_shapes(self):
        voltages = np.linspace(0, 1, 10)
        currents = np.random.rand(10) * 1e-6

        sweep = SweepData(
            name="iv_sweep",
            x_data=voltages,
            y_data=currents,
            x_label="Voltage (V)",
            y_label="Current (A)",
            metadata={"gate": "P1"},
            timestamp=1234567890.0,
            session_id="session_001",
        )

        assert len(sweep.x_data) == len(sweep.y_data)

    def test_rejects_mismatched_array_shapes(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            SweepData(
                name="bad_sweep",
                x_data=np.array([0, 1, 2]),
                y_data=np.array([0, 1]),
                x_label="X",
                y_label="Y",
                metadata={},
                timestamp=0.0,
                session_id="s1",
            )

    def test_serializes_numpy_arrays_to_lists(self):
        sweep = SweepData(
            name="test",
            x_data=np.array([1.0, 2.0]),
            y_data=np.array([3.0, 4.0]),
            x_label="X",
            y_label="Y",
            metadata={},
            timestamp=0.0,
            session_id="s1",
        )

        serialized = sweep.to_dict()
        assert serialized["x_data"] == [1.0, 2.0]
        assert serialized["y_data"] == [3.0, 4.0]


class TestSessionMetadata:
    def test_calculates_session_duration(self):
        session = SessionMetadata(
            session_id="test_session",
            start_time=100.0,
            user="alice",
            end_time=250.0,
        )

        assert session.duration == 150.0

    def test_duration_none_when_session_ongoing(self):
        session = SessionMetadata(
            session_id="ongoing",
            start_time=100.0,
            user="bob",
        )

        assert session.duration is None

    def test_serialization_includes_computed_duration(self):
        session = SessionMetadata(
            session_id="test",
            start_time=0.0,
            user="user",
            end_time=10.0,
        )

        serialized = session.to_dict()
        assert serialized["duration"] == 10.0
        assert serialized["start_time"] == 0.0
        assert serialized["end_time"] == 10.0
