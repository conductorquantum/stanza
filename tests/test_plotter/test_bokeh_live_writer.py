"""Tests for BokehLiveWriter."""

import getpass

import numpy as np
import pytest

from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData
from stanza.logger.writers.bokeh_writer import BokehLiveWriter


class MockBackend:
    """Mock backend for testing."""

    def __init__(self) -> None:
        self.figures_created: list[dict[str, str]] = []
        self.push_count = 0
        self.streamed_data: dict[str, list[dict]] = {}

    def create_figure(
        self,
        name: str,
        x_label: str,
        y_label: str,
        plot_type: str = "line",
        z_label: str | None = None,
        cell_size: tuple[float, float] | None = None,
    ) -> object:
        """Mock create_figure."""
        self.figures_created.append(
            {
                "name": name,
                "x_label": x_label,
                "y_label": y_label,
                "plot_type": plot_type,
            }
        )
        fig = type("MockFig", (), {"line": lambda *args, **kwargs: None})()
        return fig

    def stream_data(self, name: str, new_data: dict, rollover: int) -> None:
        """Mock stream_data."""
        if name not in self.streamed_data:
            self.streamed_data[name] = []
        self.streamed_data[name].append({"data": new_data, "rollover": rollover})

    def push_updates(self) -> None:
        """Mock push_updates."""
        self.push_count += 1


@pytest.fixture
def mock_backend() -> MockBackend:
    """Provide a mock backend."""
    return MockBackend()


@pytest.fixture
def writer(mock_backend: MockBackend) -> BokehLiveWriter:
    """Provide a BokehLiveWriter with mock backend."""
    return BokehLiveWriter(backend=mock_backend, max_points=100)


@pytest.fixture
def session_metadata() -> SessionMetadata:
    """Provide session metadata."""
    return SessionMetadata(
        session_id="test_session",
        routine_name="test_routine",
        start_time=1234567890.0,
        user=getpass.getuser(),
    )


def test_initialization(mock_backend: MockBackend) -> None:
    """Test writer initialization."""
    writer = BokehLiveWriter(backend=mock_backend, max_points=500)

    assert writer.backend is mock_backend
    assert writer.max_points == 500
    assert len(writer._plots) == 0


def test_initialize_session(
    writer: BokehLiveWriter, session_metadata: SessionMetadata
) -> None:
    """Test session initialization."""
    writer.initialize_session(session_metadata)

    assert writer._initialized is True


def test_write_measurement_no_op(
    writer: BokehLiveWriter, session_metadata: SessionMetadata
) -> None:
    """Test that write_measurement is a no-op."""
    writer.initialize_session(session_metadata)

    measurement = MeasurementData(
        name="test",
        data={"current": 1e-9},
        metadata={},
        timestamp=1234567890.0,
        session_id="test_session",
        routine_name="test_routine",
    )

    writer.write_measurement(measurement)


def test_write_sweep_creates_figure(
    writer: BokehLiveWriter,
    mock_backend: MockBackend,
    session_metadata: SessionMetadata,
) -> None:
    """Test that first sweep creates a figure."""
    writer.initialize_session(session_metadata)

    sweep = SweepData(
        name="G1_sweep",
        x_data=np.array([1.0, 2.0, 3.0]),
        y_data=np.array([0.1, 0.2, 0.3]),
        x_label="Voltage (V)",
        y_label="Current (A)",
        metadata={},
        timestamp=1234567890.0,
        session_id="test_session",
    )

    writer.write_sweep(sweep)

    assert len(mock_backend.figures_created) == 1
    assert mock_backend.figures_created[0]["name"] == "G1_sweep"
    assert "G1_sweep" in writer._plots


def test_flush_calls_backend_push(
    writer: BokehLiveWriter,
    mock_backend: MockBackend,
    session_metadata: SessionMetadata,
) -> None:
    """Test that flush calls backend.push_updates()."""
    writer.initialize_session(session_metadata)

    assert mock_backend.push_count == 0

    writer.flush()
    assert mock_backend.push_count == 1

    writer.flush()
    assert mock_backend.push_count == 2


def test_finalize_calls_flush(
    writer: BokehLiveWriter,
    mock_backend: MockBackend,
    session_metadata: SessionMetadata,
) -> None:
    """Test that finalize calls flush."""
    writer.initialize_session(session_metadata)

    assert mock_backend.push_count == 0

    writer.finalize_session(session_metadata)

    assert mock_backend.push_count == 1
