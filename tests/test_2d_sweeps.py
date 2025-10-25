"""Tests for 2D sweep functionality."""

import tempfile

import numpy as np
import pytest

from stanza.logger.data_logger import DataLogger
from stanza.logger.datatypes import SweepData


def test_sweep_context_2d_dimension_detection():
    """Test that dimension is detected from first append."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_2d", base_dir=tmpdir)
    session = logger.create_session()

    # 1D sweep
    with session.sweep("1d", "X", "Y") as s:
        s.append([1.0], [2.0])
        assert s._dim == 1

    # 2D sweep
    with session.sweep("2d", ["X1", "X2"], "Y") as s:
        s.append([1.0, 2.0], [3.0])
        assert s._dim == 2

    session.close()


def test_sweep_context_2d_dimension_consistency():
    """Test that dimension is enforced after first append."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_dim_consistency", base_dir=tmpdir)
    session = logger.create_session()

    # Start with 2D
    with pytest.raises(ValueError, match="2D requires len"):
        with session.sweep("inconsistent", ["X1", "X2"], "Y") as s:
            s.append([1.0, 2.0], [3.0])  # Sets dim=2
            s.append([1.0], [2.0])  # Mismatch!

    session.close()


def test_sweep_context_2d_data_accumulation():
    """Test that 2D data accumulates correctly."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_accumulation", base_dir=tmpdir)
    session = logger.create_session()

    with session.sweep("2d_accum", ["V1", "V2"], "I") as s:
        for v1 in [-1.0, 0.0, 1.0]:
            for v2 in [-0.5, 0.5]:
                i = v1**2 + v2**2
                s.append([v1, v2], [i])

    # Verify data was accumulated
    assert len(s._x_data) == 6
    assert len(s._y_data) == 6

    # Verify first and last points
    assert s._x_data[0] == [-1.0, -0.5]
    assert s._x_data[-1] == [1.0, 0.5]

    session.close()


def test_sweep_context_2d_validation():
    """Test validation of 2D sweep data."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_validation", base_dir=tmpdir)
    session = logger.create_session()

    # Test NaN rejection
    with pytest.raises(ValueError, match="NaN or inf"):
        with session.sweep("nan_test", ["X1", "X2"], "Y") as s:
            s.append([1.0, np.nan], [2.0])

    # Test empty data rejection
    with pytest.raises(ValueError, match="cannot be empty"):
        with session.sweep("empty_test", ["X1", "X2"], "Y") as s:
            s.append([], [])

    # Test y must be scalar
    with pytest.raises(ValueError, match="y_data must be length 1"):
        with session.sweep("y_scalar_test", ["X1", "X2"], "Y") as s:
            s.append([1.0, 2.0], [3.0, 4.0])

    # Test 3D not supported
    with pytest.raises(ValueError, match="Only 1D and 2D supported"):
        with session.sweep("3d_test", ["X1", "X2", "X3"], "Y") as s:
            s.append([1.0, 2.0, 3.0], [4.0])

    session.close()


def test_sweep_data_2d_validation():
    """Test SweepData validation for 2D data."""
    # Valid 2D sweep
    sweep = SweepData(
        name="test",
        x_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        y_data=np.array([5.0, 6.0]),
        x_label=["X1", "X2"],
        y_label="Y",
        metadata={},
        timestamp=0.0,
        session_id="test",
    )
    assert sweep.x_data.shape == (2, 2)

    # Invalid: x_label length mismatch
    with pytest.raises(ValueError, match="x_label length"):
        SweepData(
            name="test",
            x_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
            y_data=np.array([5.0, 6.0]),
            x_label=["X1"],  # Should be 2 labels!
            y_label="Y",
            metadata={},
            timestamp=0.0,
            session_id="test",
        )

    # Invalid: y_data not 1D
    with pytest.raises(ValueError, match="y_data must be 1D"):
        SweepData(
            name="test",
            x_data=np.array([[1.0, 2.0]]),
            y_data=np.array([[5.0]]),  # 2D!
            x_label=["X1", "X2"],
            y_label="Y",
            metadata={},
            timestamp=0.0,
            session_id="test",
        )

    # Invalid: length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        SweepData(
            name="test",
            x_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
            y_data=np.array([5.0]),  # Wrong length!
            x_label=["X1", "X2"],
            y_label="Y",
            metadata={},
            timestamp=0.0,
            session_id="test",
        )


def test_sweep_context_2d_with_live_plotting():
    """Test 2D sweep with Bokeh live plotting."""
    from stanza.plotter import enable_live_plotting

    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_2d_live", base_dir=tmpdir)

    # Enable inline backend (mock - won't actually display)
    try:
        _backend = enable_live_plotting(logger, backend="inline")
    except ImportError:
        pytest.skip("jupyter_bokeh not installed")

    session = logger.create_session()

    # Stream 2D data
    with session.sweep("heatmap", ["Gate 1 (V)", "Gate 2 (V)"], "Current (A)") as s:
        for v1 in np.linspace(-1, 1, 5):
            for v2 in np.linspace(-1, 1, 5):
                current = v1**2 + v2**2
                s.append([v1, v2], [current])

    # Verify sweep completed
    assert len(s._x_data) == 25
    assert len(s._y_data) == 25

    session.close()


def test_sweep_context_2d_metadata():
    """Test that dimension metadata is stored correctly."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_metadata", base_dir=tmpdir)
    session = logger.create_session()

    with session.sweep("2d_meta", ["X1", "X2"], "Y") as s:
        s.append([1.0, 2.0], [3.0])
        assert s.metadata["_dim"] == 2

    session.close()


def test_sweep_context_2d_cell_size_override():
    """Test cell_size metadata for explicit rect sizing."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_cell_size", base_dir=tmpdir)
    session = logger.create_session()

    # Provide explicit cell size
    with session.sweep(
        "2d_sized",
        ["X1", "X2"],
        "Y",
        metadata={"cell_size": (0.01, 0.02)},
    ) as s:
        s.append([1.0, 2.0], [3.0])
        assert s.metadata["cell_size"] == (0.01, 0.02)

    session.close()


def test_sweep_context_2d_empty_no_write():
    """Test that empty 2D sweep doesn't write."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_empty", base_dir=tmpdir)
    session = logger.create_session()

    with session.sweep("empty_2d", ["X1", "X2"], "Y") as _s:
        pass  # No appends

    # Should not raise, just skip
    session.close()


def test_sweep_context_2d_cancel():
    """Test canceling a 2D sweep."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_cancel", base_dir=tmpdir)
    session = logger.create_session()

    with session.sweep("cancel_2d", ["X1", "X2"], "Y") as s:
        s.append([1.0, 2.0], [3.0])
        s.cancel()

    # Data should be cleared
    assert len(s._x_data) == 0
    assert len(s._y_data) == 0
    assert not s._active

    session.close()


def test_sweep_context_2d_exception_no_write():
    """Test that exception in context prevents write."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_exception", base_dir=tmpdir)
    session = logger.create_session()

    try:
        with session.sweep("exception_2d", ["X1", "X2"], "Y") as s:
            s.append([1.0, 2.0], [3.0])
            raise RuntimeError("Test error")
    except RuntimeError:
        pass

    # Sweep should have been canceled
    assert not s._active

    session.close()


def test_bokeh_writer_dimension_detection():
    """Test that BokehLiveWriter correctly detects 1D vs 2D."""
    from stanza.logger.writers.bokeh_writer import BokehLiveWriter
    from stanza.plotter.backends.server import ServerBackend

    backend = ServerBackend(port=5007)
    backend.start()

    writer = BokehLiveWriter(backend=backend, max_points=1000)
    writer.initialize_session(None)

    # 1D sweep
    sweep_1d = SweepData(
        name="1d_test",
        x_data=np.array([1.0, 2.0, 3.0]),
        y_data=np.array([4.0, 5.0, 6.0]),
        x_label="X",
        y_label="Y",
        metadata={"_dim": 1},
        timestamp=0.0,
        session_id="test",
    )
    writer.write_sweep(sweep_1d)
    assert writer._plots["1d_test"]["dim"] == 1

    # 2D sweep
    sweep_2d = SweepData(
        name="2d_test",
        x_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        y_data=np.array([5.0, 6.0]),
        x_label=["X1", "X2"],
        y_label="Y",
        metadata={"_dim": 2},
        timestamp=0.0,
        session_id="test",
    )
    writer.write_sweep(sweep_2d)
    assert writer._plots["2d_test"]["dim"] == 2

    backend.stop()


def test_sweep_context_active_sweep_collision():
    """Test that same-name sweeps cannot be active simultaneously."""
    tmpdir = tempfile.mkdtemp()
    logger = DataLogger(routine_name="test_collision", base_dir=tmpdir)
    session = logger.create_session()

    with session.sweep("test", ["X1", "X2"], "Y") as _s1:
        # Try to start another with same name
        with pytest.raises(Exception, match="already active"):
            with session.sweep("test", ["X1", "X2"], "Y") as _s2:
                pass

    # After first ends, should be able to start another
    with session.sweep("test", ["X1", "X2"], "Y") as s2:
        s2.append([1.0, 2.0], [3.0])

    session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
