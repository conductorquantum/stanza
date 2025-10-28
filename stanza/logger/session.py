from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from stanza.exceptions import LoggerSessionError
from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData
from stanza.logger.writers.base import AbstractDataWriter
from stanza.logger.writers.bokeh_writer import BokehLiveWriter

logger = logging.getLogger(__name__)


class SweepContext:
    """Context manager for streaming sweep data with live plotting.

    Accumulates data points via append() calls, streams to live plot writers
    immediately, and writes the complete sweep to file writers on context exit.

    Supports 1D sweeps (scalar x, scalar y) and 2D sweeps ([x0, x1], y).

    Example:
        >>> # 1D sweep
        >>> with session.sweep("signal", "Time", "Amplitude") as s:
        ...     for x, y in data_stream:
        ...         s.append([x], [y])  # Live plot updates

        >>> # 2D sweep
        >>> with session.sweep("heatmap", ["V1", "V2"], "Current") as s:
        ...     for v1, v2 in voltage_pairs:
        ...         s.append([v1, v2], [current])  # Live heatmap updates
        >>> # Sweep written to files here
    """

    def __init__(
        self,
        session: LoggerSession,
        name: str,
        x_label: str | list[str],
        y_label: str,
        metadata: dict[str, Any],
        routine_name: str | None = None,
    ):
        """Initialize sweep context.

        Args:
            session: Parent LoggerSession
            name: Sweep name
            x_label: X-axis label (string for 1D, list for 2D)
            y_label: Y-axis label
            metadata: User metadata dict
            routine_name: Optional routine name
        """
        self.session = session
        self.name = name
        self.x_label = x_label
        self.y_label = y_label
        # Defensive copy to avoid caller mutation affecting persisted record
        self.metadata = dict(metadata) if metadata else {}
        self.routine_name = routine_name

        self._x_data: list[float] | list[list[float]] = []
        self._y_data: list[float] = []
        self._dim: int | None = None  # Detect on first append
        self._active = True

    def append(
        self, x_data: list[float] | np.ndarray, y_data: list[float] | np.ndarray
    ) -> None:
        """Append data points and stream to live plots.

        1D: append([x], [y]) or append([x1, x2, ...], [y1, y2, ...])
        2D: append([x, y], [value])
        """
        if not self._active:
            raise ValueError(f"Sweep '{self.name}' is no longer active")

        x_arr = np.asarray(x_data, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)

        if x_arr.size == 0 or y_arr.size == 0:
            raise ValueError("x_data and y_data cannot be empty")

        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError("x_data and y_data must be 1D arrays")

        if not (np.isfinite(x_arr).all() and np.isfinite(y_arr).all()):
            raise ValueError("x_data and y_data cannot contain NaN or inf")

        # Detect dimension on first append
        if self._dim is None:
            expecting_2d = isinstance(self.x_label, list) and len(self.x_label) == 2

            if len(y_arr) == 1:
                # Single y value: determine dim from x length
                self._dim = len(x_arr)
                if expecting_2d and self._dim != 2:
                    raise ValueError(
                        f"x_label has 2 elements but x_data has {self._dim}"
                    )
            elif len(x_arr) == len(y_arr):
                # Matching lengths: 1D batch mode
                if expecting_2d:
                    raise ValueError("For 2D sweeps, y_data must be length 1")
                self._dim = 1
            else:
                raise ValueError(
                    f"Ambiguous: x has {len(x_arr)} values, y has {len(y_arr)}. "
                    f"Use append([x], [y]) for 1D or append([x, y], [value]) for 2D."
                )

            if self._dim not in (1, 2):
                raise ValueError(f"Only 1D and 2D supported, got {self._dim}D")

            self.metadata["_dim"] = self._dim

        # Validate and accumulate
        if self._dim == 1:
            if len(x_arr) != len(y_arr):
                raise ValueError(f"Length mismatch: {len(x_arr)} vs {len(y_arr)}")
            self._x_data.extend(x_arr.tolist())
            self._y_data.extend(y_arr.tolist())
        else:  # 2D
            if len(x_arr) != 2 or len(y_arr) != 1:
                raise ValueError("2D requires len(x)==2, len(y)==1")
            self._x_data.append(x_arr.tolist())
            self._y_data.append(float(y_arr[0]))

        self.session._stream_live_sweep_chunk(
            self.name, x_arr, y_arr, self.x_label, self.y_label, self._dim
        )

    def end(self) -> None:
        """Write accumulated sweep to file writers."""
        if not self._active:
            return

        self._active = False
        self.session._active_sweeps.pop(self.name, None)

        if not self._x_data:
            logger.debug("Sweep '%s' empty, skipping write", self.name)
            return

        # Filter internal metadata (keys starting with _)
        user_metadata = {
            k: v for k, v in self.metadata.items() if not k.startswith("_")
        }

        sweep_data = SweepData(
            name=self.name,
            x_data=np.array(self._x_data, dtype=float),
            y_data=np.array(self._y_data, dtype=float),
            x_label=self.x_label,
            y_label=self.y_label,
            metadata=user_metadata,
            timestamp=time.time(),
            session_id=self.session.session_id,
            routine_name=self.routine_name,
        )

        self.session._write_completed_sweep(sweep_data)

    def cancel(self) -> None:
        """Cancel without persisting. Live updates remain visible."""
        self._active = False
        self.session._active_sweeps.pop(self.name, None)
        self._x_data.clear()
        self._y_data.clear()

    def __enter__(self) -> SweepContext:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Exit context manager: end() on success, cancel() on exception."""
        if not self._active:
            return
        if exc_type is None:
            self.end()
        else:
            self.cancel()


class LoggerSession:
    """Session for the logger."""

    def __init__(
        self,
        metadata: SessionMetadata,
        writer_pool: dict[str, AbstractDataWriter],
        writer_refs: list[str],
        base_dir: str | Path,
        buffer_size: int = 1000,
        auto_flush_interval: float | None = 30.0,
    ):
        if not writer_pool or not writer_refs:
            raise LoggerSessionError("Writer pool and references are required")

        base_path = Path(base_dir)
        if not base_path.exists():
            raise LoggerSessionError(f"Base directory does not exist: {base_dir}")

        self.metadata = metadata
        self._writer_pool = writer_pool
        self._writer_refs = writer_refs
        self._base_dir = base_path
        self._buffer_size = buffer_size
        self._auto_flush_interval = auto_flush_interval

        self._active = False
        self._buffer: list[MeasurementData | SweepData] = []
        self._last_flush_time = time.monotonic()
        self._buffer_size_warning_threshold = buffer_size * 10
        self._buffer_size_warned = False
        self._live_writers: list[BokehLiveWriter] = []
        self._active_sweeps: dict[str, SweepContext] = {}

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    def _check_buffer_size_warning(self) -> None:
        """Check if buffer has grown too large and log warning once."""
        if (
            not self._buffer_size_warned
            and len(self._buffer) >= self._buffer_size_warning_threshold
        ):
            logger.warning(
                "Buffer size critical for session %s: %d items (threshold: %d). "
                "Consider flushing more frequently or increasing buffer size.",
                self.session_id,
                len(self._buffer),
                self._buffer_size_warning_threshold,
            )
            self._buffer_size_warned = True

    def initialize(self) -> None:
        """Initialize the session.

        Raises:
            LoggerSessionError: If session is already initialized
        """
        if self._active:
            raise LoggerSessionError("Session is already initialized")

        initialized_refs: list[str] = []
        try:
            for writer_ref in self._writer_refs:
                writer = self._writer_pool[writer_ref]
                writer.initialize_session(self.metadata)
                initialized_refs.append(writer_ref)

            # Cache live writers for streaming sweeps
            self._live_writers = [
                w
                for ref in self._writer_refs
                if isinstance(w := self._writer_pool[ref], BokehLiveWriter)
            ]

            self._active = True
            self._last_flush_time = time.monotonic()

        except Exception as e:
            # Cleanup any writers that initialized successfully
            for ref in initialized_refs:
                try:
                    self._writer_pool[ref].finalize_session(self.metadata)
                except Exception:
                    pass
            self._live_writers = []
            self._active = False
            raise LoggerSessionError(f"Failed to initialize session: {str(e)}") from e

    def finalize(self) -> None:
        """Finalize the session."""
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        try:
            if self._active_sweeps:
                logger.warning(
                    "Auto-completing %d active sweeps: %s",
                    len(self._active_sweeps),
                    list(self._active_sweeps.keys()),
                )
                for sweep in list(self._active_sweeps.values()):
                    try:
                        sweep.end()
                    except Exception as e:
                        logger.error(
                            "Failed to auto-complete sweep '%s': %s", sweep.name, e
                        )

            self.flush()

            for ref in self._writer_refs:
                self._writer_pool[ref].finalize_session(self.metadata)

            self._active = False

        except Exception as e:
            self._active = False
            raise LoggerSessionError(f"Failed to finalize session: {str(e)}") from e

    def close(self) -> None:
        """Close the session (alias for finalize())."""
        self.finalize()

    def log_measurement(
        self,
        name: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        routine_name: str | None = None,
    ) -> None:
        """Log measurement data to a session."""
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        if not name or not name.strip():
            raise LoggerSessionError("Measurement name cannot be empty")

        if not data:
            raise LoggerSessionError("Measurement data cannot be empty")

        measurement = MeasurementData(
            name=name,
            data=data,
            metadata=metadata or {},
            timestamp=time.time(),
            session_id=self.session_id,
            routine_name=routine_name,
        )

        self._buffer.append(measurement)
        self._check_buffer_size_warning()

        if self._should_flush():
            self.flush()

    def log_analysis(
        self,
        name: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        routine_name: str | None = None,
    ) -> None:
        """Log analysis data to a session."""
        meta = metadata.copy() if metadata else {}
        meta["data_type"] = "analysis"
        self.log_measurement(name, data, meta, routine_name)

    def log_sweep(
        self,
        name: str,
        x_data: list[float] | list[list[float]] | np.ndarray,
        y_data: list[float] | np.ndarray,
        x_label: str,
        y_label: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log sweep data to a session."""
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        if not name or not name.strip():
            raise LoggerSessionError("Sweep name cannot be empty")

        x_array = np.asarray(x_data)
        y_array = np.asarray(y_data)

        if x_array.size == 0 or y_array.size == 0:
            raise LoggerSessionError("Sweep data cannot be empty")

        sweep = SweepData(
            name=name,
            x_data=x_array,
            y_data=y_array,
            x_label=x_label,
            y_label=y_label,
            metadata=metadata or {},
            timestamp=time.time(),
            session_id=self.session_id,
        )

        self._buffer.append(sweep)
        self._check_buffer_size_warning()

        if self._should_flush():
            self.flush()

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Log parameters to a session."""
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        if not parameters:
            raise LoggerSessionError("Parameters cannot be empty")

        if self.metadata.parameters is None:
            self.metadata.parameters = {}
        self.metadata.parameters.update(parameters)

    def sweep(
        self,
        name: str,
        x_label: str | list[str],
        y_label: str,
        metadata: dict[str, Any] | None = None,
    ) -> SweepContext:
        """Create streaming sweep context for live plotting.

        Accumulates data via append(), writes complete sweep to files on exit.
        Supports 1D and 2D data. Exception in context = no file write.
        One active sweep per name; HDF5 auto-uniquifies sequential same-name sweeps.

        Example:
            # 1D sweep
            with session.sweep("signal", "Time (s)", "Amplitude") as s:
                for x, y in data_stream:
                    s.append([x], [y])

            # 2D sweep
            with session.sweep("heatmap", ["V1 (V)", "V2 (V)"], "Current (A)") as s:
                for v1, v2 in voltage_pairs:
                    s.append([v1, v2], [current])
        """
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        if name in self._active_sweeps:
            raise LoggerSessionError(
                f"Sweep '{name}' is already active. Call end() or use a different name."
            )

        context = SweepContext(
            session=self,
            name=name,
            x_label=x_label,
            y_label=y_label,
            metadata=metadata or {},
            routine_name=self.metadata.routine_name,
        )

        self._active_sweeps[name] = context
        return context

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        return (
            len(self._buffer) >= self._buffer_size
            or self._auto_flush_interval is not None
            and time.monotonic() - self._last_flush_time >= self._auto_flush_interval
        )

    def _stream_live_sweep_chunk(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        x_label: str | list[str],
        y_label: str,
        dim: int,
    ) -> None:
        """Stream data chunk to live plot writers only."""
        if not self._live_writers:
            return

        # Reshape x_data for consistent SweepData format
        x_data: np.ndarray
        if dim == 1:
            x_data = x.reshape(-1)
        else:  # dim == 2
            x_data = x.reshape(1, -1)

        temp_sweep = SweepData(
            name=name,
            x_data=x_data,
            y_data=y,
            x_label=x_label,
            y_label=y_label,
            metadata={"_dim": dim},
            timestamp=time.time(),
            session_id=self.session_id,
        )

        for writer in self._live_writers:
            try:
                writer.write_sweep(temp_sweep)
            except Exception as e:
                logger.warning("Live plot stream failed: %s", e)

    def _write_completed_sweep(self, sweep_data: SweepData) -> None:
        """Write complete sweep to file writers only (skip live/bokeh)."""
        for ref in self._writer_refs:
            if ref == "bokeh":
                continue
            try:
                self._writer_pool[ref].write_sweep(sweep_data)
                self._writer_pool[ref].flush()
            except Exception as e:
                logger.error("Failed to write sweep to %s: %s", ref, e)
                raise LoggerSessionError(f"Sweep write failed: {ref}: {e}") from e

    def flush(self) -> None:
        """Flush buffered data to all writers.

        Raises:
            LoggerSessionError: If any writer fails to write or flush data
        """
        if not self._buffer:
            return

        errors = []

        for item in self._buffer:
            for ref in self._writer_refs:
                writer = self._writer_pool[ref]
                try:
                    if isinstance(item, MeasurementData):
                        writer.write_measurement(item)
                    elif isinstance(item, SweepData):
                        writer.write_sweep(item)
                    else:
                        raise LoggerSessionError(f"Invalid item type: {type(item)}")
                except Exception as e:  # noqa: BLE001
                    logger.error("Write failed to %s: %s", ref, e)
                    errors.append(f"{ref}: {e}")

        for ref in self._writer_refs:
            try:
                self._writer_pool[ref].flush()
            except Exception as e:  # noqa: BLE001
                logger.error("Flush failed for %s: %s", ref, e)
                errors.append(f"{ref}: {e}")

        if errors:
            raise LoggerSessionError(f"Flush failed: {errors[0]}")

        self._buffer.clear()
        self._last_flush_time = time.monotonic()
        self._buffer_size_warned = False

    def __enter__(self) -> LoggerSession:
        """Enter the session context."""
        if not self._active:
            self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Exit the session context."""
        try:
            if self._active:
                self.finalize()
        except Exception as e:
            logger.error("Failed to finalize session: %s", str(e))
            raise LoggerSessionError(f"Failed to finalize session: {str(e)}") from e

    def __repr__(self) -> str:
        return (
            f"LoggerSession(session_id={self.session_id}, "
            f"routine_name={self.metadata.routine_name}), active={self._active}, "
            f"buffer={len(self._buffer)}/{self._buffer_size}, "
            f"active_sweeps={len(self._active_sweeps)})"
        )
