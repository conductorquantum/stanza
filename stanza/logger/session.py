from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from stanza.exceptions import LoggerSessionError
from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData
from stanza.logger.writers.base import AbstractDataWriter

logger = logging.getLogger(__name__)


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
        self._last_flush_time = time.time()
        self._buffer_size_warning_threshold = buffer_size * 10
        self._buffer_size_warned = False

        logger.debug("Created session: %s", self.metadata.session_id)

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def routine_name(self) -> str:
        return self.metadata.routine_name or ""

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

        try:
            for writer_ref in self._writer_refs:
                writer = self._writer_pool[writer_ref]
                writer.initialize_session(self.metadata)

            self._active = True
            self._last_flush_time = time.time()
            logger.info("Initialized session: %s", self.session_id)

        except Exception as e:
            self._active = False
            raise LoggerSessionError(f"Failed to initialize session: {str(e)}") from e

    def finalize(self) -> None:
        """Finalize the session."""
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        try:
            self.flush()

            for writer_ref in self._writer_refs:
                writer = self._writer_pool[writer_ref]
                writer.finalize_session(self.metadata)

            self._active = False
            logger.info("Finalized session: %s", self.session_id)

        except Exception as e:
            self._active = False
            raise LoggerSessionError(f"Failed to finalize session: {str(e)}") from e

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

        if not name or name.strip() == "":
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

        should_flush = len(self._buffer) >= self._buffer_size or (
            self._auto_flush_interval is not None
            and time.time() - self._last_flush_time >= self._auto_flush_interval
        )

        if should_flush:
            self.flush()

        logger.debug("Logged measurement '%s' to session %s", name, self.session_id)

    def log_analysis(
        self,
        name: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        routine_name: str | None = None,
    ) -> None:
        """Log analysis data to a session."""
        analysis_metadata = (metadata or {}).copy()
        analysis_metadata["data_type"] = "analysis"
        return self.log_measurement(name, data, analysis_metadata, routine_name)

    def log_sweep(
        self,
        name: str,
        x_data: list[float] | np.ndarray,
        y_data: list[float] | np.ndarray,
        x_label: str,
        y_label: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log sweep data to a session."""
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        if not name or name.strip() == "":
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

        should_flush = len(self._buffer) >= self._buffer_size or (
            self._auto_flush_interval is not None
            and time.time() - self._last_flush_time >= self._auto_flush_interval
        )

        if should_flush:
            self.flush()

        logger.debug("Logged sweep '%s' to session %s", name, self.session_id)

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Log parameters to a session."""
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        if not parameters:
            raise LoggerSessionError("Parameters cannot be empty")

        if self.metadata.parameters is None:
            self.metadata.parameters = {}
        self.metadata.parameters.update(parameters)

    def flush(self) -> None:
        """Flush buffered data to all writers.

        Raises:
            LoggerSessionError: If any writer fails to write or flush data
        """
        if not self._buffer:
            return

        write_errors = []
        flush_errors = []

        for item in self._buffer:
            for writer_ref in self._writer_refs:
                writer = self._writer_pool[writer_ref]
                try:
                    if isinstance(item, MeasurementData):
                        writer.write_measurement(item)
                    elif isinstance(item, SweepData):
                        writer.write_sweep(item)
                    else:
                        raise LoggerSessionError(f"Invalid item type: {type(item)}")
                except Exception as e:  # noqa: BLE001
                    error_msg = f"Failed to write data to writer {writer_ref}: {str(e)}"
                    logger.error(error_msg)
                    write_errors.append(error_msg)

        for writer_ref in self._writer_refs:
            writer = self._writer_pool[writer_ref]
            try:
                writer.flush()
            except Exception as e:  # noqa: BLE001
                error_msg = f"Failed to flush data to writer {writer_ref}: {str(e)}"
                logger.error(error_msg)
                flush_errors.append(error_msg)

        # If there were any errors, don't clear buffer and raise exception
        if write_errors or flush_errors:
            all_errors = write_errors + flush_errors
            raise LoggerSessionError(
                f"Flush failed with {len(all_errors)} error(s): {all_errors[0]}"
            )

        # Clear buffer and mark success
        count = len(self._buffer)
        self._buffer.clear()
        self._last_flush_time = time.time()
        self._buffer_size_warned = False  # Reset warning flag when buffer clears
        logger.debug("Flushed %s items to session %s", count, self.session_id)

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
            f"routine_name={self.routine_name}), active={self._active}, "
            f"buffer={len(self._buffer)}/{self._buffer_size})"
        )
