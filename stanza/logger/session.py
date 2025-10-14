from __future__ import annotations

import logging
import threading
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
        max_auto_flush_failures: int = 5,
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
        self._max_auto_flush_failures = max_auto_flush_failures

        self._active = False
        self._buffer: list[MeasurementData | SweepData] = []
        self._flush_lock = threading.Lock()
        self._flush_timer: threading.Timer | None = None
        self._consecutive_flush_failures = 0

        logger.debug("Created session: %s", self.metadata.session_id)

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def routine_name(self) -> str:
        return self.metadata.routine_name or ""

    def _schedule_auto_flush(self) -> None:
        """Schedule the next auto-flush if enabled."""
        if self._auto_flush_interval is not None and self._auto_flush_interval > 0:
            self._flush_timer = threading.Timer(
                self._auto_flush_interval, self._auto_flush_callback
            )
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _cancel_auto_flush(self) -> None:
        """Cancel any pending auto-flush timer."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def _auto_flush_callback(self) -> None:
        """Callback for periodic auto-flush."""
        should_reschedule = True
        try:
            if self._active and len(self._buffer) > 0:
                logger.debug(
                    "Auto-flushing %s buffered items for session %s",
                    len(self._buffer),
                    self.session_id,
                )
                self.flush()
                self._consecutive_flush_failures = 0  # Reset on success
        except Exception as e:  # noqa: BLE001
            self._consecutive_flush_failures += 1
            logger.error(
                "Auto-flush failed for session %s (%d/%d): %s",
                self.session_id,
                self._consecutive_flush_failures,
                self._max_auto_flush_failures,
                str(e),
            )
            if self._consecutive_flush_failures >= self._max_auto_flush_failures:
                logger.critical(
                    "Max auto-flush failures reached for session %s, stopping auto-flush",
                    self.session_id,
                )
                should_reschedule = False
        finally:
            if self._active and should_reschedule:
                self._schedule_auto_flush()

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
            self._consecutive_flush_failures = 0  # Reset on initialization
            self._schedule_auto_flush()
            logger.info("Initialized session: %s", self.session_id)

        except Exception as e:
            self._active = False
            raise LoggerSessionError(f"Failed to initialize session: {str(e)}") from e

    def finalize(self) -> None:
        """Finalize the session."""
        if not self._active:
            raise LoggerSessionError("Session is not initialized")

        try:
            self._cancel_auto_flush()
            self.flush()

            for writer_ref in self._writer_refs:
                writer = self._writer_pool[writer_ref]
                writer.finalize_session(self.metadata)

            self._active = False
            logger.info("Finalized session: %s", self.session_id)

        except Exception as e:
            self._active = False
            self._cancel_auto_flush()
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

        if len(self._buffer) >= self._buffer_size:
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

        if len(self._buffer) >= self._buffer_size:
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
        with self._flush_lock:
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
                        error_msg = (
                            f"Failed to write data to writer {writer_ref}: {str(e)}"
                        )
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

            count = len(self._buffer)
            self._buffer.clear()
            self._consecutive_flush_failures = 0  # Reset on successful manual flush

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
