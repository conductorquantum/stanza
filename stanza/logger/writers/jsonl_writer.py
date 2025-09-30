import gzip
import json
import logging
from pathlib import Path
from typing import IO, Any

import numpy as np

from stanza.exceptions import WriterError
from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData
from stanza.logger.writers.base import AbstractDataWriter
from stanza.timing import to_epoch

logger = logging.getLogger(__name__)


class JSONLEncoder(json.JSONEncoder):
    """Encoder for JSONL files."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, "__class__") and obj.__class__.__name__ in [
            "Timestamp",
            "datetime",
        ]:
            return to_epoch(obj)
        return super().default(obj)


class JSONLWriter(AbstractDataWriter):
    """Writer for JSONL files."""

    def __init__(
        self,
        base_directory: str | Path,
        compression: str | None = None,
        compression_level: int = 6,
    ):
        self.base_directory = Path(base_directory)
        if not self.base_directory.exists():
            self.base_directory.mkdir(parents=True, exist_ok=True)
        self.compression = compression if compression == "gzip" else None
        self.compression_level = compression_level
        self._session_dir: Path | None = None
        self._file_handles: dict[str, Path] = {}

        logger.debug(
            f"Initialized JSONL writer with base directory: {self.base_directory}"
        )

    def _get_file_extension(self, base_name: str) -> str:
        """Get file extension with optional compression suffix."""
        return f"{base_name}{'.gz' if self.compression else ''}"

    def _open_file(self, file_path: Path, mode: str = "w") -> IO[str]:
        """Open a file with optional compression."""
        if self.compression:
            return gzip.open(  # type: ignore[return-value]
                file_path, mode + "t", compresslevel=self.compression_level
            )
        return open(file_path, mode)

    def initialize_session(self, session: SessionMetadata) -> None:
        """Initialize a new session and create session directory.

        Args:
            session: Session metadata

        Raises:
            WriterError: If session is already initialized or file creation fails
        """
        if self._session_dir is not None:
            raise WriterError("Session already initialized")

        self.session_id = session.session_id
        self._session_dir = self.base_directory
        self._session_dir.mkdir(parents=True, exist_ok=True)

        try:
            metadata_file = self._session_dir / self._get_file_extension(
                "session_metadata.json"
            )
            with self._open_file(metadata_file, "w") as f:
                json.dump(session.to_dict(), f, indent=4, cls=JSONLEncoder)
            self._file_handles["session_metadata"] = metadata_file
            logger.info(f"Initialized JSONL writer for session: {session.session_id}")
        except Exception as e:
            self._session_dir = None
            raise WriterError(f"Error creating session directory: {str(e)}") from e

    def finalize_session(self, session: SessionMetadata | None = None) -> None:
        """Finalize the writer for a session.

        Args:
            session: Optional updated session metadata to write

        Raises:
            WriterError: If no active session or finalization fails
        """
        if self._session_dir is None:
            raise WriterError("No active session")

        try:
            if session is not None:
                metadata_file = self._session_dir / self._get_file_extension(
                    "session_metadata.json"
                )
                with self._open_file(metadata_file, "w") as f:
                    json.dump(session.to_dict(), f, indent=4, cls=JSONLEncoder)

            for handle in self._file_handles.values():
                if hasattr(handle, "close"):
                    handle.close()
            self._file_handles.clear()
            logger.info(f"Finalized JSONL writer for session: {self._session_dir}")
            self._session_dir = None
        except Exception as e:
            raise WriterError(f"Error finalizing session: {str(e)}") from e

    def write_measurement(self, measurement: MeasurementData) -> None:
        """Write a single measurement data point.
        Args:
            measurement: Measurement data to write
        Raises:
            WriterError: If no active session or write operation fails
        """
        if self._session_dir is None:
            raise WriterError("No active session")

        try:
            self._write_measurement(measurement)
            logger.debug(f"Wrote measurement: {measurement.name}")
        except Exception as e:
            raise WriterError(f"Error writing measurement: {str(e)}") from e

    def _write_measurement(self, measurement: MeasurementData) -> None:
        """Write a single measurement data point in structured format."""
        if self._session_dir is None:
            raise WriterError("No active session")
        is_analysis = measurement.metadata.get("data_type") == "analysis"
        base_name = "analysis.jsonl" if is_analysis else "measurement.jsonl"
        jsonl_file = self._session_dir / self._get_file_extension(base_name)

        with self._open_file(jsonl_file, "a") as f:
            json.dump(measurement.to_dict(), f, cls=JSONLEncoder)
            f.write("\n")

    def write_sweep(self, sweep: SweepData) -> None:
        """Write a sweep of measurement data.
        Args:
            sweep: Sweep data to write
        Raises:
            WriterError: If no active session or write operation fails
        """
        if self._session_dir is None:
            raise WriterError("No active session")

        try:
            self._write_sweep(sweep)
            logger.debug(f"Wrote sweep: {sweep.name}")
        except Exception as e:
            raise WriterError(f"Error writing sweep: {str(e)}") from e

    def _write_sweep(self, sweep: SweepData) -> None:
        """Write a sweep of measurement data in structured format."""
        if self._session_dir is None:
            raise WriterError("No active session")
        base_name = "sweep.jsonl"
        jsonl_file = self._session_dir / self._get_file_extension(base_name)

        with self._open_file(jsonl_file, "a") as f:
            json.dump(sweep.to_dict(), f, cls=JSONLEncoder)
            f.write("\n")

    def flush(self) -> None:
        """Flush the writer to ensure data is written to storage."""
        pass
