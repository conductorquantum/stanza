from __future__ import annotations

import logging
from pathlib import Path

try:
    import h5py  # type: ignore[import-untyped]
except ImportError:
    h5py = None

import numpy as np

from stanza.exceptions import WriterError
from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData
from stanza.logger.writers.base import AbstractDataWriter
from stanza.timing import to_epoch

logger = logging.getLogger(__name__)


class HDF5Writer(AbstractDataWriter):
    """Writer for HDF5 files."""

    def __init__(
        self,
        base_directory: str | Path,
        compression: str | None = None,
        compression_level: int | None = None,
    ):
        if h5py is None:
            raise ImportError(
                "h5py is not installed. Install with: pip install stanza[hdf5]"
            )
        self.base_directory = Path(base_directory)
        if not self.base_directory.exists():
            self.base_directory.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.compression_level = compression_level
        self._session_file: Path | None = None
        self._h5_file: h5py.File | None = None
        self._measurement_counters: dict[str, int] = {}

        logger.info(
            "Initialized HDF5 writer with base directory: %s", self.base_directory
        )

    def initialize_session(self, session: SessionMetadata) -> None:
        """Initialize the writer for a new session.

        Args:
            session: Session metadata to write

        Raises:
            WriterError: If session is already initialized or file creation fails
        """
        try:
            self.session_id = session.session_id
            session_filename = f"{self.session_id}.h5"
            self._session_file = self.base_directory / session_filename

            # Open HDF5 file and write session metadata
            self._h5_file = h5py.File(self._session_file, "w")

            metadata_group = self._h5_file.create_group("metadata")
            metadata_group.attrs["routine_name"] = (
                session.routine_name if session.routine_name is not None else ""
            )
            metadata_group.attrs["start_time"] = session.start_time
            metadata_group.attrs["session_id"] = session.session_id
            metadata_group.attrs["device_config"] = str(session.device_config)

            self._h5_file.create_group("measurements")
            self._h5_file.create_group("sweeps")
            self._h5_file.create_group("analysis")

            logger.info("Initialized HDF5 writer for session: %s", self.session_id)
        except Exception as e:
            if self._h5_file is not None:
                self._h5_file.close()
                self._h5_file = None
            self._session_file = None
            raise WriterError(
                f"Failed to initialize HDF5 writer for session: {self.session_id}"
            ) from e

    def finalize_session(self, session: SessionMetadata | None = None) -> None:
        """Finalize the writer for a session.

        Args:
            session: Optional updated session metadata to write

        Raises:
            WriterError: If no active session or finalization fails
        """
        if self._session_file is None:
            raise WriterError("No active session")

        try:
            if session is not None and self._h5_file is not None:
                metadata_group = self._h5_file["metadata"]
                if session.end_time is not None:
                    metadata_group.attrs["end_time"] = session.end_time
                if session.parameters is not None:
                    metadata_group.attrs["parameters"] = str(session.parameters)

            if hasattr(self, "_h5_file") and self._h5_file is not None:
                self._h5_file.close()
                self._h5_file = None

            logger.debug("Session finalized: %s", self._session_file)
            self._session_file = None
            self._measurement_counters.clear()

        except Exception as e:
            raise WriterError(
                f"Failed to finalize HDF5 writer for session: {self.session_id}"
            ) from e

    def write_measurement(self, measurement: MeasurementData) -> None:
        """Write a single measurement data point.

        Args:
            measurement: Measurement data to write

        Raises:
            WriterError: If no active session or write operation fails
        """
        if self._session_file is None or self._h5_file is None:
            raise WriterError("No active session")

        try:
            measurement_name = measurement.name
            if measurement_name in self._measurement_counters:
                counter = self._measurement_counters[measurement_name]
                self._measurement_counters[measurement_name] += 1
                measurement_name = f"{measurement_name}_{counter}"
            else:
                self._measurement_counters[measurement_name] = 1

            is_analysis = measurement.metadata.get("data_type") == "analysis"
            target_group = self._h5_file["analysis" if is_analysis else "measurements"]

            measurement_group = target_group.create_group(measurement_name)

            data_group = measurement_group.create_group("data")
            metadata_group = measurement_group.create_group("metadata")

            for key, value in measurement.data.items():
                if isinstance(value, (np.ndarray, list, tuple)):
                    self._create_dataset(data_group, key, np.array(value))
                else:
                    data_group.attrs[key] = value

            for key, value in measurement.metadata.items():
                try:
                    metadata_group.attrs[key] = value
                except (TypeError, ValueError):
                    metadata_group.attrs[key] = str(value)

            # Store measurement-level metadata
            metadata_group.attrs["timestamp"] = to_epoch(measurement.timestamp)
            metadata_group.attrs["session_id"] = measurement.session_id
            metadata_group.attrs["routine_name"] = (
                measurement.routine_name if measurement.routine_name is not None else ""
            )

            logger.info("Wrote measurement: %s", measurement_name)
        except Exception as e:
            raise WriterError(f"Failed to write measurement: {measurement_name}") from e

    def write_sweep(self, sweep: SweepData) -> None:
        """Write sweep data to HDF5 file.

        Args:
            sweep: Sweep data to write

        Raises:
            WriterError: If no active session or write operation fails
        """
        if self._session_file is None or self._h5_file is None:
            raise WriterError("No active session")

        try:
            sweeps_group = self._h5_file["sweeps"]
            sweep_name = self._unique_child_name(sweeps_group, sweep.name)
            sweep_group = sweeps_group.create_group(sweep_name)

            data_group = sweep_group.create_group("data")
            metadata_group = sweep_group.create_group("metadata")

            # Handle x_label which can be str (1D) or list[str] (2D)
            if isinstance(sweep.x_label, list):
                # 2D sweep: x_data has shape (N, 2)
                for i, label in enumerate(sweep.x_label):
                    self._create_dataset(data_group, label, sweep.x_data[:, i])
            else:
                # 1D sweep
                self._create_dataset(data_group, sweep.x_label, sweep.x_data)
            self._create_dataset(data_group, sweep.y_label, sweep.y_data)

            sweep_group.attrs["x_label"] = sweep.x_label
            sweep_group.attrs["y_label"] = sweep.y_label

            for key, value in sweep.metadata.items():
                try:
                    metadata_group.attrs[key] = value
                except (TypeError, ValueError):
                    metadata_group.attrs[key] = str(value)

            metadata_group.attrs["timestamp"] = to_epoch(sweep.timestamp)
            metadata_group.attrs["session_id"] = sweep.session_id

            logger.debug("Wrote sweep: %s", sweep_name)

        except Exception as e:
            raise WriterError(f"Error writing sweep data: {str(e)}") from e

    def flush(self) -> None:
        """Flush data to disk.

        Raises:
            WriterError: If flush operation fails
        """
        if self._h5_file is None:
            raise WriterError("No active session")

        try:
            self._h5_file.flush()
            logger.debug("Flushed data to disk")
        except Exception as e:
            raise WriterError(f"Error flushing data: {str(e)}") from e

    def _create_dataset(self, group: h5py.Group, name: str, data: np.ndarray) -> None:
        """Create a dataset with the configured compression settings."""
        if self.compression is None:
            group.create_dataset(name, data=data)
        else:
            group.create_dataset(
                name,
                data=data,
                compression=self.compression,
                compression_opts=self.compression_level,
            )

    def _unique_child_name(self, group: h5py.Group, base: str) -> str:
        """Return a unique child name within a group based on base.

        If `base` exists, returns `base_<n>` with the smallest available n.
        """
        if base not in group:
            return base
        i = 0
        while f"{base}_{i}" in group:
            i += 1
        return f"{base}_{i}"
