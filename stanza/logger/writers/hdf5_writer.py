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
            f"Initialized HDF5 writer with base directory: {self.base_directory}"
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

            logger.info(f"Initialized HDF5 writer for session: {self.session_id}")
        except Exception as e:
            if self._h5_file is not None:
                self._h5_file.close()
                self._h5_file = None
            self._session_file = None
            raise WriterError(
                f"Failed to initialize HDF5 writer for session: {self.session_id}"
            ) from e

    def finalize_session(self) -> None:
        """Finalize the writer for a session.

        Raises:
            WriterError: If no active session or finalization fails
        """
        if self._session_file is None:
            raise WriterError("No active session")

        try:
            if hasattr(self, "_h5_file") and self._h5_file is not None:
                self._h5_file.close()
                self._h5_file = None

            logger.debug(f"Session finalized: {self._session_file}")
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
                measurement_name = f"{measurement_name}_{counter}"
                self._measurement_counters[measurement_name] += 1
            else:
                self._measurement_counters[measurement_name] = 0

            is_analysis = measurement.metadata.get("data_type") == "analysis"
            target_group = self._h5_file["analysis" if is_analysis else "measurements"]

            measurement_group = target_group.create_group(measurement_name)

            data_group = measurement_group.create_group("data")
            metadata_group = measurement_group.create_group("metadata")

            for key, value in measurement.data.items():
                if isinstance(value, (np.ndarray, list, tuple)):
                    data_array = np.array(value)
                    if self.compression is None:
                        data_group.create_dataset(
                            key,
                            data=data_array,
                        )
                    else:
                        data_group.create_dataset(
                            key,
                            data=data_array,
                            compression=self.compression,
                            compression_opts=self.compression_level,
                        )
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

            logger.info(f"Wrote measurement: {measurement_name}")
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
            sweep_group = sweeps_group.create_group(sweep.name)

            data_group = sweep_group.create_group("data")
            metadata_group = sweep_group.create_group("metadata")

            if self.compression is None:
                data_group.create_dataset(
                    sweep.x_label,
                    data=sweep.x_data,
                )
                data_group.create_dataset(
                    sweep.y_label,
                    data=sweep.y_data,
                )
            else:
                data_group.create_dataset(
                    sweep.x_label,
                    data=sweep.x_data,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                )
                data_group.create_dataset(
                    sweep.y_label,
                    data=sweep.y_data,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                )

            sweep_group.attrs["x_label"] = sweep.x_label
            sweep_group.attrs["y_label"] = sweep.y_label

            for key, value in sweep.metadata.items():
                try:
                    metadata_group.attrs[key] = value
                except (TypeError, ValueError):
                    metadata_group.attrs[key] = str(value)

            metadata_group.attrs["timestamp"] = to_epoch(sweep.timestamp)
            metadata_group.attrs["session_id"] = sweep.session_id

            logger.debug(f"Wrote sweep: {sweep.name}")

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
