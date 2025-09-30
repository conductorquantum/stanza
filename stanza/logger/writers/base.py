from __future__ import annotations

from abc import ABC, abstractmethod

from stanza.logger.datatypes import MeasurementData, SessionMetadata, SweepData


class AbstractDataWriter(ABC):
    """Abstract base class for data writers."""

    @abstractmethod
    def initialize_session(self, session: SessionMetadata) -> None:
        """Initialize the writer for a new session.

        Args:
            session: Session metadata containing routine info

        Raises:
            WriterError: If initialization fails
        """
        pass

    @abstractmethod
    def write_measurement(self, data: MeasurementData) -> None:
        """Write a single measurement data point.

        Args:
            data: Measurement data to write

        Raises:
            WriterError: If write operation fails
        """
        pass

    @abstractmethod
    def write_sweep(self, data: SweepData) -> None:
        """Write a sweep of measurement data.

        Args:
            data: Sweep data to write

        Raises:
            WriterError: If write operation fails
        """
        pass

    @abstractmethod
    def finalize_session(self, session: SessionMetadata | None = None) -> None:
        """Finalize the writer for a session.

        Args:
            session: Optional updated session metadata to write before finalizing

        Raises:
            WriterError: If finalization fails
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush the writer to ensure data is written to storage.

        Raises:
            WriterError: If flush operation fails
        """
        pass
