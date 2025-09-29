from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MeasurementData:
    """Represents a single measurement point."""

    name: str
    data: dict[str, Any]
    metadata: dict[str, Any]
    timestamp: float
    session_id: str
    routine_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert measurement data to dictionary for serialization."""

        return {
            "name": self.name,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "routine_name": self.routine_name,
        }


@dataclass
class SweepData:
    """Represents a sweep of measurement data."""

    name: str
    x_data: np.ndarray
    y_data: np.ndarray
    x_label: str
    y_label: str
    metadata: dict[str, Any]
    timestamp: float
    session_id: str

    def __post_init__(self) -> None:
        """Initialize the sweep data."""
        if self.x_data.shape != self.y_data.shape:
            raise ValueError("x_data and y_data must have the same shape")

    def to_dict(self) -> dict[str, Any]:
        """Convert sweep data to dictionary for serialization."""
        return {
            "name": self.name,
            "x_data": self.x_data.tolist(),
            "y_data": self.y_data.tolist(),
            "x_label": self.x_label,
            "y_label": self.y_label,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }


@dataclass
class SessionMetadata:
    """Session-level metadata."""

    session_id: str
    start_time: float
    user: str
    routine_name: str | None = None
    device_config: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    end_time: float | None = None
    git_commit: str | None = None

    @property
    def duration(self) -> float | None:
        """Calculate the duration of the session."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert session metadata to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "routine_name": self.routine_name,
            "start_time": self.start_time,
            "user": self.user,
            "device_config": self.device_config,
            "parameters": self.parameters,
            "end_time": self.end_time,
            "git_commit": self.git_commit,
            "duration": self.duration,
        }
