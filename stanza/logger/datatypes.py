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
        """Convert to dictionary for serialization."""
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
    """Represents a sweep of measurement data.

    Supports both 1D and 2D sweeps:
    - 1D: x_data.shape = (N,), y_data.shape = (N,)
    - 2D: x_data.shape = (N, 2), y_data.shape = (N,)
    """

    name: str
    x_data: np.ndarray
    y_data: np.ndarray
    x_label: str | list[str]
    y_label: str
    metadata: dict[str, Any]
    timestamp: float
    session_id: str
    routine_name: str | None = None

    def __post_init__(self) -> None:
        """Validate sweep data dimensions."""
        if self.y_data.ndim != 1:
            raise ValueError(f"y_data must be 1D, got shape {self.y_data.shape}")

        if len(self.x_data) != len(self.y_data):
            raise ValueError(
                f"Length mismatch: x_data={len(self.x_data)}, y_data={len(self.y_data)}"
            )

        if self.x_data.ndim == 2 and isinstance(self.x_label, list):
            if len(self.x_label) != self.x_data.shape[1]:
                raise ValueError(
                    f"x_label length {len(self.x_label)} != x_data dims {self.x_data.shape[1]}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "x_data": self.x_data.tolist(),
            "y_data": self.y_data.tolist(),
            "x_label": self.x_label,
            "y_label": self.y_label,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "routine_name": self.routine_name,
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
        """Session duration in seconds."""
        return None if self.end_time is None else self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
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
