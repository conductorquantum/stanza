"""Shared utilities for plotting backends."""

from __future__ import annotations

from typing import Any


def prepare_heatmap_data(
    data: dict[str, Any],
    existing_data: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Calculate rect sizes and update color range for heatmap data.

    Args:
        data: New data to add (x, y, value)
        existing_data: Existing data in the data source
        spec: Plot specification with dx, dy, value_min, value_max

    Returns:
        Updated data with width and height fields added
    """
    import numpy as np

    def calc_delta(key: str) -> float:
        """Calculate minimum delta from existing + new data."""
        if key not in data or len(data[key]) == 0:
            return 0.1
        existing_vals = list(existing_data.get(key, []))
        all_vals = existing_vals + data[key]
        if len(all_vals) > 1:
            unique = sorted(set(all_vals))
            if len(unique) > 1:
                return float(min(np.diff(unique)))
        return 0.1

    # Calculate cell sizes if not set
    if spec["dx"] is None:
        spec["dx"] = calc_delta("x")
    if spec["dy"] is None:
        spec["dy"] = calc_delta("y")

    # Add width/height to data
    n = len(data.get("value", []))
    data["width"] = [spec["dx"]] * n
    data["height"] = [spec["dy"]] * n

    # Update value range for color mapping
    if "value" in data:
        values = np.array(data["value"])
        spec["value_min"] = min(spec["value_min"], float(values.min()))
        spec["value_max"] = max(spec["value_max"], float(values.max()))

    return data
