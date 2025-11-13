"""Utility functions for built-in routines."""

from typing import Any

from stanza.logger.session import LoggerSession
from stanza.routines.core import RoutineContext


def filter_gates_by_group(ctx: RoutineContext, gate_list: list[str]) -> list[str]:
    """Filter a list of gates to only include those in the current group.

    If group is available in ctx.resources, filters gates to only
    include those present in the group. Otherwise, returns the original list.

    Args:
        ctx: Routine context containing device resources and optional group.
        gate_list: List of gate names to filter.

    Returns:
        Filtered list of gates that are in the current group, or original list
        if no group filtering is active.
    """
    group = getattr(ctx.resources, "group", None)
    if group is not None:
        group_gates = set(group.keys())
        return [gate for gate in gate_list if gate in group_gates]
    return gate_list


def get_routine_result(
    ctx: RoutineContext,
    routine_name: str,
    session: LoggerSession | None = None,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get routine result for the current group, with fallback to non-grouped result.

    Results are stored with group suffix if group is active (e.g., "leakage_test_side_A").
    This function tries to find the group-specific result first, then falls back to
    the non-grouped result.

    Args:
        ctx: Routine context containing results.
        routine_name: Name of the routine whose result to retrieve (e.g., "leakage_test").
        session: Optional logger session to get group name from.
        default: Default value to return if result is not found. Defaults to empty dict.

    Returns:
        Dictionary containing the routine result, or default value if not found.
    """
    if default is None:
        default = {}

    # Get group name from session metadata if available
    group_name = None
    if session is not None:
        group_name = getattr(session.metadata, "group_name", None)
    elif hasattr(ctx.resources, "group") and ctx.resources.group is not None:
        # Fallback: try to infer group name from results keys
        # Look for routine_name_* keys
        all_keys = ctx.results.list_results()
        routine_keys = [
            k
            for k in all_keys
            if k.startswith(f"{routine_name}_") and k != routine_name
        ]
        if routine_keys:
            # Extract group name from first matching key
            group_name = routine_keys[0].replace(f"{routine_name}_", "", 1)

    # Try group-specific result first, then fall back to non-grouped result
    if group_name:
        group_key = f"{routine_name}_{group_name}"
        result = ctx.results.get(group_key)
        if result:
            return result

    # Fall back to non-grouped result
    return ctx.results.get(routine_name, default)
