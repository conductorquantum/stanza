"""Utility functions for built-in routines."""

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
