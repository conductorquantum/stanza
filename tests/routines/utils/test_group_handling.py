"""Tests for group handling utilities."""

from unittest.mock import Mock

import pytest

from stanza.logger.datatypes import SessionMetadata
from stanza.registry import ResultsRegistry
from stanza.routines.builtins.utils.group_handling import (
    filter_gates_by_group,
    get_routine_result,
)
from stanza.routines.core import RoutineContext


class MockResources:
    """Mock resources object for testing."""

    def __init__(self, group=None):
        self.group = group


def test_filter_gates_by_group_with_group():
    """Test filter_gates_by_group when group is present."""
    group = {"G1": {}, "G2": {}, "G3": {}}
    resources = MockResources(group=group)
    ctx = RoutineContext(resources=resources, results=ResultsRegistry())

    gate_list = ["G1", "G2", "G4", "G5"]
    filtered = filter_gates_by_group(ctx, gate_list)

    # Should only include G1 and G2 (present in group)
    assert filtered == ["G1", "G2"]


def test_filter_gates_by_group_without_group():
    """Test filter_gates_by_group when group is None."""
    resources = MockResources(group=None)
    ctx = RoutineContext(resources=resources, results=ResultsRegistry())

    gate_list = ["G1", "G2", "G3"]
    filtered = filter_gates_by_group(ctx, gate_list)

    # Should return all gates unchanged
    assert filtered == ["G1", "G2", "G3"]


def test_get_routine_result_prefers_group_specific_entry():
    """Populate both grouped and ungrouped results and ensure the helper returns
    the group-specific data before falling back."""
    results = ResultsRegistry()
    results.store("leakage_test", {"ungrouped": "data"})
    results.store("leakage_test_side_A", {"grouped": "data_A"})

    # Mock resources with group
    group = {"G1": {}, "G2": {}}
    resources = MockResources(group=group)
    ctx = RoutineContext(resources=resources, results=results)

    # Mock session with group_name
    session = Mock()
    session.metadata = SessionMetadata(
        session_id="123",
        start_time=0.0,
        user="test_user",
        routine_name="test",
        group_name="side_A",
    )

    result = get_routine_result(ctx, "leakage_test", session=session)

    # Should return the grouped result
    assert result == {"grouped": "data_A"}


def test_get_routine_result_fallback_to_ungrouped():
    """Test that get_routine_result falls back to ungrouped result when grouped is missing."""
    results = ResultsRegistry()
    results.store("leakage_test", {"ungrouped": "data"})

    resources = MockResources(group=None)
    ctx = RoutineContext(resources=resources, results=results)

    result = get_routine_result(ctx, "leakage_test", session=None)

    # Should return the ungrouped result
    assert result == {"ungrouped": "data"}


def test_get_routine_result_default_value():
    """Test that get_routine_result returns default when result is missing."""
    results = ResultsRegistry()
    resources = MockResources(group=None)
    ctx = RoutineContext(resources=resources, results=results)

    default = {"default": "value"}
    result = get_routine_result(ctx, "nonexistent_routine", default=default)

    # Should return the default
    assert result == {"default": "value"}


def test_get_routine_result_empty_dict_default():
    """Test that get_routine_result returns empty dict when no default provided."""
    results = ResultsRegistry()
    resources = MockResources(group=None)
    ctx = RoutineContext(resources=resources, results=results)

    result = get_routine_result(ctx, "nonexistent_routine")

    # Should return empty dict
    assert result == {}


def test_get_routine_result_uses_session_metadata_group():
    """Supply a session metadata group_name and make sure the helper resolves to
    routine_{group} even when ctx has multiple groups cached."""
    results = ResultsRegistry()
    results.store("test_routine", {"base": "data"})
    results.store("test_routine_group_A", {"group_A": "data"})
    results.store("test_routine_group_B", {"group_B": "data"})

    resources = MockResources(group={"G1": {}})
    ctx = RoutineContext(resources=resources, results=results)

    # Mock session with specific group_name
    session = Mock()
    session.metadata = SessionMetadata(
        session_id="456",
        start_time=0.0,
        user="test_user",
        routine_name="test",
        group_name="group_B",
    )

    result = get_routine_result(ctx, "test_routine", session=session)

    # Should return group_B specific result
    assert result == {"group_B": "data"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
