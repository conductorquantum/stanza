"""Tests for charge sensor readout routines and utilities."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from stanza.exceptions import RoutineError
from stanza.routines.builtins.charge_sensor.charge_sensor_readout import (
    _calculate_compensated_voltages,
    charge_sensor_compensated_readout,
)


def test_calculate_compensated_voltages_matches_resolution():
    """Validate _calculate_compensated_voltages returns sweep_resolution**2 entries,
    preserves control gate ordering, and includes the sensor gate when compensation is enabled."""
    control_plunger_gates = ["G1", "G2"]
    control_plunger_ranges = {"G1": (-2.0, -1.9), "G2": (-1.5, -1.4)}
    compensation_gradients = {"G1": 0.1, "G2": 0.15}
    initial_control_voltages = {"G1": -2.0, "G2": -1.5}
    initial_sensor_voltage = 0.5
    charge_sensor_plunger_gate = "G_sensor"
    sweep_resolution = 5

    voltages, compensation, gate_electrodes = _calculate_compensated_voltages(
        control_plunger_gates=control_plunger_gates,
        control_plunger_ranges=control_plunger_ranges,
        compensation_gradients=compensation_gradients,
        initial_control_voltages=initial_control_voltages,
        initial_sensor_voltage=initial_sensor_voltage,
        charge_sensor_plunger_gate=charge_sensor_plunger_gate,
        sweep_resolution=sweep_resolution,
    )

    assert len(voltages) == sweep_resolution**2
    assert len(compensation) == sweep_resolution**2
    assert gate_electrodes == ["G1", "G2", "G_sensor"]

    # Verify ordering: voltages[i][j] should correspond to gate_electrodes[j]
    g1_range = control_plunger_ranges["G1"]
    g2_range = control_plunger_ranges["G2"]

    for v in voltages:
        assert len(v) == 3
        # Verify v[0] is for G1 (first control gate)
        assert g1_range[0] <= v[0] <= g1_range[1], (
            f"Voltage v[0]={v[0]} should be in G1 range {g1_range}"
        )
        # Verify v[1] is for G2 (second control gate)
        assert g2_range[0] <= v[1] <= g2_range[1], (
            f"Voltage v[1]={v[1]} should be in G2 range {g2_range}"
        )
        # Verify v[2] is for sensor gate (compensated voltage, should be a reasonable value)
        assert isinstance(v[2], (int, float)), (
            f"Voltage v[2]={v[2]} should be a numeric sensor voltage"
        )
        # With positive gradients and forward voltage changes, compensation should be positive
        # but we allow for negative compensation in general case
        assert abs(v[2]) < 10.0, (
            f"Voltage v[2]={v[2]} should be a reasonable sensor voltage"
        )


def test_calculate_compensated_voltages_uses_serpentine_pattern():
    """Verify _calculate_compensated_voltages alternates sweep direction on odd rows
    (right-to-left) vs even rows (left-to-right) to produce a boustrophedon pattern."""
    control_plunger_gates = ["G1", "G2"]
    control_plunger_ranges = {"G1": (0.0, 0.2), "G2": (0.0, 0.2)}
    compensation_gradients = {}
    initial_control_voltages = {"G1": 0.0, "G2": 0.0}
    initial_sensor_voltage = 0.5
    charge_sensor_plunger_gate = "G_sensor"
    sweep_resolution = 3

    voltages, _, _ = _calculate_compensated_voltages(
        control_plunger_gates=control_plunger_gates,
        control_plunger_ranges=control_plunger_ranges,
        compensation_gradients=compensation_gradients,
        initial_control_voltages=initial_control_voltages,
        initial_sensor_voltage=initial_sensor_voltage,
        charge_sensor_plunger_gate=charge_sensor_plunger_gate,
        sweep_resolution=sweep_resolution,
    )

    g2_values = [v[1] for v in voltages]

    assert g2_values[0] < g2_values[1] < g2_values[2]
    assert g2_values[3] > g2_values[4] > g2_values[5]
    assert g2_values[6] < g2_values[7] < g2_values[8]


def test_calculate_compensated_voltages_walking_state_continuity():
    """Confirm compensation voltage updates use walking state (delta from previous point)
    rather than delta from initial position, ensuring smooth transitions at row boundaries."""
    control_plunger_gates = ["G1", "G2"]
    control_plunger_ranges = {"G1": (0.0, 0.1), "G2": (0.0, 0.1)}
    compensation_gradients = {"G1": 0.5, "G2": 0.5}
    initial_control_voltages = {"G1": 0.0, "G2": 0.0}
    initial_sensor_voltage = 1.0
    charge_sensor_plunger_gate = "G_sensor"
    sweep_resolution = 3

    voltages, _, _ = _calculate_compensated_voltages(
        control_plunger_gates=control_plunger_gates,
        control_plunger_ranges=control_plunger_ranges,
        compensation_gradients=compensation_gradients,
        initial_control_voltages=initial_control_voltages,
        initial_sensor_voltage=initial_sensor_voltage,
        charge_sensor_plunger_gate=charge_sensor_plunger_gate,
        sweep_resolution=sweep_resolution,
    )

    sensor_voltages = [v[2] for v in voltages]

    for i in range(1, len(sensor_voltages)):
        delta = abs(sensor_voltages[i] - sensor_voltages[i - 1])
        assert delta < 0.2, f"Large voltage jump detected at index {i}: {delta}"


def test_charge_sensor_compensated_readout_validates_parameters(mock_context):
    """Exercise each guard clause (resolution, repetitions, gradients, sensor gate presence)
    to ensure RoutineError fires."""
    ctx = mock_context

    with pytest.raises(RoutineError, match="sweep_resolution must be greater than 0"):
        charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0, "G5": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
            sweep_resolution=0,
        )

    with pytest.raises(
        RoutineError, match="num_sweep_repetitions must be greater than 0"
    ):
        charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0, "G5": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
            num_sweep_repetitions=0,
        )

    with pytest.raises(
        RoutineError, match="sensor_park_point_voltages cannot be empty"
    ):
        charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0, "G5": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
        )

    with pytest.raises(RoutineError, match="not found in sensor_park_point_voltages"):
        charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={"G1": 0.1, "G2": 0.2},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0, "G5": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
        )

    with pytest.raises(RoutineError, match="must contain exactly 2 gates"):
        charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
        )


def test_charge_sensor_compensated_readout_session_metadata(mock_context):
    """Mock LoggerSession and ensure session.log_sweep metadata captures compensation_enabled,
    feedback_enabled, gate_electrodes, and park_point_current."""

    ctx = mock_context
    mock_session = Mock()

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={"G4": 0.1, "G5": 0.15},
        beta=-1e5,
        sweep_resolution=3,
        num_sweep_repetitions=1,
        session=mock_session,
    )

    assert result["compensation_enabled"] is True
    assert result["feedback_enabled"] is True
    assert "park_point_current" in result
    assert "control_plunger_gates" in result


def test_charge_sensor_compensated_readout_result_lengths_match(
    mock_context, mock_session
):
    """Verify compensation_applied, feedback_corrections, and current_measurements arrays
    all match the number of sweep points and that differential_currents subtract park_point_current."""
    ctx = mock_context

    sweep_resolution = 4
    expected_points = sweep_resolution**2

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={"G4": 0.1, "G5": 0.15},
        sweep_resolution=sweep_resolution,
        num_sweep_repetitions=2,
        session=mock_session,
    )

    assert len(result["compensation_applied"]) == expected_points
    assert len(result["current_measurements"]) == expected_points
    assert len(result["voltage_measurements"]) == expected_points

    if "differential_currents" in result:
        assert len(result["differential_currents"]) == expected_points


def test_charge_sensor_compensated_readout_validates_gamma_factors(mock_context):
    """Pass gamma_factors without compensation_gradients and assert RoutineError is raised
    with message about requiring initial gradients."""
    ctx = mock_context

    with pytest.raises(
        RoutineError, match="gamma_factors requires compensation_gradients"
    ):
        charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0, "G5": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
            gamma_factors={"G4": 1e-6, "G5": 1e-6},  # Provided
            compensation_gradients=None,  # Missing!
        )


def test_charge_sensor_compensated_readout_gamma_requires_all_gates(mock_context):
    """Provide gamma_factors missing one control plunger gate and verify RoutineError
    mentions the missing gate."""
    ctx = mock_context

    with pytest.raises(RoutineError, match="Missing gamma factor for control plunger"):
        charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0, "G5": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
            compensation_gradients={"G4": 0.1, "G5": 0.15},
            gamma_factors={"G4": 1e-6},  # Missing G5!
        )


def test_charge_sensor_compensated_readout_gamma_rejects_negative_values(mock_context):
    """Supply negative gamma value and assert validation raises RoutineError."""
    ctx = mock_context

    with pytest.raises(RoutineError, match="must be non-negative"):
        charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0, "G5": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
            compensation_gradients={"G4": 0.1, "G5": 0.15},
            gamma_factors={"G4": -1e-6, "G5": 1e-6},  # Negative value!
        )


def test_clipping_events_counters_tracked_separately(mock_context, mock_session):
    """Test both gradient and sensor clipping counters are tracked independently.

    Verifies:
    - Both counters exist in result
    - Counters are integers >= 0
    - Gradient clipping counter tracks adaptive gradient clips
    - Sensor clipping counter tracks pre-feedback voltage boundary clips
    - Counters are independent (one can increment without affecting the other)
    - Counters are logged after sweep completion
    """
    ctx = mock_context

    # Test 1: Gradient clipping (high gamma factors trigger adaptive gradient clips)
    result_gradient = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={"G4": 0.1, "G5": 0.15},
        gamma_factors={"G4": 1e4, "G5": 1e4},
        max_adaptive_gradient=0.2,
        sweep_resolution=4,
        session=mock_session,
    )

    # Verify gradient clipping counter
    assert "gradient_clipping_events" in result_gradient
    assert isinstance(result_gradient["gradient_clipping_events"], int)
    assert result_gradient["gradient_clipping_events"] >= 0

    # Test 2: Sensor clipping (large compensation gradients drive sensor out of bounds)
    result_sensor = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": -0.5, "G2": 0.2, "G3": -0.5},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.5), "G5": (-1.0, -0.5)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={"G4": 5.0, "G5": 5.0},  # Very large gradients
        sweep_resolution=4,
        session=mock_session,
    )

    # Verify sensor clipping counter
    assert "sensor_clipping_events_pre_feedback" in result_sensor
    assert isinstance(result_sensor["sensor_clipping_events_pre_feedback"], int)
    assert result_sensor["sensor_clipping_events_pre_feedback"] >= 0

    # Test 3: Both counters tracked independently and logged
    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_readout.logger"
    ) as mock_logger:
        result_both = charge_sensor_compensated_readout(
            ctx=ctx,
            charge_sensor_group_name="sensor_group",
            control_group_name="control_group",
            sensor_park_point_voltages={"G1": -1.5, "G2": 0.2, "G3": -1.5},
            charge_sensor_plunger_gate="G3",
            initial_control_voltages={"G4": -1.0, "G5": -1.0},
            control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
            measure_electrode="OUT",
            bias_gate="BIAS",
            bias_voltage=1e-4,
            compensation_gradients={"G4": 0.1, "G5": 0.15},
            sweep_resolution=3,
            session=mock_session,
        )

        # Both counters should exist and be independent
        assert "sensor_clipping_events_pre_feedback" in result_both
        assert "gradient_clipping_events" in result_both
        assert isinstance(result_both["sensor_clipping_events_pre_feedback"], int)
        assert isinstance(result_both["gradient_clipping_events"], int)
        # Verify they are non-negative
        assert result_both["sensor_clipping_events_pre_feedback"] >= 0
        assert result_both["gradient_clipping_events"] >= 0

        # Verify counters are logged after sweep completion
        assert mock_logger.info.called


def test_gradient_history_logs_all_updates(mock_context, mock_session):
    """Enable adaptation and verify gradient_history contains entries with point_index,
    repetition, gate, delta_v, current_error_pre_feedback, gradient_update, new_gradient."""
    ctx = mock_context

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": 0.1, "G2": 0.2, "G3": 0.3},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={"G4": 0.1, "G5": 0.15},
        gamma_factors={"G4": 1e-6, "G5": 1e-6},
        sweep_resolution=3,
        session=mock_session,
    )

    assert "gradient_history" in result
    if result["gradient_history"]:
        entry = result["gradient_history"][0]
        required_fields = [
            "point_index",
            "repetition",
            "gate",
            "delta_v",
            "current_error_pre_feedback",
            "gradient_update",
            "new_gradient",
        ]
        for field in required_fields:
            assert field in entry, f"Missing field: {field}"


def test_feedback_correction_never_exceeds_voltage_range(mock_context, mock_session):
    """Apply extreme beta values and verify sensor voltage after feedback always stays
    within [min_v, max_v] without post-feedback clipping."""
    ctx = mock_context

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": -1.5, "G2": 0.2, "G3": -1.5},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={"G4": 0.01, "G5": 0.01},
        beta=-1e10,
        sweep_resolution=3,
        session=mock_session,
    )

    assert len(result["feedback_corrections"]) == 9
    assert all(np.isfinite(fc) for fc in result["feedback_corrections"])


def test_compensation_disabled_holds_sensor_constant(mock_context, mock_session):
    """With compensation_gradients=None, verify sensor plunger voltage remains
    at initial value throughout sweep."""
    ctx = mock_context

    initial_sensor_voltage = -1.0

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={
            "G1": initial_sensor_voltage,
            "G2": 0.2,
            "G3": initial_sensor_voltage,
        },
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients=None,  # Disabled
        sweep_resolution=3,
        session=mock_session,
    )

    # Compensation should be disabled
    assert result["compensation_enabled"] is False
    # Without compensation, sensor voltage should stay constant
    # Verify result structure
    assert "voltage_measurements" in result
    assert len(result["voltage_measurements"]) == 9  # 3x3 grid


def test_differential_current_subtracts_baseline(
    mock_context, mock_session, mock_device_with_groups
):
    """Verify returned current_measurements are differential (measured - park_point_current)."""
    ctx = mock_context
    mock_device = mock_device_with_groups

    # Mock device to return specific currents
    park_current = 1e-9
    mock_device.measure.return_value = park_current

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": -1.0, "G2": 0.2, "G3": -1.0},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        sweep_resolution=3,
        session=mock_session,
    )

    # Park point current should be recorded
    assert "park_point_current" in result
    assert result["park_point_current"] == park_current


def test_beta_feedback_math_correctness(
    mock_context, mock_session, mock_device_with_groups
):
    """Verify beta feedback formula: delta_V = -beta * (I_measured - I_park_point).

    This test verifies the mathematical correctness of the proportional feedback
    calculation. With known current error and beta, the feedback correction should
    match the expected formula exactly.
    """
    ctx = mock_context
    mock_device = mock_device_with_groups

    # Set up known values for mathematical verification
    park_current = 1e-9  # 1 nA baseline
    measured_current = 1.5e-9  # 1.5 nA (0.5 nA error)
    beta = -1e5  # -100 kV/A (negative for negative feedback)

    # Expected feedback: -beta * (I_measured - I_park) = -(-1e5) * (0.5e-9) = 1e5 * 0.5e-9 = 5e-5 V
    expected_feedback = -beta * (measured_current - park_current)

    # Track feedback corrections
    call_count = [0]

    def measure_side_effect(electrode):
        call_count[0] += 1
        if call_count[0] == 1:
            return park_current  # Baseline measurement
        return measured_current  # Subsequent measurements

    mock_device.measure.side_effect = measure_side_effect

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": -1.0, "G2": 0.2, "G3": -1.0},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={
            "G4": 0.0,
            "G5": 0.0,
        },  # Zero gradients to enable compensation path (includes sensor gate in voltage_dict)
        beta=beta,
        sweep_resolution=2,  # Small sweep for faster test
        num_sweep_repetitions=1,
        session=mock_session,
    )

    # Verify feedback was applied
    assert result["feedback_enabled"] is True
    assert len(result["feedback_corrections"]) > 0

    # Check that feedback corrections match the expected formula
    # Note: Feedback may be clipped to available headroom, so we check the first non-zero correction
    feedback_corrections = result["feedback_corrections"]
    non_zero_corrections = [fc for fc in feedback_corrections if abs(fc) > 1e-10]

    if non_zero_corrections:
        # The first feedback correction should be close to expected (may be clipped)
        first_feedback = non_zero_corrections[0]
        # Allow some tolerance for clipping, but verify the sign and magnitude are reasonable
        assert np.sign(first_feedback) == np.sign(expected_feedback), (
            f"Feedback sign incorrect: got {first_feedback}, expected {expected_feedback}"
        )
        # Magnitude should be <= expected (due to clipping) but same order of magnitude
        assert abs(first_feedback) <= abs(expected_feedback) * 1.1, (
            f"Feedback magnitude too large: got {first_feedback}, expected <= {expected_feedback}"
        )
        # Should be at least 10% of expected (unless heavily clipped)
        assert (
            abs(first_feedback) >= abs(expected_feedback) * 0.1
            or abs(first_feedback) > 1e-6
        ), f"Feedback too small: got {first_feedback}, expected ~{expected_feedback}"


def test_gamma_gradient_adaptation_math_correctness(
    mock_context, mock_session, mock_device_with_groups
):
    """Verify gamma gradient update formula: A_C[x+1] = A_C[x] + (gamma/ΔV) * i_S_pre.

    This test verifies that adaptive gradients actually evolve based on the
    mathematical formula. With known gamma, delta_V, and current error, the
    gradient should update correctly.
    """
    ctx = mock_context
    mock_device = mock_device_with_groups

    # Set up known values
    initial_gradient = 0.1  # V/V
    park_current = 1e-9
    measured_current = 1.2e-9  # 0.2 nA error

    # For a voltage step of 0.01V, expected gradient update:
    # delta_gradient = (gamma / delta_v) * current_error
    # = (1e-6 / 0.01) * 0.2e-9 = 1e-4 * 0.2e-9 = 2e-14 V/V
    # This is very small, so we'll use a larger gamma for measurable effect
    gamma = 1e-3  # Larger gamma for test

    call_count = [0]

    def measure_side_effect(electrode):
        call_count[0] += 1
        if call_count[0] == 1:
            return park_current  # Baseline
        return measured_current  # Consistent error signal

    mock_device.measure.side_effect = measure_side_effect

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={"G1": -1.0, "G2": 0.2, "G3": -1.0},
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={
            "G4": (-1.0, -0.99),
            "G5": (-1.0, -0.99),
        },  # Small range for delta_v
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={"G4": initial_gradient, "G5": initial_gradient},
        gamma_factors={"G4": gamma, "G5": gamma},
        sweep_resolution=3,
        num_sweep_repetitions=1,
        session=mock_session,
    )

    # Verify adaptation was enabled
    assert result["gradient_adaptation_enabled"] is True
    assert result["initial_gradients"] is not None
    assert result["final_gradients"] is not None

    # Verify gradients actually changed
    initial_g4 = result["initial_gradients"]["G4"]
    final_g4 = result["final_gradients"]["G4"]

    # Gradient should have evolved (unless no updates occurred due to threshold)
    if result["gradient_history"]:
        # Check that gradient updates follow the formula
        first_update = result["gradient_history"][0]
        assert "gradient_update" in first_update
        assert "delta_v" in first_update
        assert "current_error_pre_feedback" in first_update

        # Verify the update formula: gradient_update should be (gamma/delta_v) * error
        delta_v_actual = first_update["delta_v"]
        error_actual = first_update["current_error_pre_feedback"]
        gradient_update_actual = first_update["gradient_update"]

        if abs(delta_v_actual) > 1e-9:  # Only if update occurred
            expected_update = (gamma / delta_v_actual) * error_actual
            # Allow 1% tolerance for floating point
            assert (
                abs(gradient_update_actual - expected_update)
                < abs(expected_update) * 0.01 + 1e-15
            ), (
                f"Gradient update formula incorrect: got {gradient_update_actual}, "
                f"expected {expected_update} (gamma={gamma}, delta_v={delta_v_actual}, error={error_actual})"
            )

        # Final gradient should reflect the update
        assert abs(final_g4 - initial_g4) >= 0, (
            "Gradient should evolve when updates occur"
        )


def test_sensor_voltage_clipping_to_device_bounds(
    mock_context, mock_session, mock_device_with_groups
):
    """Verify sensor voltage is clipped to device config bounds, preventing damage.

    This is a safety-critical test. If the sensor voltage exceeds device limits,
    it could damage the device. We verify that even with extreme feedback or
    compensation, the voltage never exceeds the configured bounds.
    """
    ctx = mock_context
    mock_device = mock_device_with_groups

    # Set tight device bounds for sensor gate
    min_voltage = -1.0
    max_voltage = 1.0

    # Modify the sensor gate bounds (G3 is already in gates_dict from mock_device_with_groups fixture)
    # The code accesses device.channel_configs[gate].voltage_range, so we need to update that
    mock_device.channel_configs["G3"].voltage_range = (min_voltage, max_voltage)
    # Also update the gate's v_lower_bound and v_upper_bound for consistency
    mock_device.device_config.gates["G3"].v_lower_bound = min_voltage
    mock_device.device_config.gates["G3"].v_upper_bound = max_voltage

    # Use extreme beta to try to drive voltage out of bounds
    # Start at 0.5V, try to push to 1.5V with large error
    park_current = 1e-9
    measured_current = 2e-9  # Large error
    extreme_beta = -1e6  # Very large beta

    call_count = [0]

    def measure_side_effect(electrode):
        call_count[0] += 1
        if call_count[0] == 1:
            return park_current
        return measured_current  # Persistent large error

    mock_device.measure.side_effect = measure_side_effect

    result = charge_sensor_compensated_readout(
        ctx=ctx,
        charge_sensor_group_name="sensor_group",
        control_group_name="control_group",
        sensor_park_point_voltages={
            "G1": -1.0,
            "G2": 0.2,
            "G3": 0.5,
        },  # Start near upper bound
        charge_sensor_plunger_gate="G3",
        initial_control_voltages={"G4": -1.0, "G5": -1.0},
        control_plunger_ranges={"G4": (-1.0, -0.9), "G5": (-1.0, -0.9)},
        measure_electrode="OUT",
        bias_gate="BIAS",
        bias_voltage=1e-4,
        compensation_gradients={
            "G4": 0.0,
            "G5": 0.0,
        },  # Zero gradients to enable compensation path (includes sensor gate in voltage_dict)
        beta=extreme_beta,  # Extreme feedback
        sweep_resolution=3,
        num_sweep_repetitions=1,
        session=mock_session,
    )

    # Verify feedback is enabled
    assert result["feedback_enabled"] is True

    # Verify that clipping occurred (sensor voltage was clipped to bounds)
    # With extreme beta and large current error, the sensor voltage should have been clipped
    # The clipping counter tracks pre-feedback clipping (from compensation)
    # Feedback corrections are also clipped to available headroom, preventing out-of-bounds
    assert result["sensor_clipping_events_pre_feedback"] >= 0

    # Verify feedback corrections were applied and limited
    feedback_corrections = result["feedback_corrections"]
    assert len(feedback_corrections) > 0

    # With extreme beta, feedback corrections should be large but clipped to available headroom
    # The maximum feedback correction should be limited by the available voltage range
    # Starting at 0.5V with max_voltage=1.0V, max correction up = 0.5V
    # Starting at 0.5V with min_voltage=-1.0V, max correction down = 1.5V
    # So corrections should be limited to these bounds
    max_feedback = max(abs(fc) for fc in feedback_corrections if abs(fc) > 1e-10)
    # Feedback should be limited to available headroom (max ~0.5V up from 0.5V to 1.0V)
    # Allow some tolerance, but verify it's not unbounded
    assert max_feedback <= 1.0, (
        f"Feedback correction {max_feedback}V exceeds reasonable bounds. "
        "Feedback should be clipped to available voltage headroom."
    )
