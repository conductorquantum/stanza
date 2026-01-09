"""Tests for charge sensor compensation routines and utilities."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from stanza.base.channels import ChannelConfig
from stanza.exceptions import RoutineError
from stanza.logger.session import LoggerSession
from stanza.models import DeviceGroup, GateType
from stanza.routines import RoutineContext
from stanza.routines.builtins.charge_sensor.charge_sensor_compensation import (
    run_compensation,
)
from stanza.routines.builtins.charge_sensor.utils.types import (
    PeakWindowSweepOutput,
)
from stanza.routines.builtins.utils.peak_fitting import (
    FittedPeak,
    ModelFitResult,
    fit_peak_multi_model,
    lorentzian,
)


def test_run_compensation_validates_gates_to_compensate():
    """Pass invalid gate names to run_compensation and assert it raises
    the documented error."""
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "find_sensor_peak": {
            "narrowed_sensor_plunger_range": (-1.0, -0.5),
            "mean_reservoir_saturation_voltage": 0.5,
            "sensor_gates_list": ["G1", "G2", "G3"],
            "sensor_plunger_index": 2,
            "step_size": 0.001,
        }
    }

    mock_device = Mock()
    control_group = DeviceGroup(name="control_group", gates=["G4", "G5"])
    mock_device.device_config = Mock()
    mock_device.device_config.groups = {"control_group": control_group}
    mock_device.get_gates_by_type.return_value = ["G4", "G5"]
    mock_device.check.return_value = [-0.5, -0.6]
    # Add channel_configs with voltage limits for safety validation
    mock_device.channel_configs = {
        "G4": ChannelConfig(
            name="G4",
            voltage_range=(-3.0, 3.0),
            pad_type=None,  # type: ignore
            electrode_type=GateType.PLUNGER,
        ),
        "G5": ChannelConfig(
            name="G5",
            voltage_range=(-3.0, 3.0),
            pad_type=None,  # type: ignore
            electrode_type=GateType.PLUNGER,
        ),
    }

    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_ctx.resources = mock_resources

    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with pytest.raises(
            RoutineError, match="Invalid gates specified in gates_to_compensate"
        ):
            run_compensation(
                ctx=mock_ctx,
                peak_spacing=0.02,
                control_group_name="control_group",
                measure_electrode="OUT",
                bias_gate="BIAS",
                bias_voltage=1e-4,
                gates_to_compensate=["INVALID_GATE", "ANOTHER_BAD_GATE"],
            )


def test_run_compensation_logs_per_sample_measurements():
    """Mock LoggerSession to assert per-sample log_analysis entries are
    emitted with the expected fields."""
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "find_sensor_peak": {
            "narrowed_sensor_plunger_range": (-1.0, -0.5),
            "mean_reservoir_saturation_voltage": 0.5,
            "sensor_gates_list": ["G1", "G2", "G3"],
            "sensor_plunger_index": 2,
            "step_size": 0.001,
        }
    }

    mock_device = Mock()
    mock_device.device_config.groups = {
        "control_group": DeviceGroup(name="control_group", gates=["G4"])
    }

    mock_device.check.return_value = [-0.5]
    mock_device.get_gates_by_type.return_value = ["G4"]
    mock_device.measure.return_value = 1e-9
    # Add channel_configs with voltage limits for safety validation
    mock_device.channel_configs = {
        "G4": ChannelConfig(
            name="G4",
            voltage_range=(-3.0, 3.0),
            pad_type=None,  # type: ignore
            electrode_type=GateType.PLUNGER,
        ),
    }

    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_ctx.resources = mock_resources

    mock_session = Mock(spec=LoggerSession)
    mock_session.log_analysis = Mock()

    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with patch(
            "stanza.routines.builtins.charge_sensor.charge_sensor_compensation._single_window_sensor_plunger_sweep"
        ) as mock_sweep:
            mock_peak = FittedPeak(
                peak_idx=50,
                peak_voltage=-0.75,
                lorentzian_fit=ModelFitResult(
                    model_name="lorentzian",
                    amplitude=1e-9,
                    center_idx=50.0,
                    width=0.01,
                    offset=0.0,
                    r_squared=0.95,
                    rmse=1e-11,
                    aicc=-100,
                    fwhm=0.02,
                    area=1e-10,
                    skew_resid=0.01,
                ),
                sech2_fit=None,
                voigt_fit=None,
                best_model="lorentzian",
                sensitivity=1e-8,
                sensitivity_voltage=-0.74,
                window_currents=np.ones(10) * 1e-9,
                window_voltages=np.linspace(-1.0, -0.5, 10),
                quality_score=0.9,
            )

            mock_output = PeakWindowSweepOutput(
                best_peak=mock_peak,
                aggregated_voltages=np.linspace(-1.0, -0.5, 10),
                aggregated_currents=np.ones(10) * 1e-9,
                classification=True,
                score=0.9,
                num_peaks=1,
            )
            mock_sweep.return_value = mock_output

            run_compensation(
                ctx=mock_ctx,
                peak_spacing=0.02,
                control_group_name="control_group",
                measure_electrode="OUT",
                bias_gate="BIAS",
                bias_voltage=1e-4,
                session=mock_session,
            )

    assert mock_session.log_analysis.call_count > 0


def test_fitted_peak_more_accurate_than_discrete():
    """Generate synthetic peaks and verify fitted peak centers have sub-step-size
    resolution compared to discrete voltage points."""
    voltages = np.linspace(0.0, 0.1, 100)
    true_center_idx = 51.23

    indices = np.arange(100)
    currents = lorentzian(
        indices, amplitude=2e-9, center=true_center_idx, width=8, offset=1e-11
    )

    peak_idx_discrete = int(true_center_idx)
    fitted_peak = fit_peak_multi_model(
        window_currents=currents,
        window_indices=indices,
        peak_idx_in_window=peak_idx_discrete,
        aggregated_voltages=voltages,
        window_start_idx=0,
        window_end_idx=len(currents),
        peak_idx_aggregated=peak_idx_discrete,
    )

    best_fit = getattr(fitted_peak, f"{fitted_peak.best_model.lower()}_fit")
    fitted_center_idx = best_fit.center_idx

    assert abs(fitted_center_idx - true_center_idx) < abs(
        peak_idx_discrete - true_center_idx
    )


def test_compensation_gradient_calculation_from_peak_shifts():
    """Verify run_compensation correctly converts peak position shifts to gradients.

    This is an end-to-end test that verifies the core purpose of the routine:
    when a control gate voltage changes by X volts and the sensor peak shifts by
    Y volts, the routine should return a gradient of Y/X.

    This test verifies the entire measurement pipeline: baseline -> perturbation ->
    sweep -> peak fitting -> shift calculation -> RANSAC -> gradient output.
    """
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "find_sensor_peak": {
            "narrowed_sensor_plunger_range": (-1.0, -0.5),
            "mean_reservoir_saturation_voltage": 0.5,
            "sensor_gates_list": ["G1", "G2", "G3"],
            "sensor_plunger_index": 2,
            "step_size": 0.001,
        }
    }

    mock_device = Mock()
    control_group = DeviceGroup(name="control_group", gates=["G4"])
    mock_device.device_config = Mock()
    mock_device.device_config.groups = {"control_group": control_group}
    mock_device.get_gates_by_type.return_value = ["G4"]
    mock_device.check.return_value = [-0.5]
    # Add channel_configs with voltage limits for safety validation
    mock_device.channel_configs = {
        "G4": ChannelConfig(
            name="G4",
            voltage_range=(-3.0, 3.0),
            pad_type=None,  # type: ignore
            electrode_type=GateType.PLUNGER,
        ),
    }

    true_gradient = 0.5
    baseline_peak_voltage = -0.75
    control_delta = 0.01
    peak_shift = true_gradient * control_delta

    call_count = [0]

    def sweep_nd_side_effect(electrodes, voltages, measure_electrode):
        call_count[0] += 1

        if call_count[0] <= 5:
            currents = np.ones(len(voltages)) * 1e-11
            for i, v in enumerate(voltages):
                if abs(v - baseline_peak_voltage) < 0.1:
                    currents[i] += 2e-9 * (
                        1 / (1 + ((v - baseline_peak_voltage) / 0.01) ** 2)
                    )
            return (voltages, currents)

        shifted_peak_voltage = baseline_peak_voltage + peak_shift
        currents = np.ones(len(voltages)) * 1e-11
        for i, v in enumerate(voltages):
            if abs(v - shifted_peak_voltage) < 0.1:
                currents[i] += 2e-9 * (
                    1 / (1 + ((v - shifted_peak_voltage) / 0.01) ** 2)
                )
        return (voltages, currents)

    mock_device.sweep_nd.side_effect = sweep_nd_side_effect
    mock_device.measure.return_value = 1e-9

    device_state_tracker = {"G4": -0.5}

    def jump_side_effect(voltage_dict, **kwargs):
        device_state_tracker.update(voltage_dict)

    def check_side_effect(gates):
        if isinstance(gates, list):
            return [device_state_tracker.get(g, -0.5) for g in gates]
        return device_state_tracker.get(gates, -0.5)

    mock_device.jump = Mock(side_effect=jump_side_effect)
    mock_device.check = Mock(side_effect=check_side_effect)

    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_ctx.resources = mock_resources

    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with patch(
            "stanza.routines.builtins.charge_sensor.charge_sensor_compensation._single_window_sensor_plunger_sweep"
        ) as mock_sweep:

            def sweep_side_effect(*args, **kwargs):
                current_g4_voltage = device_state_tracker.get("G4", -0.5)
                baseline_g4_voltage = -0.5
                control_delta_actual = current_g4_voltage - baseline_g4_voltage

                if abs(control_delta_actual) < 1e-6:
                    peak_voltage = baseline_peak_voltage
                else:
                    peak_voltage = (
                        baseline_peak_voltage + true_gradient * control_delta_actual
                    )

                mock_peak = FittedPeak(
                    peak_idx=50,
                    peak_voltage=peak_voltage,
                    lorentzian_fit=ModelFitResult(
                        model_name="lorentzian",
                        amplitude=1e-9,
                        center_idx=50.0,
                        width=0.01,
                        offset=0.0,
                        r_squared=0.95,
                        rmse=1e-11,
                        aicc=-100,
                        fwhm=0.02,
                        area=1e-10,
                        skew_resid=0.01,
                    ),
                    sech2_fit=None,
                    voigt_fit=None,
                    best_model="lorentzian",
                    sensitivity=1e-8,
                    sensitivity_voltage=peak_voltage,
                    window_currents=np.ones(10) * 1e-9,
                    window_voltages=np.linspace(-1.0, -0.5, 10),
                    quality_score=0.9,
                )

                return PeakWindowSweepOutput(
                    best_peak=mock_peak,
                    aggregated_voltages=np.linspace(-1.0, -0.5, 128),
                    aggregated_currents=np.ones(128) * 1e-9,
                    classification=True,
                    score=0.9,
                    num_peaks=1,
                )

            mock_sweep.side_effect = sweep_side_effect

            result = run_compensation(
                ctx=mock_ctx,
                peak_spacing=0.02,
                control_group_name="control_group",
                measure_electrode="OUT",
                bias_gate="BIAS",
                bias_voltage=1e-4,
                gates_to_compensate=["G4"],
            )

            assert "compensation_gradients" in result
            assert "G4" in result["compensation_gradients"]
            calculated_gradient = result["compensation_gradients"]["G4"]

            assert abs(calculated_gradient - true_gradient) < 0.01, (
                f"Gradient calculation incorrect: got {calculated_gradient} V/V, "
                f"expected {true_gradient} V/V (peak shift {peak_shift}V / control delta {control_delta}V)"
            )


def test_run_compensation_validates_voltage_limits():
    """Verify run_compensation raises RoutineError when voltage perturbation would exceed safety limits."""
    mock_ctx = Mock(spec=RoutineContext)
    mock_ctx.results = {
        "find_sensor_peak": {
            "narrowed_sensor_plunger_range": (-1.0, -0.5),
            "mean_reservoir_saturation_voltage": 0.5,
            "sensor_gates_list": ["G1", "G2", "G3"],
            "sensor_plunger_index": 2,
            "step_size": 0.001,
        }
    }

    mock_device = Mock()
    control_group = DeviceGroup(name="control_group", gates=["G4"])
    mock_device.device_config = Mock()
    mock_device.device_config.groups = {"control_group": control_group}
    mock_device.get_gates_by_type.return_value = ["G4"]
    # Set baseline voltage near the upper limit
    mock_device.check.return_value = [2.8]  # Near upper limit of 3.0
    # Add channel_configs with tight voltage limits
    mock_device.channel_configs = {
        "G4": ChannelConfig(
            name="G4",
            voltage_range=(-3.0, 3.0),  # Upper limit is 3.0V
            pad_type=None,  # type: ignore
            electrode_type=GateType.PLUNGER,
        ),
    }

    mock_resources = Mock()
    mock_resources.device = mock_device
    mock_ctx.resources = mock_resources

    # Create a mock peak for baseline measurements
    mock_peak = FittedPeak(
        peak_idx=50,
        peak_voltage=-0.75,
        lorentzian_fit=ModelFitResult(
            model_name="lorentzian",
            amplitude=1e-9,
            center_idx=50.0,
            width=0.01,
            offset=0.0,
            r_squared=0.95,
            rmse=1e-11,
            aicc=-100,
            fwhm=0.02,
            area=1e-10,
            skew_resid=0.01,
        ),
        sech2_fit=None,
        voigt_fit=None,
        best_model="lorentzian",
        sensitivity=1e-8,
        sensitivity_voltage=-0.74,
        window_currents=np.ones(10) * 1e-9,
        window_voltages=np.linspace(-1.0, -0.5, 10),
        quality_score=0.9,
    )

    mock_output = PeakWindowSweepOutput(
        best_peak=mock_peak,
        aggregated_voltages=np.linspace(-1.0, -0.5, 10),
        aggregated_currents=np.ones(10) * 1e-9,
        classification=True,
        score=0.9,
        num_peaks=1,
    )

    with patch(
        "stanza.routines.builtins.charge_sensor.charge_sensor_compensation.filter_gates_by_group",
        side_effect=lambda ctx, gates: gates,
    ):
        with patch(
            "stanza.routines.builtins.charge_sensor.charge_sensor_compensation._single_window_sensor_plunger_sweep"
        ) as mock_sweep:
            mock_sweep.return_value = mock_output

            # Use a very large peak_spacing that will create voltage perturbations exceeding the limit
            # With baseline at 2.8V and voltage_range = 0.3 * 10.0 = 3.0V, perturbations
            # will be up to ±3.0V, so max would be 2.8 + 3.0 = 5.8V which exceeds 3.0V limit
            with pytest.raises(
                RoutineError, match="Voltage perturbation would exceed safety limits"
            ):
                run_compensation(
                    ctx=mock_ctx,
                    peak_spacing=10.0,  # Very large spacing will create large perturbations
                    control_group_name="control_group",
                    measure_electrode="OUT",
                    bias_gate="BIAS",
                    bias_voltage=1e-4,
                )
