from unittest.mock import Mock

import numpy as np
import pytest

from stanza.models import DeviceGroup, Gate, GateType
from stanza.registry import ResultsRegistry
from stanza.routines import RoutineContext
from stanza.routines.builtins.charge_sensor.charge_sensor_find_sensor_peak import (
    StabilityMeasurement,
)
from stanza.routines.builtins.utils.peak_fitting import FittedPeak, ModelFitResult


@pytest.fixture
def model_fit_result():
    """Baseline ModelFitResult for charge sensor unit tests."""
    return ModelFitResult(
        model_name="Lorentzian",
        amplitude=1.0,
        center_idx=50.0,
        width=5.0,
        offset=0.0,
        r_squared=0.95,
        rmse=0.01,
        aicc=10.0,
        fwhm=10.0,
        area=100.0,
        skew_resid=0.0,
        eta=0.5,
    )


@pytest.fixture
def fitted_peak_factory(model_fit_result):
    """Factory fixture for constructing FittedPeak instances with sensible defaults."""

    def _factory(
        *,
        best_model: str = "Lorentzian",
        sensitivity: float = 1e-6,
        sensitivity_voltage: float = 0.0,
        peak_idx: int = 0,
        peak_voltage: float = 0.0,
        quality_score: float | None = None,
        sensitivity_score: float | None = None,
        window_currents: np.ndarray | None = None,
        window_voltages: np.ndarray | None = None,
        lorentzian_fit: ModelFitResult | None = None,
        sech2_fit: ModelFitResult | None = None,
        voigt_fit: ModelFitResult | None = None,
    ) -> FittedPeak:
        return FittedPeak(
            best_model=best_model,
            lorentzian_fit=lorentzian_fit or model_fit_result,
            sech2_fit=sech2_fit or model_fit_result,
            voigt_fit=voigt_fit or model_fit_result,
            sensitivity=sensitivity,
            sensitivity_voltage=sensitivity_voltage,
            peak_idx=peak_idx,
            peak_voltage=peak_voltage,
            window_currents=window_currents
            if window_currents is not None
            else np.ones(10),
            window_voltages=window_voltages
            if window_voltages is not None
            else np.linspace(0, 0.1, 10),
            quality_score=quality_score,
            sensitivity_score=sensitivity_score,
        )

    return _factory


@pytest.fixture
def stability_measurement_factory():
    """Factory fixture for constructing StabilityMeasurement instances."""

    def _factory(
        *,
        peak_index: int = 0,
        peak_voltage: float = -0.7,
        max_gradient_voltage: float = -0.68,
        time_array: np.ndarray | None = None,
        current_array: np.ndarray | None = None,
        current_mean: float = 1e-9,
        current_std: float = 1e-11,
        local_slope: float = 1e-6,
        voltage_noise: float = 1e-5,
    ) -> StabilityMeasurement:
        time_array_np = (
            np.array(time_array, dtype=np.float64)
            if time_array is not None
            else np.array([0.0, 1.0], dtype=np.float64)
        )
        current_array_np = (
            np.array(current_array, dtype=np.float64)
            if current_array is not None
            else np.array([1e-9, 1e-9], dtype=np.float64)
        )

        return StabilityMeasurement(
            peak_index=peak_index,
            peak_voltage=peak_voltage,
            max_gradient_voltage=max_gradient_voltage,
            time_array=time_array_np,
            current_array=current_array_np,
            current_mean=current_mean,
            current_std=current_std,
            local_slope=local_slope,
            voltage_noise=voltage_noise,
        )

    return _factory


@pytest.fixture
def mock_device_with_groups():
    """Create a mock device with sensor and control groups."""
    mock_device = Mock()

    sensor_group = DeviceGroup(
        name="sensor_group", gates=["G1", "G2", "G3"], description="Sensor group"
    )
    control_group = DeviceGroup(
        name="control_group",
        gates=["G4", "G5", "G6", "G7"],
        description="Control group",
    )

    mock_device.device_config.groups = {
        "sensor_group": sensor_group,
        "control_group": control_group,
    }

    mock_device.control_gates = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]

    mock_device.check.return_value = {
        "G1": 0.0,
        "G2": 0.0,
        "G3": 0.0,
        "G4": 0.0,
        "G5": 0.0,
        "G6": 0.0,
        "G7": 0.0,
    }

    mock_device.jump = Mock()
    mock_device.measure.return_value = 1e-9

    def mock_sweep_nd(electrodes, voltages, measure_electrode):
        return np.random.normal(1e-9, 1e-11, len(voltages))

    mock_device.sweep_nd = Mock(side_effect=mock_sweep_nd)

    gates_dict = {
        f"G{i}": Gate(
            name=f"G{i}",
            type=GateType.PLUNGER,
            control_channel=i,
            v_lower_bound=-3.0,
            v_upper_bound=0.0,
        )
        for i in range(1, 8)
    }

    mock_device.device_config.gates = gates_dict

    mock_channel_configs = {}
    for gate_name in gates_dict:
        mock_channel = Mock()
        mock_channel.voltage_range = (-3.0, 0.0)
        mock_channel_configs[gate_name] = mock_channel

    mock_device.channel_configs = mock_channel_configs

    return mock_device


@pytest.fixture
def mock_context(mock_device_with_groups):
    """Create a mock routine context with device."""
    resources = Mock()
    resources.device = mock_device_with_groups
    resources.group = None

    results = ResultsRegistry()
    ctx = RoutineContext(resources=resources, results=results)
    return ctx


@pytest.fixture
def mock_session():
    """Create a mock logger session."""
    mock_session = Mock()
    mock_session.log_sweep = Mock()
    mock_session.log_analysis = Mock()
    mock_session.log_measurement = Mock()
    return mock_session


@pytest.fixture
def mock_device_for_sensor_routines():
    """Create a comprehensive mock device for sensor routine tests."""
    mock_device = Mock()

    sensor_group = DeviceGroup(name="sensor_group", gates=["G1", "G2", "G3"])
    control_group = DeviceGroup(name="control_group", gates=["G4", "G5"])

    mock_device.device_config = Mock()
    mock_device.device_config.groups = {
        "sensor_group": sensor_group,
        "control_group": control_group,
    }

    mock_device.control_gates = ["G1", "G2", "G3", "G4", "G5"]
    mock_device.get_gates_by_type.return_value = ["G1", "G2", "G3", "G4", "G5"]

    mock_device.check.return_value = [0.0, 0.0, 0.0]
    mock_device.jump = Mock()
    mock_device.measure.return_value = 1e-9

    voltages = np.linspace(-1.0, -0.5, 128)
    currents = np.ones(128) * 1e-9
    mock_device.sweep_nd.return_value = (voltages, currents)

    return mock_device


@pytest.fixture
def mock_context_for_sensor_routines(mock_device_for_sensor_routines):
    """Create a mock RoutineContext with required prerequisite results."""
    mock_ctx = Mock(spec=RoutineContext)

    mock_ctx.results = {
        "global_accumulation_sensor_group": {"global_turn_on_voltage": -0.8},
        "finger_gate_characterization_sensor_group": {
            "G3": {
                "saturation_voltage": 0.5,
                "cutoff_voltage": -1.5,
                "pinch_off_voltage": -2.0,
            }
        },
    }

    mock_models_client = Mock()
    mock_models = Mock()
    mock_execute_result = Mock()
    mock_execute_result.output = {
        "classification": True,
        "score": 0.95,
        "peak_indices": [64],
    }
    mock_models.execute.return_value = mock_execute_result
    mock_models_client.models = mock_models

    mock_resources = Mock()
    mock_resources.device = mock_device_for_sensor_routines
    mock_resources.models_client = mock_models_client
    mock_ctx.resources = mock_resources
    mock_ctx.session_metadata = {}

    return mock_ctx, mock_device_for_sensor_routines
