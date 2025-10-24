from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from stanza.models import ContactType, GateType, GPIOType, PadType


@dataclass
class ChannelConfig:
    """Configuration for an instrument channel.

    Attributes:
        name: Unique channel string identifier, used to reference
            the gate or contact on the device
        voltage_range: Min and max voltage limits (V)
        control_channel: Physical control channel number
        measure_channel: Physical measurement channel number
        output_mode: Output mode (dc, ac, etc.)
        enabled: Whether channel is enabled
        unit: Channel units
        metadata: Additional channel metadata
    """

    name: str
    voltage_range: tuple[float | None, float | None]
    pad_type: PadType
    electrode_type: GateType | ContactType | GPIOType
    control_channel: int | None = None
    measure_channel: int | None = None
    breakout_channel: int | None = None
    output_mode: str = "dc"
    enabled: bool = True
    unit: str = "V"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Parameter:
    """Instrument parameter with validation and metadata.

    Attributes:
        name: Parameter name
        value: Current parameter value
        unit: Parameter unit
        validator: Value validation function
        getter: Value getter function
        setter: Value setter function
        metadata: Additional parameter metadata
    """

    name: str
    value: Any = None
    unit: str = ""
    validator: Callable[[Any], bool] | None = None
    getter: Callable[[], Any] | None = None
    setter: Callable[[Any], None] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def validate(self, value: Any) -> bool:
        """Validate parameter value."""
        return self.validator(value) if self.validator else True

    def set(self, value: Any) -> None:
        """Set parameter value with validation."""
        if not self.validate(value):
            raise ValueError(f"Invalid value {value} for parameter {self.name}")
        if self.setter:
            self.setter(value)
        self.value = value

    def get(self) -> Any:
        """Get parameter value."""
        if self.getter:
            self.value = self.getter()
        return self.value


class Validators:
    """Common validators for instrument parameters."""

    @staticmethod
    def range_validator(min_val: float, max_val: float) -> Callable[[float], bool]:
        """Create a range validator.

        Args:
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validator function that checks if value is within range
        """

        def validator(value: float) -> bool:
            return min_val <= value <= max_val

        return validator

    @staticmethod
    def positive_validator(value: float) -> bool:
        """Validate that value is positive."""
        return value > 0

    @staticmethod
    def negative_validator(value: float) -> bool:
        """Validate that value is negative."""
        return value < 0

    @staticmethod
    def non_zero_validator(value: float) -> bool:
        """Validate that value is non-zero."""
        return value != 0


class InstrumentChannel(ABC):
    """Base class for instrument channels with parameter management.

    Attributes:
        name: Unique channel string identifier
        channel_id: Physical channel identifier
        config: Channel configuration
        parameters: Dictionary of channel parameters
    """

    def __init__(self, config: ChannelConfig):
        """Initialize instrument channel."""
        self.name = config.name
        self.channel_id: int | None = None  # Will be set by subclasses
        self.config = config
        self.parameters: dict[str, Parameter] = {}
        self._setup_parameters()

    def __str__(self) -> str:
        """Return a string representation of the channel."""
        return f"Channel(name={self.name}, channel_id={self.channel_id}, config={self.config}, parameters={self.parameters})"

    @property
    def channel_info(self) -> dict[str, Any]:
        """Return a dictionary of channel information."""
        return {
            "name": self.name,
            "channel_id": self.channel_id,
            "config": self.config,
            "parameters": self.parameters,
        }

    @abstractmethod
    def _setup_parameters(self) -> None:
        """Setup channel-specific parameters.

        Must be implemented by subclasses to define their specific parameters.
        """
        pass

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to this channel."""
        if param.name in self.parameters:
            raise ValueError(f"Parameter '{param.name}' already exists")
        self.parameters[param.name] = param

    def get_parameter(self, name: str) -> Parameter:
        """Get parameter by name."""
        if name not in self.parameters:
            raise KeyError(f"Parameter '{name}' not found")
        return self.parameters[name]

    def set_parameter(self, name: str, value: Any) -> None:
        """Set parameter value."""
        try:
            self.get_parameter(name).set(value)
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"Set parameter {name} failed with: {e}") from e

    def get_parameter_value(self, name: str) -> Any:
        """Get parameter value."""
        return self.get_parameter(name).get()


class ControlChannel(InstrumentChannel):
    """Standard control channel implementation with voltage parameter.

    Provides voltage control functionality with range validation based on
    the channel configuration.
    """

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.channel_id = config.control_channel

    def _setup_parameters(self) -> None:
        """Setup voltage parameter with range validation."""
        voltage_param = Parameter(
            name="voltage",
            value=0.0,
            unit="V",
            validator=Validators.range_validator(
                self.config.voltage_range[0], self.config.voltage_range[1]
            )
            if self.config.voltage_range[0] is not None
            and self.config.voltage_range[1] is not None
            else None,
        )
        self.add_parameter(voltage_param)

        slew_rate_param = Parameter(
            name="slew_rate",
            value=10.0,
            unit="V/s",
            validator=Validators.range_validator(0.001, 1000.0),
        )
        self.add_parameter(slew_rate_param)


class MeasurementChannel(InstrumentChannel):
    """Standard measurement channel implementation with measurement parameters.

    Provides current measurement functionality with conversion factor support
    for ADC-based measurements.
    """

    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self.channel_id = config.measure_channel

    def _setup_parameters(self) -> None:
        """Setup measurement parameters with validation."""
        current_param = Parameter(
            name="current",
            value=None,
            unit="A",
            getter=None,
            setter=None,
            metadata={
                "description": "Measured DC current (mean over integration time)"
            },
        )
        self.add_parameter(current_param)

        conversion_factor_param = Parameter(
            name="conversion_factor",
            value=1e-6,
            unit="A/count",
            validator=Validators.non_zero_validator,
            metadata={"description": "Conversion factor from ADC counts to amperes"},
        )
        self.add_parameter(conversion_factor_param)
