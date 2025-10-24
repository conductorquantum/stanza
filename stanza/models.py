from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, model_validator
from pydantic.version import VERSION as PYDANTIC_VERSION

PYDANTIC_VERSION_MINOR_TUPLE = tuple(int(x) for x in PYDANTIC_VERSION.split(".")[:2])
PYDANTIC_V2 = PYDANTIC_VERSION_MINOR_TUPLE[0] == 2


class BaseModelWithConfig(BaseModel):
    if PYDANTIC_V2:
        model_config = {"extra": "allow"}

    else:

        class Config:
            extra = "allow"


class Electrode(BaseModel):
    control_channel: int | None = Field(
        None, ge=0, le=1024, description="Control channel for control signals"
    )
    measure_channel: int | None = Field(
        None, ge=0, le=1024, description="Measurement channel for measurement signals"
    )
    breakout_channel: int | None = Field(
        None, ge=0, le=1024, description="Breakout channel for breakout box lines"
    )
    v_lower_bound: float | None
    v_upper_bound: float | None

    @model_validator(mode="after")
    def validate_control_channel_required_when_no_measure_channel(self) -> "Electrode":
        if self.measure_channel is None and self.control_channel is None:
            raise ValueError(
                "Either `control_channel` or `measure_channel` must be specified"
            )
        return self

    @model_validator(mode="after")
    def validate_control_channel_requires_v_bounds(
        self,
    ) -> "Electrode":
        if self.control_channel is not None and self.v_lower_bound is None:
            raise ValueError(
                "`v_lower_bound` must be specified when control_channel is set"
            )
        if self.control_channel is not None and self.v_upper_bound is None:
            raise ValueError(
                "`v_upper_bound` must be specified when control_channel is set"
            )
        return self


class PadType(str, Enum):
    """Pad type"""

    GATE = "GATE"
    CONTACT = "CONTACT"
    GPIO = "GPIO"
    ALL = "ALL"

    def __str__(self) -> str:
        return self.value


class GateType(str, Enum):
    PLUNGER = "PLUNGER"
    BARRIER = "BARRIER"
    RESERVOIR = "RESERVOIR"
    SCREEN = "SCREEN"


class ContactType(str, Enum):
    SOURCE = "SOURCE"
    DRAIN = "DRAIN"


class GPIOType(str, Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class InstrumentType(str, Enum):
    CONTROL = "CONTROL"
    MEASUREMENT = "MEASUREMENT"
    GENERAL = "GENERAL"
    BREAKOUT_BOX = "BREAKOUT_BOX"


class Gate(Electrode):
    """Gate pads for inductive channels on the device"""

    type: GateType


class Contact(Electrode):
    """Contact pads for conductive channels on the device"""

    type: ContactType


class GPIO(Electrode):
    """General Purpose Input/Output pins for digital signals"""

    type: GPIOType


class RoutineConfig(BaseModelWithConfig):
    name: str
    parameters: dict[str, Any] | None = None
    routines: list["RoutineConfig"] | None = None

    @model_validator(mode="after")
    def convert_to_number(self) -> "RoutineConfig":
        """Convert float values to int if they have no fractional part."""
        if self.parameters:
            for key, value in self.parameters.items():
                if isinstance(value, str):
                    try:
                        numeric_value = float(value)
                        if numeric_value.is_integer():
                            self.parameters[key] = int(numeric_value)
                        else:
                            self.parameters[key] = numeric_value
                    except ValueError:
                        pass
        return self


class BaseInstrumentConfig(BaseModelWithConfig):
    """Base instrument configuration with discriminator."""

    name: str
    ip_addr: str | None = None
    serial_addr: str | None = None
    port: int | None = None
    type: InstrumentType
    breakout_line: int | None = None
    driver: str | None = Field(
        None,
        description="Driver file name (e.g., 'qdac2' maps to stanza.drivers.qdac2.QDAC2)",
    )

    @model_validator(mode="after")
    def check_comm_type(self) -> "BaseInstrumentConfig":
        if not (self.ip_addr or self.serial_addr):
            raise ValueError("Either 'ip_addr' or 'serial_addr' must be provided")
        return self


class MeasurementInstrumentConfigMixin(BaseModel):
    """Mixin for instruments with measurement capabilities."""

    measurement_duration: float = Field(
        gt=0, description="Total measurement duration per point in seconds"
    )
    sample_time: float = Field(gt=0, description="Individual sample time in seconds")
    conversion_factor: float = Field(
        gt=0, default=1, description="The conversion factor from ADC counts to amperes"
    )

    # OPX-specific fields
    machine_type: str | None = Field(None, description="OPX machine type")
    cluster_name: str | None = Field(None, description="OPX cluster name")
    measurement_channels: list[int] | None = Field(
        None, description="OPX measurement channels"
    )
    connection_headers: dict[str, str] | None = Field(
        None, description="OPX connection headers"
    )
    octave: str | None = Field(None, description="OPX octave configuration")

    @model_validator(mode="after")
    def validate_timing_constraints(self) -> "MeasurementInstrumentConfigMixin":
        """Validate logical constraints between timing parameters."""
        if self.sample_time > self.measurement_duration:
            raise ValueError(
                f"sample_time ({self.sample_time}s) cannot be larger than "
                f"measurement_duration ({self.measurement_duration}s)"
            )
        return self


class ControlInstrumentConfigMixin(BaseModel):
    """Mixin for instruments with control capabilities."""

    slew_rate: float = Field(gt=0, description="Slew rate in V/s")


class MeasurementInstrumentConfig(
    MeasurementInstrumentConfigMixin, BaseInstrumentConfig
):
    """Instrument configuration for measurement instruments with required timing parameters."""

    type: Literal[InstrumentType.MEASUREMENT] = InstrumentType.MEASUREMENT


class ControlInstrumentConfig(ControlInstrumentConfigMixin, BaseInstrumentConfig):
    """Instrument configuration for control instruments."""

    type: Literal[InstrumentType.CONTROL] = InstrumentType.CONTROL


class GeneralInstrumentConfig(
    ControlInstrumentConfigMixin, MeasurementInstrumentConfigMixin, BaseInstrumentConfig
):
    """Instrument configuration for general instruments with both control and measurement capabilities."""

    model_config = ConfigDict(extra="allow")
    type: Literal[InstrumentType.GENERAL] = InstrumentType.GENERAL


class BreakoutBoxInstrumentConfig(BaseInstrumentConfig):
    """Instrument configuration for breakout box instruments."""

    model_config = ConfigDict(extra="allow")
    type: Literal[InstrumentType.BREAKOUT_BOX] = InstrumentType.BREAKOUT_BOX


InstrumentConfig = Annotated[
    ControlInstrumentConfig
    | MeasurementInstrumentConfig
    | GeneralInstrumentConfig
    | BreakoutBoxInstrumentConfig,
    Discriminator("type"),
]


class DeviceConfig(BaseModel):
    """Configuration for a quantum device (Device Under Test)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str
    gates: dict[str, Gate] = {}
    contacts: dict[str, Contact] = {}
    gpios: dict[str, GPIO] = {}
    routines: list[RoutineConfig] = []
    instruments: list[InstrumentConfig]

    @model_validator(mode="after")
    def validate_unique_channels(self) -> "DeviceConfig":
        """Ensure that all channels are unique across gates, contacts, and gpios"""
        control_channel_users: dict[int, list[str]] = {}
        measure_channel_users: dict[int, list[str]] = {}
        duplicates = []

        all_electrodes = {
            **{f"gate '{name}'": electrode for name, electrode in self.gates.items()},
            **{
                f"contact '{name}'": electrode
                for name, electrode in self.contacts.items()
            },
            **{f"gpio '{name}'": electrode for name, electrode in self.gpios.items()},
        }

        # Track which electrodes use each channel
        for electrode_name, electrode in all_electrodes.items():
            # Track control_channel usage
            if electrode.control_channel is not None:
                if electrode.control_channel not in control_channel_users:
                    control_channel_users[electrode.control_channel] = []
                control_channel_users[electrode.control_channel].append(electrode_name)

            # Track measure_channel usage
            if electrode.measure_channel is not None:
                if electrode.measure_channel not in measure_channel_users:
                    measure_channel_users[electrode.measure_channel] = []
                measure_channel_users[electrode.measure_channel].append(electrode_name)

        # Find duplicates
        for channel, users in control_channel_users.items():
            if len(users) > 1:
                duplicates.extend(
                    [f"{user} control_channel {channel}" for user in users]
                )

        for channel, users in measure_channel_users.items():
            if len(users) > 1:
                duplicates.extend(
                    [f"{user} measure_channel {channel}" for user in users]
                )

        if duplicates:
            raise ValueError(f"Duplicate channels found: {', '.join(duplicates)}")

        return self

    @model_validator(mode="after")
    def validate_required_instruments(self) -> "DeviceConfig":
        control_instruments = [
            i for i in self.instruments if i.type == InstrumentType.CONTROL
        ]
        measurement_instruments = [
            i for i in self.instruments if i.type == InstrumentType.MEASUREMENT
        ]
        general_instruments = [
            i for i in self.instruments if i.type == InstrumentType.GENERAL
        ]

        has_control = len(control_instruments) > 0 or len(general_instruments) > 0
        has_measurement = (
            len(measurement_instruments) > 0 or len(general_instruments) > 0
        )

        if not has_control:
            raise ValueError("At least one CONTROL or GENERAL instrument is required")
        if not has_measurement:
            raise ValueError(
                "At least one MEASUREMENT or GENERAL instrument is required"
            )

        return self
