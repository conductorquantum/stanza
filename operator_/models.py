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
    readout: bool
    v_lower_bound: float | None
    v_upper_bound: float | None

    @model_validator(mode="after")
    def validate_readout_requires_measure_channel(self):
        if self.readout and self.measure_channel is None:
            raise ValueError("`measure_channel` must be specified when readout=True")
        return self

    @model_validator(mode="after")
    def validate_control_channel_requires_measure_channel(self):
        if not self.readout and self.control_channel is None:
            raise ValueError("`control_channel` must be specified when readout=False")
        return self

    @model_validator(mode="after")
    def validate_control_channel_requires_v_lower_bound_and_v_upper_bound(self):
        if (
            not self.readout
            and self.control_channel is not None
            and self.v_lower_bound is None
        ):
            raise ValueError(
                "`v_lower_bound` must be specified when control_channel is set"
            )
        if (
            not self.readout
            and self.control_channel is not None
            and self.v_upper_bound is None
        ):
            raise ValueError(
                "`v_upper_bound` must be specified when control_channel is set"
            )
        return self


class GateType(str, Enum):
    PLUNGER = "PLUNGER"
    BARRIER = "BARRIER"
    RESERVOIR = "RESERVOIR"
    SCREEN = "SCREEN"


class ContactType(str, Enum):
    SOURCE = "SOURCE"
    DRAIN = "DRAIN"


class InstrumentType(str, Enum):
    CONTROL = "CONTROL"
    MEASUREMENT = "MEASUREMENT"


class Gate(Electrode):
    type: GateType


class Contact(Electrode):
    type: ContactType


class ExperimentConfig(BaseModelWithConfig):
    name: str
    parameters: dict[str, Any | str] | None = None
    experiments: list["ExperimentConfig"] | None = None


class BaseInstrumentConfig(BaseModelWithConfig):
    """Base instrument configuration with discriminator."""

    name: str
    ip_addr: str | None = None
    serial_addr: str | None = None
    port: int | None = None
    type: InstrumentType

    @model_validator(mode="after")
    def check_comm_type(cls, properties):
        if not (properties.ip_addr or properties.serial_addr):
            raise ValueError("Either 'ip_addr' or 'serial_addr' must be provided")
        return properties


class MeasurementInstrumentConfig(BaseInstrumentConfig):
    """Instrument configuration for measurement instruments with required timing parameters."""

    type: Literal[InstrumentType.MEASUREMENT] = InstrumentType.MEASUREMENT
    measurement_duration: float = Field(
        gt=0, description="Total measurement duration per point in seconds"
    )
    sample_time: float = Field(gt=0, description="Individual sample time in seconds")
    conversion_factor: float = Field(
        gt=0, default=1, description="The conversion factor from ADC counts to amperes"
    )

    @model_validator(mode="after")
    def validate_timing_constraints(self):
        """Validate logical constraints between timing parameters."""
        if self.sample_time > self.measurement_duration:
            raise ValueError(
                f"sample_time ({self.sample_time}s) cannot be larger than "
                f"measurement_duration ({self.measurement_duration}s)"
            )
        return self


class ControlInstrumentConfig(BaseInstrumentConfig):
    """Instrument configuration for control instruments."""

    type: Literal[InstrumentType.CONTROL] = InstrumentType.CONTROL
    slew_rate: float = Field(gt=0, description="Slew rate in V/s")


InstrumentConfig = Annotated[
    ControlInstrumentConfig | MeasurementInstrumentConfig, Discriminator("type")
]


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str
    gates: dict[str, Gate]
    contacts: dict[str, Contact]
    experiments: list[ExperimentConfig]
    instruments: list[InstrumentConfig]

    @model_validator(mode="after")
    def validate_unique_channels(self):
        """Ensure that all channels are unique across gates and contacts"""
        control_channel_users = {}
        measure_channel_users = {}
        duplicates = []

        all_electrodes = {
            **{f"gate '{name}'": electrode for name, electrode in self.gates.items()},
            **{
                f"contact '{name}'": electrode
                for name, electrode in self.contacts.items()
            },
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
                duplicates.extend([f"{user} control_channel {channel}" for user in users])

        for channel, users in measure_channel_users.items():
            if len(users) > 1:
                duplicates.extend([f"{user} measure_channel {channel}" for user in users])

        if duplicates:
            raise ValueError(f"Duplicate channels found: {', '.join(duplicates)}")

        return self

    @model_validator(mode="after")
    def validate_required_instruments(self):
        """Ensure at least one control and one measurement instrument"""
        control_instruments = [
            i for i in self.instruments if i.type == InstrumentType.CONTROL
        ]
        measurement_instruments = [
            i for i in self.instruments if i.type == InstrumentType.MEASUREMENT
        ]

        if not control_instruments:
            raise ValueError("At least one control instrument is required")
        if not measurement_instruments:
            raise ValueError("At least one measurement instrument is required")

        return self
