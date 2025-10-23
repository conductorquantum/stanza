from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from stanza.base.channels import ChannelConfig
from stanza.base.registry import load_driver_class, validate_driver_protocols
from stanza.device import Device
from stanza.models import DeviceConfig, InstrumentType, PadType


def substitute_parameters(template: str, substitutions: dict[str, Any]) -> str:
    """Substitute parameters in a template string.

    Args:
        template: The template string to substitute parameters in.
        substitutions: A dictionary of substitutions to make.

    Returns:
        The substituted string.
    """

    content = template
    for placeholder, value in substitutions.items():
        # Match <PLACEHOLDER> patterns and substitute with value
        pattern = f"<{re.escape(placeholder)}>"
        content = re.sub(pattern, str(value), content)

    return content


def get_config_resource(config_path: str | Path, encoding: str = "utf-8") -> str:
    """Get config file content

    Args:
        config_path: The path to the config file.
        encoding: The encoding of the config file.

    Returns:
        The config file content.
    """
    resource = files("stanza.configs").joinpath(str(config_path))
    return resource.read_text(encoding)


def load_device_config(
    config_path: str | Path, is_stanza_config: bool = False
) -> DeviceConfig:
    """Load a device configuration YAML file.

    Args:
        file_content: The content of the device configuration YAML file.
        is_stanza_config: Whether the config is a stanza config.

    Raises:
        ValueError: If the file cannot be loaded.

    Returns:
        The device configuration.
    """
    try:
        file_content = (
            get_config_resource(config_path)
            if is_stanza_config
            else Path(config_path).read_text()
        )
        yaml_file = yaml.safe_load(file_content)
        return DeviceConfig.model_validate(yaml_file)
    except Exception as e:
        raise ValueError(f"Failed to load device config: {e}") from e


def generate_channel_configs(device_config: DeviceConfig) -> dict[str, ChannelConfig]:
    """Generate ChannelConfigs for the device.

    Args:
        device_config: The device configuration.

    Returns:
        A dictionary mapping of gate/contact name to ChannelConfigs.
    """
    channel_configs = {}
    for gate_name, gate in device_config.gates.items():
        channel_configs[gate_name] = ChannelConfig(
            name=gate_name,
            control_channel=gate.control_channel,
            measure_channel=gate.measure_channel,
            breakout_channel=gate.breakout_channel,
            voltage_range=(gate.v_lower_bound, gate.v_upper_bound),
            pad_type=PadType.GATE,
            electrode_type=gate.type,
            output_mode="dc",
            enabled=True,
        )

    for contact_name, contact in device_config.contacts.items():
        channel_configs[contact_name] = ChannelConfig(
            name=contact_name,
            control_channel=contact.control_channel,
            measure_channel=contact.measure_channel,
            breakout_channel=contact.breakout_channel,
            voltage_range=(contact.v_lower_bound, contact.v_upper_bound),
            pad_type=PadType.CONTACT,
            electrode_type=contact.type,
            output_mode="dc",
            enabled=True,
        )

    for gpio_name, gpio in device_config.gpios.items():
        channel_configs[gpio_name] = ChannelConfig(
            name=gpio_name,
            control_channel=gpio.control_channel,
            measure_channel=gpio.measure_channel,
            breakout_channel=gpio.breakout_channel,
            voltage_range=(gpio.v_lower_bound, gpio.v_upper_bound),
            pad_type=PadType.GPIO,
            electrode_type=gpio.type,
            output_mode="digital",
            enabled=True,
        )
    return channel_configs


def device_from_config(
    device_config: DeviceConfig,
    **driver_kwargs: Any,
) -> Device:
    """Create a device from a DeviceConfig object.

    Args:
        device_config: The device configuration object.
        **driver_kwargs: Additional keyword arguments to pass to driver constructors.

    Returns:
        A configured Device instance with instantiated instruments.

    Raises:
        ValueError: If required driver field is missing or instruments cannot be instantiated.
    """
    channel_configs = generate_channel_configs(device_config)

    control_instrument = None
    measurement_instrument = None
    breakout_box_instrument = None

    for instrument_config in device_config.instruments:
        if instrument_config.driver is None:
            raise ValueError(
                f"Instrument '{instrument_config.name}' missing driver field"
            )

        driver_class = load_driver_class(instrument_config.driver)
        validate_driver_protocols(driver_class, instrument_config.type)
        instrument = driver_class(instrument_config, channel_configs, **driver_kwargs)

        match instrument_config.type:
            case InstrumentType.BREAKOUT_BOX if breakout_box_instrument is None:
                breakout_box_instrument = instrument
            case InstrumentType.CONTROL if control_instrument is None:
                control_instrument = instrument
            case InstrumentType.MEASUREMENT if measurement_instrument is None:
                measurement_instrument = instrument
            case InstrumentType.GENERAL if (
                control_instrument is None and measurement_instrument is None
            ):
                control_instrument = instrument
                measurement_instrument = instrument
            case _:
                raise ValueError(
                    f"Instrument '{instrument_config.name}' is not a valid instrument type"
                )

    return Device(
        name=device_config.name,
        device_config=device_config,
        channel_configs=channel_configs,
        control_instrument=control_instrument,
        measurement_instrument=measurement_instrument,
        breakout_box_instrument=breakout_box_instrument,
    )


def device_from_yaml(
    config_path: str | Path,
    is_stanza_config: bool = False,
    **driver_kwargs: Any,
) -> Device:
    """Load a device from a YAML configuration file.

    Args:
        config_path: Path to the device configuration YAML file.
        is_stanza_config: Whether the config is a stanza config.
        **driver_kwargs: Additional keyword arguments to pass to driver constructors.

    Returns:
        A configured Device instance with instantiated instruments.

    Raises:
        ValueError: If required driver field is missing or instruments cannot be instantiated.
    """
    device_config = load_device_config(config_path, is_stanza_config)
    return device_from_config(device_config, **driver_kwargs)
