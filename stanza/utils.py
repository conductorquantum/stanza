import re
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from stanza.instruments.channels import ChannelConfig
from stanza.models import DeviceConfig, PadType


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


def load_device_config(file_path: str | Path) -> DeviceConfig:
    """Load a device configuration YAML file.

    Args:
        file_path: The path to the device configuration YAML file.

    Raises:
        ValueError: If the file extension is not .yaml or .yml.
        ValueError: If the file cannot be loaded.

    Returns:
        The device configuration.
    """
    if PurePosixPath(file_path).suffix not in [".yaml", ".yml"]:
        raise ValueError(
            f"Invalid file extension for device config: {file_path}. Expected .yaml or .yml"
        )

    try:
        with open(file_path) as f:
            yaml_file = yaml.safe_load(f)
        return DeviceConfig.model_validate(yaml_file)
    except Exception as e:
        raise ValueError(f"Failed to load device config from {file_path}: {e}") from e


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
            voltage_range=(gate.v_lower_bound, gate.v_upper_bound),
            pad_type=PadType.GATE,
            electrode_type=gate.type,
            output_mode="dc",
            enabled=True,
            readout=gate.readout,
        )

    for contact_name, contact in device_config.contacts.items():
        channel_configs[contact_name] = ChannelConfig(
            name=contact_name,
            control_channel=contact.control_channel,
            measure_channel=contact.measure_channel,
            voltage_range=(contact.v_lower_bound, contact.v_upper_bound),
            pad_type=PadType.CONTACT,
            electrode_type=contact.type,
            output_mode="dc",
            enabled=True,
            readout=contact.readout,
        )
    return channel_configs
