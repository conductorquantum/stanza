import re
from importlib.resources import files
from pathlib import Path
from typing import Any


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
