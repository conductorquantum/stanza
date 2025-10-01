from pathlib import Path
from unittest.mock import patch

import pytest

from stanza.utils import get_config_resource, load_device_config, substitute_parameters


@pytest.fixture
def mock_resource():
    with patch("stanza.utils.files") as mock_files:
        mock_res = mock_files.return_value.joinpath.return_value
        yield mock_files, mock_res


class TestSubstituteParameters:
    def test_single_placeholder(self):
        assert (
            substitute_parameters("Hello <NAME>!", {"NAME": "World"}) == "Hello World!"
        )

    def test_multiple_placeholders(self):
        template = "Connect to <HOST>:<PORT> using <PROTOCOL>"
        subs = {"HOST": "localhost", "PORT": 8080, "PROTOCOL": "HTTP"}
        assert (
            substitute_parameters(template, subs)
            == "Connect to localhost:8080 using HTTP"
        )

    def test_repeated_placeholder(self):
        assert (
            substitute_parameters(
                "<VALUE> + <VALUE> = <RESULT>", {"VALUE": 5, "RESULT": 10}
            )
            == "5 + 5 = 10"
        )

    def test_no_placeholders(self):
        assert (
            substitute_parameters("No placeholders here", {"UNUSED": "value"})
            == "No placeholders here"
        )

    def test_unused_substitutions(self):
        assert (
            substitute_parameters(
                "Only <USED>", {"USED": "active", "UNUSED": "ignored"}
            )
            == "Only active"
        )

    def test_missing_substitution(self):
        assert (
            substitute_parameters("Missing <PLACEHOLDER>", {})
            == "Missing <PLACEHOLDER>"
        )

    def test_special_regex_characters(self):
        assert (
            substitute_parameters("Pattern <REGEX>", {"REGEX": ".*+?^${}[]\\|()"})
            == "Pattern .*+?^${}[]\\|()"
        )

    def test_numeric_values(self):
        assert (
            substitute_parameters(
                "Number: <NUM>, Float: <FLOAT>", {"NUM": 42, "FLOAT": 3.14}
            )
            == "Number: 42, Float: 3.14"
        )

    def test_boolean_values(self):
        assert (
            substitute_parameters(
                "Debug: <DEBUG>, Enabled: <ENABLED>", {"DEBUG": True, "ENABLED": False}
            )
            == "Debug: True, Enabled: False"
        )

    def test_none_value(self):
        assert substitute_parameters("Value: <VALUE>", {"VALUE": None}) == "Value: None"

    def test_empty_template(self):
        assert substitute_parameters("", {}) == ""

    def test_case_sensitive(self):
        assert (
            substitute_parameters(
                "<host> vs <HOST>", {"host": "lower", "HOST": "UPPER"}
            )
            == "lower vs UPPER"
        )


class TestGetConfigResource:
    def test_reads_config_with_default_encoding(self, mock_resource):
        mock_files, mock_res = mock_resource
        mock_res.read_text.return_value = "config content"

        result = get_config_resource("test_config.json")

        assert result == "config content"
        mock_files.assert_called_once_with("stanza.configs")
        mock_res.read_text.assert_called_once_with("utf-8")

    def test_reads_config_with_custom_encoding(self, mock_resource):
        _, mock_res = mock_resource
        mock_res.read_text.return_value = "config content"

        result = get_config_resource("test_config.json", encoding="ascii")

        assert result == "config content"
        mock_res.read_text.assert_called_once_with("ascii")

    def test_handles_pathlib_path(self, mock_resource):
        mock_files, mock_res = mock_resource
        mock_res.read_text.return_value = "config content"

        result = get_config_resource(Path("templates/config.yaml"))

        assert result == "config content"
        mock_files.return_value.joinpath.assert_called_once_with(
            "templates/config.yaml"
        )

    def test_handles_nested_paths(self, mock_resource):
        _, mock_res = mock_resource
        mock_res.read_text.return_value = "nested config"
        assert get_config_resource("templates/instruments/opx.json") == "nested config"

    def test_propagates_file_not_found_error(self, mock_resource):
        _, mock_res = mock_resource
        mock_res.read_text.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(FileNotFoundError, match="Config not found"):
            get_config_resource("nonexistent.json")

    def test_propagates_encoding_error(self, mock_resource):
        _, mock_res = mock_resource
        mock_res.read_text.side_effect = UnicodeDecodeError(
            "utf-8", b"", 0, 1, "invalid start byte"
        )

        with pytest.raises(UnicodeDecodeError):
            get_config_resource("invalid_encoding.json")

    def test_handles_empty_config(self, mock_resource):
        _, mock_res = mock_resource
        mock_res.read_text.return_value = ""
        assert get_config_resource("empty_config.json") == ""


class TestLoadDeviceConfig:
    def test_loads_yaml_config(self, valid_device_yaml, tmp_path):
        config_file = tmp_path / "device.yaml"
        config_file.write_text(valid_device_yaml)

        result = load_device_config(str(config_file))

        assert result.name == "test_device"
        assert "G1" in result.gates
        assert result.gates["G1"].control_channel == 1
        assert "C1" in result.contacts
        assert result.contacts["C1"].readout is True

    def test_loads_yml_extension(self, valid_device_yaml, tmp_path):
        config_file = tmp_path / "device.yml"
        config_file.write_text(valid_device_yaml)
        assert load_device_config(str(config_file)).name == "test_device"

    def test_handles_pathlib_path(self, valid_device_yaml, tmp_path):
        config_file = tmp_path / "device.yaml"
        config_file.write_text(valid_device_yaml)
        assert load_device_config(config_file).name == "test_device"

    def test_handles_nested_path(self, valid_device_yaml, tmp_path):
        nested_dir = tmp_path / "configs" / "devices"
        nested_dir.mkdir(parents=True)
        config_file = nested_dir / "device.yaml"
        config_file.write_text(valid_device_yaml)
        assert load_device_config(str(config_file)).name == "test_device"

    def test_rejects_invalid_extension(self):
        with pytest.raises(
            ValueError, match="Invalid file extension.*Expected .yaml or .yml"
        ):
            load_device_config("/path/to/config.json")

    def test_rejects_no_extension(self):
        with pytest.raises(
            ValueError, match="Invalid file extension.*Expected .yaml or .yml"
        ):
            load_device_config("/path/to/config")

    def test_rejects_txt_extension(self):
        with pytest.raises(
            ValueError, match="Invalid file extension.*Expected .yaml or .yml"
        ):
            load_device_config("/path/to/device.txt")

    def test_raises_error_for_nonexistent_file(self):
        with pytest.raises(ValueError, match="Failed to load device config"):
            load_device_config("nonexistent.yaml")

    def test_raises_error_for_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("{ invalid: yaml: content:")

        with pytest.raises(ValueError, match="Failed to load device config"):
            load_device_config(str(config_file))

    def test_raises_error_for_invalid_schema(self, tmp_path):
        config_file = tmp_path / "invalid_schema.yaml"
        config_file.write_text("name: test\ngates: {}\ncontacts: {}\nroutines: []\n")

        with pytest.raises(ValueError, match="Failed to load device config"):
            load_device_config(str(config_file))
