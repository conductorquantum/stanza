from pathlib import Path
from unittest.mock import patch

import pytest

from stanza.utils import get_config_resource, substitute_parameters


class TestSubstituteParameters:
    """Test template parameter substitution utility."""

    def test_single_placeholder_substitution(self):
        template = "Hello <NAME>!"
        substitutions = {"NAME": "World"}

        result = substitute_parameters(template, substitutions)

        assert result == "Hello World!"

    def test_multiple_placeholder_substitution(self):
        template = "Connect to <HOST>:<PORT> using <PROTOCOL>"
        substitutions = {"HOST": "localhost", "PORT": 8080, "PROTOCOL": "HTTP"}

        result = substitute_parameters(template, substitutions)

        assert result == "Connect to localhost:8080 using HTTP"

    def test_repeated_placeholder_substitution(self):
        template = "<VALUE> + <VALUE> = <RESULT>"
        substitutions = {"VALUE": 5, "RESULT": 10}

        result = substitute_parameters(template, substitutions)

        assert result == "5 + 5 = 10"

    def test_no_placeholders_returns_original(self):
        template = "No placeholders here"
        substitutions = {"UNUSED": "value"}

        result = substitute_parameters(template, substitutions)

        assert result == "No placeholders here"

    def test_unused_substitutions_ignored(self):
        template = "Only <USED> placeholder"
        substitutions = {"USED": "active", "UNUSED": "ignored"}

        result = substitute_parameters(template, substitutions)

        assert result == "Only active placeholder"

    def test_missing_substitution_leaves_placeholder(self):
        assert (
            substitute_parameters("Missing <PLACEHOLDER> stays", {})
            == "Missing <PLACEHOLDER> stays"
        )

    def test_special_regex_characters_escaped(self):
        subs = {"REGEX": ".*+?^${}[]\\|()"}
        assert (
            substitute_parameters("Pattern <REGEX> test", subs)
            == "Pattern .*+?^${}[]\\|() test"
        )

    def test_numeric_values_converted_to_string(self):
        subs = {"NUM": 42, "FLOAT": 3.14159}
        assert (
            substitute_parameters("Number: <NUM>, Float: <FLOAT>", subs)
            == "Number: 42, Float: 3.14159"
        )

    def test_boolean_values_converted_to_string(self):
        subs = {"DEBUG": True, "ENABLED": False}
        assert (
            substitute_parameters("Debug: <DEBUG>, Enabled: <ENABLED>", subs)
            == "Debug: True, Enabled: False"
        )

    def test_none_value_converted_to_string(self):
        assert substitute_parameters("Value: <VALUE>", {"VALUE": None}) == "Value: None"

    def test_complex_nested_placeholders(self):
        template = "Config: {host: <HOST>, port: <PORT>, ssl: <SSL>}"
        subs = {"HOST": "api.example.com", "PORT": 443, "SSL": True}
        assert (
            substitute_parameters(template, subs)
            == "Config: {host: api.example.com, port: 443, ssl: True}"
        )

    def test_empty_template_and_substitutions(self):
        result = substitute_parameters("", {})
        assert result == ""

    def test_case_sensitive_placeholders(self):
        template = "<host> vs <HOST>"
        substitutions = {"host": "lower", "HOST": "UPPER"}

        result = substitute_parameters(template, substitutions)

        assert result == "lower vs UPPER"


class TestGetConfigResource:
    """Test configuration file resource loading."""

    @patch("stanza.utils.files")
    def test_reads_config_file_with_default_encoding(self, mock_files):
        mock_resource = mock_files.return_value.joinpath.return_value
        mock_resource.read_text.return_value = "config content"

        result = get_config_resource("test_config.json")

        assert result == "config content"
        mock_files.assert_called_once_with("stanza.configs")
        mock_files.return_value.joinpath.assert_called_once_with("test_config.json")
        mock_resource.read_text.assert_called_once_with("utf-8")

    @patch("stanza.utils.files")
    def test_reads_config_file_with_custom_encoding(self, mock_files):
        mock_resource = mock_files.return_value.joinpath.return_value
        mock_resource.read_text.return_value = "config content"

        result = get_config_resource("test_config.json", encoding="ascii")

        assert result == "config content"
        mock_resource.read_text.assert_called_once_with("ascii")

    @patch("stanza.utils.files")
    def test_handles_pathlib_path_input(self, mock_files):
        mock_resource = mock_files.return_value.joinpath.return_value
        mock_resource.read_text.return_value = "config content"
        path_input = Path("templates/config.yaml")

        result = get_config_resource(path_input)

        assert result == "config content"
        mock_files.return_value.joinpath.assert_called_once_with(
            "templates/config.yaml"
        )

    @patch("stanza.utils.files")
    def test_handles_nested_paths(self, mock_files):
        mock_resource = mock_files.return_value.joinpath.return_value
        mock_resource.read_text.return_value = "nested config"

        result = get_config_resource("templates/instruments/opx.json")

        assert result == "nested config"
        mock_files.return_value.joinpath.assert_called_once_with(
            "templates/instruments/opx.json"
        )

    @patch("stanza.utils.files")
    def test_propagates_file_not_found_error(self, mock_files):
        mock_resource = mock_files.return_value.joinpath.return_value
        mock_resource.read_text.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(FileNotFoundError, match="Config not found"):
            get_config_resource("nonexistent.json")

    @patch("stanza.utils.files")
    def test_propagates_encoding_error(self, mock_files):
        mock_resource = mock_files.return_value.joinpath.return_value
        mock_resource.read_text.side_effect = UnicodeDecodeError(
            "utf-8", b"", 0, 1, "invalid start byte"
        )

        with pytest.raises(UnicodeDecodeError):
            get_config_resource("invalid_encoding.json")

    @patch("stanza.utils.files")
    def test_handles_empty_config_file(self, mock_files):
        mock_resource = mock_files.return_value.joinpath.return_value
        mock_resource.read_text.return_value = ""

        result = get_config_resource("empty_config.json")

        assert result == ""
