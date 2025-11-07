"""Tests for setup routines."""

from unittest.mock import MagicMock, patch

import pytest

from stanza.registry import ResourceRegistry, ResultsRegistry
from stanza.routines import RoutineContext
from stanza.routines.builtins.setup import setup_models_sdk


class MockLoggerSession:
    def __init__(self):
        self.measurements = []
        self.analyses = []

    def log_measurement(self, name, data):
        self.measurements.append((name, data))

    def log_analysis(self, name, data):
        self.analyses.append((name, data))


@pytest.fixture
def routine_context():
    return RoutineContext(ResourceRegistry(), ResultsRegistry())


class TestSetupModelsSDK:
    @patch("stanza.routines.builtins.setup.ConductorQuantum")
    def test_basic_setup(self, mock_cq_class, routine_context):
        """Test that setup_models_sdk creates and registers a client."""
        mock_client = MagicMock()
        mock_cq_class.return_value = mock_client

        token = "test-token-123"
        setup_models_sdk(routine_context, token=token)

        mock_cq_class.assert_called_once_with(token=token)

        assert hasattr(routine_context.resources, "models_client")
        assert routine_context.resources.models_client == mock_client

    @patch("stanza.routines.builtins.setup.ConductorQuantum")
    def test_setup_with_session(self, mock_cq_class, routine_context):
        """Test that setup_models_sdk works with a logger session."""
        mock_client = MagicMock()
        mock_cq_class.return_value = mock_client
        session = MockLoggerSession()

        token = "test-token-456"
        setup_models_sdk(routine_context, token=token, session=session)

        mock_cq_class.assert_called_once_with(token=token)

        assert hasattr(routine_context.resources, "models_client")
        assert routine_context.resources.models_client == mock_client

    @patch("stanza.routines.builtins.setup.ConductorQuantum")
    def test_setup_returns_none(self, mock_cq_class, routine_context):
        """Test that setup_models_sdk returns None."""
        mock_client = MagicMock()
        mock_cq_class.return_value = mock_client

        result = setup_models_sdk(routine_context, token="test-token")

        assert result is None

    @patch("stanza.routines.builtins.setup.ConductorQuantum")
    def test_client_accessible_after_setup(self, mock_cq_class, routine_context):
        """Test that the client can be accessed from context after setup."""
        mock_client = MagicMock()
        mock_client.some_method = MagicMock(return_value="test_value")
        mock_cq_class.return_value = mock_client

        setup_models_sdk(routine_context, token="test-token")

        client = routine_context.resources.models_client
        assert client.some_method() == "test_value"
        client.some_method.assert_called_once()

    @patch("stanza.routines.builtins.setup.ConductorQuantum")
    def test_setup_with_kwargs(self, mock_cq_class, routine_context):
        """Test that setup_models_sdk handles additional kwargs."""
        mock_client = MagicMock()
        mock_cq_class.return_value = mock_client

        setup_models_sdk(
            routine_context, token="test-token", extra_param="value", another_param=123
        )

        mock_cq_class.assert_called_once_with(token="test-token")
        assert hasattr(routine_context.resources, "models_client")
