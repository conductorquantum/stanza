from unittest.mock import Mock, patch

import pytest

from stanza.pyvisa import PyVisaDriver


@pytest.fixture
def mock_visa():
    with patch("stanza.pyvisa.visa") as mock:
        mock_rm, mock_resource = Mock(), Mock()
        mock.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_resource
        yield mock, mock_rm, mock_resource


class TestPyVisaDriver:
    def test_initialization_tcp_socket(self, mock_visa):
        _, mock_rm, mock_resource = mock_visa

        PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")

        mock_rm.open_resource.assert_called_once_with(
            "TCPIP::192.168.1.1::5025::SOCKET"
        )
        assert (mock_resource.write_termination, mock_resource.read_termination) == (
            "\n",
            "\n",
        )

    def test_initialization_serial(self, mock_visa):
        _, mock_rm, mock_resource = mock_visa

        PyVisaDriver("ASRL2::INSTR")

        assert (mock_resource.baud_rate, mock_resource.send_end) == (921600, False)

    def test_query(self, mock_visa):
        _, _, mock_resource = mock_visa
        mock_resource.query.return_value = "Test Response"

        driver = PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")

        assert driver.query("*IDN?") == "Test Response"
        mock_resource.query.assert_called_once_with("*IDN?")

    def test_write(self, mock_visa):
        _, _, mock_resource = mock_visa

        driver = PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
        driver.write("SOUR:VOLT 1.5")

        mock_resource.write.assert_called_once_with("SOUR:VOLT 1.5")

    def test_close(self, mock_visa):
        _, _, mock_resource = mock_visa

        driver = PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
        driver.close()

        mock_resource.close.assert_called_once()

    def test_write_binary_values(self, mock_visa):
        _, _, mock_resource = mock_visa

        driver = PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
        values = [1, 2, 3, 4]
        driver.write_binary_values("DATA:BINARY", values)

        mock_resource.write_binary_values.assert_called_once_with("DATA:BINARY", values)

    def test_context_manager_exit(self, mock_visa):
        _, _, mock_resource = mock_visa

        driver = PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
        driver.__exit__(None, None, None)

        mock_resource.close.assert_called_once()

    def test_initialization_with_sim_file(self, mock_visa):
        mock_visa_module, mock_rm, _ = mock_visa

        PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET", sim_file="/path/to/sim")

        mock_visa_module.ResourceManager.assert_called_once_with("/path/to/sim@sim")

    def test_visa_not_available(self):
        with patch("stanza.pyvisa.visa", None):
            with pytest.raises(ImportError):
                PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
