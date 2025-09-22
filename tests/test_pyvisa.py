from unittest.mock import Mock, patch

import pytest

from stanza.pyvisa import PyVisaDriver


class TestPyVisaDriver:
    @patch("stanza.pyvisa.visa")
    def test_initialization_tcp_socket(self, mock_visa):
        mock_rm = Mock()
        mock_resource = Mock()
        mock_visa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_resource

        PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")

        mock_visa.ResourceManager.assert_called_once_with("@py")
        mock_rm.open_resource.assert_called_once_with(
            "TCPIP::192.168.1.1::5025::SOCKET"
        )
        assert mock_resource.write_termination == "\n"
        assert mock_resource.read_termination == "\n"

    @patch("stanza.pyvisa.visa")
    def test_initialization_serial(self, mock_visa):
        mock_rm = Mock()
        mock_resource = Mock()
        mock_visa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_resource

        PyVisaDriver("ASRL2::INSTR")

        mock_rm.open_resource.assert_called_once_with("ASRL2::INSTR")
        assert mock_resource.baud_rate == 921600
        assert mock_resource.send_end is False

    @patch("stanza.pyvisa.visa")
    def test_query(self, mock_visa):
        mock_rm = Mock()
        mock_resource = Mock()
        mock_visa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_resource
        mock_resource.query.return_value = "Test Response"

        driver = PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
        result = driver.query("*IDN?")

        mock_resource.query.assert_called_once_with("*IDN?")
        assert result == "Test Response"

    @patch("stanza.pyvisa.visa")
    def test_write(self, mock_visa):
        mock_rm = Mock()
        mock_resource = Mock()
        mock_visa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_resource

        driver = PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
        driver.write("SOUR:VOLT 1.5")

        mock_resource.write.assert_called_once_with("SOUR:VOLT 1.5")

    @patch("stanza.pyvisa.visa")
    def test_close(self, mock_visa):
        mock_rm = Mock()
        mock_resource = Mock()
        mock_visa.ResourceManager.return_value = mock_rm
        mock_rm.open_resource.return_value = mock_resource

        driver = PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
        driver.close()

        mock_resource.close.assert_called_once()

    def test_visa_not_available(self):
        with patch("stanza.pyvisa.visa", None):
            with pytest.raises(AttributeError):
                PyVisaDriver("TCPIP::192.168.1.1::5025::SOCKET")
