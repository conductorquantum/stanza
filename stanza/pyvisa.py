from __future__ import annotations

from typing import cast

try:
    import pyvisa as visa
    from pyvisa.resources import MessageBasedResource
except ImportError:
    visa = None  # type: ignore[assignment]


class PyVisaDriver:
    def __init__(self, visa_addr: str, lib: str = "@py", sim_file: str | None = None):
        if sim_file:
            lib = f"{sim_file}@sim"
        if visa is None:
            raise ImportError(
                "pyvisa is not installed. Install with: pip install stanza[pyvisa]"
            )
        rm: visa.ResourceManager = visa.ResourceManager(lib)
        resource = rm.open_resource(visa_addr)
        self._visa: MessageBasedResource = cast(MessageBasedResource, resource)
        self._visa.write_termination = "\n"
        self._visa.read_termination = "\n"

        if visa_addr.find("ASRL") != -1:
            self._visa.baud_rate = 921600  # type: ignore[attr-defined]
            self._visa.send_end = False

    def query(self, cmd: str) -> str:
        result = self._visa.query(cmd)
        return str(result)

    def write(self, cmd: str) -> None:
        self._visa.write(cmd)

    def write_binary_values(self, cmd: str, values: list[int]) -> None:
        self._visa.write_binary_values(cmd, values)

    def close(self) -> None:
        self._visa.close()

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object | None
    ) -> None:
        self.close()
