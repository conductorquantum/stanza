class InstrumentError(RuntimeError):
    """Exception raised when an instrument operation fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
