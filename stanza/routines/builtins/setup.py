from typing import Any

from conductorquantum import ConductorQuantum

from stanza.logger.session import LoggerSession
from stanza.routines.core import RoutineContext, routine


@routine
def setup_models_sdk(
    ctx: RoutineContext, token: str, session: LoggerSession | None = None, **kwargs: Any
) -> None:
    """
    Instantiate the Conductor Quantum SDK and add it to the context.

    Args:
        ctx: The context object
        token: The token for the conductor quantum models client
    """
    client = ConductorQuantum(token=token)
    ctx.resources.add("models_client", client)
