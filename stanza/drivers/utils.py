import time
from warnings import warn

import numpy as np

try:
    from qm.jobs.running_qm_job import RunningQmJob
except ImportError:
    RunningQmJob = None  # type: ignore[misc,assignment]


def wait_until_job_is_paused(
    job: RunningQmJob, timeout: int = 30, strict_timeout: bool = True
) -> bool:
    """Waits until the OPX FPGA reaches a "pause" statement.

    Used when the OPX sequence needs to be synchronized with an external parameter sweep and to ensure that the OPX
    sequence is done before proceeding to the next iteration of the external loop, or when using IO variables:

    Args:
        running_job: the QM running job object.
        timeout: Duration in seconds after which the console will be freed even if the pause statement has not been
                 reached to prevent from being stuck here forever.
        strict_timeout: Will throw and exception is set to True, otherwise it will just a print a warning.

    Returns:
        True when the pause statement has been reached.
    """
    start = time.time()
    delay = 0.0
    while (not job.is_paused()) and (delay < timeout):
        time.sleep(0.1)
        delay = time.time() - start
    if delay > timeout:
        if strict_timeout:
            raise TimeoutError(
                f"Timeout ({timeout}s) was reached, consider extending it if it was not intended."
            )
        else:
            warn(
                f"Timeout ({timeout}s) was reached, consider extending it if it was not intended.",
                stacklevel=2,
            )
    return True


def demod2volts(
    data: float | np.ndarray,
    duration: float | int,
    single_demod: bool = False,
) -> float | np.ndarray:
    """Converts the demodulated data to volts.

    Args:
        data: demodulated data. Must be a python variable or array.
        duration: demodulation duration in ns. **WARNING**: this must be the duration of one slice in
                  the case of ```demod.sliced``` and ```demod.accumulated```.
        single_demod: Flag to add the additional factor of 2 needed for single demod.

    Returns:
        The demodulated data in volts.
    """
    if single_demod:
        return 2 * 4096 * data * 1 / duration
    else:
        return 4096 * data * 1 / duration
