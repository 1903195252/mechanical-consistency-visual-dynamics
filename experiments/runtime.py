"""Optional compute limits shared by the experiment command-line programs."""
import os
import time


def deadline():
    """Return an optional UNIX deadline supplied by the person running the job."""
    return float(os.environ.get("WAM_STOP_UNIX", "inf"))


def check_deadline():
    if time.time() >= deadline():
        raise TimeoutError("Configured compute deadline reached")
