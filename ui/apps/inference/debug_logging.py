"""Debug logging utilities to trace app freeze points.

Add extensive logging to identify exactly where the app freezes.
"""

import logging
import sys
import time
from functools import wraps

# Configure logging to write to file AND console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("/tmp/streamlit_debug.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

LOGGER = logging.getLogger("STREAMLIT_DEBUG")


def trace_call(func):
    """Decorator to trace function entry/exit and timing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        LOGGER.info(f">>> ENTERING {func_name}")
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            LOGGER.info(f"<<< EXITING {func_name} (took {elapsed:.3f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start
            LOGGER.error(f"!!! EXCEPTION in {func_name} after {elapsed:.3f}s: {e}")
            raise

    return wrapper


def log_checkpoint(message: str):
    """Log a checkpoint message."""
    LOGGER.info(f"⬤ CHECKPOINT: {message}")


def log_memory_usage():
    """Log current memory usage."""
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        LOGGER.info(f"📊 Memory: {mem_mb:.1f} MB")
    except Exception:
        pass


# Enable at module import
LOGGER.info("=" * 80)
LOGGER.info("STREAMLIT DEBUG LOGGING ENABLED")
LOGGER.info("Log file: /tmp/streamlit_debug.log")
LOGGER.info("=" * 80)
