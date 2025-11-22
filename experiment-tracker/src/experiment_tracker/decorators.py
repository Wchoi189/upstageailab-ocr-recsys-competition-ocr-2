import functools
import traceback
from pathlib import Path

from .core import ExperimentTracker
from .utils.path_utils import ExperimentPaths


def track_experiment(experiment_id: str | None = None, task_id: str | None = None):
    """
    Decorator to automatically track experiment execution.

    Args:
        experiment_id: ID of the experiment. If None, attempts to auto-detect.
        task_id: Optional task ID to associate with this run.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal experiment_id

            # Auto-detect experiment ID if not provided
            if experiment_id is None:
                # Try detecting from CWD
                experiment_id = ExperimentPaths.detect_experiment_id()

                # If not found, try detecting from the script file path
                if experiment_id is None and hasattr(func, "__code__"):
                    script_path = Path(func.__code__.co_filename)
                    experiment_id = ExperimentPaths.detect_experiment_id(script_path)

            if experiment_id is None:
                print("Warning: Could not detect experiment ID. Tracking disabled.")
                return func(*args, **kwargs)

            tracker = ExperimentTracker()

            # Log start
            tracker.log_insight(f"Started execution of {func.__name__}", category="execution", experiment_id=experiment_id)

            try:
                result = func(*args, **kwargs)
                tracker.log_insight(f"Completed execution of {func.__name__}", category="execution", experiment_id=experiment_id)
                return result
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                tracker.log_insight(error_msg, category="error", experiment_id=experiment_id)
                raise e

        return wrapper

    return decorator
