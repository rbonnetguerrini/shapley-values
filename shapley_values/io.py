"""Save and load Shapley value results.

Handles JSON serialization of results dicts, including numpy arrays.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_results(
    results: Dict[str, Any],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save Shapley value results to JSON.

    Parameters
    ----------
    results : dict
        As returned by ExactShapley.compute(). Matplotlib figure objects (if any) are silently skipped.
    path : str
        Output file path. Parent directories are created if needed.
    metadata : dict, optional
        Extra metadata to include (e.g. application-specific parameters).

    Returns
    -------
    path : str
        The path written to.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Filter out non-serializable objects (figures, etc.)
    serializable = {}
    for k, v in results.items():
        try:
            json.dumps(v, cls=_NumpyEncoder)
            serializable[k] = v
        except (TypeError, ValueError):
            continue

    output = {
        "timestamp": datetime.now().isoformat(),
        "results": serializable,
    }
    if metadata is not None:
        output["metadata"] = metadata

    with open(path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    return path


def load_results(path: str) -> Dict[str, Any]:
    """Load Shapley value results from JSON.

    Parameters
    ----------
    path : str

    Returns
    -------
    data : dict
        Contains 'results' (with 'shapley_values' as np.ndarray), 'timestamp', and optionally 'metadata'.
    """
    with open(path) as f:
        data = json.load(f)

    # Convert lists back to numpy arrays where appropriate
    res = data.get("results", {})
    if "shapley_values" in res:
        res["shapley_values"] = np.array(res["shapley_values"])

    return data
