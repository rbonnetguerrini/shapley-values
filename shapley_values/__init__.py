"""shapley_values: Compute "exact" Shapley value as defined in game theory.

Problem-agnostic implementation of exact Shapley values for cooperative
game theory.
"""

from .core import ExactShapley, power_set
from .io import save_results, load_results


def plot_shapley_bar(*args, **kwargs):
    from .plotting import plot_shapley_bar as _plot_shapley_bar
    return _plot_shapley_bar(*args, **kwargs)


def plot_shapley_pie(*args, **kwargs):
    from .plotting import plot_shapley_pie as _plot_shapley_pie
    return _plot_shapley_pie(*args, **kwargs)


def plot_shapley_comparison(*args, **kwargs):
    from .plotting import plot_shapley_comparison as _plot_shapley_comparison
    return _plot_shapley_comparison(*args, **kwargs)


def plot_marginal_contributions(*args, **kwargs):
    from .plotting import plot_marginal_contributions as _plot_marginal_contributions
    return _plot_marginal_contributions(*args, **kwargs)

__all__ = [
    "ExactShapley",
    "power_set",
    "plot_shapley_bar",
    "plot_shapley_pie",
    "plot_shapley_comparison",
    "plot_marginal_contributions",
    "save_results",
    "load_results",
]
