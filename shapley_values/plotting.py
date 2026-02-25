"""Generic Shapley value visualizations.

All functions take a results dict (as returned by ExactShapley.compute()) or raw arrays. No application-specific knowledge.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_shapley_bar(
    shapley_values: np.ndarray,
    player_labels: List[str],
    title: Optional[str] = None,
    positive_color: str = "green",
    negative_color: str = "red",
    alpha: float = 0.7,
    ylabel: str = "Shapley Value",
    figsize: Tuple[int, int] = (12, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar chart of Shapley values, colored by sign.

    Parameters
    ----------
    shapley_values : array-like of shape (n_players,)
    player_labels : list of str
    title : str, optional
    positive_color, negative_color : str
    alpha : float
    ylabel : str
    figsize : tuple
    ax : matplotlib Axes, optional
        If provided, plot on this axes instead of creating a new figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    sv = np.asarray(shapley_values)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bars = ax.bar(player_labels, sv)
    for bar, val in zip(bars, sv):
        bar.set_color(positive_color if val > 0 else negative_color)
        bar.set_alpha(alpha)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", ls="--", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if title:
        ax.set_title(title)

    pos_patch = mpatches.Patch(
        color=positive_color, alpha=alpha, label="SV > 0"
    )
    neg_patch = mpatches.Patch(
        color=negative_color, alpha=alpha, label="SV < 0"
    )
    ax.legend(handles=[pos_patch, neg_patch], loc="upper right")

    fig.tight_layout()
    return fig


def plot_shapley_pie(
    shapley_values: np.ndarray,
    player_labels: List[str],
    title: Optional[str] = "Relative Importance (|SV|)",
    figsize: Tuple[int, int] = (8, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Pie chart of |SV| proportions.

    Parameters
    ----------
    shapley_values : array-like of shape (n_players,)
    player_labels : list of str
    title : str, optional
    figsize : tuple
    ax : matplotlib Axes, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    sv = np.abs(np.asarray(shapley_values))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = plt.cm.Set3(np.linspace(0, 1, len(sv)))
    ax.pie(sv, labels=player_labels, autopct="%1.1f%%", colors=colors)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_shapley_comparison(
    results_list: List[Dict],
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Side-by-side bar charts comparing multiple Shapley analyses.

    Parameters
    ----------
    results_list : list of dict
        Each dict has keys 'shapley_values' and 'player_short' (or
        'player_labels').
    titles : list of str, optional
        Title for each subplot.
    figsize : tuple, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(results_list)
    if figsize is None:
        figsize = (7 * n, 5)
    if titles is None:
        titles = [f"Analysis {i+1}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, res, ttl in zip(axes, results_list, titles):
        sv = res["shapley_values"]
        labels = res.get("player_short", res.get("player_labels"))
        plot_shapley_bar(sv, labels, title=ttl, ax=ax)

    fig.tight_layout()
    return fig


def plot_marginal_contributions(
    deltas: np.ndarray,
    player_labels: List[str],
    title: str = "Single-player marginal contributions",
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Bar chart of single-player marginal contributions.

    Parameters
    ----------
    deltas : array-like of shape (n_players,)
        v({i}) - v({}) for each player.
    player_labels : list of str
    title : str
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    return plot_shapley_bar(
        deltas, player_labels, title=title, figsize=figsize,
        ylabel="Delta (marginal contribution)",
    )
