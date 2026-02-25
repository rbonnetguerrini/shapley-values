"""Exact Shapley value computation.

Problem-agnostic implementation of exact Shapley values for cooperative game theory. Given N players and a value function v(S) -> float, computes the marginal contribution of each player using the combinatorial formula:

    SV_j = sum_{S not containing j} [|S|!(n-|S|-1)!/n!] * [v(S u {j}) - v(S)]

This module is model agnostic.
The user supplies a value function and player labels.
"""

import math
import random
import time
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Coalition enumeration
# ---------------------------------------------------------------------------

def power_set(
    players: Sequence[int],
    num_sets: Optional[int] = None,
) -> List[Set[int]]:
    """Enumerate all coalitions (subsets) of a set of players.

    Parameters
    ----------
    players : sequence of int
        Player indices (e.g. range(n)).
    num_sets : int, optional
        If given, return a random sample of this many subsets instead of the full power set. Useful for approximate SV computation on large player counts.

    Returns
    -------
    list of set
        All subsets of ``players`` excluding the full set, or a random
        sample thereof.
    """
    s = list(players)
    if num_sets is None:
        return [
            set(comb) for r in range(0, len(s)) for comb in combinations(s, r)
        ]
    subsets: set = {frozenset()}
    while len(subsets) < num_sets:
        r = random.randint(0, len(s) - 1)
        subsets.add(frozenset(random.sample(s, r)) if r > 0 else frozenset())
    return [set(sub) for sub in subsets]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ExactShapley:
    """Exact Shapley value calculator.

    Parameters
    ----------
    n_players : int
        Number of players in the cooperative game.
    value_function : callable
        Maps a list of player indices (the coalition) to a scalar value.
        Signature: ``v(List[int]) -> float``.
        The function should accept an empty list for the baseline (empty
        coalition) value.
    player_labels : list of str, optional
        Human-readable labels for each player (length ``n_players``).
        Defaults to ``["P0", "P1", ...]``.
    player_short : list of str, optional
        Short labels for plots (length ``n_players``).
        Defaults to ``player_labels``.
    """

    def __init__(
        self,
        n_players: int,
        value_function: Callable[[List[int]], float],
        player_labels: Optional[List[str]] = None,
        player_short: Optional[List[str]] = None,
    ):
        self.n_players = n_players
        self.value_function = value_function

        if player_labels is None:
            player_labels = [f"P{i}" for i in range(n_players)]
        if len(player_labels) != n_players:
            raise ValueError(
                f"player_labels length ({len(player_labels)}) != "
                f"n_players ({n_players})"
            )
        self.player_labels = list(player_labels)
        self.player_short = (
            list(player_short) if player_short is not None
            else list(player_labels)
        )

    # -------------------------------------------------------------------
    # Core computation
    # -------------------------------------------------------------------

    def compute(self, verbose: bool = True) -> Dict[str, object]:
        """Compute exact Shapley values for all players.

        Enumerates all 2^n coalitions and applies the combinatorial Shapley formula. Caches value function evaluations to avoid redundant computations.

        Parameters
        ----------
        verbose : bool
            Print progress and results to stdout.

        Returns
        -------
        results : dict
            shapley_values : np.ndarray of shape (n_players,)
            baseline : float
                Value of the empty coalition.
            player_labels : list of str
            player_short : list of str
            coalitions_evaluated : int
            total_coalitions : int (2^n)
            elapsed_seconds : float
            value_cache : dict
                Map from sorted coalition tuple to value.
        """
        n = self.n_players
        shapley_vals = np.zeros(n)
        all_subsets = power_set(range(n))
        value_cache: Dict[Tuple[int, ...], float] = {}

        def get_value(subset: set) -> float:
            key = tuple(sorted(subset))
            if key not in value_cache:
                value_cache[key] = self.value_function(list(subset))
            return value_cache[key]

        t0 = time.time()

        baseline = get_value(set())

        if verbose:
            print(f"Computing exact Shapley values for {n} players "
                  f"({2**n} coalitions)...")
            print(f"Baseline (empty coalition) = {baseline:.6f}\n")

        for i in range(n):
            if verbose:
                print(f"  Player {i}: {self.player_labels[i]}", end="",
                      flush=True)

            for subset in all_subsets:
                subset = set(subset)
                if i in subset: # Run only on subsets that do not contain player i
                    continue
                s = len(subset)
                weight = (
                    math.factorial(s) * math.factorial(n - s - 1)
                ) / math.factorial(n)

                v_with = get_value(subset | {i}) # Add player i to the coalition 
                v_without = get_value(subset)
                shapley_vals[i] += weight * (v_with - v_without)

            if verbose:
                print(f"  ->  SV = {shapley_vals[i]:+.6f}")

        elapsed = time.time() - t0

        if verbose:
            print(f"Baseline        : {baseline:.6f}")
            print(f"Max |SV|        : {np.max(np.abs(shapley_vals)):.6f}")
            print(f"Mean |SV|       : {np.mean(np.abs(shapley_vals)):.6f}")
            print(f"Sum SV          : {np.sum(shapley_vals):.6f}")
            print(f"Sum |SV|        : {np.sum(np.abs(shapley_vals)):.6f}")
            print(f"Elapsed         : {elapsed:.1f}s")


        return {
            "shapley_values": shapley_vals,
            "baseline": baseline,
            "player_labels": self.player_labels,
            "player_short": self.player_short,
            "coalitions_evaluated": len(value_cache),
            "total_coalitions": 2 ** n,
            "elapsed_seconds": elapsed,
            "value_cache": {
                str(k): v for k, v in value_cache.items()
            },
        }

    # -------------------------------------------------------------------
    # Single-player marginal contributions
    # -------------------------------------------------------------------

    def marginal_contributions(self) -> Dict[str, object]:
        """Compute the marginal contribution of each player alone.

        Evaluates v({i}) - v({}) for each player i. No interpretation can be found into this. Useful as a quick diagnostic before running the full exact Shapley computation.

        Returns
        -------
        dict with:
            baseline : float
            deltas : np.ndarray of shape (n_players,)
            player_labels : list of str
        """
        baseline = self.value_function([])
        deltas = np.zeros(self.n_players)
        for i in range(self.n_players):
            deltas[i] = self.value_function([i]) - baseline
        return {
            "baseline": baseline,
            "deltas": deltas,
            "player_labels": self.player_labels,
        }
