"""Exact Shapley value computation.

Problem-agnostic implementation of exact Shapley values for cooperative game theory. Given N players and a value function v(S) -> float, computes the marginal contribution of each player using the combinatorial formula:

    SV_j = sum_{S not containing j} [|S|!(n-|S|-1)!/n!] * [v(S u {j}) - v(S)]

This module is model agnostic.
The user supplies a value function and player labels.
"""

import math
import random
import sys
import time
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def compute(self, verbose: bool = True, n_jobs: int = 1) -> Dict[str, object]:
        """Compute exact Shapley values for all players.

        Enumerates all 2^n coalitions and applies the combinatorial Shapley formula. Caches value function evaluations to avoid redundant computations.

        Parameters
        ----------
        verbose : bool
            Print progress and results to stdout.
        n_jobs : int
            Number of worker threads to use for coalition evaluation.
            If ``n_jobs=1``, computation is serial.

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
        if int(n_jobs) < 1:
            raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
        n_jobs = int(n_jobs)

        n = self.n_players
        shapley_vals = np.zeros(n)
        all_subsets = power_set(range(n))
        value_cache: Dict[Tuple[int, ...], float] = {}

        t0 = time.time()

        if n_jobs == 1:
            def get_value(subset: set) -> float:
                key = tuple(sorted(subset))
                if key not in value_cache:
                    value_cache[key] = self.value_function(list(subset))
                return value_cache[key]

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
                    if i in subset:  # Run only on subsets that do not contain player i
                        continue
                    s = len(subset)
                    weight = (
                        math.factorial(s) * math.factorial(n - s - 1)
                    ) / math.factorial(n)

                    v_with = get_value(subset | {i})  # Add player i to the coalition
                    v_without = get_value(subset)
                    shapley_vals[i] += weight * (v_with - v_without)

                if verbose:
                    print(f"  ->  SV = {shapley_vals[i]:+.6f}")

        else:
            all_coalitions = [tuple(sorted(s)) for s in all_subsets]
            full_coalition = tuple(range(n))
            if full_coalition not in all_coalitions:
                all_coalitions.append(full_coalition)

            total_coalitions = len(all_coalitions)

            if verbose:
                print(
                    f"Computing exact Shapley values for {n} players "
                    f"({total_coalitions} coalitions) with {n_jobs} worker(s)..."
                )

            progress_start = time.time()
            interactive_stderr = sys.stderr.isatty()
            log_every = max(1, total_coalitions // 100)

            def _evaluate_one(coalition: Tuple[int, ...]) -> Tuple[Tuple[int, ...], float]:
                value = self.value_function(list(coalition))
                return coalition, value

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_map = {
                    executor.submit(_evaluate_one, coalition): coalition
                    for coalition in all_coalitions
                }
                completed = 0
                for future in as_completed(future_map):
                    coalition, value = future.result()
                    value_cache[coalition] = value
                    completed += 1

                    if verbose:
                        elapsed = time.time() - progress_start
                        if completed == 1:
                            msg = (
                                f"    [{completed}/{total_coalitions}] "
                                f"elapsed {elapsed:.0f}s ..."
                            )
                        else:
                            rate = elapsed / completed
                            remaining = rate * (total_coalitions - completed)
                            mins, secs = divmod(int(remaining), 60)
                            msg = (
                                f"    [{completed}/{total_coalitions}] "
                                f"elapsed {elapsed:.0f}s | "
                                f"~{rate:.1f}s/eval | ETA {mins}m{secs:02d}s   "
                            )
                        if interactive_stderr:
                            sys.stderr.write(f"\r{msg}")
                            sys.stderr.flush()
                        elif (
                            completed == 1
                            or completed == total_coalitions
                            or completed % log_every == 0
                        ):
                            sys.stderr.write(f"{msg}\n")
                            sys.stderr.flush()

            if verbose:
                sys.stderr.write("\n")
                sys.stderr.flush()

            baseline = value_cache[tuple()]

            if verbose:
                print(f"Baseline (empty coalition) = {baseline:.6f}\n")

            for i in range(n):
                if verbose:
                    print(f"  Player {i}: {self.player_labels[i]}", end="",
                          flush=True)

                for subset in all_subsets:
                    if i in subset:
                        continue
                    s = len(subset)
                    weight = (
                        math.factorial(s) * math.factorial(n - s - 1)
                    ) / math.factorial(n)

                    coalition_without = tuple(sorted(subset))
                    coalition_with = tuple(sorted(subset | {i}))
                    v_with = value_cache[coalition_with]
                    v_without = value_cache[coalition_without]
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
