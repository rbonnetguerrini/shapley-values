# shapley-values

Problem-agnostic "exact" Shapley value computation inspiring from cooperative game theory.

## Game theory background
The original paper by Shapley (1953) can be found here: https://doi.org/10.1515/9781400881970-018

In game theory, **Shapley Values** represent the contribution of a player \(i\), denoted \(\phi_i\), within a coalition \(S\) by comparing the outcomes of scenarios where the player is present \(v(S \cup \{i\})\) versus absent \(v(S)\):

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} w(S)\\bigl[\ v(S \cup \{i\}) - v(S) \\bigr].
$$

where

$$
w(S) = \frac{|S|!\ (|N| - |S| - 1)!}{|N|!},
$$

weights the importance of the tested coalition \(S\). It is the probability that the set of players who come before \(i\) is exactly \(S\).

- \(|S|!\) is the number of ways to order the predecessors of \(i\)
- \((|N| - |S| - 1)!\) is the number of ways to order the players after \(i\)
- \(|N|!\) is the number of ways to order all players

To understand this intuitively: picking a random sequence of all players, there are many more possible coalitions in which \(i\) appears early or late in the sequence than in the middle. The weight \(w(S)\) accounts for this.

The weight of the Shapley Values can be derived as follows. Given \(|N| = n\), for a fixed player \(i\), let \(S \subseteq N \setminus \{i\}\) with \(|S| = s\). The probability that the players before \(i\) are exactly \(S\) can be decomposed into two conditions:

First, \(i\) can be in any position in \(N\), so:

$$
p(\text{pos}_i = s+1) = \frac{1}{n}.
$$

Second, given that \(i\) is in position \(s+1\), the \(s\) players before \(i\) form a uniformly random subset of the remaining \(n-1\) players of size \(\binom{n-1}{s}\):

$$
p(\text{predecessors} = S \mid \text{pos}_i = s+1) = \frac{1}{\binom{n-1}{s}}.
$$

To fulfill these conditions, we multiply these probabilities:

$$
\frac{1}{n}\cdot \frac{1}{\binom{n-1}{s}}
= \frac{1}{n}\cdot \frac{s!(n-1-s)!}{(n-1)!}
= w(S).
$$

## Game theory properties

Shapley Value satisfy these axioms:
- **Efficiency**: sum of SVs = v(N) - v({})
- **Symmetry**: symmetric players get equal values
- **Null player**: non-contributing player gets SV = 0
- **Additivity**: SV(v + w) = SV(v) + SV(w)


# Repository 
## Installation

```bash
pip install -e .
```

## Quick start

```python
from shapley_values import ExactShapley, plot_shapley_bar

# Define a value function: coalition (list of player indices) -> scalar
def value_function(coalition):
    """Example: each player contributes their weight."""
    weights = [3.0, 1.0, 2.0, 4.0]
    return sum(weights[i] for i in coalition)

# Compute exact Shapley values
sv = ExactShapley(
    n_players=4,
    value_function=value_function,
    player_labels=["Alice", "Bob", "Carol", "Dave"],
)
results = sv.compute()

# Visualize
plot_shapley_bar(results["shapley_values"], results["player_labels"])
```

## API

### `ExactShapley(n_players, value_function, player_labels=None, player_short=None)`

- `n_players`: number of players
- `value_function`: callable `f(List[int]) -> float` mapping a coalition to a scalar value
- `player_labels`: human-readable names (optional)
- `player_short`: short names for plots (optional, defaults to `player_labels`)

#### Methods

- `compute(verbose=True)` — exact Shapley values via combinatorial formula (2^n coalitions)
- `marginal_contributions()` — quick diagnostic: v({i}) - v({}) for each player

## Plotting

- `plot_shapley_bar(sv, labels, ...)` — bar chart colored by sign
- `plot_shapley_pie(sv, labels, ...)` — pie chart of |SV| proportions
- `plot_shapley_comparison(results_list, ...)` — side-by-side comparison
- `plot_marginal_contributions(deltas, labels, ...)` — bar chart of marginals

## I/O

- `save_results(results, path, metadata=None)` — save to JSON
- `load_results(path)` — load from JSON

## Example notebook

See [`notebook/sv_example.ipynb`](notebook/sv_example.ipynb) for a full walkthrough using the California Housing dataset.
