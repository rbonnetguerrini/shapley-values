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
import numpy as np
from shapley_values import ExactShapley, plot_shapley_bar

# Toy dataset: 4 features, linear target
np.random.seed(42)
X = np.random.randn(100, 4)
y = 3 * X[:, 0] + 1 * X[:, 1] + 0.5 * X[:, 2] + 0 * X[:, 3]

# Value function: negative MSE using only the selected features
def value_function(coalition):
    """Return negative mean squared error for a least-squares fit."""
    if len(coalition) == 0:
        return -np.mean(y ** 2)
    Xc = X[:, coalition]
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    residuals = y - Xc @ beta
    return -np.mean(residuals ** 2)

# Compute exact Shapley values
sv = ExactShapley(
    n_players=4,
    value_function=value_function,
    player_labels=["x1", "x2", "x3", "x4"],
)
results = sv.compute()

# Visualize each feature's contribution to reducing prediction error
plot_shapley_bar(results["shapley_values"], results["player_labels"])
```

## API

### `ExactShapley(n_players, value_function, player_labels=None, player_short=None)`

- `n_players`: number of players
- `value_function`: callable `f(List[int]) -> float` mapping a coalition to a scalar value
- `player_labels`: human-readable names (optional)
- `player_short`: short names for plots (optional, defaults to `player_labels`)

#### Methods

- `compute(verbose=True, n_jobs=1)` : exact Shapley values via combinatorial formula (2^n coalitions). Set `n_jobs>1` to evaluate coalitions in parallel.
- `marginal_contributions()` : quick diagnostic: v({i}) - v({}) for each player

## Plotting

- `plot_shapley_bar(sv, labels, ...)` : bar chart colored by sign
- `plot_shapley_pie(sv, labels, ...)` : pie chart of |SV| proportions
- `plot_shapley_comparison(results_list, ...)` : side-by-side comparison
- `plot_marginal_contributions(deltas, labels, ...)` : bar chart of marginals

## I/O

- `save_results(results, path, metadata=None)` : save to JSON
- `load_results(path)` : load from JSON

## Example notebook

See [`notebook/sv_example.ipynb`](notebook/sv_example.ipynb) for a full walkthrough using the California Housing dataset.
