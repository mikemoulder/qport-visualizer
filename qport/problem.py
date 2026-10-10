"""Portfolio selection as a constrained binary problem, then as QUBO and Ising.

Decision: pick exactly B of n assets, equally weighted (w = x / B).
Objective (minimize):  f(x) = q * w^T Sigma w  -  mu^T w
    q      risk aversion (0 = chase return only)
    Sigma  annualized covariance, mu annualized mean return

Bitstring convention (matches Qiskit): integer k encodes x_i = (k >> i) & 1,
qubit i <-> asset i, and the printed label has qubit 0 on the RIGHT.
"""
from dataclasses import dataclass, field

import numpy as np
from qiskit.quantum_info import SparsePauliOp



def all_bitstrings(n):
    """Matrix of shape (2^n, n): row k holds the bits of k (bit i = asset i)."""
    k = np.arange(2 ** n)
    return ((k[:, None] >> np.arange(n)) & 1).astype(float)


def label(k, n):
    return format(int(k), f"0{n}b")


def picked(k, tickers):
    return [t for i, t in enumerate(tickers) if (int(k) >> i) & 1]


def qubo_value(Q, X, const=0.0):
    """x^T Q x + const for every row x of X."""
    return np.einsum("ki,ij,kj->k", X, Q, X) + const


def qubo_to_ising(Q, const=0.0):
    """Map x^T Q x + const onto Z-basis Pauli operators with x = (1 - z) / 2.

    Returns (SparsePauliOp H, offset) with  x^T Q x + const = <x|H|x> + offset.
    """
    n = len(Q)
    terms = []
    offset = const
    for i in range(n):
        terms.append(("Z", [i], -Q[i, i] / 2))
        offset += Q[i, i] / 2
        for j in range(i + 1, n):
            w = Q[i, j] / 2  # x^T Q x carries Q_ij + Q_ji = 2 Q_ij; times 1/4
            terms += [("ZZ", [i, j], w), ("Z", [i], -w), ("Z", [j], -w)]
            offset += w
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify(), offset


@dataclass
class PortfolioProblem:
    tickers: list
    mu: np.ndarray
    sigma: np.ndarray
    budget: int
    risk_aversion: float
    penalty: float = 0.0           # <= 0 -> the provably safe bound
    # derived
    n: int = field(init=False)
    Q_obj: np.ndarray = field(init=False)       # objective only (no constraint)
    Q_pen: np.ndarray = field(init=False)       # objective + penalty
    pen_const: float = field(init=False)
    costs: np.ndarray = field(init=False)       # f(x) for all 2^n bitstrings
    feasible: np.ndarray = field(init=False)    # bool mask, Hamming weight == B
    optimum: int = field(init=False)            # best feasible bitstring index

    def __post_init__(self):
        self.n = len(self.tickers)
        if not 1 <= self.budget < self.n:
            raise ValueError(f"budget must be between 1 and {self.n - 1}, got {self.budget}")
        n, B, q = self.n, self.budget, self.risk_aversion
        self.Q_obj = q * self.sigma / B ** 2 - np.diag(self.mu) / B
        X = all_bitstrings(n)
        self.costs = qubo_value(self.Q_obj, X)
        self.feasible = X.sum(axis=1) == B
        feas_idx = np.flatnonzero(self.feasible)
        self.optimum = int(feas_idx[np.argmin(self.costs[feas_idx])])
        # Any violation costs >= lambda and no flip gains more than the objective's
        # full range, so lambda > range is a provably safe (but loose) penalty.
        self.safe_penalty = 1.05 * float(self.costs.max() - self.costs.min())
        self.min_penalty = self._bisect_min_penalty()
        if self.penalty <= 0:
            self.penalty = self.safe_penalty
        self._set_penalty(self.penalty)
        if int(np.argmin(self.penalized_costs())) != self.optimum:
            raise ValueError(f"penalty {self.penalty:.4g} is too small: the QUBO minimum "
                             f"breaks the budget (need > {self.min_penalty:.4g})")

    def _set_penalty(self, lam):
        n, B = self.n, self.budget
        self.Q_pen = self.Q_obj + lam * (np.ones((n, n)) - np.eye(n)) + np.diag(
            np.full(n, lam * (1 - 2 * B)))
        self.pen_const = lam * B ** 2

    def _bisect_min_penalty(self, iters=40):
        """Smallest lambda whose QUBO minimum is the true constrained optimum.

        Exhaustive (2^n), so only for small n; that is exactly the regime where we
        can afford ground truth and want the gentlest landscape for QAOA.
        """
        X = all_bitstrings(self.n)
        viol2 = (X.sum(axis=1) - self.budget) ** 2
        lo, hi = 0.0, self.safe_penalty
        for _ in range(iters):
            mid = (lo + hi) / 2
            if int(np.argmin(self.costs + mid * viol2)) == self.optimum:
                hi = mid
            else:
                lo = mid
        return hi

    # --- views used throughout -------------------------------------------------
    @property
    def feasible_costs(self):
        return self.costs[self.feasible]

    @property
    def best_cost(self):
        return float(self.costs[self.optimum])

    @property
    def worst_feasible_cost(self):
        return float(self.feasible_costs.max())

    def penalized_costs(self):
        return qubo_value(self.Q_pen, all_bitstrings(self.n), self.pen_const)

    def ising(self, which="penalty"):
        """Normalized cost Hamiltonian; returns (H, scale) where true = scale * H + offset."""
        Q, c = (self.Q_pen, self.pen_const) if which == "penalty" else (self.Q_obj, 0.0)
        scale = float(np.abs(Q).max())
        H, _ = qubo_to_ising(Q / scale, c / scale)
        return H, scale

    def stats(self, k):
        """Expected return, volatility and Sharpe (rf = 0) of the equal-weight pick k."""
        x = all_bitstrings(self.n)[int(k)]
        w = x / x.sum() if x.sum() else x
        ret = float(self.mu @ w)
        vol = float(np.sqrt(w @ self.sigma @ w))
        return {"return": ret, "vol": vol, "sharpe": ret / vol if vol else float("nan")}

    def approx_ratio(self, probs):
        """1 = always the optimum, 0 = always the worst feasible pick.

        Infeasible outcomes are scored as the worst feasible pick, so a solver
        cannot look good by breaking the budget constraint.
        """
        best, worst = self.best_cost, self.worst_feasible_cost
        scored = np.where(self.feasible, self.costs, worst)
        expected = float(probs @ scored)
        return (worst - expected) / (worst - best) if worst > best else 1.0
