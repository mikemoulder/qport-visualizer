"""Classical baselines and out-of-sample evaluation.

exact        enumerate every feasible pick: the ground truth QAOA is scored against
naive        pick the B assets with the best individual Sharpe ratio
markowitz    continuous long-only mean-variance weights (no cardinality limit):
             what a practitioner would actually run, and a lower bound on f
backtest     hold each portfolio's weights through the unseen test window
"""
import time
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import minimize

from .data import TRADING_DAYS


def exact(problem):
    """Score every way to pick `budget` assets; return the best one."""
    t0 = time.perf_counter()
    best_k, best_f, count = None, np.inf, 0
    for combo in combinations(range(problem.n), problem.budget):
        w = np.zeros(problem.n)
        w[list(combo)] = 1 / problem.budget
        f = problem.risk_aversion * w @ problem.sigma @ w - problem.mu @ w
        count += 1
        if f < best_f:
            best_k, best_f = sum(1 << i for i in combo), f
    seconds = time.perf_counter() - t0
    assert best_k == problem.optimum, "enumeration disagrees with the QUBO optimum"
    return {"index": best_k, "evaluated": count, "seconds": seconds, "cost": float(best_f)}


def naive_top_sharpe(problem):
    vol = np.sqrt(np.diag(problem.sigma))
    order = np.argsort(-(problem.mu / vol))[: problem.budget]
    return int(sum(1 << int(i) for i in order))


def markowitz(mu, sigma, risk_aversion):
    n = len(mu)

    def f(w):
        return risk_aversion * w @ sigma @ w - mu @ w

    def grad(w):
        return 2 * risk_aversion * sigma @ w - mu

    res = minimize(f, np.full(n, 1 / n), jac=grad, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = np.clip(res.x, 0, None)
    return w / w.sum(), float(f(w / w.sum()))


def weights_of(k, n):
    x = np.array([(int(k) >> i) & 1 for i in range(n)], dtype=float)
    return x / x.sum()


def uniform_guess(problem):
    """Probability that blind guessing hits the optimum."""
    return {"any_bitstring": 1 / 2 ** problem.n,
            "feasible_only": 1 / comb(problem.n, problem.budget)}


def backtest(weights, test_returns):
    """Constant weights, rebalanced daily, over the out-of-sample window."""
    daily = test_returns @ weights
    growth = np.cumprod(1 + daily)
    years = len(daily) / TRADING_DAYS
    cagr = growth[-1] ** (1 / years) - 1
    vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    peak = np.maximum.accumulate(growth)
    return {"growth": growth, "cagr": float(cagr), "vol": float(vol),
            "sharpe": float(daily.mean() * TRADING_DAYS / vol) if vol else float("nan"),
            "max_drawdown": float((growth / peak - 1).min())}
