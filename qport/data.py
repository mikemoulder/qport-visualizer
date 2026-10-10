"""Bundled market data: daily adjusted closes, 2021-01-04 .. 2025-12-31.

Qollab only loads files whose paths appear as plain string literals in code,
so the CSV path below must stay a literal.
"""
import csv
from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class MarketData:
    tickers: list
    dates: list          # ISO date strings, one per price row
    prices: np.ndarray   # shape (T, n)

    def subset(self, tickers):
        idx = [self.tickers.index(t) for t in tickers]
        return MarketData(list(tickers), self.dates, self.prices[:, idx])

    def returns(self):
        """Daily simple returns, shape (T-1, n), dated by the later day."""
        return self.prices[1:] / self.prices[:-1] - 1.0, self.dates[1:]


def load_prices():
    with open("data/prices.csv", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    dates = [r[0] for r in body]
    prices = np.array([[float(v) for v in r[1:]] for r in body])
    return MarketData(header[1:], dates, prices)


def split_returns(market, train_end):
    """Split daily returns into an estimation window and an out-of-sample window."""
    rets, dates = market.returns()
    mask = np.array([d <= train_end for d in dates])
    if mask.sum() < 60 or (~mask).sum() < 20:
        raise ValueError(f"train_end={train_end} leaves too little data on one side "
                         f"({mask.sum()} train days, {(~mask).sum()} test days)")
    test_dates = [d for d, m in zip(dates, mask) if not m]
    return rets[mask], rets[~mask], test_dates


def estimate(train_returns):
    """Annualized expected returns (mean) and covariance from daily returns."""
    mu = train_returns.mean(axis=0) * TRADING_DAYS
    sigma = np.cov(train_returns, rowvar=False) * TRADING_DAYS
    return mu, sigma
