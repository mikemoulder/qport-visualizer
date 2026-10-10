# Data

`prices.csv`: daily adjusted closing prices (split- and dividend-adjusted, USD) for
AAPL, MSFT, NVDA, JPM, XOM, JNJ, WMT and NEE, 2021-01-04 to 2025-12-31, 1,255 trading days
on which all eight traded.

- **Source:** Yahoo Finance public chart endpoint, downloaded 2026-10-10.
  `tools/fetch_prices.py` repeats the download (a fresh copy can differ by about 1e-5
  relative, because the provider revises adjusted closes).
- **Why bundled:** Qollab runs in the browser and only allows cross-origin GET requests,
  which this source does not support, and nothing persists between runs.
- **Use:** returns and covariances are estimated from 2021-01-04 to `train_end`
  (default 2023-12-31); the later days are held out for the out-of-sample test.
- Market data is included for research and education. Prices remain the property of
  their respective providers and exchanges.
