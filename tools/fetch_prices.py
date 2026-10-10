"""Re-download the bundled price history (run locally, not on Qollab).

Qollab projects cannot fetch this at run time (the source sends no CORS headers),
so the CSV ships with the project. This script documents exactly how it was made:

    python tools/fetch_prices.py

Source: Yahoo Finance public chart endpoint, daily adjusted closes (split- and
dividend-adjusted), 2021-01-01 to 2025-12-31, dates kept where every ticker traded.
The bundled file is a snapshot taken on 2026-10-10. The provider occasionally
revises adjusted closes, so a fresh download can differ in the 4th decimal place.
"""
import datetime as dt
import json
import time
import urllib.request

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "WMT", "NEE"]
START, END = 1609459200, 1767225600  # 2021-01-01, 2026-01-01 (UTC)
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{t}"
       "?period1={a}&period2={b}&interval=1d&events=div,split")


def fetch(ticker):
    req = urllib.request.Request(URL.format(t=ticker, a=START, b=END),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.load(resp)["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["adjclose"][0]["adjclose"]
    return {dt.datetime.fromtimestamp(s, dt.timezone.utc).date().isoformat(): c
            for s, c in zip(stamps, closes) if c is not None}


def main():
    series = {}
    for t in TICKERS:
        series[t] = fetch(t)
        time.sleep(1)
    dates = sorted(set.intersection(*(set(s) for s in series.values())))
    with open("data/prices_refetched.csv", "w") as f:
        f.write("date," + ",".join(TICKERS) + "\n")
        for d in dates:
            f.write(d + "," + ",".join(f"{series[t][d]:.4f}" for t in TICKERS) + "\n")
    print(f"wrote data/prices_refetched.csv: {len(dates)} rows, {dates[0]} .. {dates[-1]}")


if __name__ == "__main__":
    main()
