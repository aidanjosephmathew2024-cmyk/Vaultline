import yfinance as yf
import pandas as pd
import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from institutional_conviction import get_institutional_conviction_scores

TICKER_CATEGORY = {
    "GLD": "precious_metals", "SLV": "precious_metals",
    "PPLT": "precious_metals", "PALL": "precious_metals",
    "DBB": "industrial_metals", "COPX": "industrial_metals",
    "TLT": "sovereign_debt", "IEF": "sovereign_debt",
    "LQD": "corporate_credit", "HYG": "corporate_credit",
    "MBB": "structured_credit",
    "VNQ": "real_assets", "IFRA": "real_assets",
    "BIZD": "alt_financing"
}

CATEGORY_LIQUIDITY = {
    "precious_metals": 0.9, "industrial_metals": 0.6, "sovereign_debt": 0.9,
    "corporate_credit": 0.5, "structured_credit": 0.4,
    "real_assets": 0.3, "alt_financing": 0.4
}

DRAWDOWN_THRESHOLD = -0.08   # loosened from -0.15 to -0.08 (8% drop in 30 days)
FORWARD_WINDOW = 30          # look 30 trading days ahead
ROLLING_VOL_WINDOW = 30      # volatility computed over the past 30 days


def build_real_labeled_dataset():
    conviction_scores = get_institutional_conviction_scores()
    all_rows = []

    for ticker, category in TICKER_CATEGORY.items():
        print(f"Pulling history for {ticker}...")
        hist = yf.download(ticker, period="10y", interval="1d")["Close"]
        hist = hist.dropna()

        returns = hist.pct_change()
        rolling_vol = returns.rolling(ROLLING_VOL_WINDOW).std() * (252 ** 0.5)
        forward_return = hist.shift(-FORWARD_WINDOW) / hist - 1

        df = pd.DataFrame({
            "date": hist.index,
            "ticker": ticker,
            "category": category,
            "price": hist.values.flatten(),
            "rolling_volatility": rolling_vol.values.flatten(),
            "forward_return": forward_return.values.flatten()
        })

        df["drawdown_occurred"] = (df["forward_return"] < DRAWDOWN_THRESHOLD).astype(int)
        df["institutional_conviction"] = conviction_scores.get(category, 0)
        df["category_liquidity"] = CATEGORY_LIQUIDITY[category]

        # Simulated for now — no real user data exists yet
        df["verification_confidence"] = np.random.randint(50, 100, size=len(df))
        df["asset_concentration"] = np.round(np.random.uniform(0.1, 1.0, size=len(df)), 2)

        df = df.dropna(subset=["rolling_volatility", "forward_return"])
        all_rows.append(df)

    full_df = pd.concat(all_rows).reset_index(drop=True)
    return full_df


if __name__ == "__main__":
    df = build_real_labeled_dataset()
    print(f"\nTotal real labeled samples: {len(df)}")
    print("\nLabel balance:")
    print(df["drawdown_occurred"].value_counts())
    print("\nSample rows:")
    print(df.head(10))