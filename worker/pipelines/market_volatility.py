import yfinance as yf

def label_real_drawdowns(ticker, threshold=-0.15, window_days=30):
    data = yf.download(ticker, period="2y", interval="1d")["Close"]
    future_return = data.shift(-window_days) / data - 1
    drawdown_occurred = (future_return < threshold).astype(int)
    return drawdown_occurred

CATEGORY_TICKERS = {
    "precious_metals": ["GLD", "SLV", "PPLT", "PALL"],
    "industrial_metals": ["DBB", "COPX"],
    "sovereign_debt": ["TLT", "IEF"],
    "corporate_credit": ["LQD", "HYG"],
    "structured_credit": ["MBB"],
    "real_assets": ["VNQ", "IFRA"],
    "alt_financing": ["BIZD"]
}


def get_market_volatility_scores():
    """
    Pulls 30 days of live prices for all tracked tickers, computes
    annualized volatility per ticker, then averages into a per-category score.
    """
    all_tickers = [t for tickers in CATEGORY_TICKERS.values() for t in tickers]
    data = yf.download(all_tickers, period="30d", interval="1d")["Close"]
    returns = data.pct_change()
    volatility = returns.std() * (252 ** 0.5)

    scores = {}
    for category, tickers in CATEGORY_TICKERS.items():
        scores[category] = round(volatility[tickers].mean(), 4)
    return scores


# Quick manual test — only runs when this file is executed directly
if __name__ == "__main__":
    result = get_market_volatility_scores()
    print("=== Market Volatility Scores ===")
    for category, score in result.items():
        print(f"{category}: {score}")