import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from institutional_conviction import get_institutional_conviction_scores
from market_volatility import get_market_volatility_scores
from xgboost import XGBClassifier

np.random.seed(42)

CATEGORIES = ["precious_metals", "industrial_metals", "sovereign_debt",
              "corporate_credit", "structured_credit", "real_assets", "alt_financing"]

CATEGORY_LIQUIDITY = {
    "precious_metals": 0.9,
    "industrial_metals": 0.6,
    "sovereign_debt": 0.9,
    "corporate_credit": 0.5,
    "structured_credit": 0.4,
    "real_assets": 0.3,
    "alt_financing": 0.4
}

def generate_synthetic_baskets(n_samples=2000):
    conviction_scores = get_institutional_conviction_scores()
    volatility_scores = get_market_volatility_scores()
    max_volatility = max(volatility_scores.values())

    start_date = datetime(2024, 1, 1)

    rows = []
    for i in range(n_samples):
        category = np.random.choice(CATEGORIES)

        # Spread samples chronologically across ~2 years, in order
        days_offset = int((i / n_samples) * 730)
        sample_date = start_date + timedelta(days=days_offset)

        verification_confidence = np.random.randint(50, 100)
        asset_concentration = round(np.random.uniform(0.1, 1.0), 2)
        category_liquidity = CATEGORY_LIQUIDITY[category]
        base_conviction = conviction_scores[category]
        # Real-world correlation: high volatility periods tend to see reduced institutional conviction
        volatility_drag = volatility_scores[category] * np.random.uniform(20, 60)
        conviction = max(0, base_conviction - volatility_drag * np.random.uniform(0, 0.3))
        volatility = volatility_scores[category]

        vol_norm = volatility / max_volatility if max_volatility > 0 else 0
        illiquidity = 1 - category_liquidity
        unverified = (100 - verification_confidence) / 100
        low_conviction = 1 - (conviction / 100)

        base_risk_score = (
            0.35 * vol_norm +
            0.25 * asset_concentration +
            0.20 * illiquidity +
            0.10 * unverified +
            0.10 * low_conviction
        )

        # Add realistic noise: real-world outcomes aren't a clean function of known factors —
        # unmodeled events (news shocks, liquidity crunches, etc.) add randomness
        noise = np.random.normal(loc=0, scale=0.08)
        risk_score = np.clip(base_risk_score + noise, 0, 1)

        label = 1 if np.random.rand() < risk_score else 0

        rows.append({
            "date": sample_date,
            "category": category,
            "verification_confidence": verification_confidence,
            "asset_concentration": asset_concentration,
            "category_liquidity": category_liquidity,
            "institutional_conviction": conviction,
            "market_volatility": volatility,
            "drawdown_occurred": label
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)  # ensure chronological order
    return df

def walk_forward_backtest(df, n_folds=5):
    """
    Splits data chronologically into n_folds+1 chunks.
    Each fold trains on all data up to that point, tests on the next chunk.
    This is out-of-sample by construction — the model never sees future data during training.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    from xgboost import XGBClassifier

    df_encoded = pd.get_dummies(df.drop(columns=["date"]), columns=["category"])
    n = len(df_encoded)
    fold_size = n // (n_folds + 1)

    results = []
    for fold in range(1, n_folds + 1):
        train_end = fold_size * fold
        test_end = fold_size * (fold + 1)

        train_df = df_encoded.iloc[:train_end]
        test_df = df_encoded.iloc[train_end:test_end]

        X_train = train_df.drop(columns=["drawdown_occurred"])
        y_train = train_df["drawdown_occurred"]
        X_test = test_df.drop(columns=["drawdown_occurred"])
        y_test = test_df["drawdown_occurred"]

        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results.append({"fold": fold, "train_size": len(train_df), "test_size": len(test_df),
                         "accuracy": round(acc, 3), "roc_auc": round(auc, 3)})

    return pd.DataFrame(results)
def generate_edge_case_baskets(n_samples=200):
    """
    Deliberately generates extreme/rare scenarios:
    - Fully concentrated baskets (100% one category)
    - Categories with zero institutional conviction
    - High verification but high volatility (conflicting signals)
    """
    conviction_scores = get_institutional_conviction_scores()
    volatility_scores = get_market_volatility_scores()
    max_volatility = max(volatility_scores.values())
    start_date = datetime(2024, 1, 1)

    rows = []
    for i in range(n_samples):
        category = np.random.choice(CATEGORIES)
        days_offset = int((i / n_samples) * 730)
        sample_date = start_date + timedelta(days=days_offset)

        scenario = np.random.choice(["concentrated", "zero_conviction", "conflicting_signals"])

        if scenario == "concentrated":
            asset_concentration = 1.0
            verification_confidence = np.random.randint(70, 100)
        elif scenario == "zero_conviction":
            asset_concentration = round(np.random.uniform(0.1, 0.5), 2)
            verification_confidence = np.random.randint(70, 100)
        else:  # conflicting_signals
            asset_concentration = round(np.random.uniform(0.1, 0.5), 2)
            verification_confidence = np.random.randint(90, 100)  # high verification

        category_liquidity = CATEGORY_LIQUIDITY[category]
        conviction = 0 if scenario == "zero_conviction" else conviction_scores[category]
        volatility = volatility_scores[category]
        if scenario == "conflicting_signals":
            volatility = max_volatility * np.random.uniform(0.8, 1.0)  # force high volatility

        vol_norm = volatility / max_volatility if max_volatility > 0 else 0
        illiquidity = 1 - category_liquidity
        unverified = (100 - verification_confidence) / 100
        low_conviction = 1 - (conviction / 100)

        base_risk_score = (
            0.35 * vol_norm + 0.25 * asset_concentration +
            0.20 * illiquidity + 0.10 * unverified + 0.10 * low_conviction
        )
        noise = np.random.normal(loc=0, scale=0.08)
        risk_score = np.clip(base_risk_score + noise, 0, 1)
        label = 1 if np.random.rand() < risk_score else 0

        rows.append({
            "date": sample_date, "category": category,
            "verification_confidence": verification_confidence,
            "asset_concentration": asset_concentration,
            "category_liquidity": category_liquidity,
            "institutional_conviction": conviction,
            "market_volatility": volatility,
            "drawdown_occurred": label
        })

    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True)
if __name__ == "__main__":
    normal_df = generate_synthetic_baskets(n_samples=2000)
    edge_df = generate_edge_case_baskets(n_samples=200)
    df = pd.concat([normal_df, edge_df]).sort_values("date").reset_index(drop=True)
    
    print("\n=== Edge Case Sanity Check ===")
    print("Max asset_concentration:", df["asset_concentration"].max())
    print("Rows with 0 institutional_conviction:", (df["institutional_conviction"] == 0).sum())
    print("Total rows:", len(df))

    print("=== Walk-Forward Backtest Results (Out-of-Sample) ===")
    backtest_results = walk_forward_backtest(df, n_folds=5)
    print(backtest_results)
    print(f"\nAverage out-of-sample accuracy: {backtest_results['accuracy'].mean():.3f}")
    print(f"Average out-of-sample ROC-AUC: {backtest_results['roc_auc'].mean():.3f}")

    # Train the FINAL model on ALL data (this is what gets saved/served live)
    df_encoded = pd.get_dummies(df.drop(columns=["date"]), columns=["category"])
    X = df_encoded.drop(columns=["drawdown_occurred"])
    y = df_encoded["drawdown_occurred"]

    final_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    final_model.fit(X, y)
    final_model.save_model("basket_risk_model.json")
    print("\nFinal model (trained on full history) saved to basket_risk_model.json")