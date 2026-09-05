import numpy as np
import pandas as pd
from institutional_conviction import get_institutional_conviction_scores
from market_volatility import get_market_volatility_scores

np.random.seed(42)  # reproducible results

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

    max_volatility = max(volatility_scores.values())  # for normalizing to 0-1

    rows = []
    for _ in range(n_samples):
        category = np.random.choice(CATEGORIES)

        verification_confidence = np.random.randint(50, 100)
        asset_concentration = round(np.random.uniform(0.1, 1.0), 2)
        category_liquidity = CATEGORY_LIQUIDITY[category]
        conviction = conviction_scores[category]
        volatility = volatility_scores[category]

        # Normalize each factor to 0-1
        vol_norm = volatility / max_volatility if max_volatility > 0 else 0
        illiquidity = 1 - category_liquidity
        unverified = (100 - verification_confidence) / 100
        low_conviction = 1 - (conviction / 100)

        # Weighted average (weights sum to 1) -> naturally bounded 0-1
        risk_score = (
            0.35 * vol_norm +
            0.25 * asset_concentration +
            0.20 * illiquidity +
            0.10 * unverified +
            0.10 * low_conviction
        )

        label = 1 if np.random.rand() < risk_score else 0

        rows.append({
            "category": category,
            "verification_confidence": verification_confidence,
            "asset_concentration": asset_concentration,
            "category_liquidity": category_liquidity,
            "institutional_conviction": conviction,
            "market_volatility": volatility,
            "drawdown_occurred": label
        })

    return pd.DataFrame(rows)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

if __name__ == "__main__":
    df = generate_synthetic_baskets()

    # One-hot encode the category column since XGBoost needs numeric input
    df_encoded = pd.get_dummies(df, columns=["category"])

    X = df_encoded.drop(columns=["drawdown_occurred"])
    y = df_encoded["drawdown_occurred"]

    # Walk-forward-style split is ideal, but since this is synthetic (no real time order),
    # a standard train/test split is fine here — real backtesting will use time-based splits later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("=== Model Performance ===")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 3))

    print("\n=== Feature Importance ===")
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importance)

    # Save the trained model so the FastAPI service can load it later
    model.save_model("basket_risk_model.json")
    print("\nModel saved to basket_risk_model.json")