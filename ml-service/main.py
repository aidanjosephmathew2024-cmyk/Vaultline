from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "worker", "pipelines"))

from institutional_conviction import get_institutional_conviction_scores
from market_volatility import get_market_volatility_scores

app = FastAPI(title="Vaultline Basket Risk Service")

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

# Load the REAL-DATA trained model
model = XGBClassifier()
model.load_model(os.path.join(os.path.dirname(__file__), "..", "worker", "pipelines", "real_basket_risk_model.json"))

conviction_scores = get_institutional_conviction_scores()
volatility_scores = get_market_volatility_scores()  # category-level average volatility


class BasketAsset(BaseModel):
    category: str
    value_usd: float
    verification_confidence: int


class BasketRequest(BaseModel):
    assets: list[BasketAsset]


@app.post("/risk/basket-score")
def score_basket(request: BasketRequest):
    total_value = sum(a.value_usd for a in request.assets)

    rows = []
    for asset in request.assets:
        concentration = asset.value_usd / total_value if total_value > 0 else 0
        rows.append({
            "rolling_volatility": volatility_scores.get(asset.category, 0),
            "institutional_conviction": conviction_scores.get(asset.category, 0),
            "category_liquidity": CATEGORY_LIQUIDITY.get(asset.category, 0.5),
            "verification_confidence": asset.verification_confidence,
            "asset_concentration": concentration,
            "category": asset.category
        })

    df = pd.DataFrame(rows)
    df_encoded = pd.get_dummies(df, columns=["category"], prefix="cat")

    for cat in CATEGORIES:
        col = f"cat_{cat}"
        if col not in df_encoded.columns:
            df_encoded[col] = False

    df_encoded = df_encoded[model.get_booster().feature_names]

    probabilities = model.predict_proba(df_encoded)[:, 1]
    basket_drawdown_probability = round(float(probabilities.mean()), 3)

    if basket_drawdown_probability < 0.15:
        risk_band = "LOW"
    elif basket_drawdown_probability < 0.35:
        risk_band = "MODERATE"
    else:
        risk_band = "HIGH"

    return {
        "drawdown_probability": basket_drawdown_probability,
        "risk_band": risk_band,
        "per_asset_scores": [
            {"category": r["category"], "individual_risk_contribution": round(float(p), 3)}
            for r, p in zip(rows, probabilities)
        ]
    }