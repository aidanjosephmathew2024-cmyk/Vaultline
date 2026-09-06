import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from real_historical_labels import build_real_labeled_dataset

def walk_forward_backtest_real(df, n_folds=5):
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = ["rolling_volatility", "institutional_conviction",
                     "category_liquidity", "verification_confidence", "asset_concentration"]

    df_encoded = pd.get_dummies(df, columns=["category"], prefix="cat")
    category_cols = [c for c in df_encoded.columns if c.startswith("cat_")]
    feature_cols = feature_cols + category_cols

    n = len(df_encoded)
    fold_size = n // (n_folds + 1)

    results = []
    for fold in range(1, n_folds + 1):
        train_end = fold_size * fold
        test_end = fold_size * (fold + 1)

        train_df = df_encoded.iloc[:train_end]
        test_df = df_encoded.iloc[train_end:test_end]

        X_train, y_train = train_df[feature_cols], train_df["drawdown_occurred"]
        X_test, y_test = test_df[feature_cols], test_df["drawdown_occurred"]

        # Class weighting: tell XGBoost the positive class is rare and important
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

        model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "fold": fold,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "positive_rate_test": round(y_test.mean(), 3),
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "roc_auc": round(roc_auc_score(y_test, probs), 3),
            "precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "recall": round(recall_score(y_test, preds, zero_division=0), 3),
            "f1": round(f1_score(y_test, preds, zero_division=0), 3)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = build_real_labeled_dataset()

    print("\n=== Walk-Forward Backtest on REAL Historical Data ===")
    results = walk_forward_backtest_real(df, n_folds=5)
    print(results.to_string(index=False))

    print(f"\nAverage ROC-AUC: {results['roc_auc'].mean():.3f}")
    print(f"Average Precision: {results['precision'].mean():.3f}")
    print(f"Average Recall: {results['recall'].mean():.3f}")

    # Train the FINAL model on ALL available real data — this is what gets served live
    df_encoded = pd.get_dummies(df, columns=["category"], prefix="cat")
    feature_cols = ["rolling_volatility", "institutional_conviction",
                     "category_liquidity", "verification_confidence", "asset_concentration"]
    category_cols = [c for c in df_encoded.columns if c.startswith("cat_")]
    feature_cols = feature_cols + category_cols

    X = df_encoded[feature_cols]
    y = df_encoded["drawdown_occurred"]

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    final_model = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42
    )
    final_model.fit(X, y)
    final_model.save_model("real_basket_risk_model.json")
    print(f"\nFinal real-data model saved to real_basket_risk_model.json")
    print(f"Feature columns used: {feature_cols}")