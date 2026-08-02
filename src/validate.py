"""Temporal validation of the freight rate model.

Primary split: train on Jan-Aug 2025, hold out Sep-Oct 2025. This mirrors the
real task (the scored validation set is Nov-Dec 2025, strictly after all
labeled data), so the holdout measures genuine forward-in-time generalization
including unseen-month market conditions.

A shuffled 80/20 split is also reported for comparison to quantify how
optimistic a random split would be.

Usage: python src/validate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    build_features,
    clean_training_rows,
    daily_market_series,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

LGB_PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": SEED,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_pct": float(np.mean(np.abs(err) / y_true) * 100),
        "R2": float(1 - np.sum(err**2) / np.sum((y_true - y_true.mean()) ** 2)),
        "n": int(len(y_true)),
    }


def fit_lgb(train_df, valid_df):
    dtrain = lgb.Dataset(
        train_df[ALL_FEATURES],
        label=np.log(train_df["posted_rate"]),
        categorical_feature=CATEGORICAL_FEATURES,
    )
    dvalid = lgb.Dataset(
        valid_df[ALL_FEATURES],
        label=np.log(valid_df["posted_rate"]),
        reference=dtrain,
    )
    booster = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=4000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return booster


def align_categories(frames: list[pd.DataFrame]) -> None:
    """Give every frame the same category levels so LightGBM's integer codes
    line up across train / early-stopping / prediction frames."""
    for col in CATEGORICAL_FEATURES:
        union = sorted(set().union(*[set(f[col].cat.categories) for f in frames]))
        for f in frames:
            f[col] = f[col].cat.set_categories(union)


def run_split(name: str, train_raw: pd.DataFrame, holdout_raw: pd.DataFrame, daily) -> dict:
    weight_median = train_raw["weight"].abs().median()
    train_clean = clean_training_rows(train_raw)
    train_f = build_features(train_clean, daily, weight_median)
    holdout_f = build_features(holdout_raw, daily, weight_median)
    stop_f = build_features(clean_training_rows(holdout_raw), daily, weight_median)
    align_categories([train_f, holdout_f, stop_f])

    booster = fit_lgb(train_f, stop_f)
    pred = np.exp(booster.predict(holdout_f[ALL_FEATURES]))

    # Baseline: median rate-per-mile by equipment x distance decile.
    rpm = train_clean["posted_rate"] / train_clean["distance"]
    bins = np.quantile(train_clean["distance"], np.linspace(0, 1, 11))
    tb = train_clean.assign(rpm=rpm, dbin=np.digitize(train_clean["distance"], bins[1:-1]))
    lookup = tb.groupby(["equipment", "dbin"], observed=True)["rpm"].median()
    hb = holdout_raw.assign(dbin=np.digitize(holdout_raw["distance"], bins[1:-1]))
    base_rpm = (
        hb.set_index(["equipment", "dbin"]).index.map(lookup).to_numpy(dtype=float)
    )
    base_rpm = np.where(np.isnan(base_rpm), rpm.median(), base_rpm)
    base_pred = base_rpm * holdout_raw["distance"].to_numpy()

    y = holdout_raw["posted_rate"].to_numpy()
    clean_mask = ((y / holdout_raw["distance"]) >= 0.8) & ((y / holdout_raw["distance"]) <= 6.0)

    result = {
        "split": name,
        "best_iteration": booster.best_iteration,
        "model_all_rows": metrics(y, pred),
        "model_clean_rows": metrics(y[clean_mask.to_numpy()], pred[clean_mask.to_numpy()]),
        "baseline_all_rows": metrics(y, base_pred),
    }

    imp = pd.Series(
        booster.feature_importance(importance_type="gain"), index=ALL_FEATURES
    ).sort_values(ascending=False)
    result["top_features"] = {k: round(float(v), 1) for k, v in imp.head(12).items()}
    return result


def main() -> None:
    train_test = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    daily = daily_market_series([train_test, validation])

    cutoff = pd.Timestamp("2025-09-01")
    temporal = run_split(
        "temporal (train Jan-Aug, holdout Sep-Oct)",
        train_test[train_test["date"] < cutoff],
        train_test[train_test["date"] >= cutoff],
        daily,
    )

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train_test))
    n_holdout = len(train_test) // 5
    random_split = run_split(
        "random 80/20",
        train_test.iloc[idx[n_holdout:]],
        train_test.iloc[idx[:n_holdout]],
        daily,
    )

    results = {"temporal": temporal, "random": random_split}
    out_path = ROOT / "report" / "validation_metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    for res in results.values():
        print(f"\n=== {res['split']} (best_iter={res['best_iteration']}) ===")
        for key in ("model_all_rows", "model_clean_rows", "baseline_all_rows"):
            m = res[key]
            print(
                f"{key:18s} MAE ${m['MAE']:7.2f} | RMSE ${m['RMSE']:7.2f} "
                f"| MAPE {m['MAPE_pct']:5.2f}% | R2 {m['R2']:.4f} | n={m['n']}"
            )
        print("top features:", ", ".join(list(res["top_features"])[:8]))
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
