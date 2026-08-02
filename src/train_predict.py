"""Train the final model on all labeled data and produce both deliverables:

1. validation_predictions.csv  - one predicted_rate per load in data/validation.csv
2. data/december_chart_inputs.csv - predicted_rate filled for the fixed
   Lexington -> Fort Wayne Dry Van lane, one row per December 2025 day.

The number of boosting rounds is chosen by early stopping against the
Sep-Oct temporal holdout, then the model is refit on all of Jan-Oct with the
round count scaled up proportionally to the extra data.

Usage: python src/train_predict.py
"""
from __future__ import annotations

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
from validate import LGB_PARAMS, align_categories, fit_lgb

ROOT = Path(__file__).resolve().parents[1]

DEC_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")
DEC_LANE = {
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32_000.0,
}


def city_coords(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Each city has a single fixed lat/lon across the dataset."""
    parts = []
    for f in frames:
        parts.append(
            f[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
            )
        )
        parts.append(
            f[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
            )
        )
    return pd.concat(parts).groupby("city")[["lat", "lon"]].first()


def build_december_frame(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    coords = city_coords([train, validation])
    lane = train[
        (train["pickup"] == DEC_LANE["pickup"])
        & (train["delivery"] == DEC_LANE["delivery"])
        & (train["equipment"] == DEC_LANE["equipment"])
    ]
    # Expected per-mile quote for a typical load on this lane; the daily
    # component of the quote market enters separately through qs_daily.
    lane_quote_signal = lane["quote_signal"].mean()

    dec = pd.DataFrame({"date": DEC_DATES})
    for key, value in DEC_LANE.items():
        dec[key] = value
    dec["pickup_lat"] = coords.loc[DEC_LANE["pickup"], "lat"]
    dec["pickup_lon"] = coords.loc[DEC_LANE["pickup"], "lon"]
    dec["delivery_lat"] = coords.loc[DEC_LANE["delivery"], "lat"]
    dec["delivery_lon"] = coords.loc[DEC_LANE["delivery"], "lon"]
    # market_index is left missing and is imputed from the per-date mean of
    # the December loads in validation.csv (features are given there).
    dec["market_index"] = np.nan
    dec["quote_signal"] = lane_quote_signal
    return dec


def main() -> None:
    train_test = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    daily = daily_market_series([train_test, validation])
    weight_median = train_test["weight"].abs().median()

    # Stage 1: pick the boosting round count on the Sep-Oct temporal holdout.
    cutoff = pd.Timestamp("2025-09-01")
    early = clean_training_rows(train_test[train_test["date"] < cutoff])
    late = clean_training_rows(train_test[train_test["date"] >= cutoff])
    early_f = build_features(early, daily, weight_median)
    late_f = build_features(late, daily, weight_median)
    align_categories([early_f, late_f])
    stage1 = fit_lgb(early_f, late_f)
    n_rounds = int(stage1.best_iteration * len(train_test) / len(early) )
    print(f"Stage 1 best_iteration={stage1.best_iteration} -> final rounds={n_rounds}")

    # Stage 2: refit on all cleaned labeled data.
    full_clean = clean_training_rows(train_test)
    full_f = build_features(full_clean, daily, weight_median)
    valid_f = build_features(validation, daily, weight_median)
    december_f = build_features(build_december_frame(train_test, validation), daily, weight_median)
    align_categories([full_f, valid_f, december_f])

    dtrain = lgb.Dataset(
        full_f[ALL_FEATURES],
        label=np.log(full_f["posted_rate"]),
        categorical_feature=CATEGORICAL_FEATURES,
    )
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=n_rounds)

    # Deliverable 1: validation predictions in template order.
    valid_pred = np.exp(booster.predict(valid_f[ALL_FEATURES]))
    template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
    rates = pd.Series(valid_pred, index=validation["load_id"].values)
    template["predicted_rate"] = template["load_id"].map(rates).round(2)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    out1 = ROOT / "validation_predictions.csv"
    template.to_csv(out1, index=False)
    print(f"Wrote {out1} ({len(template)} rows, "
          f"mean ${template['predicted_rate'].mean():.2f})")

    # Deliverable 2: December fixed-lane predictions.
    dec_pred = np.exp(booster.predict(december_f[ALL_FEATURES]))
    dec_out = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    assert (dec_out["predicted_rate"] > 0).all()
    out2 = ROOT / "data/december_chart_inputs.csv"
    dec_out.to_csv(out2, index=False)
    print(f"Wrote {out2}: min ${dec_pred.min():.2f}, max ${dec_pred.max():.2f}, "
          f"mean ${dec_pred.mean():.2f}")


if __name__ == "__main__":
    main()
