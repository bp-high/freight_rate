"""Final predictions from the LightGBM + TabPFN v2 + TabM blend.

Recipe selection (see report/tabfm_metrics.json and
report/rolling_validation.json):

* The Sep-Oct temporal holdout showed single-full-context TabPFN v2 beats
  the earlier 4x8k bagging by ~$9 MAE, and TabM (parameter-efficient MLP
  ensemble) beats LightGBM as a single model.
* Because one window is not enough to pick blend weights, the weights were
  re-validated rolling-origin across four forward-in-time folds (predict
  Jul, Aug, Sep, Oct from strictly earlier months). Per-month regimes vary
  a lot (TabPFN wins Jul/Sep, collapses Aug/Oct), so the selected weights
  sit on the plateau that minimizes MEAN clean-row MAE across folds:

      0.4 * LightGBM + 0.1 * TabPFN v2 (single context) + 0.5 * TabM
      (log space; rolling mean $51.74 vs $64.98 for the previous
       0.5/0.5 LightGBM+bagged-TabPFN blend; Sep-Oct holdout $53.35
       vs $58.75)

All three components are license-clean for this deliverable (TabPFN v2 is
Prior Labs' license-free tier; the newer v2.5/v3 checkpoints are gated
behind a non-commercial license and are deliberately not used here).

This script regenerates both deliverables with that blend:

1. validation_predictions.csv
2. data/december_chart_inputs.csv (predicted_rate column)

Slow steps cache their log-predictions under .tabfm_cache/final/ so an
interrupted run resumes. Everything is seeded.

Usage:
  TABPFN_ALLOW_CPU_LARGE_DATASET=1 python src/final_blend_predict.py
"""
from __future__ import annotations

import json
import os
import time
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
from train_predict import build_december_frame
from validate import LGB_PARAMS, align_categories, fit_lgb

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".tabfm_cache" / "final"
SEED = 42
N_ESTIMATORS = 2
W_LGB, W_TABPFN, W_TABM = 0.4, 0.1, 0.5
TABPFN_CKPT = "tabpfn-v2-regressor.ckpt"  # resolved/downloaded by the tabpfn package


def tabpfn_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].astype(str)
    return x


def lgb_final_log(full_f, target_f, train_test, daily, weight_median) -> np.ndarray:
    cache = CACHE / "lgb_final_log.npy"
    if cache.exists():
        print("lightgbm predictions loaded from cache", flush=True)
        return np.load(cache)
    cutoff = pd.Timestamp("2025-09-01")
    early = clean_training_rows(train_test[train_test["date"] < cutoff])
    late = clean_training_rows(train_test[train_test["date"] >= cutoff])
    early_f = build_features(early, daily, weight_median)
    late_f = build_features(late, daily, weight_median)
    align_categories([early_f, late_f])
    stage1 = fit_lgb(early_f, late_f)
    n_rounds = int(stage1.best_iteration * len(train_test) / len(early))
    dtrain = lgb.Dataset(
        full_f[ALL_FEATURES],
        label=np.log(full_f["posted_rate"]),
        categorical_feature=CATEGORICAL_FEATURES,
    )
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=n_rounds)
    log_pred = booster.predict(target_f[ALL_FEATURES])
    np.save(cache, log_pred)
    print(f"lightgbm predictions done (rounds={n_rounds})", flush=True)
    return log_pred


def tabpfn_final_log(full_f, target_f) -> np.ndarray:
    """Single context holding the full cleaned training data; one predict
    call for all targets (each call redoes the context forward, so chunking
    the queries would multiply the dominant cost)."""
    from tabpfn import TabPFNRegressor

    cache = CACHE / "tabpfn_single_final_log.npy"
    if cache.exists():
        print("tabpfn predictions loaded from cache", flush=True)
        return np.load(cache)
    model = TabPFNRegressor(
        model_path=TABPFN_CKPT,
        device="cpu",
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        ignore_pretraining_limits=True,
    )
    t0 = time.time()
    model.fit(tabpfn_frame(full_f), np.log(full_f["posted_rate"].to_numpy()))
    log_pred = model.predict(tabpfn_frame(target_f))
    np.save(cache, log_pred)
    print(
        f"tabpfn single-context ({len(full_f)} rows -> {len(target_f)} targets) "
        f"done in {time.time() - t0:.0f}s",
        flush=True,
    )
    return log_pred


def tabm_final_log(train_f, stop_f, target_f) -> np.ndarray:
    """Mirrors the rolling-fold protocol: train on everything before the
    last labeled month, early-stop on that month (TabM has no round count
    to rescale, so there is no refit-on-all step)."""
    cache = CACHE / "tabm_final_log.npy"
    if cache.exists():
        print("tabm predictions loaded from cache", flush=True)
        return np.load(cache)
    from tabm_model import train_tabm

    t0 = time.time()
    model = train_tabm(train_f, stop_f, seed=SEED)
    log_pred = model.predict_log(target_f)
    np.save(cache, log_pred)
    print(f"tabm done in {time.time() - t0:.0f}s", flush=True)
    return log_pred


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    train_test = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    daily = daily_market_series([train_test, validation])
    weight_median = train_test["weight"].abs().median()

    full_clean = clean_training_rows(train_test)
    full_f = build_features(full_clean, daily, weight_median)
    valid_f = build_features(validation, daily, weight_median)
    december_f = build_features(
        build_december_frame(train_test, validation), daily, weight_median
    )
    last_month = pd.Timestamp("2025-10-01")
    tabm_train_f = build_features(
        full_clean[full_clean["date"] < last_month], daily, weight_median
    )
    tabm_stop_f = build_features(
        full_clean[full_clean["date"] >= last_month], daily, weight_median
    )
    align_categories([full_f, valid_f, december_f, tabm_train_f, tabm_stop_f])
    target_f = pd.concat([valid_f, december_f], ignore_index=True)

    lgb_log = lgb_final_log(full_f, target_f, train_test, daily, weight_median)
    tab_log = tabpfn_final_log(full_f, target_f)
    tabm_log = tabm_final_log(tabm_train_f, tabm_stop_f, target_f)
    blend = np.exp(W_LGB * lgb_log + W_TABPFN * tab_log + W_TABM * tabm_log)

    valid_pred, dec_pred = blend[: len(valid_f)], blend[len(valid_f) :]

    template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
    rates = pd.Series(valid_pred, index=validation["load_id"].values)
    template["predicted_rate"] = template["load_id"].map(rates).round(2)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    template.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv (mean ${template['predicted_rate'].mean():.2f})")

    dec_out = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    assert (dec_out["predicted_rate"] > 0).all()
    dec_out.to_csv(ROOT / "data/december_chart_inputs.csv", index=False)
    print(
        f"Wrote december_chart_inputs.csv: min ${dec_pred.min():.2f}, "
        f"max ${dec_pred.max():.2f}, mean ${dec_pred.mean():.2f}"
    )

    summary = {
        "method": (
            "0.4 * LightGBM + 0.1 * TabPFN v2 (single full context) "
            "+ 0.5 * TabM (log space)"
        ),
        "weights": {"lightgbm": W_LGB, "tabpfn_v2_single": W_TABPFN, "tabm": W_TABM},
        "selection": "rolling-origin mean clean MAE across Jul-Oct folds "
        "(report/rolling_validation.json); Sep-Oct holdout $53.35 vs $58.75 "
        "for the previous LGB+bagged-TabPFN 50/50 blend",
        "validation_mean": float(valid_pred.mean()),
        "december_min": float(dec_pred.min()),
        "december_max": float(dec_pred.max()),
    }
    (ROOT / "report" / "final_blend_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
