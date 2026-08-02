"""Final predictions from the LightGBM + TabPFN v2 blend.

The temporal-holdout experiment (src/tabfm_experiment.py) showed a 50/50
log-space blend of LightGBM and bagged TabPFN v2 improves clean-row MAE by
~7% over LightGBM alone ($58.75 vs $63.23 on identical rows). This script
regenerates both deliverables with that blend:

1. validation_predictions.csv
2. data/december_chart_inputs.csv (predicted_rate column)

TabPFN inference on CPU is slow, so predictions are computed per
(bag, chunk) and cached under .tabfm_cache/final/; an interrupted run
resumes where it stopped. LightGBM predictions are cached too.

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
N_BAGS = 4
BAG_SIZE = 8000
N_ESTIMATORS = 2
CHUNK = 1500
LGB_WEIGHT = 0.5
TABPFN_MODEL = os.path.expanduser("~/.cache/tabpfn/tabpfn-v2-regressor.ckpt")


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
    from tabpfn import TabPFNRegressor

    x_target = tabpfn_frame(target_f)
    n_chunks = int(np.ceil(len(x_target) / CHUNK))
    bag_means = []
    for bag in range(N_BAGS):
        chunk_files = [CACHE / f"bag{bag}_chunk{i}.npy" for i in range(n_chunks)]
        missing = [i for i, f in enumerate(chunk_files) if not f.exists()]
        if missing:
            rng = np.random.default_rng(SEED + bag)
            idx = rng.choice(len(full_f), BAG_SIZE, replace=False)
            ctx = full_f.iloc[idx]
            model = TabPFNRegressor(
                model_path=TABPFN_MODEL,
                device="cpu",
                n_estimators=N_ESTIMATORS,
                random_state=SEED + bag,
                ignore_pretraining_limits=True,
            )
            model.fit(tabpfn_frame(ctx), np.log(ctx["posted_rate"].to_numpy()))
            for i in missing:
                t0 = time.time()
                sl = x_target.iloc[i * CHUNK : (i + 1) * CHUNK]
                np.save(chunk_files[i], model.predict(sl))
                print(
                    f"bag {bag + 1}/{N_BAGS} chunk {i + 1}/{n_chunks} "
                    f"done in {time.time() - t0:.0f}s",
                    flush=True,
                )
        bag_means.append(np.concatenate([np.load(f) for f in chunk_files]))
    return np.mean(bag_means, axis=0)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    train_test = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    daily = daily_market_series([train_test, validation])
    weight_median = train_test["weight"].abs().median()

    full_f = build_features(clean_training_rows(train_test), daily, weight_median)
    valid_f = build_features(validation, daily, weight_median)
    december_f = build_features(
        build_december_frame(train_test, validation), daily, weight_median
    )
    align_categories([full_f, valid_f, december_f])
    target_f = pd.concat([valid_f, december_f], ignore_index=True)

    lgb_log = lgb_final_log(full_f, target_f, train_test, daily, weight_median)
    tab_log = tabpfn_final_log(full_f, target_f)
    blend = np.exp(LGB_WEIGHT * lgb_log + (1 - LGB_WEIGHT) * tab_log)

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
        "method": "0.5 * LightGBM + 0.5 * bagged TabPFN v2 (log space)",
        "bags": N_BAGS,
        "bag_size": BAG_SIZE,
        "validation_mean": float(valid_pred.mean()),
        "december_min": float(dec_pred.min()),
        "december_max": float(dec_pred.max()),
    }
    (ROOT / "report" / "final_blend_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
