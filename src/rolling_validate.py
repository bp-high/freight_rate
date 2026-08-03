"""Rolling-origin validation of the blend weights.

The Sep-Oct holdout picks blend weights from a single forward-in-time
window; before those weights change the shipped predictions they must hold
up across several forward-in-time folds, or the grid was just fit to one
window's noise.

Folds evaluate one month each (Jul, Aug, Sep, Oct 2025) using only strictly
earlier months for training:

* LightGBM mirrors the production recipe: early-stop on the last training
  month, then refit on the whole training window with proportionally scaled
  rounds.
* Bagged TabPFN v2 mirrors the previously shipped 4 x 8,000-row contexts,
  and single-context TabPFN v2 uses the whole training window as one
  context (the holdout experiment showed the single context is worth ~$9
  MAE over bagging).
* TabM early-stops on the last training month (no refit; it has no
  round-count to scale).

A fixed 1,500-row subsample per eval month bounds TabPFN CPU cost. Every
model's log-predictions are cached under .tabfm_cache/rolling/ so an
interrupted run resumes. The selection criterion is mean clean-row MAE
across folds for every weight vector on the LGB/TabPFN-single/TabM simplex;
the LGB/TabPFN-bagged/TabM simplex is reported alongside for comparison.

Usage:
  TABPFN_ALLOW_CPU_LARGE_DATASET=1 python src/rolling_validate.py
"""
from __future__ import annotations

import json
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
from tabfm_experiment import tabpfn_frame
from validate import LGB_PARAMS, align_categories, fit_lgb, metrics

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".tabfm_cache" / "rolling"
SEED = 42
N_EVAL = 1500
N_BAGS = 4
BAG_SIZE = 8000
N_ESTIMATORS = 2
EVAL_MONTHS = [7, 8, 9, 10]


def lgb_fold_log(tag, train_f, stop_f, full_f, eval_f):
    cache = CACHE / f"{tag}_lgb.npy"
    if cache.exists():
        return np.load(cache)
    stage1 = fit_lgb(train_f, stop_f)
    n_rounds = max(1, int(stage1.best_iteration * len(full_f) / len(train_f)))
    dtrain = lgb.Dataset(
        full_f[ALL_FEATURES],
        label=np.log(full_f["posted_rate"]),
        categorical_feature=CATEGORICAL_FEATURES,
    )
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=n_rounds)
    log_pred = booster.predict(eval_f[ALL_FEATURES])
    np.save(cache, log_pred)
    return log_pred


def tabpfn_fold_log(tag, full_f, eval_f):
    from tabpfn import TabPFNRegressor

    x_eval = tabpfn_frame(eval_f)
    bag_logs = []
    for bag in range(N_BAGS):
        cache = CACHE / f"{tag}_tabpfn_bag{bag}.npy"
        if cache.exists():
            bag_logs.append(np.load(cache))
            continue
        rng = np.random.default_rng(SEED + bag)
        idx = rng.choice(len(full_f), min(BAG_SIZE, len(full_f)), replace=False)
        ctx = full_f.iloc[idx]
        model = TabPFNRegressor(
            model_path="tabpfn-v2-regressor.ckpt",
            device="cpu",
            n_estimators=N_ESTIMATORS,
            random_state=SEED + bag,
            ignore_pretraining_limits=True,
        )
        t0 = time.time()
        model.fit(tabpfn_frame(ctx), np.log(ctx["posted_rate"].to_numpy()))
        log_pred = model.predict(x_eval)
        np.save(cache, log_pred)
        bag_logs.append(log_pred)
        print(f"{tag} tabpfn bag {bag + 1}/{N_BAGS} in {time.time() - t0:.0f}s", flush=True)
    return np.mean(bag_logs, axis=0)


def tabpfn_single_fold_log(tag, full_f, eval_f):
    """Single full-training-window context; one predict call covers the whole
    eval subsample (the context forward dominates cost, so chunking the
    queries would just repeat it)."""
    from tabpfn import TabPFNRegressor

    cache = CACHE / f"{tag}_tabpfn_single.npy"
    if cache.exists():
        return np.load(cache)
    model = TabPFNRegressor(
        model_path="tabpfn-v2-regressor.ckpt",
        device="cpu",
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        ignore_pretraining_limits=True,
    )
    t0 = time.time()
    model.fit(tabpfn_frame(full_f), np.log(full_f["posted_rate"].to_numpy()))
    log_pred = model.predict(tabpfn_frame(eval_f))
    np.save(cache, log_pred)
    print(f"{tag} tabpfn single-context ({len(full_f)} rows) in {time.time() - t0:.0f}s", flush=True)
    return log_pred


def tabm_fold_log(tag, train_f, stop_f, eval_f):
    cache = CACHE / f"{tag}_tabm.npy"
    if cache.exists():
        return np.load(cache)
    from tabm_model import train_tabm

    model = train_tabm(train_f, stop_f, seed=SEED)
    log_pred = model.predict_log(eval_f)
    np.save(cache, log_pred)
    return log_pred


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    train_test = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    daily = daily_market_series([train_test, validation])
    weight_median = train_test["weight"].abs().median()

    folds = {}
    weight_maes_bagged: dict[str, list[float]] = {}
    weight_maes_single: dict[str, list[float]] = {}
    for m in EVAL_MONTHS:
        tag = f"2025-{m:02d}"
        eval_start = pd.Timestamp(2025, m, 1)
        stop_start = pd.Timestamp(2025, m - 1, 1)
        full_raw = clean_training_rows(train_test[train_test["date"] < eval_start])
        train_raw = full_raw[full_raw["date"] < stop_start]
        stop_raw = full_raw[full_raw["date"] >= stop_start]
        eval_month = train_test[
            (train_test["date"] >= eval_start)
            & (train_test["date"] < eval_start + pd.offsets.MonthBegin(1))
        ]
        rng = np.random.default_rng(SEED + m)
        eval_raw = eval_month.iloc[
            rng.choice(len(eval_month), min(N_EVAL, len(eval_month)), replace=False)
        ]

        train_f = build_features(train_raw, daily, weight_median)
        stop_f = build_features(stop_raw, daily, weight_median)
        full_f = build_features(full_raw, daily, weight_median)
        eval_f = build_features(eval_raw, daily, weight_median)
        align_categories([train_f, stop_f, full_f, eval_f])

        y = eval_f["posted_rate"].to_numpy()
        rpm = y / eval_f["distance"].to_numpy()
        clean = (rpm >= 0.8) & (rpm <= 6.0)

        print(f"=== fold {tag}: train {len(full_f)}, eval {len(eval_f)} ===", flush=True)
        lgb_log = lgb_fold_log(tag, train_f, stop_f, full_f, eval_f)
        bag_log = tabpfn_fold_log(tag, full_f, eval_f)
        single_log = tabpfn_single_fold_log(tag, full_f, eval_f)
        tabm_log = tabm_fold_log(tag, train_f, stop_f, eval_f)

        grids = {}
        for name, pfn_log, store in (
            ("bagged", bag_log, weight_maes_bagged),
            ("single", single_log, weight_maes_single),
        ):
            grid = {}
            for i in range(11):
                for j in range(11 - i):
                    k = 10 - i - j
                    pred = np.exp(
                        (i / 10) * lgb_log + (j / 10) * pfn_log + (k / 10) * tabm_log
                    )
                    key = f"{i / 10:.1f}/{j / 10:.1f}/{k / 10:.1f}"
                    grid[key] = metrics(y[clean], pred[clean])["MAE"]
                    store.setdefault(key, []).append(grid[key])
            grids[name] = grid

        single = grids["single"]
        folds[tag] = {
            "train_rows": len(full_f),
            "eval_rows": len(eval_f),
            "lightgbm_mae": single["1.0/0.0/0.0"],
            "tabpfn_bagged_mae": grids["bagged"]["0.0/1.0/0.0"],
            "tabpfn_single_mae": single["0.0/1.0/0.0"],
            "tabm_mae": single["0.0/0.0/1.0"],
            "old_blend_lgb_bagged_5050_mae": grids["bagged"]["0.5/0.5/0.0"],
            "candidate_single_best_weights": min(single, key=single.get),
            "candidate_single_best_mae": min(single.values()),
        }
        print(json.dumps(folds[tag], indent=2), flush=True)

    mean_single = {k: float(np.mean(v)) for k, v in weight_maes_single.items()}
    mean_bagged = {k: float(np.mean(v)) for k, v in weight_maes_bagged.items()}
    ranked = sorted(mean_single, key=mean_single.get)
    results = {
        "setup": {
            "eval_months": EVAL_MONTHS,
            "eval_rows_per_fold": N_EVAL,
            "models": "LightGBM / TabPFN v2 (bagged 4x8k and single full context) / TabM",
            "weights_order": "lgb/tabpfn/tabm",
            "selection": "mean clean-row MAE across folds on the single-context simplex",
        },
        "folds": folds,
        "mean_mae_single_top10": {k: mean_single[k] for k in ranked[:10]},
        "mean_mae_selected": {
            "best_overall_single": {"weights": ranked[0], "mean_mae": mean_single[ranked[0]]},
            "holdout_winner_0.1/0.5/0.4_single": mean_single["0.1/0.5/0.4"],
            "old_blend_lgb_bagged_5050": mean_bagged["0.5/0.5/0.0"],
            "lightgbm_only": mean_single["1.0/0.0/0.0"],
            "tabm_only": mean_single["0.0/0.0/1.0"],
            "tabpfn_single_only": mean_single["0.0/1.0/0.0"],
        },
    }
    out = ROOT / "report" / "rolling_validation.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["mean_mae_selected"], indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
