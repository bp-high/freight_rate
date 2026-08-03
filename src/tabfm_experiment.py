"""Tabular foundation model comparison on the temporal holdout.

Models compared on an identical fixed evaluation subsample of the Sep-Oct
temporal holdout:

* LightGBM (the production model) as the reference.
* TabPFN v2, fit as a bagged ensemble of context subsamples (the model is an
  in-context learner capped at ~10k training rows per fit). Weights are
  fetched from the public GCS fallback, so no HuggingFace access is needed.
* TabICL v2 (regressor), if its checkpoint can be obtained. TabICL weights
  are hosted only on HuggingFace (jingang/TabICL); in environments where
  huggingface.co is unreachable the stage records itself as unavailable
  instead of failing the run.
* TabPFN v2 with a single full-train context (vs the bagged contexts above),
  to separate "bigger context" from "bagging" effects.
* TabM (parameter-efficient MLP ensemble, ICLR 2025) trained from scratch —
  a strong neural model with errors decorrelated from both trees and
  in-context transformers, i.e. a natural third blend member.
* TabPFN v2.5 / v3 checkpoints, which are license-gated: Prior Labs requires
  a one-time license acceptance (register at https://ux.priorlabs.ai, accept
  the license, export TABPFN_TOKEN=<api key>). Without the token these
  stages record themselves as unavailable; with it they run automatically.
* Log-space blends: LightGBM+TabPFN (2-way) and every 3-way simplex over
  the available models, gridded in 0.1 steps on the identical holdout rows.

CPU inference is slow, so the holdout comparison uses a fixed random
subsample of 3,000 holdout loads (MAE standard error ~ $4); every model is
evaluated on the same rows for an apples-to-apples comparison.

Usage:
  TABPFN_ALLOW_CPU_LARGE_DATASET=1 python src/tabfm_experiment.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    build_features,
    clean_training_rows,
    daily_market_series,
)
from validate import align_categories, fit_lgb, metrics

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_EVAL = 3000          # holdout rows used for the comparison
N_BAGS = 4             # TabPFN context subsamples
BAG_SIZE = 8000        # rows per TabPFN context
N_ESTIMATORS = 2       # TabPFN internal ensemble size per bag
MODEL_PATH = os.path.expanduser("~/.cache/tabpfn/tabpfn-v2-regressor.ckpt")
TABICL_MODEL_PATH = os.path.expanduser("~/.cache/tabicl/tabicl-regressor-v2-20260212.ckpt")
BAG_CACHE = Path(__file__).resolve().parents[1] / ".tabfm_cache"


def tabpfn_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].astype(str)
    return x


def report(log_pred, y_eval, clean_mask, **extra) -> dict:
    pred = np.exp(log_pred)
    return {
        "status": "ok",
        **extra,
        "all_rows": metrics(y_eval, pred),
        "clean_rows": metrics(y_eval[clean_mask], pred[clean_mask]),
    }


def grid3(a_log, b_log, c_log, y_eval, clean_mask):
    """MAE over the full 3-model log-space weight simplex in 0.1 steps.
    Boundary points cover the single models and every 2-way blend."""
    out = {}
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            pred = np.exp((i / 10) * a_log + (j / 10) * b_log + (k / 10) * c_log)
            out[f"{i / 10:.1f}/{j / 10:.1f}/{k / 10:.1f}"] = metrics(
                y_eval[clean_mask], pred[clean_mask]
            )["MAE"]
    best = min(out, key=out.get)
    return out, best


def try_tabm(train_f, stop_f, eval_f, y_eval, clean_mask):
    """TabM trained on the temporal-split training months, early-stopped on
    the same frame LightGBM uses. Predictions cached like every other stage."""
    cache_file = BAG_CACHE / f"tabm_eval_seed{SEED}.npy"
    try:
        if cache_file.exists():
            return report(np.load(cache_file), y_eval, clean_mask, cached=True), np.load(cache_file)
        from tabm_model import train_tabm

        t0 = time.time()
        model = train_tabm(train_f, stop_f, seed=SEED)
        log_pred = model.predict_log(eval_f)
        np.save(cache_file, log_pred)
        return report(
            log_pred, y_eval, clean_mask, seconds=round(time.time() - t0, 1)
        ), log_pred
    except Exception as exc:  # noqa: BLE001 - record any failure and move on
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}, None


def try_tabpfn_ckpt(tag, ckpt, train_f, x_eval, y_eval, clean_mask, chunk=750):
    """A TabPFN checkpoint fit with the full training window as one context.

    v2 is license-free; v2.5/v3 checkpoints require Prior Labs' one-time
    license acceptance (TABPFN_TOKEN) and record themselves as unavailable
    without it — never bypassed here. Predictions are cached per chunk so an
    interrupted run resumes.
    """
    from tabpfn import TabPFNRegressor

    n_chunks = int(np.ceil(len(x_eval) / chunk))
    chunk_files = [BAG_CACHE / f"tabpfn_{tag}_eval_seed{SEED}_chunk{i}.npy" for i in range(n_chunks)]
    try:
        missing = [i for i, f in enumerate(chunk_files) if not f.exists()]
        seconds = None
        if missing:
            model = TabPFNRegressor(
                model_path=ckpt,
                device="cpu",
                n_estimators=N_ESTIMATORS,
                random_state=SEED,
                ignore_pretraining_limits=True,
            )
            t0 = time.time()
            model.fit(tabpfn_frame(train_f), np.log(train_f["posted_rate"].to_numpy()))
            for i in missing:
                np.save(chunk_files[i], model.predict(x_eval.iloc[i * chunk : (i + 1) * chunk]))
                print(f"{tag} chunk {i + 1}/{n_chunks} done", flush=True)
            seconds = round(time.time() - t0, 1)
        log_pred = np.concatenate([np.load(f) for f in chunk_files])
        extra = {"context_rows": len(train_f), "checkpoint": ckpt}
        if seconds is not None:
            extra["seconds"] = seconds
        else:
            extra["cached"] = True
        return report(log_pred, y_eval, clean_mask, **extra), log_pred
    except Exception as exc:  # noqa: BLE001 - record any failure and move on
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}, None


def try_tabicl(train_f: pd.DataFrame, x_eval: pd.DataFrame, y_eval, clean_mask):
    """Evaluate TabICL v2 if its checkpoint is available; report why not
    otherwise. TabICL is built for large in-context training sets, so the
    full cleaned Jan-Aug data is passed as a single context (capped for CPU).

    Returns (report_dict, log_pred-or-None); log predictions are cached so
    the slow CPU inference runs once and blend grids can reuse them."""
    cache_file = BAG_CACHE / f"tabicl_eval_seed{SEED}.npy"
    try:
        if cache_file.exists():
            log_pred = np.load(cache_file)
            pred = np.exp(log_pred)
            return {
                "status": "ok",
                "cached": True,
                "all_rows": metrics(y_eval, pred),
                "clean_rows": metrics(y_eval[clean_mask], pred[clean_mask]),
            }, log_pred

        from tabicl import TabICLRegressor

        model_path = TABICL_MODEL_PATH if os.path.exists(TABICL_MODEL_PATH) else None
        model = TabICLRegressor(
            device="cpu", n_estimators=2, random_state=SEED, model_path=model_path
        )
        # Cap the in-context training set so CPU inference stays tractable.
        ctx = train_f
        if len(ctx) > 20_000:
            idx = np.random.default_rng(SEED).choice(len(ctx), 20_000, replace=False)
            ctx = ctx.iloc[idx]
        t0 = time.time()
        model.fit(tabpfn_frame(ctx), np.log(ctx["posted_rate"].to_numpy()))
        log_pred = model.predict(x_eval)
        np.save(cache_file, log_pred)
        pred = np.exp(log_pred)
        return {
            "status": "ok",
            "seconds": round(time.time() - t0, 1),
            "all_rows": metrics(y_eval, pred),
            "clean_rows": metrics(y_eval[clean_mask], pred[clean_mask]),
        }, log_pred
    except Exception as exc:  # noqa: BLE001 - record any failure and move on
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
        }, None


def main() -> None:
    from tabpfn import TabPFNRegressor

    train_test = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    daily = daily_market_series([train_test, validation])
    weight_median = train_test["weight"].abs().median()

    cutoff = pd.Timestamp("2025-09-01")
    train_clean = clean_training_rows(train_test[train_test["date"] < cutoff])
    holdout = train_test[train_test["date"] >= cutoff]
    train_f = build_features(train_clean, daily, weight_median)
    holdout_f = build_features(holdout, daily, weight_median)
    stop_f = build_features(clean_training_rows(holdout), daily, weight_median)
    align_categories([train_f, holdout_f, stop_f])

    rng = np.random.default_rng(SEED)
    eval_idx = rng.choice(len(holdout_f), N_EVAL, replace=False)
    eval_f = holdout_f.iloc[eval_idx]
    y_eval = eval_f["posted_rate"].to_numpy()
    rpm = y_eval / eval_f["distance"].to_numpy()
    clean_mask = (rpm >= 0.8) & (rpm <= 6.0)

    # --- LightGBM reference on the identical rows (cached across reruns) ---
    BAG_CACHE.mkdir(exist_ok=True)
    lgb_cache = BAG_CACHE / f"lgb_eval_seed{SEED}.npy"
    if lgb_cache.exists():
        lgb_log = np.load(lgb_cache)
        print("lightgbm reference loaded from cache", flush=True)
    else:
        booster = fit_lgb(train_f, stop_f)
        lgb_log = booster.predict(eval_f[ALL_FEATURES])
        np.save(lgb_cache, lgb_log)
        print("lightgbm reference done", flush=True)
    lgb_pred = np.exp(lgb_log)

    # --- TabPFN v2: bagged context subsamples ---
    x_eval = tabpfn_frame(eval_f)
    bag_logs = []
    for bag in range(N_BAGS):
        # Bag predictions are cached to disk so an interrupted run resumes
        # instead of recomputing finished bags (deterministic per seed).
        cache_file = BAG_CACHE / f"tabpfn_bag{bag}_seed{SEED}.npy"
        if cache_file.exists():
            bag_logs.append(np.load(cache_file))
            print(f"bag {bag + 1}/{N_BAGS} loaded from cache", flush=True)
            continue
        bag_rng = np.random.default_rng(SEED + bag)
        idx = bag_rng.choice(len(train_f), BAG_SIZE, replace=False)
        ctx = train_f.iloc[idx]
        model = TabPFNRegressor(
            model_path=MODEL_PATH,
            device="cpu",
            n_estimators=N_ESTIMATORS,
            random_state=SEED + bag,
            ignore_pretraining_limits=True,
        )
        t0 = time.time()
        model.fit(tabpfn_frame(ctx), np.log(ctx["posted_rate"].to_numpy()))
        log_pred = model.predict(x_eval)
        np.save(cache_file, log_pred)
        bag_logs.append(log_pred)
        print(f"bag {bag + 1}/{N_BAGS} done in {time.time() - t0:.0f}s", flush=True)
    tab_log = np.mean(bag_logs, axis=0)
    tab_pred = np.exp(tab_log)
    tab_pred_1bag = np.exp(bag_logs[0])

    # --- TabICL v2 (skips gracefully when weights are unreachable) ---
    tabicl_result, tabicl_log = try_tabicl(train_f, x_eval, y_eval, clean_mask)
    print("tabicl:", tabicl_result.get("status"), flush=True)

    # --- TabM: trained neural ensemble, no pretrained weights needed ---
    tabm_result, tabm_log = try_tabm(train_f, stop_f, eval_f, y_eval, clean_mask)
    print("tabm:", tabm_result.get("status"), flush=True)

    # --- TabPFN v2 with a single full-train context (vs bagging) ---
    single_result, single_log = try_tabpfn_ckpt(
        "v2_single", "tabpfn-v2-regressor.ckpt", train_f, x_eval, y_eval, clean_mask
    )
    print("tabpfn v2 single-context:", single_result.get("status"), flush=True)

    # --- License-gated newer checkpoints; run automatically if the user has
    # accepted Prior Labs' license and exported TABPFN_TOKEN ---
    gated, gated_logs = {}, {}
    for tag, ckpt in (
        ("v2.5_default", "tabpfn-v2.5-regressor-v2.5_default.ckpt"),
        ("v2.5_real", "tabpfn-v2.5-regressor-v2.5_real.ckpt"),
        ("v3_default", "tabpfn-v3-regressor-v3_default.ckpt"),
    ):
        gated[tag], gated_logs[tag] = try_tabpfn_ckpt(
            tag, ckpt, train_f, x_eval, y_eval, clean_mask
        )
        print(f"tabpfn {tag}:", gated[tag].get("status"), flush=True)

    # --- Blend in log space ---
    blend_results = {}
    for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
        pred = np.exp(w * lgb_log + (1 - w) * tab_log)
        blend_results[float(w)] = metrics(y_eval[clean_mask], pred[clean_mask])["MAE"]
    best_w = min(blend_results, key=blend_results.get)

    # --- 3-way simplex grids over the available models ---
    blend3_results, best3 = ({}, None)
    if tabicl_log is not None:
        blend3_results, best3 = grid3(lgb_log, tab_log, tabicl_log, y_eval, clean_mask)
    tabm3_results, tabm3_best = ({}, None)
    if tabm_log is not None:
        tabm3_results, tabm3_best = grid3(lgb_log, tab_log, tabm_log, y_eval, clean_mask)
    single3_results, single3_best = ({}, None)
    if tabm_log is not None and single_log is not None:
        single3_results, single3_best = grid3(
            lgb_log, single_log, tabm_log, y_eval, clean_mask
        )

    results = {
        "setup": {
            "eval_rows": N_EVAL,
            "bags": N_BAGS,
            "bag_size": BAG_SIZE,
            "n_estimators_per_bag": N_ESTIMATORS,
            "checkpoint": "tabpfn-v2-regressor.ckpt",
            "device": "cpu",
        },
        "lightgbm": {
            "all_rows": metrics(y_eval, lgb_pred),
            "clean_rows": metrics(y_eval[clean_mask], lgb_pred[clean_mask]),
        },
        "tabpfn_v2_single_bag": {
            "clean_rows": metrics(y_eval[clean_mask], tab_pred_1bag[clean_mask]),
        },
        "tabpfn_v2_bagged": {
            "all_rows": metrics(y_eval, tab_pred),
            "clean_rows": metrics(y_eval[clean_mask], tab_pred[clean_mask]),
        },
        "tabicl_v2": tabicl_result,
        "tabm": tabm_result,
        "tabpfn_v2_single_context": single_result,
        "tabpfn_license_gated": gated,
        "blend_mae_by_lgb_weight": blend_results,
        "best_blend": {
            "lgb_weight": best_w,
            "clean_rows_mae": blend_results[best_w],
        },
    }
    if best3 is not None:
        results["blend3_mae_by_weights_lgb_tabpfn_tabicl"] = blend3_results
        results["best_blend3"] = {
            "weights_lgb_tabpfn_tabicl": best3,
            "clean_rows_mae": blend3_results[best3],
        }
    if tabm3_best is not None:
        results["blend3_mae_by_weights_lgb_tabpfn_tabm"] = tabm3_results
        results["best_blend3_tabm"] = {
            "weights_lgb_tabpfn_tabm": tabm3_best,
            "clean_rows_mae": tabm3_results[tabm3_best],
        }
    if single3_best is not None:
        results["blend3_mae_by_weights_lgb_tabpfnsingle_tabm"] = single3_results
        results["best_blend3_single"] = {
            "weights_lgb_tabpfnsingle_tabm": single3_best,
            "clean_rows_mae": single3_results[single3_best],
        }
    out = ROOT / "report" / "tabfm_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
