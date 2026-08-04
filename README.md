# Freight Rate Prediction Challenge

Predicts the posted rate for truckload freight from load attributes
(lane, distance, equipment, weight, date, and two market signals). The
shipped predictions come from a **log-space blend of LightGBM (0.4), a
single-full-context TabPFN v2 tabular foundation model (0.1), and a TabM
neural ensemble (0.5)**. The weights were selected by rolling-origin
validation across four forward-in-time folds — not on a single holdout —
because per-month regimes vary enough that any one window misleads.

## Repository layout

```
data/
  train_test.csv                     labeled development data (Jan-Oct 2025)
  validation.csv                     12,000 loads to predict (Nov-Dec 2025)
  validation_predictions_template.csv
  december_chart_inputs.csv          fixed Lexington->Fort Wayne lane, predicted_rate filled
src/
  features.py                        cleaning + feature engineering (shared)
  validate.py                        temporal + random split validation, metrics
  train_predict.py                   LightGBM-only pipeline (baseline deliverables)
  tabm_model.py                      TabM (parameter-efficient MLP ensemble) wrapper
  tabfm_experiment.py                TabPFN / TabICL / TabM vs LightGBM on the temporal holdout
  rolling_validate.py                rolling-origin (Jul-Oct folds) blend-weight validation
  final_blend_predict.py             final model: LGB + TabPFN v2 + TabM blend; writes both outputs
score.py                             provided scorer (validates outputs, draws chart)
validation_predictions.csv           final predictions for the 12,000 loads (blend)
report/
  validation_metrics.json            LightGBM validation metrics (src/validate.py)
  tabfm_metrics.json                 foundation-model comparison (src/tabfm_experiment.py)
  rolling_validation.json            rolling-origin fold results (src/rolling_validate.py)
  final_blend_summary.json           what the shipped predictions are
  Freight_Rate_Report.docx           written report (validation approach + chart)
  loom_outline.md                    talking points for the walkthrough video
scorer_results/
  candidate_december.png             December chart produced by score.py
```

## Setup

Python 3.10+.

```bash
python -m pip install -r requirements.txt          # core pipeline + scorer
python -m pip install -r requirements-tabfm.txt    # optional: foundation models
```

TabPFN v2 weights auto-download (license-free tier; a public GCS mirror
also exists, so no HuggingFace account is needed). TabICL v2 weights are
fetched from HuggingFace on first use where that host is reachable. The
newer TabPFN v2.5 / v3 checkpoints are gated behind Prior Labs'
non-commercial license: accepting it at https://ux.priorlabs.ai and
exporting `TABPFN_TOKEN=<api key>` lets `src/tabfm_experiment.py` benchmark
them automatically; without a token those stages record themselves as
unavailable, and the shipped blend deliberately uses only the license-free
v2 weights.

## Reproduce everything

```bash
# 1. Validation experiments (temporal Jan-Aug -> Sep-Oct holdout, plus a
#    shuffled 80/20 comparison). Writes report/validation_metrics.json.
python src/validate.py

# 2. Model comparison on the same holdout (LightGBM vs TabPFN v2 bagged
#    and single-context vs TabICL v2 vs TabM vs blends). Writes
#    report/tabfm_metrics.json. Slow on CPU (~3h cold); resumes if interrupted.
TABPFN_ALLOW_CPU_LARGE_DATASET=1 python src/tabfm_experiment.py

# 3. Rolling-origin validation of the blend weights (predict Jul, Aug, Sep,
#    Oct from strictly earlier months). Writes report/rolling_validation.json.
#    Slow on CPU (~3h cold); resumes if interrupted.
TABPFN_ALLOW_CPU_LARGE_DATASET=1 python src/rolling_validate.py

# 4. Final predictions with the selected 0.4/0.1/0.5 blend. Writes
#    validation_predictions.csv and fills data/december_chart_inputs.csv.
#    Slow on CPU (~2h cold); resumes if interrupted.
TABPFN_ALLOW_CPU_LARGE_DATASET=1 python src/final_blend_predict.py

# (LightGBM-only alternative for both outputs, ~2 min: python src/train_predict.py)

# 5. Run the provided scorer.
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

All randomness is seeded; reruns reproduce the committed outputs.

## Results (temporal holdout: train Jan-Aug, predict Sep-Oct, clean rows)

| Model | MAE | MAPE | Notes |
|---|---|---|---|
| Rate-per-mile lookup baseline | $134.89 | 6.19% | equipment x distance-decile median |
| LightGBM (full holdout, 9.4k rows) | $58.14 | 2.46% | R² 0.967 |
| LightGBM (3k comparison subsample) | $63.23 | 2.50% | same rows as all rows below |
| TabPFN v2, single 8k context | $78.53 | 3.22% | in-context, no training |
| TabPFN v2, bagged 4 x 8k contexts | $65.27 | 2.58% | bagging closes most of the gap |
| TabPFN v2, single 38k context | $56.40 | 2.24% | context size beats bagging by ~$9 |
| TabICL v2, single 20k context | $86.85 | 3.36% | not competitive; blend grids assign it zero weight |
| TabM (MLP ensemble, trained) | $57.90 | 2.35% | beats LightGBM as a single model |
| Blend 0.5 LGB + 0.5 TabPFN-bagged | $58.75 | — | previously shipped 2-way blend |
| Blend 0.1/0.5/0.4 LGB/TabPFN-38k/TabM | $50.97 | — | this window's optimum — but see rolling table |
| **Blend 0.4/0.1/0.5 LGB/TabPFN-38k/TabM** | **$53.35** | — | **shipped: selected by rolling-origin, not this window** |

## Rolling-origin validation (predict month m from months < m, clean rows)

Blend weights picked on one window can overfit that window's regime, so
the final weights minimize MEAN clean-row MAE across four forward-in-time
folds (`src/rolling_validate.py`). Regimes swing hard month to month —
single-context TabPFN is the best model in Jul/Sep and the worst in
Aug/Oct — which is exactly why the mean, not any single fold, chooses.

| Candidate (LGB/TabPFN-38k/TabM) | Jul | Aug | Sep | Oct | Mean |
|---|---|---|---|---|---|
| LightGBM only | $66.67 | $70.81 | $57.99 | $53.13 | $62.15 |
| TabPFN v2 single-context only | $47.75 | $129.91 | $40.49 | $74.43 | $73.15 |
| TabM only | $63.57 | $45.19 | $53.02 | $67.51 | $57.32 |
| Previous blend (0.5 LGB + 0.5 bagged) | $64.73 | $95.47 | $52.10 | $47.64 | $64.98 |
| Sep-Oct holdout optimum 0.1/0.5/0.4 | — | — | — | — | $56.64 |
| **Shipped 0.4/0.1/0.5** | — | — | — | — | **$51.74** |

## Approach in one paragraph

Rates are modeled as `log(posted_rate)`. Features: distance (raw, log, and
straight-line haversine), equipment, origin/destination cities and
coordinates, lane bearing, cleaned weight, the two market signals plus
pooled per-date market aggregates (daily mean and trailing 7-day mean of
`market_index`, daily mean of `quote_signal`), and calendar fields. Data
cleaning: negative weights are sign errors and are absolute-valued; missing
weight is median-imputed with a flag; missing `market_index` is imputed
with its per-date mean; ~1.1% of training rows whose rate-per-mile falls
outside $0.80–$6.00/mile are treated as corrupted labels and dropped from
training only. Validation is primarily **temporal** — train on Jan–Aug,
hold out Sep–Oct — because the scored set (Nov–Dec) lies strictly after all
labeled data, and blend weights are additionally re-validated
rolling-origin across Jul–Oct folds. The final predictor combines
LightGBM (0.4), single-full-context TabPFN v2 (0.1), and TabM (0.5) in
log space. For the fixed December lane (which has no market
signal columns), `market_index` per day is taken from the December loads
present in `validation.csv`, and `quote_signal` uses the lane's Dry Van
average with daily market variation entering through the pooled per-date
aggregate features.
