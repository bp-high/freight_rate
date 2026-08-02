# Freight Rate Prediction Challenge

Predicts the posted rate for truckload freight from load attributes
(lane, distance, equipment, weight, date, and two market signals). The
shipped predictions come from a **50/50 log-space blend of LightGBM and a
bagged TabPFN v2 tabular foundation model**, which beat either model alone
on a strictly forward-in-time holdout.

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
  tabfm_experiment.py                TabPFN v2 / TabICL v2 vs LightGBM on the temporal holdout
  final_blend_predict.py             final model: LightGBM + TabPFN v2 blend; writes both outputs
score.py                             provided scorer (validates outputs, draws chart)
validation_predictions.csv           final predictions for the 12,000 loads (blend)
report/
  validation_metrics.json            LightGBM validation metrics (src/validate.py)
  tabfm_metrics.json                 foundation-model comparison (src/tabfm_experiment.py)
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

TabPFN v2 weights auto-download (a public GCS mirror is used, no
HuggingFace account needed). TabICL v2 weights are fetched from
HuggingFace on first use where that host is reachable.

## Reproduce everything

```bash
# 1. Validation experiments (temporal Jan-Aug -> Sep-Oct holdout, plus a
#    shuffled 80/20 comparison). Writes report/validation_metrics.json.
python src/validate.py

# 2. Foundation-model comparison on the same holdout (LightGBM vs bagged
#    TabPFN v2 vs TabICL v2 vs blend). Writes report/tabfm_metrics.json.
#    Slow on CPU (~30 min); resumes if interrupted.
TABPFN_ALLOW_CPU_LARGE_DATASET=1 python src/tabfm_experiment.py

# 3. Final predictions with the winning 50/50 blend. Writes
#    validation_predictions.csv and fills data/december_chart_inputs.csv.
#    Slow on CPU (~90 min); resumes if interrupted.
TABPFN_ALLOW_CPU_LARGE_DATASET=1 python src/final_blend_predict.py

# (LightGBM-only alternative for both outputs, ~2 min: python src/train_predict.py)

# 4. Run the provided scorer.
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

All randomness is seeded; reruns reproduce the committed outputs.

## Results (temporal holdout: train Jan-Aug, predict Sep-Oct, clean rows)

| Model | MAE | MAPE | Notes |
|---|---|---|---|
| Rate-per-mile lookup baseline | $134.89 | 6.19% | equipment x distance-decile median |
| LightGBM (full holdout, 9.4k rows) | $58.14 | 2.46% | R² 0.967 |
| LightGBM (3k comparison subsample) | $63.23 | 2.50% | same rows as below |
| TabPFN v2, single 8k context | $78.53 | 3.22% | in-context, no training |
| TabPFN v2, bagged 4 x 8k contexts | $65.27 | 2.58% | approaches LightGBM |
| **Blend 0.5 LGB + 0.5 TabPFN (log)** | **$58.75** | — | **-7.1% MAE vs LightGBM on same rows** |
| TabICL v2 | n/a | n/a | weights unreachable in this environment (HF-gated hosting) |

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
labeled data. The final predictor averages LightGBM and a 4-bag TabPFN v2
ensemble in log space. For the fixed December lane (which has no market
signal columns), `market_index` per day is taken from the December loads
present in `validation.csv`, and `quote_signal` uses the lane's Dry Van
average with daily market variation entering through the pooled per-date
aggregate features.
