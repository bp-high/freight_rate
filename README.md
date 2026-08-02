# Freight Rate Prediction Challenge

Predicts the posted rate for truckload freight from load attributes
(lane, distance, equipment, weight, date, and two market signals) using a
LightGBM gradient-boosted model trained on 48,000 labeled loads
(Jan–Oct 2025).

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
  train_predict.py                   final model; writes both prediction files
score.py                             provided scorer (validates outputs, draws chart)
validation_predictions.csv           final predictions for the 12,000 loads
report/
  validation_metrics.json            metrics emitted by src/validate.py
  Freight_Rate_Report.docx           written report (validation approach + chart)
scorer_results/
  candidate_december.png             December chart produced by score.py
```

## Setup

Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

## Reproduce everything

```bash
# 1. Validation experiments (temporal Jan-Aug -> Sep-Oct holdout, plus a
#    shuffled 80/20 comparison). Writes report/validation_metrics.json.
python src/validate.py

# 2. Train the final model on all labeled data and write
#    validation_predictions.csv and the predicted_rate column of
#    data/december_chart_inputs.csv.
python src/train_predict.py

# 3. Run the provided scorer.
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

All randomness is seeded; reruns reproduce the committed outputs.

## Approach in one paragraph

Rates are modeled as `log(posted_rate)` with LightGBM. Features: distance
(raw, log, and straight-line haversine), equipment, origin/destination
cities and coordinates, lane bearing, cleaned weight, the two market
signals plus pooled per-date market aggregates (daily mean and trailing
7-day mean of `market_index`, daily mean of `quote_signal`), and calendar
fields. Data cleaning: negative weights are sign errors (magnitudes match
the positive distribution) and are absolute-valued; missing weight is
median-imputed with a flag; missing `market_index` is imputed with its
per-date mean; ~1.1% of training rows whose rate-per-mile falls outside
$0.80–$6.00/mile are treated as corrupted labels and dropped from training
only. Validation is primarily **temporal** — train on Jan–Aug, hold out
Sep–Oct — because the scored set (Nov–Dec) lies strictly after all labeled
data. Holdout MAE is **$58 (2.5% MAPE, R² 0.967)** on clean rows versus
$135 MAE for a rate-per-mile lookup baseline. For the fixed December lane
(which has no market signal columns), `market_index` per day is taken from
the December loads present in `validation.csv`, and `quote_signal` uses the
lane's Dry Van average with daily market variation entering through the
pooled per-date aggregate features.
