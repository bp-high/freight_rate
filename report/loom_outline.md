# Loom walkthrough outline (2–3 minutes)

Talking points matching the assessment's requested structure. Suggested
screen: the README, then `src/features.py` / `src/train_predict.py`, then the
chart.

## 1. Key findings from exploring the data (~40s)
- 48k labeled loads Jan–Oct 2025; the 12k scored loads are Nov–Dec 2025 —
  so this is a *forecasting* problem, and validation must respect time.
- Distance dominates (corr 0.91) with a tapering rate-per-mile
  ($2.67/mi short-haul → $1.91/mi long-haul).
- `quote_signal` is a per-mile quote: `quote_signal × distance` correlates
  0.968 with the rate.
- `market_index` is a daily market state: near-constant within a day, strong
  weekly cycle (weekend dips) and annual cycle (May peak, Sep trough); its
  daily mean correlates 0.58 with daily mean rate-per-mile.
- Equipment premiums (Reefer > Flatbed > Dry Van), heavier loads pay more.

## 2. Data-quality issues and treatment (~30s)
- 0.6% negative weights = sign flips (magnitudes match positives) → abs().
- Missing weight → median + flag; missing market_index → per-date mean + flag.
- 1.1% of labels have rate-per-mile outside $0.80–$6.00, uniform across
  months/equipment → random corruption; dropped from *training only*.

## 3. Why a three-family blend (~30s)
- LightGBM captures the nonlinear distance taper, equipment × distance
  interactions, and market nonlinearities with little tuning; trained on
  log(rate) so errors are multiplicative. Halves the baseline error
  ($58 vs $135 MAE).
- Benchmarked four model families on the same temporal holdout
  (report/tabfm_metrics.json): TabPFN v2 with the whole 38k-row training
  window as one context hits $56.40 (bagged 4×8k contexts: $65.27 —
  context size beats bagging); TabM, a trained MLP ensemble, is the best
  single model at $57.90; TabICL v2 trails at $86.85 and gets zero weight
  in every grid.
- Shipped: 0.4 LightGBM + 0.1 TabPFN + 0.5 TabM in log space — three
  genuinely different error profiles (trees / in-context transformer /
  trained MLPs).
- TabPFN v2.5/v3 exist but sit behind a non-commercial license gate
  (TABPFN_TOKEN); the experiment script picks them up automatically where
  a token is present — the shipped blend uses only license-free weights.

## 4. Training and validation split (~40s)
- Primary split is temporal: train Jan–Aug, hold out Sep–Oct — mirrors
  predicting Nov–Dec from all labeled data.
- Temporal MAE $58 / MAPE 2.5% / R² 0.967 (clean rows); a shuffled 80/20
  split reads $46 — ~26% optimistic, which is why the temporal number is the
  one to trust.
- Boosting rounds picked by early stopping on the temporal holdout, then the
  model is refit on all ten months.
- Blend weights get a stricter test: rolling-origin across four folds
  (predict Jul/Aug/Sep/Oct from earlier months only,
  report/rolling_validation.json). Regimes swing hard — TabPFN wins
  Jul/Sep, collapses Aug/Oct — so weights minimize the MEAN across folds
  ($51.74) instead of one window's optimum ($50.97 on Sep–Oct, but $56.64
  on the rolling test).

## 5. Code walkthrough (~30s)
- `src/features.py`: one shared cleaning + feature-engineering path for
  train, validation, and the December lane (no leakage — features only).
- `src/validate.py`: both splits + metrics + baseline.
- `src/tabfm_experiment.py` + `src/tabm_model.py`: the model comparison;
  `src/rolling_validate.py`: the fold-based weight selection.
- `src/final_blend_predict.py`: final blend fit, writes
  `validation_predictions.csv` and fills `data/december_chart_inputs.csv`
  (`src/train_predict.py` is the LightGBM-only variant).
- December lane trick: the chart inputs carry no market columns, but
  validation.csv has ~200 real December loads per day — their per-date mean
  market_index supplies the market state; quote_signal uses the lane's
  Dry Van average. Result: $851–$882, inside the lane's historical
  $758–$974 range and close to its $857 mean, with the weekly cycle
  (midweek peaks) visible in the final chart.
