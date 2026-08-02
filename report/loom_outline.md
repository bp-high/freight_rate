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

## 3. Why LightGBM (~20s)
- Needs the nonlinear distance taper, equipment × distance interactions, and
  market nonlinearities — exactly what boosted trees capture with little
  tuning; trained on log(rate) so errors are multiplicative.
- Halves the error of a rate-per-mile lookup baseline ($58 vs $135 MAE).
- Also benchmarked tabular foundation models on the same temporal holdout
  (report/tabfm_metrics.json): bagged TabPFN v2 nearly matches LightGBM
  ($65.27 vs $63.23 MAE on identical rows), and a 50/50 log-space blend
  beats both at $58.75 (−7.1% vs LightGBM) — the shipped predictions use
  the blend. TabICL v2 (single 20k-row context) trails at $86.85, and a
  full 3-way blend-weight grid assigns it zero weight, so it adds nothing
  here and the shipped blend stays LightGBM + TabPFN.

## 4. Training and validation split (~40s)
- Primary split is temporal: train Jan–Aug, hold out Sep–Oct — mirrors
  predicting Nov–Dec from all labeled data.
- Temporal MAE $58 / MAPE 2.5% / R² 0.967 (clean rows); a shuffled 80/20
  split reads $46 — ~26% optimistic, which is why the temporal number is the
  one to trust.
- Boosting rounds picked by early stopping on the temporal holdout, then the
  model is refit on all ten months.

## 5. Code walkthrough (~30s)
- `src/features.py`: one shared cleaning + feature-engineering path for
  train, validation, and the December lane (no leakage — features only).
- `src/validate.py`: both splits + metrics + baseline.
- `src/train_predict.py`: final fit, writes `validation_predictions.csv` and
  fills `data/december_chart_inputs.csv`.
- December lane trick: the chart inputs carry no market columns, but
  validation.csv has ~200 real December loads per day — their per-date mean
  market_index supplies the market state; quote_signal uses the lane's
  Dry Van average. Result: $820–$843, inside the lane's historical range,
  with the weekly cycle visible in the final chart.
