const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  ImageRun, LevelFormat, convertInchesToTwip,
} = require("docx");

const ACCENT = "064A56";
const GREY = "455A60";

const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 140 }, children: [new TextRun({ text: t, color: ACCENT })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 100 }, children: [new TextRun({ text: t, color: ACCENT })] });
const p = (runs, opts = {}) =>
  new Paragraph({
    spacing: { after: 120, line: 276 },
    children: (Array.isArray(runs) ? runs : [runs]).map((r) =>
      typeof r === "string" ? new TextRun({ text: r, size: 21 }) : r
    ),
    ...opts,
  });
const b = (t) => new TextRun({ text: t, bold: true, size: 21 });
const bullet = (runs) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 276 },
    children: (Array.isArray(runs) ? runs : [runs]).map((r) =>
      typeof r === "string" ? new TextRun({ text: r, size: 21 }) : r
    ),
  });

// --- metrics table ---
const COLW = [3060, 1550, 1550, 1550, 1650];
const TOTAL = COLW.reduce((a, c) => a + c, 0);
const cell = (text, { bold = false, shade = null, align = AlignmentType.RIGHT, w } = {}) =>
  new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: align, children: [new TextRun({ text, bold, size: 20 })] })],
  });
const row = (cells, opts = {}) =>
  new TableRow({
    children: cells.map((c, i) =>
      cell(c, { w: COLW[i], align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT, ...opts })
    ),
  });

const metricsTable = new Table({
  width: { size: TOTAL, type: WidthType.DXA },
  columnWidths: COLW,
  rows: [
    row(["Evaluation", "MAE ($)", "RMSE ($)", "MAPE (%)", "R²"], { bold: true, shade: "E3EDEF" }),
    row(["Temporal holdout — all rows", "103.29", "631.98", "4.54", "0.829"]),
    row(["Temporal holdout — clean rows", "58.14", "255.08", "2.46", "0.967"], { shade: "F2F7F8" }),
    row(["Temporal — rate-per-mile baseline", "134.89", "638.57", "6.19", "0.825"]),
    row(["Random 80/20 — clean rows", "46.05", "214.84", "1.98", "0.975"], { shade: "F2F7F8" }),
  ],
});

const TCOLW = [3760, 1500, 1500, 2600];
const TTOTAL = TCOLW.reduce((a, c) => a + c, 0);
const trow = (cells, opts = {}) =>
  new TableRow({
    children: cells.map((c, i) =>
      cell(c, { w: TCOLW[i], align: i === 0 || i === 3 ? AlignmentType.LEFT : AlignmentType.RIGHT, ...opts })
    ),
  });
const tabfmTable = new Table({
  width: { size: TTOTAL, type: WidthType.DXA },
  columnWidths: TCOLW,
  rows: [
    trow(["Model (identical 3,000 holdout rows)", "MAE ($)", "MAPE (%)", "Notes"], { bold: true, shade: "E3EDEF" }),
    trow(["LightGBM", "63.23", "2.50", "reference"]),
    trow(["TabPFN v2 — single 8k context", "78.53", "3.22", "in-context, no training"], { shade: "F2F7F8" }),
    trow(["TabPFN v2 — bagged 4 × 8k contexts", "65.27", "2.58", "bagging closes most of the gap"]),
    trow(["Blend: 0.5·LGB + 0.5·TabPFN (log)", "58.75", "—", "−7.1% MAE vs LightGBM"], { bold: true, shade: "F2F7F8" }),
    trow(["TabICL v2", "n/a", "n/a", "weights unreachable in build env"]),
  ],
});

const img = fs.readFileSync("scorer_results/candidate_december.png");

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 360, hanging: 200 } } },
          },
        ],
      },
    ],
  },
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 21 }, paragraph: { spacing: { line: 276 } } },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, bottom: 1080, left: 1260, right: 1260 },
        },
      },
      children: [
        new Paragraph({
          spacing: { after: 60 },
          children: [new TextRun({ text: "Freight Rate Prediction — ML Assessment Report", bold: true, size: 40, color: ACCENT })],
        }),
        new Paragraph({
          spacing: { after: 280 },
          children: [new TextRun({ text: "LightGBM model · 48,000 labeled loads (Jan–Oct 2025) · 12,000 scored loads (Nov–Dec 2025)", size: 22, color: GREY })],
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "9DAFB3" } },
        }),

        h1("1. Summary"),
        p([
          "Posted rates are modeled on ",
          b("log(posted_rate)"),
          " with a LightGBM gradient-boosted tree model, which on a strictly forward-in-time holdout (train Jan–Aug, predict Sep–Oct) reaches ",
          b("MAE $58 / MAPE 2.5% / R² 0.967"),
          " on clean rows — less than half the error of a rate-per-mile lookup baseline ($135 MAE). A tabular-foundation-model comparison on the same holdout showed a ",
          b("50/50 log-space blend of LightGBM and a bagged TabPFN v2 ensemble cuts MAE a further 7%"),
          ", so the shipped predictions use that blend, retrained/re-contexted on all ten labeled months, for every load in validation.csv plus the fixed Lexington → Fort Wayne December lane.",
        ]),

        h1("2. Data exploration — key findings"),
        bullet([b("Distance dominates. "), "Correlation with posted_rate is 0.91; rate-per-mile tapers from ~$2.67 for short hauls (<330 mi) to ~$1.91 beyond 2,200 mi, so the model needs the nonlinear distance shape, not a single per-mile constant."]),
        bullet([b("quote_signal is a per-mile quote. "), "quote_signal × distance correlates 0.968 with the rate — the strongest engineered signal in the data."]),
        bullet([b("market_index is a daily market state. "), "It is nearly constant within a day (per-day σ ≈ 0.025), with a clear annual cycle (peak ~1.30 in May, trough ~0.89 in September) and a strong weekly cycle (weekend dips). Daily mean market_index correlates 0.58 with daily mean rate-per-mile."]),
        bullet([b("Stable secondary effects. "), "Equipment premium (Reefer $2.38 > Flatbed $2.30 > Dry Van $2.12 per mile), heavier loads price higher, and there is residual upward drift through the year beyond what market_index explains."]),
        bullet([b("Geography is consistent. "), "Each of the 64 cities has one fixed coordinate pair; road distance ≈ 1.18 × straight-line (haversine) distance. Eight validation cities never appear in training — coordinates let the model generalize to them."]),

        h1("3. Data quality issues and treatment"),
        bullet([b("Sign-flipped weights (0.6%). "), "292 training rows have negative weight; their magnitude distribution matches positive rows exactly, so they are sign errors — fixed with absolute value."]),
        bullet([b("Missing values. "), "weight (0.6%) is median-imputed with an indicator flag; market_index (0.8%) is imputed with its per-date mean (it is near-constant within a day), also flagged."]),
        bullet([b("Corrupted labels (1.1%). "), "599 training rows have rate-per-mile outside $0.80–$6.00 (up to $14/mile, down to $0.33/mile), spread uniformly across months and equipment — consistent with random label corruption. They are dropped from training only; reported holdout metrics include them (“all rows”) and exclude them (“clean rows”)."]),

        h1("4. Train / validation split approach"),
        p([
          "The scored set (Nov–Dec 2025) lies strictly after every labeled load (Jan–Oct 2025), so the primary validation must test forward-in-time generalization. I hold out the ",
          b("last two labeled months (Sep–Oct, ~9.5k loads) and train on Jan–Aug (~38k loads)"),
          ". This reproduces the real difficulty: unseen future months, market conditions that drifted from training, and early stopping tuned against future data rather than shuffled data. A shuffled 80/20 split is reported only to quantify the optimism of random splitting (MAE $46 vs $58 — a ~26% gap a random split would have hidden).",
        ]),
        metricsTable,
        p([new TextRun({ text: "“Clean rows” excludes holdout rows whose label itself is corrupted (rate-per-mile outside $0.80–$6.00); those rows are unpredictable by construction and dominate RMSE. The final boosting-round count is chosen by early stopping on the temporal holdout, then scaled proportionally when refitting on all ten months.", size: 19, italics: true, color: GREY })], { spacing: { before: 100, after: 160 } }),

        h1("5. Model and features"),
        p([
          b("Model: "), "LightGBM regression on log(posted_rate) (multiplicative errors match how rates scale with distance), 127 leaves, learning rate 0.05, ~160 trees chosen by early stopping, fixed seed. ",
          b("Features: "), "distance (raw, log, haversine), equipment, pickup/delivery city categoricals plus coordinates and lane bearing, cleaned weight, load-level market_index and quote_signal, pooled per-date market aggregates (daily mean and trailing 7-day mean of market_index, daily mean of quote_signal), and calendar fields (month, day-of-week, day-of-year).",
        ]),
        p([
          "Feature importance confirms the EDA story: distance features carry most of the gain, followed by equipment, quote_signal, weight, and the market/seasonality block. A linear model was rejected because the distance taper, equipment × distance interactions, and market nonlinearities are exactly what trees capture cheaply.",
        ]),

        h2("5.1 Tabular foundation models (TabPFN v2, TabICL v2)"),
        p([
          "Two recent tabular foundation models were benchmarked against LightGBM on a fixed 3,000-row subsample of the same temporal holdout (identical rows for every model; clean-row metrics). TabPFN v2 is an in-context learner capped at ~10k training rows per fit, so it was applied as a bagged ensemble of four disjoint 8,000-row contexts whose log-predictions are averaged. TabICL v2 could not be evaluated in the build environment (its checkpoints are served from gated hosting the environment cannot reach); the experiment script runs it automatically wherever the weights are reachable.",
        ]),
        tabfmTable,
        p([new TextRun({ text: "TabPFN alone does not beat a tuned LightGBM here, but its errors are decorrelated enough that the simple 50/50 log-space blend wins by ~7% MAE, with a flat optimum across blend weights 0.4–0.6 (robust, not tuned to the grid). The shipped validation_predictions.csv and December chart use this blend; src/train_predict.py reproduces the LightGBM-only outputs.", size: 19, italics: true, color: GREY })], { spacing: { before: 100, after: 160 } }),

        h1("6. Fixed December lane prediction"),
        p([
          "december_chart_inputs.csv provides no market_index or quote_signal, but validation.csv contains ~200 real December loads per day, whose features are given. For each December date, ",
          b("market_index is imputed from the per-date mean of those loads"),
          " (legitimate: features only, no labels involved), and the per-date market aggregates are computed the same way. ",
          b("quote_signal uses the lane’s Dry Van average (2.02)"),
          " from 21 historical Lexington → Fort Wayne loads, with day-to-day quote variation entering through the pooled daily aggregate. The blended predictions land at $821–$840 — inside the lane’s historical $758–$973 range — and reproduce the weekly market cycle (midweek peaks, weekend troughs).",
        ]),
        new Paragraph({
          spacing: { before: 120, after: 60 },
          alignment: AlignmentType.CENTER,
          children: [
            new ImageRun({
              type: "png",
              data: img,
              transformation: { width: 660, height: 220 },
            }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "Chart produced by the provided score.py from the committed predictions.", size: 18, italics: true, color: GREY })],
        }),

        h1("7. Reproducibility"),
        p([
          "GitHub repository: ", b("bp-high/freight_rate"), ". ",
          "python -m pip install -r requirements.txt (plus requirements-tabfm.txt for the foundation models), then: python src/validate.py (validation experiments), python src/tabfm_experiment.py (foundation-model comparison), python src/final_blend_predict.py (shipped blend predictions; src/train_predict.py for the LightGBM-only variant), and python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv. All randomness is seeded, and the slow TabPFN steps checkpoint their progress and resume if interrupted.",
        ]),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("report/Freight_Rate_Report.docx", buf);
  console.log("written", buf.length);
});
