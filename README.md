# F1 Race Predictions with FastF1

Predicts Formula 1 finishing orders from live session data pulled with [FastF1](https://docs.fastf1.dev/), and compares a range of models (including AutoML) on the 2026 season.

## Scripts

### predict_race.py — Predict any race
- Trains on every completed race of the season: qualifying position, gap to pole, best sectors (ultimate lap), championship points share, recent form, and race-day weather, with finishing position as the target
- XGBoost with monotone constraints, validated by leaving one race out at a time
- Race-day weather comes from the free [Open-Meteo](https://open-meteo.com/) forecast (no API key). Past races use the weather measured at the track
- Defaults to the next upcoming race. Pass `--round N` for a past race to backtest it against the real result

### compare_models.py — Model comparison incl. AutoML
Runs the same dataset through 8 models: a grid-position baseline, Linear Regression, Ridge, Random Forest, Gradient Boosting, LightGBM, XGBoost, and FLAML AutoML.
- **Holdout test:** trains on the early races and tests on the last 4 (`--test-races`)
- **Walk-forward:** predicts each race using only the races before it, which matches real use
- Metrics: MAE, RMSE, R², Spearman rank correlation, winner hit rate, podium hit rate
- Saves `outputs/model_comparison.csv` and `.png`. `--automl-budget` sets AutoML seconds per fit (default 30)

### f1data.py — Data loading
FastF1 helpers for qualifying, race and sprint results, standings, form, measured race weather and forecasts. Everything is cached in `f1_cache/`, so only the first run downloads data.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# macOS: XGBoost and LightGBM also need OpenMP
brew install libomp
```

## Usage

```bash
.venv/bin/python predict_race.py             # Next race
.venv/bin/python predict_race.py --round 16  # Backtest a completed race
.venv/bin/python compare_models.py           # Compare models incl. AutoML (~6 min)
```

Inspired by [mar-antaya/2026_f1_predictions](https://github.com/mar-antaya/2026_f1_predictions).
