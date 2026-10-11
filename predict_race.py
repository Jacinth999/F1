"""Predict any 2026 race using live FastF1 data.

Trains on every completed race this season (qualifying pace, championship form,
recent results, weather -> finishing position), then predicts the target race
from its qualifying session.

    python predict_race.py              # next upcoming race
    python predict_race.py --round 16   # backtest a past race against its real result
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

import f1data

warnings.filterwarnings("ignore")

FEATURES = [
    "QualiPosition",      # starting position (from qualifying)
    "GapToPole_pct",      # single-lap pace, % slower than pole
    "UltimateGap_pct",    # best S1+S2+S3, % slower than fastest
    "DriverPointsShare",  # championship points / leader's points
    "TeamPointsShare",    # constructor points / leading team's points
    "RecentForm",         # mean finish over last 3 races
    "AirTemp",            # race-day air temperature
    "Humidity",           # race-day relative humidity
    "RainChance",         # 1/0 for past races, forecast probability for upcoming
]
# +1: higher value -> worse finish, -1: higher value -> better finish, 0: free
MONOTONE = (1, 1, 1, -1, -1, 1, 0, 0, 0)


def build_round(year, rnd, results, weather):
    """Feature rows for one round (qualifying + form going into that round)."""
    df = f1data.load_qualifying(year, rnd)
    df = df.merge(f1data.form_features(results, rnd), on="Driver", how="left")
    # New drivers / first race: neutral form values
    df["DriverPointsShare"] = df["DriverPointsShare"].fillna(0)
    team_share = df.groupby("Team")["TeamPointsShare"].transform("max")
    df["TeamPointsShare"] = df["TeamPointsShare"].fillna(team_share).fillna(0)
    df["RecentForm"] = df["RecentForm"].fillna(df["QualiPosition"])
    for col in ["AirTemp", "Humidity", "RainChance"]:
        df[col] = weather[col]
    df["Round"] = rnd
    return df


def build_training_set(year, before_round):
    """Feature rows + actual finish for every completed race before `before_round`.

    Returns (season results, training rows). Retirements are dropped from the
    training rows since they're mostly bad luck rather than pace.
    """
    results = f1data.season_results(year, before_round)
    race_results = results[results["Session"] == "R"]
    frames = []
    for rnd in sorted(race_results["Round"].unique()):
        rows = build_round(year, rnd, results, f1data.race_weather(year, rnd))
        finish = race_results[race_results["Round"] == rnd][["Driver", "FinishPosition", "Finished"]]
        frames.append(rows.merge(finish, on="Driver", how="inner"))
    train = pd.concat(frames, ignore_index=True)
    return results, train[train["Finished"]].reset_index(drop=True)


def make_model():
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        monotone_constraints=MONOTONE,
        random_state=42,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, help="round number (default: next race)")
    args = parser.parse_args()

    year = args.year
    target = args.round or f1data.next_round(year)
    name = f1data.event_name(year, target)
    if target < 2:
        raise SystemExit("Need at least one completed race to train on; pick round >= 2.")

    print(f"Loading {year} season data up to {name} (round {target})...")
    results, train = build_training_set(year, target)

    # Past race: use what was measured. Upcoming race: use the forecast.
    weather, weather_source = None, "measured during the race"
    try:
        weather = f1data.race_weather(year, target)
    except Exception:
        pass
    if weather is None:
        try:
            weather = f1data.forecast_weather(year, target)
            weather_source = f"Open-Meteo forecast for {weather['Place']}, race start {weather['Start']:%a %d %b %H:%M} UTC"
        except Exception as e:
            weather = train[["AirTemp", "Humidity", "RainChance"]].median().to_dict()
            weather_source = f"season median (forecast unavailable: {e})"

    try:
        pred = build_round(year, target, results, weather)
    except Exception as e:
        raise SystemExit(f"Qualifying for {name} isn't available yet ({e}).")

    # Leave-one-race-out validation
    errors = []
    for rnd in train["Round"].unique():
        tr, te = train[train["Round"] != rnd], train[train["Round"] == rnd]
        m = make_model().fit(tr[FEATURES], tr["FinishPosition"])
        errors.append(mean_absolute_error(te["FinishPosition"], m.predict(te[FEATURES])))

    model = make_model().fit(train[FEATURES], train["FinishPosition"])
    pred["PredictedScore"] = model.predict(pred[FEATURES])
    pred = pred.sort_values(["PredictedScore", "QualiPosition"]).reset_index(drop=True)
    pred["Predicted"] = np.arange(1, len(pred) + 1)

    # If the race already happened, compare with what really happened
    actual = None
    try:
        actual = f1data.load_results(year, target, "R")[["Driver", "FinishPosition"]].dropna()
    except Exception:
        pass
    if actual is not None and actual.empty:
        actual = None
    if actual is not None:
        pred = pred.merge(actual.rename(columns={"FinishPosition": "Actual"}), on="Driver", how="left")

    print("\n" + "=" * 60)
    print(f"   PREDICTED FINISHING ORDER — {year} {name.upper()}")
    print(f"   Trained on {train['Round'].nunique()} races, {len(train)} driver results")
    print("=" * 60)
    header = f"  {'P':<4}{'DRV':<6}{'Team':<18}{'Grid':>5}"
    print(header + (f"{'Actual':>8}" if actual is not None else ""))
    for _, r in pred.iterrows():
        line = f"  {r['Predicted']:<4}{r['Driver']:<6}{r['Team']:<18}{r['QualiPosition']:>5.0f}"
        if actual is not None:
            a = r["Actual"]
            line += f"{'-' if pd.isna(a) else 'DNF' if a == f1data.DNF_POSITION else int(a):>8}"
        print(line)

    podium = pred.head(3)
    print()
    for medal, (_, r) in zip(["🥇", "🥈", "🥉"], podium.iterrows()):
        print(f"  {medal}  {r['Driver']} ({r['Team']})")
    print(f"\n  Leave-one-race-out MAE: {np.mean(errors):.2f} positions")
    print(f"  Race weather: {weather['AirTemp']:.1f}°C, humidity {weather['Humidity']:.0f}%, "
          f"rain {weather['RainChance']:.0%}  ({weather_source})")
    if actual is not None:
        finished = pred[pred["Actual"] < f1data.DNF_POSITION]
        print(f"  Actual-race MAE (finishers): {mean_absolute_error(finished['Actual'], finished['Predicted']):.2f} positions")

    print("\nFeature importances:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<18} {imp:.3f}  {'█' * int(imp * 50)}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{year} {name} — prediction from FastF1 data", fontweight="bold")
    order = np.argsort(model.feature_importances_)
    axes[0].barh(np.array(FEATURES)[order], model.feature_importances_[order], color="#1f77b4")
    axes[0].set_title("Feature importance")
    axes[1].barh(pred["Driver"][::-1], pred["PredictedScore"][::-1], color="#e10600")
    axes[1].set_title("Predicted finishing score (lower = better)")
    plt.tight_layout()
    out = os.path.join("outputs", f"{year}_r{target:02d}_prediction.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\nChart saved to {out}")


if __name__ == "__main__":
    main()
