"""Compare models (including FLAML AutoML) on the 2026 FastF1 dataset.

Uses the same features as predict_race.py and reports test metrics two ways:

  1. Holdout test   - train on the early races, test on the last N races
  2. Walk-forward   - for each race, train only on the races before it and
                      predict it (exactly how the model is used for real)

    python compare_models.py                    # all completed 2026 races
    python compare_models.py --automl-budget 60 # give AutoML more time per fit
"""
import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flaml import AutoML
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import f1data
from predict_race import FEATURES, build_training_set, make_model

warnings.filterwarnings("ignore")
TARGET = "FinishPosition"


class GridBaseline:
    """No learning: predicts you finish where you started."""
    def fit(self, X, y, groups=None):
        return self

    def predict(self, X):
        return X["QualiPosition"].to_numpy()


class SklearnModel:
    def __init__(self, factory):
        self.factory = factory

    def fit(self, X, y, groups=None):
        self.model = self.factory().fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class FlamlAutoML:
    """FLAML searches model types and hyperparameters within a time budget,
    validating with races kept together (group k-fold) so it can't peek."""
    def __init__(self, budget):
        self.budget = budget

    def fit(self, X, y, groups=None):
        self.automl = AutoML()
        self.automl.fit(
            X, y,
            task="regression",
            metric="mae",
            time_budget=self.budget,
            estimator_list=["lgbm", "xgboost", "xgb_limitdepth", "rf", "extra_tree"],
            eval_method="cv",
            split_type="group",
            groups=groups,
            n_splits=min(5, len(np.unique(groups))),
            seed=42,
            verbose=0,
        )
        return self

    def predict(self, X):
        return self.automl.predict(X)


def models(automl_budget):
    return {
        "Grid baseline": lambda: GridBaseline(),
        "Linear regression": lambda: SklearnModel(lambda: make_pipeline(StandardScaler(), LinearRegression())),
        "Ridge": lambda: SklearnModel(lambda: make_pipeline(StandardScaler(), Ridge(alpha=10))),
        "Random forest": lambda: SklearnModel(lambda: RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        "Gradient boosting": lambda: SklearnModel(lambda: GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)),
        "LightGBM": lambda: SklearnModel(lambda: LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
            subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)),
        "XGBoost (predict_race.py)": lambda: SklearnModel(make_model),
        "FLAML AutoML": lambda: FlamlAutoML(automl_budget),
    }


def metrics(df):
    """df has Round, actual TARGET and Pred columns for the test rows."""
    per_race = []
    for _, race in df.groupby("Round"):
        pred_rank = race["Pred"].rank(method="first")
        actual_rank = race[TARGET].rank(method="first")
        top3_pred = set(race.loc[pred_rank <= 3, "Driver"])
        top3_actual = set(race.loc[actual_rank <= 3, "Driver"])
        per_race.append({
            "Spearman": spearmanr(race["Pred"], race[TARGET]).correlation,
            "Winner": float(race.loc[pred_rank.idxmin(), "Driver"] == race.loc[actual_rank.idxmin(), "Driver"]),
            "Podium": len(top3_pred & top3_actual) / 3,
        })
    per_race = pd.DataFrame(per_race)
    return {
        "MAE": mean_absolute_error(df[TARGET], df["Pred"]),
        "RMSE": np.sqrt(mean_squared_error(df[TARGET], df["Pred"])),
        "R2": r2_score(df[TARGET], df["Pred"]),
        "Spearman": per_race["Spearman"].mean(),
        "Winner %": per_race["Winner"].mean() * 100,
        "Podium %": per_race["Podium"].mean() * 100,
    }


def evaluate(make, train, test):
    model = make().fit(train[FEATURES], train[TARGET], groups=train["Round"].to_numpy())
    out = test[["Round", "Driver", TARGET]].copy()
    out["Pred"] = model.predict(test[FEATURES])
    return out, model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--test-races", type=int, default=4, help="races in the holdout test set")
    parser.add_argument("--min-train-races", type=int, default=5, help="races needed before walk-forward starts")
    parser.add_argument("--automl-budget", type=int, default=30, help="seconds per AutoML fit")
    args = parser.parse_args()

    target = f1data.next_round(args.year)
    print(f"Loading completed {args.year} races (rounds 1-{target - 1})...")
    _, data = build_training_set(args.year, target)
    rounds = sorted(data["Round"].unique())
    print(f"{len(data)} driver results from {len(rounds)} races, {len(FEATURES)} features\n")

    holdout_rounds = rounds[-args.test_races:]
    train, test = data[~data["Round"].isin(holdout_rounds)], data[data["Round"].isin(holdout_rounds)]
    wf_rounds = rounds[args.min_train_races:]

    holdout_rows, wf_rows, automl_info = [], [], None
    for name, make in models(args.automl_budget).items():
        t0 = time.time()
        preds, model = evaluate(make, train, test)
        holdout_rows.append({"Model": name, **metrics(preds)})
        if isinstance(model, FlamlAutoML):
            automl_info = (model.automl.best_estimator, model.automl.best_config)

        wf = pd.concat([evaluate(make, data[data["Round"] < r], data[data["Round"] == r])[0] for r in wf_rounds])
        wf_rows.append({"Model": name, **metrics(wf)})
        print(f"  {name:<27} done in {time.time() - t0:5.1f}s")

    holdout = pd.DataFrame(holdout_rows).sort_values("MAE").reset_index(drop=True)
    walk = pd.DataFrame(wf_rows).sort_values("MAE").reset_index(drop=True)

    fmt = {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}", "Spearman": "{:.3f}",
           "Winner %": "{:.0f}", "Podium %": "{:.0f}"}
    def show(title, df):
        print("\n" + "=" * 92)
        print(f"  {title}")
        print("=" * 92)
        print(df.to_string(index=False, formatters={k: v.format for k, v in fmt.items()}))

    names = ", ".join(f1data.event_name(args.year, r).replace(" Grand Prix", "") for r in holdout_rounds)
    show(f"HOLDOUT TEST — trained on rounds {rounds[0]}-{holdout_rounds[0] - 1}, tested on {names}", holdout)
    show(f"WALK-FORWARD — each of rounds {wf_rounds[0]}-{wf_rounds[-1]} predicted from only the races before it", walk)
    print("\n  MAE/RMSE: positions off (lower is better)   R2, Spearman: 1.0 is perfect")
    print("  Winner %: races where the winner was called   Podium %: share of podium drivers called")
    if automl_info:
        print(f"\n  AutoML picked (holdout fit): {automl_info[0]}  {automl_info[1]}")

    os.makedirs("outputs", exist_ok=True)
    pd.concat([holdout.assign(Evaluation="holdout"), walk.assign(Evaluation="walk-forward")]) \
        .to_csv("outputs/model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (title, df) in zip(axes, [("Holdout test", holdout), ("Walk-forward", walk)]):
        df = df.sort_values("MAE", ascending=False)
        colors = ["#e10600" if m == "FLAML AutoML" else "#888888" if m == "Grid baseline" else "#1f77b4"
                  for m in df["Model"]]
        ax.barh(df["Model"], df["MAE"], color=colors)
        ax.set_title(f"{title} — MAE (positions, lower is better)")
        for y, v in enumerate(df["MAE"]):
            ax.text(v, y, f" {v:.2f}", va="center")
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=120)
    print("\n  Saved outputs/model_comparison.csv and outputs/model_comparison.png")


if __name__ == "__main__":
    main()
