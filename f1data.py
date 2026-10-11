"""FastF1 data loading helpers.

Everything is cached in ./f1_cache, so only the first run for a session hits the network.
"""
import os
from datetime import datetime, timezone

import fastf1
import numpy as np
import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.set_log_level("ERROR")

# Finishing position used for drivers who did not finish / did not start
DNF_POSITION = 22


def schedule(year):
    return fastf1.get_event_schedule(year, include_testing=False)


def next_round(year):
    """First round whose race hasn't happened yet (race day counts as upcoming)."""
    sched = schedule(year)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    upcoming = sched[sched["EventDate"].dt.normalize() >= today]
    if upcoming.empty:
        raise SystemExit(f"No upcoming races left in {year}; pass --round explicitly.")
    return int(upcoming.iloc[0]["RoundNumber"])


def event_name(year, rnd):
    return fastf1.get_event(year, rnd)["EventName"]


def _seconds(td):
    return td.dt.total_seconds() if hasattr(td, "dt") else td.total_seconds()


def load_qualifying(year, rnd):
    """One row per driver: position, best lap, sector bests and session weather."""
    q = fastf1.get_session(year, rnd, "Q")
    q.load(laps=True, telemetry=False, weather=True, messages=False)

    res = q.results.copy()
    df = pd.DataFrame({
        "Driver": res["Abbreviation"],
        "Team": res["TeamName"],
        "QualiPosition": res["Position"].astype(float),
    })
    best = res[["Q1", "Q2", "Q3"]].apply(_seconds).min(axis=1)
    df["QualiTime_s"] = best.values

    sectors = (
        q.laps.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]]
        .min()
        .apply(_seconds)
    )
    df["UltimateLap_s"] = df["Driver"].map(sectors.sum(axis=1, min_count=3))

    # Gaps as % of pole so different circuits are comparable
    pole = df["QualiTime_s"].min()
    best_ultimate = df["UltimateLap_s"].min()
    df["GapToPole_pct"] = (df["QualiTime_s"] - pole) / pole * 100
    df["UltimateGap_pct"] = (df["UltimateLap_s"] - best_ultimate) / best_ultimate * 100
    # Drivers who set no time go to the back of the field
    for col in ["GapToPole_pct", "UltimateGap_pct"]:
        df[col] = df[col].fillna(df[col].max() + 0.5)
    df["QualiPosition"] = df["QualiPosition"].fillna(len(df))
    return df.reset_index(drop=True)


def race_weather(year, rnd):
    """Weather measured at the track during the race. None if the race hasn't run."""
    r = fastf1.get_session(year, rnd, "R")
    r.load(laps=False, telemetry=False, weather=True, messages=False)
    w = r.weather_data
    if w is None or len(w) == 0:
        return None
    return {
        "AirTemp": w["AirTemp"].mean(),
        "Humidity": w["Humidity"].mean(),
        "RainChance": float(w["Rainfall"].any()),
    }


def forecast_weather(year, rnd, hours=2):
    """Open-Meteo hourly forecast (free, no API key) for the race window."""
    ev = fastf1.get_event(year, rnd)
    start = pd.Timestamp(ev["Session5DateUtc"])
    place = None
    for name in [ev["Location"], ev["Country"]]:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1}, timeout=10,
        ).json()
        if geo.get("results"):
            place = geo["results"][0]
            break
    if place is None:
        raise ValueError(f"could not geocode {ev['Location']}")

    fc = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability",
            "start_date": start.date().isoformat(), "end_date": start.date().isoformat(),
            "timezone": "UTC",
        },
        timeout=10,
    ).json()
    if "hourly" not in fc:
        raise ValueError(fc.get("reason", "no forecast available"))
    hourly = pd.DataFrame(fc["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    window = hourly[(hourly["time"] >= start.floor("h")) & (hourly["time"] < start + pd.Timedelta(hours=hours))]
    if window.empty or window["temperature_2m"].isna().all():
        raise ValueError("race is outside the forecast range")
    return {
        "AirTemp": window["temperature_2m"].mean(),
        "Humidity": window["relative_humidity_2m"].mean(),
        "RainChance": window["precipitation_probability"].max() / 100,
        "Start": start,
        "Place": place["name"],
    }


def load_results(year, rnd, session="R"):
    """Classified results for a race ('R') or sprint ('S')."""
    s = fastf1.get_session(year, rnd, session)
    s.load(laps=False, telemetry=False, weather=False, messages=False)
    res = s.results
    finished = res["ClassifiedPosition"].astype(str).str.isdigit()
    return pd.DataFrame({
        "Driver": res["Abbreviation"].values,
        "Team": res["TeamName"].values,
        "FinishPosition": np.where(finished, res["Position"], DNF_POSITION).astype(float),
        "Finished": finished.values,
        "Points": res["Points"].fillna(0).values,
    })


def season_results(year, before_round):
    """Race and sprint results for every completed round before `before_round`."""
    sched = schedule(year)
    rows = []
    for _, ev in sched[sched["RoundNumber"] < before_round].iterrows():
        rnd = int(ev["RoundNumber"])
        race = load_results(year, rnd, "R")
        race["Round"], race["Session"] = rnd, "R"
        rows.append(race)
        if "sprint" in str(ev["EventFormat"]):
            sprint = load_results(year, rnd, "S")
            sprint["Round"], sprint["Session"] = rnd, "S"
            rows.append(sprint)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def form_features(results, rnd, window=3):
    """Championship share and recent race form using only rounds before `rnd`."""
    prev = results[results["Round"] < rnd]
    if prev.empty:
        return pd.DataFrame(columns=["Driver", "DriverPointsShare", "TeamPointsShare", "RecentForm"])

    driver_pts = prev.groupby("Driver")["Points"].sum()
    team_pts = prev.groupby("Team")["Points"].sum()
    races = prev[prev["Session"] == "R"].sort_values("Round")
    recent_rounds = races["Round"].unique()[-window:]
    recent = races[races["Round"].isin(recent_rounds)].groupby("Driver")["FinishPosition"].mean()

    drivers = races.drop_duplicates("Driver", keep="last")[["Driver", "Team"]]
    out = drivers.assign(
        DriverPointsShare=drivers["Driver"].map(driver_pts / max(driver_pts.max(), 1)),
        TeamPointsShare=drivers["Team"].map(team_pts / max(team_pts.max(), 1)),
        RecentForm=drivers["Driver"].map(recent),
    )
    return out.drop(columns="Team")
