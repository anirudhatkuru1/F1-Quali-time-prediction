"""
Fetch real 2025 qualifying session data (laps + weather) via FastF1
and build data_2025.csv matching the schema of data.csv
"""

import fastf1
import pandas as pd
import numpy as np
import os

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

YEAR = 2025

TEAM_MAP = {
    "Alfa Romeo Racing": "Alfa Romeo", "AlphaTauri": "RB", "Toro Rosso": "RB",
    "Racing Point": "Aston Martin", "Renault": "Alpine",
    "Haas F1 Team": "Haas", "Kick Sauber": "Alfa Romeo",
}

real25 = pd.read_csv("data/real_lap_time_2025.csv")
tracks = pd.read_csv("data/tracks.csv")
races = real25["race"].unique().tolist()

print(f"Fetching {len(races)} 2025 qualifying sessions via FastF1...")

all_rows = []

for race in races:
    print(f"  -> {race} ...", end=" ")
    try:
        session = fastf1.get_session(YEAR, race, "Q")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as e:
        print(f"FAILED ({e})")
        continue

    laps = session.laps.copy()
    if laps.empty:
        print("no lap data, skipped")
        continue

    # Split into Q1 / Q2 / Q3
    try:
        q1, q2, q3 = laps.split_qualifying_sessions()
        q1 = q1.copy(); q1["QualiSegment"] = "Q1"
        q2 = q2.copy(); q2["QualiSegment"] = "Q2"
        q3 = q3.copy(); q3["QualiSegment"] = "Q3"
        laps = pd.concat([q1, q2, q3], ignore_index=True)
    except Exception as e:
        print(f"(segment split failed: {e}, defaulting all to Q1) ", end="")
        laps["QualiSegment"] = "Q1"

    # Attach nearest weather reading by session time
    weather = session.weather_data.copy()
    if not weather.empty:
        laps = laps.sort_values("Time")
        weather = weather.sort_values("Time")
        laps = pd.merge_asof(laps, weather, on="Time", direction="nearest")
    else:
        for col in ["AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp", "WindDirection", "WindSpeed"]:
            laps[col] = np.nan

    # Push-lap heuristic: real attempt, not an in/out lap, clean timing, green flag
    laps["IsPushLap"] = (
        laps["PitOutTime"].isna() &
        laps["PitInTime"].isna() &
        laps["IsAccurate"].fillna(False) &
        laps["LapTime"].notna() &
        (laps["TrackStatus"].astype(str) == "1")
    ).astype(int)

    laps["LapTime_sec"] = laps["LapTime"].dt.total_seconds()
    laps["Year"] = YEAR
    laps["Event"] = race
    laps["Session"] = "Q"

    all_rows.append(laps)
    print(f"ok ({len(laps)} laps)")

if not all_rows:
    raise SystemExit("No sessions were successfully loaded — check race names / internet connection.")

raw2025 = pd.concat(all_rows, ignore_index=True)

# Normalise team names to match historical naming
raw2025["Team"] = raw2025["Team"].replace(TEAM_MAP)

# Attach circuit metadata
raw2025 = raw2025.merge(tracks, on="Event", how="left")

FINAL_COLS = [
    "Driver", "DriverNumber", "Team", "LapTime", "LapNumber", "Sector1Time", "Sector2Time", "Sector3Time",
    "Compound", "TyreLife", "FreshTyre", "Stint", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "TrackStatus", "IsPersonalBest", "LapStartTime", "Year", "Event", "Session", "LapTime_sec", "Time",
    "AirTemp", "Humidity", "Pressure", "Rainfall", "TrackTemp", "WindDirection", "WindSpeed",
    "QualiSegment", "IsPushLap", "CircuitName", "Country", "TrackType", "Altitude_m", "DRSZones",
    "LapSpeedClass", "TrackLength_m", "NumCorners", "CornerDensity", "AvgCornerSpacing_m",
]

for col in FINAL_COLS:
    if col not in raw2025.columns:
        raw2025[col] = np.nan

data_2025 = raw2025[FINAL_COLS]
data_2025.to_csv("data/data_2025.csv", index=False)

print(f"\n✓ Saved {len(data_2025):,} rows to data/data_2025.csv")
print(f"✓ Push laps: {data_2025['IsPushLap'].sum():,}")
print(f"✓ Races with rain detected: {data_2025.groupby('Event')['Rainfall'].max().sum()}")