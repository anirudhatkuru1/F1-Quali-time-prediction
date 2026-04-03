"""
F1 Qualifying Time Predictor — Model Training Script
Run this ONCE locally to train and save the model.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# ─────────────────────────────────────────────
# 2025 OFFICIAL GRID
# ─────────────────────────────────────────────
F1_2025_GRID = {
    "Red Bull Racing": ["VER", "LAW"],
    "McLaren":         ["NOR", "PIA"],
    "Ferrari":         ["LEC", "HAM"],
    "Mercedes":        ["RUS", "ANT"],
    "Aston Martin":    ["ALO", "STR"],
    "Alpine":          ["GAS", "DOO"],
    "Williams":        ["ALB", "SAI"],
    "Haas F1 Team":    ["BEA", "OCO"],
    "RB":              ["TSU", "HAD"],
    "Kick Sauber":     ["HUL", "BOR"],
}

# Rookies with no historical data — imputed from team average
ROOKIES_2025 = {"ANT", "HAD", "BOR"}

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/data.csv")
print(f"  Raw rows: {len(df)}")

# ─────────────────────────────────────────────
# 2. CLEAN & FILTER
# ─────────────────────────────────────────────
df = df[df["IsPushLap"] == 1].copy()
df = df.dropna(subset=["LapTime_sec", "Team", "Driver", "Event", "TrackType"])

q_low  = df["LapTime_sec"].quantile(0.01)
q_high = df["LapTime_sec"].quantile(0.99)
df = df[(df["LapTime_sec"] >= q_low) & (df["LapTime_sec"] <= q_high)]
print(f"  Clean rows after filtering: {len(df)}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
segment_map  = {"Q1": 1, "Q2": 2, "Q3": 3}
compound_map = {"SOFT": 3, "MEDIUM": 2, "HARD": 1, "INTER": 0, "WET": -1}
speed_map    = {"Slow": 1, "Medium": 2, "Fast": 3}

df["QualiSegment_num"] = df["QualiSegment"].map(segment_map).fillna(1)
df["IsStreet"]         = (df["TrackType"] == "Street").astype(int)
df["SpeedClass_num"]   = df["LapSpeedClass"].map(speed_map).fillna(2)
df["FreshTyre_int"]    = df["FreshTyre"].astype(int)
df["Rainfall_int"]     = df["Rainfall"].astype(int)
df["Compound_num"]     = df["Compound"].map(compound_map).fillna(2)

# Team avg per circuit type (downforce proxy)
team_circuit_avg = (
    df.groupby(["Team", "TrackType"])["LapTime_sec"]
    .mean().reset_index()
    .rename(columns={"LapTime_sec": "TeamCircuitAvg"})
)
df = df.merge(team_circuit_avg, on=["Team", "TrackType"], how="left")

# Session best → driver delta from best
session_best = (
    df.groupby(["Year", "Event"])["LapTime_sec"]
    .min().reset_index()
    .rename(columns={"LapTime_sec": "SessionBest"})
)
df = df.merge(session_best, on=["Year", "Event"], how="left")
df["LapDeltaFromBest"] = df["LapTime_sec"] - df["SessionBest"]

driver_skill = (
    df.groupby("Driver")["LapDeltaFromBest"]
    .mean().reset_index()
    .rename(columns={"LapDeltaFromBest": "DriverAvgDelta"})
)
df = df.merge(driver_skill, on="Driver", how="left")

# Team-level avg delta (used for rookie imputation)
team_avg_delta = (
    df.groupby("Team")["LapDeltaFromBest"]
    .mean().reset_index()
    .rename(columns={"LapDeltaFromBest": "TeamAvgDelta"})
)

# Add rookie rows to driver_skill using their team's average
rookie_rows = []
for team, drivers in F1_2025_GRID.items():
    for drv in drivers:
        if drv in ROOKIES_2025:
            team_delta = team_avg_delta[team_avg_delta["Team"] == team]["TeamAvgDelta"]
            delta_val  = team_delta.values[0] if not team_delta.empty else driver_skill["DriverAvgDelta"].mean()
            rookie_rows.append({"Driver": drv, "DriverAvgDelta": delta_val})

if rookie_rows:
    driver_skill = pd.concat([driver_skill, pd.DataFrame(rookie_rows)], ignore_index=True)
    print(f"  Rookie imputation: {[r['Driver'] for r in rookie_rows]}")

# ─────────────────────────────────────────────
# 4. LABEL ENCODERS
# Include all 2025 teams/drivers so the app never hits unseen labels
# ─────────────────────────────────────────────
all_2025_teams   = list(F1_2025_GRID.keys())
all_2025_drivers = [d for drivers in F1_2025_GRID.values() for d in drivers]

all_teams   = sorted(set(df["Team"].unique().tolist()   + all_2025_teams))
all_drivers = sorted(set(df["Driver"].unique().tolist() + all_2025_drivers))
all_events  = sorted(df["Event"].unique().tolist())

le_team   = LabelEncoder().fit(all_teams)
le_driver = LabelEncoder().fit(all_drivers)
le_event  = LabelEncoder().fit(all_events)

df["Team_enc"]   = le_team.transform(df["Team"].astype(str))
df["Driver_enc"] = le_driver.transform(df["Driver"].astype(str))
df["Event_enc"]  = le_event.transform(df["Event"].astype(str))

# ─────────────────────────────────────────────
# 5. DEFINE FEATURES
# ─────────────────────────────────────────────
FEATURES = [
    "Team_enc", "Driver_enc", "Event_enc", "Year",
    "QualiSegment_num",
    "Compound_num", "TyreLife", "FreshTyre_int",
    "IsStreet", "SpeedClass_num", "DRSZones", "Altitude_m",
    "NumCorners", "CornerDensity", "TrackLength_m", "AvgCornerSpacing_m",
    "AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall_int",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "TeamCircuitAvg", "DriverAvgDelta",
]

for col in FEATURES:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

X = df[FEATURES]
y = df["LapTime_sec"]

# ─────────────────────────────────────────────
# 6. TIME-BASED SPLIT
# ─────────────────────────────────────────────
X_train = X[df["Year"] <= 2023]
y_train = y[df["Year"] <= 2023]
X_test  = X[df["Year"] == 2024]
y_test  = y[df["Year"] == 2024]
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 7. TRAIN XGBOOST
# ─────────────────────────────────────────────
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20,
    eval_metric="mae",
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# ─────────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"\n📊 Model Performance (2024 holdout):")
print(f"  MAE : {mae:.3f}s")
print(f"  R²  : {r2:.4f}")

# ─────────────────────────────────────────────
# 9. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,            "model/xgb_model.pkl")
joblib.dump(le_team,          "model/le_team.pkl")
joblib.dump(le_driver,        "model/le_driver.pkl")
joblib.dump(le_event,         "model/le_event.pkl")
joblib.dump(FEATURES,         "model/features.pkl")
joblib.dump(team_circuit_avg, "model/team_circuit_avg.pkl")
joblib.dump(driver_skill,     "model/driver_skill.pkl")
joblib.dump(team_avg_delta,   "model/team_avg_delta.pkl")
joblib.dump(F1_2025_GRID,     "model/f1_2025_grid.pkl")
joblib.dump(ROOKIES_2025,     "model/rookies_2025.pkl")
joblib.dump({"mae": round(mae, 3), "r2": round(r2, 4)}, "model/metrics.pkl")

print("\n✅ All artifacts saved to /model/")
print("Run: python -m streamlit run app.py")