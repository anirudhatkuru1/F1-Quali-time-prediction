"""
F1 Qualifying Predictor — Model Training Script
Run once: python train_model.py

KEY FIXES vs previous version:
1. Train on BEST LAP per driver per session (not all laps)
   - Old data had 6-7 laps per driver per session; only the best matters
2. Predict DELTA FROM POLE, not absolute lap time
   - Absolute times change year-to-year (regs, car pace, track resurfacing)
   - Delta from pole is far more stable and transferable to 2025
3. For final output: predicted_delta + real_2025_pole = accurate absolute time
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
ROOKIES_2025 = {"ANT", "HAD", "BOR"}

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("Loading data...")
df     = pd.read_csv("data/data.csv")
tracks = pd.read_csv("data/tracks.csv")

raw = df[df["IsPushLap"] == 1].copy()
raw = raw.dropna(subset=["LapTime_sec", "Team", "Driver", "Event", "TrackType"])
print(f"  Raw push laps: {len(raw)}")

# ─────────────────────────────────────────────
# 2. BEST LAP PER DRIVER PER SESSION
# Previous bug: model was trained on all 6-7 laps per driver per session
# but 2025 data has only the single best time → huge mismatch
# ─────────────────────────────────────────────
best = (raw.sort_values("LapTime_sec")
           .groupby(["Year", "Event", "Driver"])
           .first()
           .reset_index())
print(f"  Best-lap rows (1 per driver per session): {len(best)}")

# ─────────────────────────────────────────────
# 3. COMPUTE DELTA FROM POLE (prediction target)
# Predicting absolute times fails because car pace changes year to year.
# Delta from pole is stable: VER is ~0.0s from pole, Haas is ~1.5s, etc.
# ─────────────────────────────────────────────
pole = (best.groupby(["Year", "Event"])["LapTime_sec"]
            .min()
            .reset_index()
            .rename(columns={"LapTime_sec": "PoleTime"}))
best = best.merge(pole, on=["Year", "Event"], how="left")
best["DeltaFromPole"] = best["LapTime_sec"] - best["PoleTime"]

# Remove outliers (crashed laps, very slow laps that snuck through)
best = best[best["DeltaFromPole"] <= 6.0].copy()
print(f"  After removing delta > 6s outliers: {len(best)}")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
segment_map  = {"Q1": 1, "Q2": 2, "Q3": 3}
compound_map = {"SOFT": 3, "MEDIUM": 2, "HARD": 1, "INTER": 0, "WET": -1}
speed_map    = {"Slow": 1, "Medium": 2, "Fast": 3}

best["QualiSegment_num"] = best["QualiSegment"].map(segment_map).fillna(2)
best["IsStreet"]         = (best["TrackType"] == "Street").astype(int)
best["SpeedClass_num"]   = best["LapSpeedClass"].map(speed_map).fillna(2)
best["FreshTyre_int"]    = best["FreshTyre"].astype(int)
best["Rainfall_int"]     = best["Rainfall"].astype(int)
best["Compound_num"]     = best["Compound"].map(compound_map).fillna(3)

# Team avg delta per circuit type (team competitiveness proxy)
team_circuit_avg_delta = (
    best.groupby(["Team", "TrackType"])["DeltaFromPole"]
    .mean().reset_index()
    .rename(columns={"DeltaFromPole": "TeamCircuitAvgDelta"})
)
best = best.merge(team_circuit_avg_delta, on=["Team", "TrackType"], how="left")

# Team avg absolute time (for absolute prediction anchor)
team_circuit_avg_abs = (
    best.groupby(["Team", "TrackType"])["LapTime_sec"]
    .mean().reset_index()
    .rename(columns={"LapTime_sec": "TeamCircuitAvg"})
)
best = best.merge(team_circuit_avg_abs, on=["Team", "TrackType"], how="left")

# Driver historical avg delta from pole (driver skill)
driver_skill = (
    best.groupby("Driver")["DeltaFromPole"]
    .mean().reset_index()
    .rename(columns={"DeltaFromPole": "DriverAvgDelta"})
)
best = best.merge(driver_skill, on="Driver", how="left")

# Team avg delta (for rookie imputation)
team_avg_delta = (
    best.groupby("Team")["DeltaFromPole"]
    .mean().reset_index()
    .rename(columns={"DeltaFromPole": "TeamAvgDelta"})
)

# Rookie imputation: use team's historical avg delta
for team, drivers in F1_2025_GRID.items():
    for drv in drivers:
        if drv in ROOKIES_2025 and drv not in driver_skill["Driver"].values:
            td  = team_avg_delta[team_avg_delta["Team"] == team]["TeamAvgDelta"]
            val = td.values[0] if not td.empty else driver_skill["DriverAvgDelta"].mean()
            driver_skill = pd.concat(
                [driver_skill, pd.DataFrame([{"Driver": drv, "DriverAvgDelta": val}])],
                ignore_index=True
            )
print(f"  Rookies imputed: {ROOKIES_2025}")

# Weather defaults per circuit
weather_defaults = df.groupby("Event").agg(
    AirTemp=("AirTemp","mean"), TrackTemp=("TrackTemp","mean"),
    Humidity=("Humidity","mean"), WindSpeed=("WindSpeed","mean"),
    Pressure=("Pressure","mean"), Rainfall=("Rainfall","mean"),
).round(3).reset_index()

# ─────────────────────────────────────────────
# 5. LABEL ENCODERS (include all 2025 entities)
# ─────────────────────────────────────────────
all_teams   = sorted(set(best["Team"].tolist()   + list(F1_2025_GRID.keys())))
all_drivers = sorted(set(best["Driver"].tolist() + [d for drvs in F1_2025_GRID.values() for d in drvs]))
all_events  = sorted(best["Event"].unique().tolist())

le_team   = LabelEncoder().fit(all_teams)
le_driver = LabelEncoder().fit(all_drivers)
le_event  = LabelEncoder().fit(all_events)

best["Team_enc"]   = le_team.transform(best["Team"].astype(str))
best["Driver_enc"] = le_driver.transform(best["Driver"].astype(str))
best["Event_enc"]  = le_event.transform(best["Event"].astype(str))

# ─────────────────────────────────────────────
# 6. FEATURES AND TARGET
# Target is DeltaFromPole (not absolute lap time)
# ─────────────────────────────────────────────
FEATURES = [
    "Team_enc", "Driver_enc", "Event_enc", "Year",
    "QualiSegment_num", "Compound_num", "TyreLife", "FreshTyre_int",
    "IsStreet", "SpeedClass_num", "DRSZones", "Altitude_m",
    "NumCorners", "CornerDensity", "TrackLength_m", "AvgCornerSpacing_m",
    "AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall_int",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "TeamCircuitAvgDelta", "DriverAvgDelta",
]

for col in FEATURES:
    if best[col].isnull().any():
        best[col] = best[col].fillna(best[col].median())

X = best[FEATURES]
y = best["DeltaFromPole"]   # ← predicting gap to pole, not absolute time

# ─────────────────────────────────────────────
# 7. TIME-BASED SPLIT
# ─────────────────────────────────────────────
X_train = X[best["Year"] <= 2023]; y_train = y[best["Year"] <= 2023]
X_test  = X[best["Year"] == 2024]; y_test  = y[best["Year"] == 2024]
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────
print("Training XGBoost model (target: delta from pole)...")
model = xgb.XGBRegressor(
    n_estimators=1000, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
    early_stopping_rounds=40, eval_metric="mae",
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# ─────────────────────────────────────────────
# 9. EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"\n📊 Model Performance on 2024 holdout (delta from pole):")
print(f"  MAE : {mae:.3f}s  |  R² : {r2:.4f}")
print(f"  (For reference: avg pole-to-last gap is ~3.1s)")

# ─────────────────────────────────────────────
# 10. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,                 "model/xgb_model.pkl")
joblib.dump(le_team,               "model/le_team.pkl")
joblib.dump(le_driver,             "model/le_driver.pkl")
joblib.dump(le_event,              "model/le_event.pkl")
joblib.dump(FEATURES,              "model/features.pkl")
joblib.dump(team_circuit_avg_delta,"model/team_circuit_avg_delta.pkl")
joblib.dump(team_circuit_avg_abs,  "model/team_circuit_avg_abs.pkl")
joblib.dump(driver_skill,          "model/driver_skill.pkl")
joblib.dump(team_avg_delta,        "model/team_avg_delta.pkl")
joblib.dump(weather_defaults,      "model/weather_defaults.pkl")
joblib.dump(F1_2025_GRID,          "model/f1_2025_grid.pkl")
joblib.dump(ROOKIES_2025,          "model/rookies_2025.pkl")
joblib.dump({"mae": round(mae,3), "r2": round(r2,4), "target": "delta_from_pole"}, "model/metrics.pkl")

print("\n✅ All artifacts saved to /model/")
print("Run: python -m streamlit run app.py")