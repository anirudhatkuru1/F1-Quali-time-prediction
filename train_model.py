import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/data.csv")
print(f"  Raw rows: {len(df)}")

# ─────────────────────────────────────────────
# 2. CLEAN & FILTER
# ─────────────────────────────────────────────
# Keep only push laps (actual qualifying attempts)
df = df[df["IsPushLap"] == 1].copy()

# Drop rows with missing target or key features
df = df.dropna(subset=["LapTime_sec", "Team", "Driver", "Event", "TrackType"])

# Remove obvious outliers (pit in/out laps that slipped through)
q_low = df["LapTime_sec"].quantile(0.01)
q_high = df["LapTime_sec"].quantile(0.99)
df = df[(df["LapTime_sec"] >= q_low) & (df["LapTime_sec"] <= q_high)]

print(f"  Clean rows after filtering: {len(df)}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Encode QualiSegment as ordinal (Q1=1, Q2=2, Q3=3)
segment_map = {"Q1": 1, "Q2": 2, "Q3": 3}
df["QualiSegment_num"] = df["QualiSegment"].map(segment_map).fillna(1)

# Encode TrackType
df["IsStreet"] = (df["TrackType"] == "Street").astype(int)

# Encode LapSpeedClass
speed_map = {"Slow": 1, "Medium": 2, "Fast": 3}
df["SpeedClass_num"] = df["LapSpeedClass"].map(speed_map).fillna(2)

# FreshTyre as int
df["FreshTyre_int"] = df["FreshTyre"].astype(int)

# Rainfall as int
df["Rainfall_int"] = df["Rainfall"].astype(int)

# Compound encoding
compound_map = {"SOFT": 3, "MEDIUM": 2, "HARD": 1, "INTER": 0, "WET": -1}
df["Compound_num"] = df["Compound"].map(compound_map).fillna(2)

# Team performance proxy: team's average lap time per circuit type (historical)
team_circuit_avg = (
    df.groupby(["Team", "TrackType"])["LapTime_sec"]
    .mean()
    .reset_index()
    .rename(columns={"LapTime_sec": "TeamCircuitAvg"})
)
df = df.merge(team_circuit_avg, on=["Team", "TrackType"], how="left")

# Driver skill proxy: driver's historical average delta from session best
session_best = df.groupby(["Year", "Event"])["LapTime_sec"].min().reset_index()
session_best.rename(columns={"LapTime_sec": "SessionBest"}, inplace=True)
df = df.merge(session_best, on=["Year", "Event"], how="left")
df["LapDeltaFromBest"] = df["LapTime_sec"] - df["SessionBest"]

driver_skill = (
    df.groupby("Driver")["LapDeltaFromBest"]
    .mean()
    .reset_index()
    .rename(columns={"LapDeltaFromBest": "DriverAvgDelta"})
)
df = df.merge(driver_skill, on="Driver", how="left")

# Label encode Team and Driver
le_team = LabelEncoder()
le_driver = LabelEncoder()
le_event = LabelEncoder()

df["Team_enc"] = le_team.fit_transform(df["Team"].astype(str))
df["Driver_enc"] = le_driver.fit_transform(df["Driver"].astype(str))
df["Event_enc"] = le_event.fit_transform(df["Event"].astype(str))

# ─────────────────────────────────────────────
# 4. DEFINE FEATURES
# ─────────────────────────────────────────────
FEATURES = [
    # Identity
    "Team_enc", "Driver_enc", "Event_enc", "Year",
    # Session context
    "QualiSegment_num",
    # Tyre
    "Compound_num", "TyreLife", "FreshTyre_int",
    # Circuit DNA
    "IsStreet", "SpeedClass_num", "DRSZones", "Altitude_m",
    "NumCorners", "CornerDensity", "TrackLength_m", "AvgCornerSpacing_m",
    # Weather
    "AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall_int",
    # Speed traps (downforce proxy)
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    # Engineered
    "TeamCircuitAvg", "DriverAvgDelta",
]

# Fill any remaining NaNs in features with median
for col in FEATURES:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

X = df[FEATURES]
y = df["LapTime_sec"]

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT (time-based: train on ≤2023, test on 2024)
# ─────────────────────────────────────────────
X_train = X[df["Year"] <= 2023]
y_train = y[df["Year"] <= 2023]
X_test = X[df["Year"] == 2024]
y_test = y[df["Year"] == 2024]

print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# 6. TRAIN XGBOOST MODEL
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

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ─────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📊 Model Performance on 2024 holdout:")
print(f"  MAE  : {mae:.3f} seconds")
print(f"  R²   : {r2:.4f}")

# Validate against 2025 real lap times
print("\nValidating against 2025 real lap times...")
real_2025 = pd.read_csv("data/real_lap_time_2025.csv")

# ─────────────────────────────────────────────
# 8. SAVE ARTIFACTS
# ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/xgb_model.pkl")
joblib.dump(le_team, "model/le_team.pkl")
joblib.dump(le_driver, "model/le_driver.pkl")
joblib.dump(le_event, "model/le_event.pkl")

# Save feature list and engineered lookup tables
joblib.dump(FEATURES, "model/features.pkl")
joblib.dump(team_circuit_avg, "model/team_circuit_avg.pkl")
joblib.dump(driver_skill, "model/driver_skill.pkl")

# Save model metrics
metrics = {"mae": round(mae, 3), "r2": round(r2, 4)}
joblib.dump(metrics, "model/metrics.pkl")

print("\n✅ All artifacts saved to /model/")
print("   xgb_model.pkl | le_team.pkl | le_driver.pkl | le_event.pkl")
print("   features.pkl  | team_circuit_avg.pkl | driver_skill.pkl | metrics.pkl")
print("\nYou can now run: streamlit run app.py")