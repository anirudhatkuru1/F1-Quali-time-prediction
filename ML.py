"""
F1 Qualifying Predictor — Step 2 (Enhanced v2): Model Training & Validation
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

os.makedirs("model", exist_ok=True)

print("=" * 60)
print("STEP 2 (ENHANCED v2): MODEL TRAINING & VALIDATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. LOAD & REBUILD FEATURES
# ─────────────────────────────────────────────────────────────
raw    = pd.read_csv("Data/data.csv")
real25 = pd.read_csv("Data/real_lap_time_2025.csv")
tracks = pd.read_csv("Data/tracks.csv")

TEAM_MAP = {
    "Alfa Romeo Racing": "Alfa Romeo", "AlphaTauri": "RB", "Toro Rosso": "RB",
    "Racing Point": "Aston Martin",   "Renault":    "Alpine",
    "Haas F1 Team": "Haas",           "Kick Sauber":"Alfa Romeo",
}
raw["Team"] = raw["Team"].replace(TEAM_MAP)
raw = raw[raw["IsPushLap"] == 1].copy()

raw["SessionAvgSpeedST"] = raw.groupby(["Year","Event"])["SpeedST"].transform("mean")
raw["RelSpeedST"]        = raw["SpeedST"] - raw["SessionAvgSpeedST"]

best = raw.sort_values("LapTime_sec").groupby(["Driver","Team","Year","Event"], as_index=False).first()

print(f"\n✓ Loaded raw laps        : {len(raw):,}")
print(f"✓ Best-lap table         : {len(best):,} rows")

# ─────────────────────────────────────────────────────────────
# 2. CORE FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
pole = best.groupby(["Year","Event"])["LapTime_sec"].min().reset_index().rename(columns={"LapTime_sec":"PoleLapTime"})
best = best.merge(pole, on=["Year","Event"])
best["GapToPole"] = best["LapTime_sec"] - best["PoleLapTime"]

SPEED_MAP = {"Slow":0,"Medium":1,"Fast":2}
best["SpeedClassNum"]    = best["LapSpeedClass"].map(SPEED_MAP).fillna(1)
best["IsStreetCircuit"]  = (best["TrackType"] == "Street").astype(int)
best["CircuitArchetype"] = best["TrackType"].apply(lambda x: "Street" if x=="Street" else "Permanent") + "-" + best["LapSpeedClass"]

team_perf = best.groupby(["Team","Year","Event"])["GapToPole"].mean().reset_index().rename(columns={"GapToPole":"TeamAvgGap"})
team_year_strength = team_perf.groupby(["Team","Year"])["TeamAvgGap"].median().reset_index().rename(columns={"TeamAvgGap":"TeamYearStrength"})
best = best.merge(team_year_strength, on=["Team","Year"], how="left")

# FIX 3: Per-circuit-type team strength
team_circuit_strength = (
    best.groupby(["Team","Year","CircuitArchetype"])["GapToPole"]
        .median().reset_index()
        .rename(columns={"GapToPole":"TeamCircuitStrength"})
)
best = best.merge(team_circuit_strength, on=["Team","Year","CircuitArchetype"], how="left")
best["TeamCircuitStrength"] = best["TeamCircuitStrength"].fillna(best["TeamYearStrength"])

teammate_avg       = best.groupby(["Team","Year","Event"])["LapTime_sec"].transform("mean")
best["DriverDelta"] = best["LapTime_sec"] - teammate_avg
driver_skill       = best.groupby(["Driver","Year"])["DriverDelta"].median().reset_index().rename(columns={"DriverDelta":"DriverSkillDelta"})
best               = best.merge(driver_skill, on=["Driver","Year"], how="left")

COMPOUND_BASE = {"SOFT":0.0,"MEDIUM":0.4,"HARD":0.8,"INTERMEDIATE":1.5,"WET":2.5,"UNKNOWN":0.4}
best["CompoundPenalty"]    = best["Compound"].map(COMPOUND_BASE).fillna(0.4)
best["TyreLifePenalty"]    = (best["TyreLife"] - 1).clip(lower=0) * 0.05
best["TyreConditionScore"] = best["CompoundPenalty"] + best["TyreLifePenalty"]

best["TrackTempPenalty"]   = ((40 - best["TrackTemp"]).clip(lower=0) * 0.02)
best["RainfallPenalty"]    = best["Rainfall"].astype(int) * 3.5
best["HumidityPenalty"]    = ((best["Humidity"] - 50).clip(lower=0) * 0.005)
best["WeatherGripPenalty"] = best["TrackTempPenalty"] + best["RainfallPenalty"] + best["HumidityPenalty"]

tl_mean, tl_std = best["TrackLength_m"].mean(), best["TrackLength_m"].std()
nc_mean, nc_std = best["NumCorners"].mean(),    best["NumCorners"].std()
cd_mean, cd_std = best["CornerDensity"].mean(), best["CornerDensity"].std()

best["NormTrackLength"]   = (best["TrackLength_m"] - tl_mean) / tl_std
best["NormNumCorners"]    = (best["NumCorners"]    - nc_mean) / nc_std
best["NormCornerDensity"] = (best["CornerDensity"] - cd_mean) / cd_std
best["QualiSegmentNum"]   = best["QualiSegment"].map({"Q1":1,"Q2":2,"Q3":3})

for sec in ["Sector1Time","Sector2Time","Sector3Time"]:
    best[sec+"_s"] = pd.to_timedelta(best[sec]).dt.total_seconds()
total_s = best["Sector1Time_s"] + best["Sector2Time_s"] + best["Sector3Time_s"]
best["S1_share"] = best["Sector1Time_s"] / total_s
best["S2_share"] = best["Sector2Time_s"] / total_s
best["S3_share"] = best["Sector3Time_s"] / total_s

driver_circuit_perf = best.groupby(["Driver","CircuitArchetype"])["GapToPole"].median().reset_index().rename(columns={"GapToPole":"DriverCircuitAffinity"})
best = best.merge(driver_circuit_perf, on=["Driver","CircuitArchetype"], how="left")

# ─────────────────────────────────────────────────────────────
# 3. NEW FEATURES
# ─────────────────────────────────────────────────────────────
best = best.drop(columns=["RelSpeedST"], errors="ignore")
speed_trap = raw.groupby(["Driver","Year","Event"])["RelSpeedST"].max().reset_index()
best = best.merge(speed_trap, on=["Driver","Year","Event"], how="left")
best["RelSpeedST"] = best["RelSpeedST"].fillna(0)

driver_team_year = best.groupby(["Driver","Year"])["Team"].first().reset_index().sort_values(["Driver","Year"])
driver_team_year["PrevTeam"]   = driver_team_year.groupby("Driver")["Team"].shift(1)
driver_team_year["TeamChange"] = (
    (driver_team_year["Team"] != driver_team_year["PrevTeam"]) &
    driver_team_year["PrevTeam"].notna()
).astype(int)
best = best.merge(driver_team_year[["Driver","Year","TeamChange"]], on=["Driver","Year"], how="left")
best["TeamChange"] = best["TeamChange"].fillna(0).astype(int)

best_sorted = best.sort_values(["Driver","Year","Event"])
best["Rolling3Form"] = (
    best_sorted.groupby(["Driver","Year"])["GapToPole"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
best["Rolling3Form"] = best["Rolling3Form"].fillna(best["GapToPole"].median())

driver_yearly = best.groupby(["Driver","Year"])["GapToPole"].median().reset_index()
driver_yearly["PrevYearGap"]    = driver_yearly.groupby("Driver")["GapToPole"].shift(1)
driver_yearly["YoYImprovement"] = driver_yearly["PrevYearGap"] - driver_yearly["GapToPole"]
driver_yearly = driver_yearly[["Driver","Year","YoYImprovement"]]
best = best.merge(driver_yearly, on=["Driver","Year"], how="left")
best["YoYImprovement"] = best["YoYImprovement"].fillna(0)

print("✓ New features added: RelSpeedST, TeamChange, Rolling3Form, YoYImprovement, TeamCircuitStrength")

# ─────────────────────────────────────────────────────────────
# 4. FEATURE SET DEFINITION (needed before 2025 training build)
# ─────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    "TeamYearStrength",
    "TeamCircuitStrength",
    "DriverSkillDelta", "DriverCircuitAffinity",
    "Rolling3Form", "YoYImprovement", "TeamChange",
    "TyreConditionScore", "CompoundPenalty", "TyreLifePenalty",
    "WeatherGripPenalty", "TrackTempPenalty", "RainfallPenalty", "TrackTemp", "Humidity",
    "IsStreetCircuit", "SpeedClassNum",
    "NormTrackLength", "NormNumCorners", "NormCornerDensity",
    "Altitude_m", "DRSZones",
    "QualiSegmentNum",
    "S1_share", "S2_share", "S3_share",
    "RelSpeedST",
]

# ─────────────────────────────────────────────────────────────
# 5. FIX 2: BUILD 2025 TRAINING ROWS
# ─────────────────────────────────────────────────────────────
raw2025_train = pd.read_csv("Data/data_2025.csv")
raw2025_train["Team"] = raw2025_train["Team"].replace(TEAM_MAP)
raw2025_train = raw2025_train[raw2025_train["IsPushLap"] == 1].copy()

raw2025_train["SessionAvgSpeedST"] = raw2025_train.groupby(["Year","Event"])["SpeedST"].transform("mean")
raw2025_train["RelSpeedST"]        = raw2025_train["SpeedST"] - raw2025_train["SessionAvgSpeedST"]

best2025_train = (
    raw2025_train.sort_values("LapTime_sec")
                 .groupby(["Driver","Team","Year","Event"], as_index=False)
                 .first()
)

# FIX 1: Override lap times with real ground truth
real25_for_train = real25[["driver","race","real_time_seconds"]].rename(
    columns={"driver":"Driver","race":"Event","real_time_seconds":"RealBestLap"}
)
best2025_train = best2025_train.merge(real25_for_train, on=["Driver","Event"], how="left")
best2025_train["LapTime_sec"] = best2025_train["RealBestLap"].combine_first(best2025_train["LapTime_sec"])

pole2025 = best2025_train.groupby("Event")["LapTime_sec"].min().reset_index().rename(columns={"LapTime_sec":"PoleLapTime"})
best2025_train = best2025_train.merge(pole2025, on="Event")
best2025_train["GapToPole"] = best2025_train["LapTime_sec"] - best2025_train["PoleLapTime"]

best2025_train["SpeedClassNum"]    = best2025_train["LapSpeedClass"].map(SPEED_MAP).fillna(1)
best2025_train["IsStreetCircuit"]  = (best2025_train["TrackType"] == "Street").astype(int)
best2025_train["CircuitArchetype"] = best2025_train["TrackType"].apply(lambda x: "Street" if x=="Street" else "Permanent") + "-" + best2025_train["LapSpeedClass"]

team_perf_25 = best2025_train.groupby(["Team","Event"])["GapToPole"].mean().reset_index().rename(columns={"GapToPole":"TeamAvgGap"})
team_strength_25 = team_perf_25.groupby("Team")["TeamAvgGap"].median().reset_index().rename(columns={"TeamAvgGap":"TeamYearStrength"})
best2025_train = best2025_train.merge(team_strength_25, on="Team", how="left")

team_circuit_25 = (
    best2025_train.groupby(["Team","CircuitArchetype"])["GapToPole"]
    .median().reset_index()
    .rename(columns={"GapToPole":"TeamCircuitStrength"})
)
best2025_train = best2025_train.merge(team_circuit_25, on=["Team","CircuitArchetype"], how="left")
best2025_train["TeamCircuitStrength"] = best2025_train["TeamCircuitStrength"].fillna(best2025_train["TeamYearStrength"])

teammate_avg_25 = best2025_train.groupby(["Team","Event"])["LapTime_sec"].transform("mean")
best2025_train["DriverDelta"] = best2025_train["LapTime_sec"] - teammate_avg_25
driver_skill_25 = best2025_train.groupby("Driver")["DriverDelta"].median().reset_index().rename(columns={"DriverDelta":"DriverSkillDelta"})
best2025_train = best2025_train.merge(driver_skill_25, on="Driver", how="left")

best2025_train["CompoundPenalty"]    = best2025_train["Compound"].map(COMPOUND_BASE).fillna(0.4)
best2025_train["TyreLifePenalty"]    = (best2025_train["TyreLife"] - 1).clip(lower=0) * 0.05
best2025_train["TyreConditionScore"] = best2025_train["CompoundPenalty"] + best2025_train["TyreLifePenalty"]
best2025_train["TrackTempPenalty"]   = ((40 - best2025_train["TrackTemp"]).clip(lower=0) * 0.02)
best2025_train["RainfallPenalty"]    = best2025_train["Rainfall"].astype(str).str.strip().str.lower().map({"true":1,"false":0,"1":1,"0":0}).fillna(0) * 3.5
best2025_train["HumidityPenalty"]    = ((best2025_train["Humidity"] - 50).clip(lower=0) * 0.005)
best2025_train["WeatherGripPenalty"] = best2025_train["TrackTempPenalty"] + best2025_train["RainfallPenalty"] + best2025_train["HumidityPenalty"]

best2025_train["NormTrackLength"]   = (best2025_train["TrackLength_m"] - tl_mean) / tl_std
best2025_train["NormNumCorners"]    = (best2025_train["NumCorners"]    - nc_mean) / nc_std
best2025_train["NormCornerDensity"] = (best2025_train["CornerDensity"] - cd_mean) / cd_std
best2025_train["QualiSegmentNum"]   = best2025_train["QualiSegment"].map({"Q1":1,"Q2":2,"Q3":3})

for sec in ["Sector1Time","Sector2Time","Sector3Time"]:
    best2025_train[sec+"_s"] = pd.to_timedelta(best2025_train[sec]).dt.total_seconds()
total_s_25 = best2025_train["Sector1Time_s"] + best2025_train["Sector2Time_s"] + best2025_train["Sector3Time_s"]
best2025_train["S1_share"] = best2025_train["Sector1Time_s"] / total_s_25
best2025_train["S2_share"] = best2025_train["Sector2Time_s"] / total_s_25
best2025_train["S3_share"] = best2025_train["Sector3Time_s"] / total_s_25

driver_circuit_25 = best2025_train.groupby(["Driver","CircuitArchetype"])["GapToPole"].median().reset_index().rename(columns={"GapToPole":"DriverCircuitAffinity"})
best2025_train = best2025_train.merge(driver_circuit_25, on=["Driver","CircuitArchetype"], how="left")


speed_trap_25 = raw2025_train.groupby(["Driver","Event"])["RelSpeedST"].max().reset_index().rename(columns={"RelSpeedST":"RelSpeedST_new"})
best2025_train = best2025_train.merge(speed_trap_25, on=["Driver","Event"], how="left")
best2025_train["RelSpeedST"] = best2025_train["RelSpeedST_new"].fillna(0)
best2025_train = best2025_train.drop(columns=["RelSpeedST_new"], errors="ignore")

driver_team_24 = best[best["Year"]==2024].groupby("Driver")["Team"].first().reset_index().rename(columns={"Team":"Team2024"})
driver_team_25_map = best2025_train.groupby("Driver")["Team"].first().reset_index()
team_change_25 = driver_team_25_map.merge(driver_team_24, on="Driver", how="left")
team_change_25["TeamChange"] = ((team_change_25["Team"] != team_change_25["Team2024"]) & team_change_25["Team2024"].notna()).astype(int)
best2025_train = best2025_train.merge(team_change_25[["Driver","TeamChange"]], on="Driver", how="left")
best2025_train["TeamChange"] = best2025_train["TeamChange"].fillna(0).astype(int)

best2025_train_sorted = best2025_train.sort_values(["Driver","Event"])
best2025_train["Rolling3Form"] = (
    best2025_train_sorted.groupby("Driver")["GapToPole"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
best2025_train["Rolling3Form"] = best2025_train["Rolling3Form"].fillna(best2025_train["GapToPole"].median())
best2025_train["YoYImprovement"] = 0.0
best2025_train["Year"] = 2025

shared_cols = MODEL_FEATURES + ["GapToPole","Year"]
best2025_train_aligned = best2025_train[[c for c in shared_cols if c in best2025_train.columns]].copy()
for c in shared_cols:
    if c not in best2025_train_aligned.columns:
        best2025_train_aligned[c] = 0.0

best_combined = pd.concat([best, best2025_train_aligned], ignore_index=True)
print(f"✓ 2025 training rows added : {len(best2025_train_aligned)}")
print(f"✓ Total training rows      : {len(best_combined)}")

# ─────────────────────────────────────────────────────────────
# 6. RECENCY WEIGHTS
# ─────────────────────────────────────────────────────────────
YEAR_WEIGHTS = {2019: 1.0, 2020: 1.5, 2021: 2.0, 2022: 2.5, 2023: 3.5, 2024: 5.0, 2025: 7.0}
best_combined["SampleWeight"] = best_combined["Year"].map(YEAR_WEIGHTS).fillna(1.0)

X       = best_combined[MODEL_FEATURES]
y       = best_combined["GapToPole"]
years   = best_combined["Year"]
weights = best_combined["SampleWeight"]

# ─────────────────────────────────────────────────────────────
# 7. LEAVE-ONE-YEAR-OUT CROSS VALIDATION (exclude 2025)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("LEAVE-ONE-YEAR-OUT CROSS VALIDATION")
print("─" * 50)

xgb_params = dict(
    n_estimators=600, learning_rate=0.04, max_depth=5,
    subsample=0.8, colsample_bytree=0.75,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.5,
    random_state=42, n_jobs=-1,
)
lgb_params = dict(
    n_estimators=600, learning_rate=0.04, max_depth=5,
    subsample=0.8, colsample_bytree=0.75,
    min_child_samples=10, reg_alpha=0.1, reg_lambda=1.5,
    random_state=42, n_jobs=-1, verbose=-1,
)

# FIX 5: Only CV on historical years, 2025 is used for training only
unique_years = sorted([y for y in years.unique() if y < 2025])
fold_results = []

for test_year in unique_years:
    tr_mask = (years != test_year) & (years < 2025)
    te_mask = years == test_year

    X_tr, y_tr, w_tr = X[tr_mask], y[tr_mask], weights[tr_mask]
    X_te, y_te       = X[te_mask], y[te_mask]

    m_xgb = XGBRegressor(**xgb_params)
    m_xgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_te, y_te)], verbose=False)

    m_lgb = LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr, y_tr, sample_weight=w_tr)

    preds = (0.6 * m_xgb.predict(X_te) + 0.4 * m_lgb.predict(X_te)).clip(min=0)
    mae   = mean_absolute_error(y_te, preds)
    fold_results.append({"TestYear": test_year, "MAE": round(mae, 4), "N": len(y_te)})
    print(f"  Fold {test_year} | MAE: {mae:.4f}s | N={len(y_te)}")

cv_mae = np.mean([r["MAE"] for r in fold_results])
print(f"\n  CV Mean MAE  : {cv_mae:.4f}s")
print(f"  Baseline MAE : 0.9417s")
print(f"  Improvement vs baseline : {((0.9417 - cv_mae)/0.9417*100):.1f}%")

# ─────────────────────────────────────────────────────────────
# 8. TRAIN FINAL MODELS ON ALL DATA (2019-2025)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("TRAINING FINAL MODELS ON FULL DATASET")
print("─" * 50)

final_xgb = XGBRegressor(**xgb_params)
final_xgb.fit(X, y, sample_weight=weights, verbose=False)

final_lgb = LGBMRegressor(**lgb_params)
final_lgb.fit(X, y, sample_weight=weights)

print("✓ XGBoost trained")
print("✓ LightGBM trained")

importance_df = pd.DataFrame({
    "Feature"   : MODEL_FEATURES,
    "Importance": final_xgb.feature_importances_,
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\nFEATURE IMPORTANCE (top 12):")
print(importance_df.head(12).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 9. PREDICT ON 2025 ACTUALS
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("PREDICTING 2025 QUALIFYING GAPS")
print("─" * 50)

recent = best[best["Year"] == 2024]

driver_features_2024 = (
    recent.groupby("Driver")[["TeamYearStrength","TeamCircuitStrength","DriverSkillDelta",
                               "DriverCircuitAffinity","S1_share","S2_share","S3_share",
                               "RelSpeedST","Rolling3Form","YoYImprovement"]]
    .mean().reset_index()
)

team_medians_2024 = (
    recent.groupby("Team")[["TeamYearStrength","TeamCircuitStrength","DriverSkillDelta",
                             "DriverCircuitAffinity","S1_share","S2_share","S3_share",
                             "RelSpeedST","Rolling3Form","YoYImprovement"]]
    .median()
)

NEW_DRIVER_TEAM_MAP = {"ANT":"Mercedes","HAD":"RB","BOR":"Alfa Romeo"}
new_rows = []
for drv, team in NEW_DRIVER_TEAM_MAP.items():
    row = {"Driver": drv}
    if team in team_medians_2024.index:
        for col in team_medians_2024.columns:
            row[col] = team_medians_2024.loc[team, col]
    new_rows.append(row)
new_drivers_df = pd.DataFrame(new_rows)

all_driver_features = pd.concat(
    [driver_features_2024, new_drivers_df], ignore_index=True
).drop_duplicates(subset="Driver", keep="first")

driver_last_known_team = recent.groupby("Driver")["Team"].first()

# ─────────────────────────────────────────────────────────────
# 9a. LOAD REAL 2025 WEATHER / TYRE / SECTOR / SPEED-TRAP DATA
# ─────────────────────────────────────────────────────────────
def to_bool_int(series):
    if series.dtype == bool:
        return series.astype(int)
    return (
        series.astype(str).str.strip().str.lower()
              .map({"true":1,"false":0,"1":1,"0":0})
              .fillna(0).astype(int)
    )

raw2025 = pd.read_csv("Data/data_2025.csv")
raw2025["Team"] = raw2025["Team"].replace(TEAM_MAP)
raw2025 = raw2025[raw2025["IsPushLap"] == 1].copy()

best2025 = pd.DataFrame()
if not raw2025.empty:
    raw2025["SessionAvgSpeedST"] = raw2025.groupby(["Year","Event"])["SpeedST"].transform("mean")
    raw2025["RelSpeedST"]        = raw2025["SpeedST"] - raw2025["SessionAvgSpeedST"]

    best2025 = (
        raw2025.sort_values("LapTime_sec")
               .groupby(["Driver","Team","Year","Event"], as_index=False)
               .first()
    )

    # FIX 1: Override with real lap times
    real25_lookup = real25[["driver","race","real_time_seconds"]].rename(
        columns={"driver":"Driver","race":"Event","real_time_seconds":"RealBestLap"}
    )
    best2025 = best2025.merge(real25_lookup, on=["Driver","Event"], how="left")
    best2025["LapTime_sec"] = best2025["RealBestLap"].combine_first(best2025["LapTime_sec"])

    best2025["CompoundPenalty"]    = best2025["Compound"].map(COMPOUND_BASE).fillna(0.4)
    best2025["TyreLifePenalty"]    = (best2025["TyreLife"] - 1).clip(lower=0) * 0.05
    best2025["TyreConditionScore"] = best2025["CompoundPenalty"] + best2025["TyreLifePenalty"]
    best2025["TrackTempPenalty"]   = ((40 - best2025["TrackTemp"]).clip(lower=0) * 0.02)
    best2025["RainfallPenalty"]    = to_bool_int(best2025["Rainfall"]) * 3.5
    best2025["HumidityPenalty"]    = ((best2025["Humidity"] - 50).clip(lower=0) * 0.005)
    best2025["WeatherGripPenalty"] = best2025["TrackTempPenalty"] + best2025["RainfallPenalty"] + best2025["HumidityPenalty"]
    best2025["QualiSegmentNum"]    = best2025["QualiSegment"].map({"Q1":1,"Q2":2,"Q3":3})

    for sec in ["Sector1Time","Sector2Time","Sector3Time"]:
        best2025[sec+"_s"] = pd.to_timedelta(best2025[sec]).dt.total_seconds()
    total_s2025 = best2025["Sector1Time_s"] + best2025["Sector2Time_s"] + best2025["Sector3Time_s"]
    best2025["S1_share"] = best2025["Sector1Time_s"] / total_s2025
    best2025["S2_share"] = best2025["Sector2Time_s"] / total_s2025
    best2025["S3_share"] = best2025["Sector3Time_s"] / total_s2025

    best2025 = best2025.set_index(["Driver","Event"])

n_drivers_2025 = best2025.index.get_level_values("Driver").nunique() if not best2025.empty else 0
print(f"✓ Real 2025 lap data loaded for {n_drivers_2025} drivers")

# ─────────────────────────────────────────────────────────────
# 9b. CIRCUIT CLIMATE FALLBACK
# ─────────────────────────────────────────────────────────────
event_climate = (
    raw.groupby("Event")[["TrackTemp","Humidity"]]
       .mean()
       .rename(columns={"TrackTemp":"EventAvgTrackTemp","Humidity":"EventAvgHumidity"})
)
session_rain = (
    raw.groupby(["Year","Event"])["Rainfall"]
       .apply(lambda s: s.astype(int).max())
       .reset_index()
       .rename(columns={"Rainfall":"SessionHadRain"})
)
event_rain_prob = session_rain.groupby("Event")["SessionHadRain"].mean().rename("EventRainProb")
event_climate   = event_climate.join(event_rain_prob, how="left")

GLOBAL_AVG_TRACKTEMP = raw["TrackTemp"].mean()
GLOBAL_AVG_HUMIDITY  = raw["Humidity"].mean()
q3_tyre = best[best["QualiSegment"] == "Q3"][["CompoundPenalty","TyreLifePenalty"]].mean()
Q3_COMPOUND_PENALTY  = q3_tyre["CompoundPenalty"]
Q3_TYRELIFE_PENALTY  = q3_tyre["TyreLifePenalty"]

pred_rows = []
n_real, n_fallback = 0, 0

for _, row in real25.iterrows():
    driver = row["driver"]
    race   = row["race"]

    track_info = tracks[tracks["Event"] == race]
    if track_info.empty:
        continue
    track = track_info.iloc[0]

    drv_feat = all_driver_features[all_driver_features["Driver"] == driver]
    if drv_feat.empty:
        drv_feat = all_driver_features.mean(numeric_only=True).to_frame().T.iloc[0]
    else:
        drv_feat = drv_feat.iloc[0]

    lap_row = None
    if not best2025.empty and (driver, race) in best2025.index:
        lap_row = best2025.loc[(driver, race)]
        if isinstance(lap_row, pd.DataFrame):
            lap_row = lap_row.iloc[0]

    if lap_row is not None:
        n_real += 1
        compound_penalty   = lap_row["CompoundPenalty"]
        tyre_life_penalty  = lap_row["TyreLifePenalty"]
        tyre_condition     = lap_row["TyreConditionScore"]
        track_temp_penalty = lap_row["TrackTempPenalty"]
        rainfall_penalty   = lap_row["RainfallPenalty"]
        weather_grip       = lap_row["WeatherGripPenalty"]
        track_temp_val     = lap_row["TrackTemp"]
        humidity_val       = lap_row["Humidity"]
        quali_seg_num      = lap_row["QualiSegmentNum"]
        s1, s2, s3         = lap_row["S1_share"], lap_row["S2_share"], lap_row["S3_share"]
        rel_speed_st       = lap_row["RelSpeedST"]
        actual_team_2025   = lap_row["Team"]
    else:
        n_fallback += 1
        if race in event_climate.index:
            ev_temp = event_climate.loc[race, "EventAvgTrackTemp"]
            ev_hum  = event_climate.loc[race, "EventAvgHumidity"]
            ev_rain = event_climate.loc[race, "EventRainProb"]
        else:
            ev_temp, ev_hum, ev_rain = GLOBAL_AVG_TRACKTEMP, GLOBAL_AVG_HUMIDITY, 0.0

        track_temp_penalty = max(0.0, 40 - ev_temp) * 0.02
        rainfall_penalty   = ev_rain * 3.5
        humidity_penalty   = max(0.0, ev_hum - 50) * 0.005
        weather_grip       = track_temp_penalty + rainfall_penalty + humidity_penalty
        track_temp_val     = ev_temp
        humidity_val       = ev_hum
        compound_penalty   = Q3_COMPOUND_PENALTY
        tyre_life_penalty  = Q3_TYRELIFE_PENALTY
        tyre_condition     = Q3_COMPOUND_PENALTY + Q3_TYRELIFE_PENALTY
        quali_seg_num      = 3
        s1, s2, s3         = drv_feat["S1_share"], drv_feat["S2_share"], drv_feat["S3_share"]
        rel_speed_st       = drv_feat["RelSpeedST"]
        actual_team_2025   = None

    prev_team    = driver_last_known_team.get(driver, None)
    current_team = actual_team_2025 if actual_team_2025 is not None else NEW_DRIVER_TEAM_MAP.get(driver, prev_team)
    team_change  = int(prev_team is not None and current_team is not None and prev_team != current_team)

    feature_row = {
        "TeamYearStrength"      : drv_feat["TeamYearStrength"],
        "TeamCircuitStrength"   : drv_feat["TeamCircuitStrength"],
        "DriverSkillDelta"      : drv_feat["DriverSkillDelta"],
        "DriverCircuitAffinity" : drv_feat["DriverCircuitAffinity"],
        "Rolling3Form"          : drv_feat["Rolling3Form"],
        "YoYImprovement"        : drv_feat["YoYImprovement"],
        "TeamChange"            : team_change,
        "TyreConditionScore"    : tyre_condition,
        "CompoundPenalty"       : compound_penalty,
        "TyreLifePenalty"       : tyre_life_penalty,
        "WeatherGripPenalty"    : weather_grip,
        "TrackTempPenalty"      : track_temp_penalty,
        "RainfallPenalty"       : rainfall_penalty,
        "TrackTemp"             : track_temp_val,
        "Humidity"              : humidity_val,
        "IsStreetCircuit"       : 1 if track["TrackType"] == "Street" else 0,
        "SpeedClassNum"         : SPEED_MAP.get(track["LapSpeedClass"], 1),
        "NormTrackLength"       : (track["TrackLength_m"] - tl_mean) / tl_std,
        "NormNumCorners"        : (track["NumCorners"]    - nc_mean) / nc_std,
        "NormCornerDensity"     : (track["CornerDensity"] - cd_mean) / cd_std,
        "Altitude_m"            : track["Altitude_m"],
        "DRSZones"              : track["DRSZones"],
        "QualiSegmentNum"       : quali_seg_num,
        "S1_share"              : s1,
        "S2_share"              : s2,
        "S3_share"              : s3,
        "RelSpeedST"            : rel_speed_st,
        "driver"                : driver,
        "race"                  : race,
    }
    pred_rows.append(feature_row)

print(f"✓ Used REAL 2025 lap data for {n_real} driver-race rows, fallback for {n_fallback}")

pred_df = pd.DataFrame(pred_rows)
X_pred  = pred_df[MODEL_FEATURES]

pred_xgb = final_xgb.predict(X_pred)
pred_lgb = final_lgb.predict(X_pred)
pred_df["PredictedGapToPole"] = (0.6 * pred_xgb + 0.4 * pred_lgb).clip(min=0)

# ─────────────────────────────────────────────────────────────
# 10. EVALUATE AGAINST 2025 ACTUALS
# ─────────────────────────────────────────────────────────────
pole25 = (
    real25.groupby("race")["real_time_seconds"].min().reset_index()
    .rename(columns={"real_time_seconds":"ActualPole"})
)
real25 = real25.merge(pole25, on="race")
real25["ActualGapToPole"] = real25["real_time_seconds"] - real25["ActualPole"]

results = pred_df[["driver","race","PredictedGapToPole"]].merge(
    real25[["driver","race","ActualGapToPole","real_time_seconds","ActualPole"]],
    on=["driver","race"], how="inner"
)
results["Error"]          = abs(results["PredictedGapToPole"] - results["ActualGapToPole"])
results["PredTime"]       = results["ActualPole"] + results["PredictedGapToPole"]
results["PredPosition"]   = results.groupby("race")["PredictedGapToPole"].rank(method="min").astype(int)
results["ActualPosition"] = results.groupby("race")["ActualGapToPole"].rank(method="min").astype(int)
results["PositionError"]  = abs(results["PredPosition"] - results["ActualPosition"])

model_mae   = results["Error"].mean()
pos_exact   = (results["PositionError"] == 0).mean() * 100
pos_within3 = (results["PositionError"] <= 3).mean() * 100

print(f"\n  Model MAE on 2025 actuals : {model_mae:.4f}s")
print(f"  Baseline MAE              : 0.9417s")
print(f"  Overall improvement       : {((0.9417 - model_mae)/0.9417*100):.1f}%")
print(f"\n  Grid position accuracy:")
print(f"    Exact position  : {pos_exact:.1f}%")
print(f"    Within 3 places : {pos_within3:.1f}%")

# ─────────────────────────────────────────────────────────────
# 11. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
with open("model/model_xgb.pkl", "wb") as f:
    pickle.dump(final_xgb, f)
with open("model/model_lgb.pkl", "wb") as f:
    pickle.dump(final_lgb, f)

norm_stats = {
    "tl_mean": tl_mean, "tl_std": tl_std,
    "nc_mean": nc_mean, "nc_std": nc_std,
    "cd_mean": cd_mean, "cd_std": cd_std,
}
pd.DataFrame([norm_stats]).to_csv("model/norm_stats.csv", index=False)

results.to_csv("model/model_results.csv", index=False)
importance_df.to_csv("model/feature_importance.csv", index=False)
pd.DataFrame(fold_results).to_csv("model/cv_results.csv", index=False)
best_combined.to_csv("model/training_data.csv", index=False)
all_driver_features.to_csv("model/driver_features_2025.csv", index=False)

print("\n" + "=" * 60)
print("STEP 2 ENHANCED v2 COMPLETE")
print("Outputs saved:")
print("  model/model_xgb.pkl")
print("  model/model_lgb.pkl")
print("  model/norm_stats.csv")
print("  model/model_results.csv")
print("  model/feature_importance.csv")
print("  model/cv_results.csv")
print("  model/training_data.csv")
print("  model/driver_features_2025.csv")
print("=" * 60)