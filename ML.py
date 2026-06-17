"""
F1 Qualifying Predictor — Step 2 (Enhanced): Model Training & Validation
=========================================================================
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
print("STEP 2 (ENHANCED): MODEL TRAINING & VALIDATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. LOAD & REBUILD FEATURES (including new ones)
# ─────────────────────────────────────────────────────────────
raw    = pd.read_csv("data/data.csv")
real25 = pd.read_csv("data/real_lap_time_2025.csv")
tracks = pd.read_csv("data/tracks.csv")

TEAM_MAP = {
    "Alfa Romeo Racing": "Alfa Romeo", "AlphaTauri": "RB", "Toro Rosso": "RB",
    "Racing Point": "Aston Martin",   "Renault":    "Alpine",
    "Haas F1 Team": "Haas",           "Kick Sauber":"Alfa Romeo",
}
raw["Team"] = raw["Team"].replace(TEAM_MAP)
raw = raw[raw["IsPushLap"] == 1].copy()

# Speed trap relative to session average
raw["SessionAvgSpeedST"] = raw.groupby(["Year","Event"])["SpeedST"].transform("mean")
raw["RelSpeedST"]        = raw["SpeedST"] - raw["SessionAvgSpeedST"]

# Best lap per driver per event
best = raw.sort_values("LapTime_sec").groupby(["Driver","Team","Year","Event"], as_index=False).first()

print(f"\n✓ Loaded raw laps        : {len(raw):,}")
print(f"✓ Best-lap table         : {len(best):,} rows")

# ─────────────────────────────────────────────────────────────
# 2. CORE FEATURE ENGINEERING (same as Step 1)
# ─────────────────────────────────────────────────────────────
pole = best.groupby(["Year","Event"])["LapTime_sec"].min().reset_index().rename(columns={"LapTime_sec":"PoleLapTime"})
best = best.merge(pole, on=["Year","Event"])
best["GapToPole"] = best["LapTime_sec"] - best["PoleLapTime"]

team_perf = best.groupby(["Team","Year","Event"])["GapToPole"].mean().reset_index().rename(columns={"GapToPole":"TeamAvgGap"})
team_year_strength = team_perf.groupby(["Team","Year"])["TeamAvgGap"].median().reset_index().rename(columns={"TeamAvgGap":"TeamYearStrength"})
best = best.merge(team_year_strength, on=["Team","Year"], how="left")

teammate_avg      = best.groupby(["Team","Year","Event"])["LapTime_sec"].transform("mean")
best["DriverDelta"] = best["LapTime_sec"] - teammate_avg
driver_skill      = best.groupby(["Driver","Year"])["DriverDelta"].median().reset_index().rename(columns={"DriverDelta":"DriverSkillDelta"})
best              = best.merge(driver_skill, on=["Driver","Year"], how="left")

COMPOUND_BASE = {"SOFT":0.0,"MEDIUM":0.4,"HARD":0.8,"INTERMEDIATE":1.5,"WET":2.5,"UNKNOWN":0.4}
best["CompoundPenalty"]    = best["Compound"].map(COMPOUND_BASE).fillna(0.4)
best["TyreLifePenalty"]    = (best["TyreLife"] - 1).clip(lower=0) * 0.05
best["TyreConditionScore"] = best["CompoundPenalty"] + best["TyreLifePenalty"]

best["TrackTempPenalty"]  = ((40 - best["TrackTemp"]).clip(lower=0) * 0.02)
best["RainfallPenalty"]   = best["Rainfall"].astype(int) * 3.5
best["HumidityPenalty"]   = ((best["Humidity"] - 50).clip(lower=0) * 0.005)
best["WeatherGripPenalty"]= best["TrackTempPenalty"] + best["RainfallPenalty"] + best["HumidityPenalty"]

SPEED_MAP = {"Slow":0,"Medium":1,"Fast":2}
best["SpeedClassNum"]   = best["LapSpeedClass"].map(SPEED_MAP).fillna(1)
best["IsStreetCircuit"] = (best["TrackType"] == "Street").astype(int)
best["CircuitArchetype"]= best["TrackType"].apply(lambda x: "Street" if x=="Street" else "Permanent") + "-" + best["LapSpeedClass"]

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

# 3a. RelSpeedST — from raw laps, take driver's best speed trap per event
best = best.drop(columns=["RelSpeedST"], errors="ignore")  # FIX: drop stale copy carried over from raw before merging
speed_trap = raw.groupby(["Driver","Year","Event"])["RelSpeedST"].max().reset_index()
best = best.merge(speed_trap, on=["Driver","Year","Event"], how="left")
best["RelSpeedST"] = best["RelSpeedST"].fillna(0)

# 3b. TeamChange flag — did the driver change team vs previous year?
driver_team_year = best.groupby(["Driver","Year"])["Team"].first().reset_index().sort_values(["Driver","Year"])
driver_team_year["PrevTeam"]  = driver_team_year.groupby("Driver")["Team"].shift(1)
driver_team_year["TeamChange"] = (
    (driver_team_year["Team"] != driver_team_year["PrevTeam"]) &
    driver_team_year["PrevTeam"].notna()
).astype(int)
best = best.merge(driver_team_year[["Driver","Year","TeamChange"]], on=["Driver","Year"], how="left")
best["TeamChange"] = best["TeamChange"].fillna(0).astype(int)

# 3c. Rolling 3-race form (within season, lagged — no leakage)
best_sorted = best.sort_values(["Driver","Year","Event"])
best["Rolling3Form"] = (
    best_sorted.groupby(["Driver","Year"])["GapToPole"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
best["Rolling3Form"] = best["Rolling3Form"].fillna(best["GapToPole"].median())

# 3d. Driver momentum — gap improvement trend (last year vs year before)
driver_yearly = best.groupby(["Driver","Year"])["GapToPole"].median().reset_index()
driver_yearly["PrevYearGap"] = driver_yearly.groupby("Driver")["GapToPole"].shift(1)
driver_yearly["YoYImprovement"] = driver_yearly["PrevYearGap"] - driver_yearly["GapToPole"]
driver_yearly = driver_yearly[["Driver","Year","YoYImprovement"]]
best = best.merge(driver_yearly, on=["Driver","Year"], how="left")
best["YoYImprovement"] = best["YoYImprovement"].fillna(0)

print("✓ New features added: RelSpeedST, TeamChange, Rolling3Form, YoYImprovement")

# ─────────────────────────────────────────────────────────────
# 4. RECENCY WEIGHTS
#    Exponential decay: 2024 matters 4x more than 2019
# ─────────────────────────────────────────────────────────────
YEAR_WEIGHTS = {2019: 1.0, 2020: 1.5, 2021: 2.0, 2022: 2.5, 2023: 3.5, 2024: 5.0}
best["SampleWeight"] = best["Year"].map(YEAR_WEIGHTS).fillna(1.0)

# ─────────────────────────────────────────────────────────────
# 5. FEATURE SET
# ─────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    # Car/team pace
    "TeamYearStrength",
    # Driver skill
    "DriverSkillDelta", "DriverCircuitAffinity",
    # NEW: driver momentum & context
    "Rolling3Form", "YoYImprovement", "TeamChange",
    # Tyre
    "TyreConditionScore", "CompoundPenalty", "TyreLifePenalty",
    # Weather
    "WeatherGripPenalty", "TrackTempPenalty", "RainfallPenalty", "TrackTemp", "Humidity",
    # Circuit
    "IsStreetCircuit", "SpeedClassNum",
    "NormTrackLength", "NormNumCorners", "NormCornerDensity",
    "Altitude_m", "DRSZones",
    # Quali context
    "QualiSegmentNum",
    # Sector pace
    "S1_share", "S2_share", "S3_share",
    # NEW: straight-line speed
    "RelSpeedST",
]

X       = best[MODEL_FEATURES]
y       = best["GapToPole"]
years   = best["Year"]
weights = best["SampleWeight"]

# ─────────────────────────────────────────────────────────────
# 6. LEAVE-ONE-YEAR-OUT CROSS VALIDATION
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

unique_years = sorted(years.unique())
fold_results = []

for test_year in unique_years:
    tr_mask = years != test_year
    te_mask = years == test_year

    X_tr, y_tr, w_tr = X[tr_mask], y[tr_mask], weights[tr_mask]
    X_te, y_te       = X[te_mask], y[te_mask]

    m_xgb = XGBRegressor(**xgb_params)
    m_xgb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_te, y_te)], verbose=False)

    m_lgb = LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr, y_tr, sample_weight=w_tr)

    # Ensemble: 60% XGBoost, 40% LightGBM
    preds = 0.6 * m_xgb.predict(X_te) + 0.4 * m_lgb.predict(X_te)
    preds = preds.clip(min=0)

    mae = mean_absolute_error(y_te, preds)
    fold_results.append({"TestYear": test_year, "MAE": round(mae, 4), "N": len(y_te)})
    print(f"  Fold {test_year} | MAE: {mae:.4f}s | N={len(y_te)}")

cv_mae = np.mean([r["MAE"] for r in fold_results])
print(f"\n  CV Mean MAE  : {cv_mae:.4f}s")
print(f"  Baseline MAE : 0.9417s")
print(f"  v1 Model MAE : 0.7950s")
print(f"  Improvement vs baseline : {((0.9417 - cv_mae)/0.9417*100):.1f}%")
print(f"  Improvement vs v1       : {((0.7950 - cv_mae)/0.7950*100):.1f}%")

# ─────────────────────────────────────────────────────────────
# 7. TRAIN FINAL MODELS ON ALL DATA
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

# Feature importance (XGBoost)
importance_df = pd.DataFrame({
    "Feature"   : MODEL_FEATURES,
    "Importance": final_xgb.feature_importances_,
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\nFEATURE IMPORTANCE (top 12):")
print(importance_df.head(12).to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 8. PREDICT ON 2025 ACTUALS
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("PREDICTING 2025 QUALIFYING GAPS")
print("─" * 50)

recent = best[best["Year"] == 2024]

driver_features_2024 = (
    recent.groupby("Driver")[["TeamYearStrength","DriverSkillDelta",
                               "DriverCircuitAffinity","S1_share",
                               "S2_share","S3_share","RelSpeedST",
                               "Rolling3Form","YoYImprovement"]]
    .mean().reset_index()
)

# Team-level fallback for new drivers
team_medians_2024 = (
    recent.groupby("Team")[["TeamYearStrength","DriverSkillDelta",
                             "DriverCircuitAffinity","S1_share",
                             "S2_share","S3_share","RelSpeedST",
                             "Rolling3Form","YoYImprovement"]]
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

pred_rows = []
for _, row in real25.iterrows():
    driver = row["driver"]
    race   = row["race"]

    track_info = tracks[tracks["Event"] == race]
    if track_info.empty:
        continue
    track = track_info.iloc[0]

    drv_feat = all_driver_features[all_driver_features["Driver"] == driver]
    if drv_feat.empty:
        drv_feat = all_driver_features.mean(numeric_only=True).to_frame().T
        drv_feat = drv_feat.iloc[0]
    else:
        drv_feat = drv_feat.iloc[0]

    feature_row = {
        "TeamYearStrength"      : drv_feat["TeamYearStrength"],
        "DriverSkillDelta"      : drv_feat["DriverSkillDelta"],
        "DriverCircuitAffinity" : drv_feat["DriverCircuitAffinity"],
        "Rolling3Form"          : drv_feat["Rolling3Form"],
        "YoYImprovement"        : drv_feat["YoYImprovement"],
        "TeamChange"            : 1 if driver in ["HAM","SAI","ANT","HAD","BOR","DOO","COL","LAW","BEA"] else 0,
        "TyreConditionScore"    : 0.05,
        "CompoundPenalty"       : 0.0,
        "TyreLifePenalty"       : 0.05,
        "WeatherGripPenalty"    : 0.15,
        "TrackTempPenalty"      : 0.10,
        "RainfallPenalty"       : 0.0,
        "TrackTemp"             : 35.0,
        "Humidity"              : 55.0,
        "IsStreetCircuit"       : 1 if track["TrackType"] == "Street" else 0,
        "SpeedClassNum"         : SPEED_MAP.get(track["LapSpeedClass"], 1),
        "NormTrackLength"       : (track["TrackLength_m"] - tl_mean) / tl_std,
        "NormNumCorners"        : (track["NumCorners"]    - nc_mean) / nc_std,
        "NormCornerDensity"     : (track["CornerDensity"] - cd_mean) / cd_std,
        "Altitude_m"            : track["Altitude_m"],
        "DRSZones"              : track["DRSZones"],
        "QualiSegmentNum"       : 3,
        "S1_share"              : drv_feat["S1_share"],
        "S2_share"              : drv_feat["S2_share"],
        "S3_share"              : drv_feat["S3_share"],
        "RelSpeedST"            : drv_feat["RelSpeedST"],
    }
    feature_row["driver"] = driver
    feature_row["race"]   = race
    pred_rows.append(feature_row)

pred_df = pd.DataFrame(pred_rows)
X_pred  = pred_df[MODEL_FEATURES]

pred_xgb = final_xgb.predict(X_pred)
pred_lgb = final_lgb.predict(X_pred)
pred_df["PredictedGapToPole"] = (0.6 * pred_xgb + 0.4 * pred_lgb).clip(min=0)

# ─────────────────────────────────────────────────────────────
# 9. EVALUATE AGAINST 2025 ACTUALS
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
results["Error"]         = abs(results["PredictedGapToPole"] - results["ActualGapToPole"])
results["PredTime"]      = results["ActualPole"] + results["PredictedGapToPole"]
results["PredPosition"]  = results.groupby("race")["PredictedGapToPole"].rank(method="min").astype(int)
results["ActualPosition"]= results.groupby("race")["ActualGapToPole"].rank(method="min").astype(int)
results["PositionError"] = abs(results["PredPosition"] - results["ActualPosition"])

model_mae   = results["Error"].mean()
pos_exact   = (results["PositionError"] == 0).mean() * 100
pos_within3 = (results["PositionError"] <= 3).mean() * 100

print(f"\n  Model MAE on 2025 actuals : {model_mae:.4f}s")
print(f"  v1 MAE                    : 0.5768s")
print(f"  Baseline MAE              : 0.9417s")
print(f"  Overall improvement       : {((0.9417 - model_mae)/0.9417*100):.1f}%")
print(f"\n  Grid position accuracy:")
print(f"    Exact position  : {pos_exact:.1f}%")
print(f"    Within 3 places : {pos_within3:.1f}%")

# ─────────────────────────────────────────────────────────────
# 10. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
with open("model/model_xgb.pkl", "wb") as f:
    pickle.dump(final_xgb, f)
with open("model/model_lgb.pkl", "wb") as f:
    pickle.dump(final_lgb, f)

# Save normalisation stats needed by the web app
norm_stats = {
    "tl_mean": tl_mean, "tl_std": tl_std,
    "nc_mean": nc_mean, "nc_std": nc_std,
    "cd_mean": cd_mean, "cd_std": cd_std,
}
pd.DataFrame([norm_stats]).to_csv("model/norm_stats.csv", index=False)

results.to_csv("model/model_results.csv", index=False)
importance_df.to_csv("model/feature_importance.csv", index=False)
pd.DataFrame(fold_results).to_csv("model/cv_results.csv", index=False)
best.to_csv("model/training_data.csv", index=False)
all_driver_features.to_csv("model/driver_features_2025.csv", index=False)

print("\n" + "=" * 60)
print("STEP 2 ENHANCED COMPLETE")
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