import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: F1 QUALIFYING DATA AUDIT & FEATURE ENGINEERING")
print("=" * 60)

df      = pd.read_csv("data/data.csv")
real25  = pd.read_csv("data/real_lap_time_2025.csv")
tracks  = pd.read_csv("data/tracks.csv")

print(f"\n✓ Loaded historical laps : {len(df):,} rows, {df.shape[1]} columns")
print(f"✓ Loaded 2025 actuals    : {len(real25):,} rows")
print(f"✓ Loaded track metadata  : {len(tracks):,} circuits")

# ─────────────────────────────────────────────────────────────
# 2. CLEAN TEAM NAMES  (constructor continuity across seasons)
# ─────────────────────────────────────────────────────────────
TEAM_MAP = {
    "Alfa Romeo Racing" : "Alfa Romeo",
    "AlphaTauri"        : "RB",
    "Toro Rosso"        : "RB",
    "Racing Point"      : "Aston Martin",
    "Renault"           : "Alpine",
    "Haas F1 Team"      : "Haas",
    "Kick Sauber"       : "Alfa Romeo",
}
df["Team"] = df["Team"].replace(TEAM_MAP)
print(f"\n✓ Normalised team names  → {sorted(df['Team'].unique())}")

# ─────────────────────────────────────────────────────────────
# 3. FILTER TO PUSH LAPS ONLY  (real qualifying attempts)
# ─────────────────────────────────────────────────────────────
df = df[df["IsPushLap"] == 1].copy()
print(f"✓ Push laps only         : {len(df):,} rows remaining")

# ─────────────────────────────────────────────────────────────
# 4. BEST LAP PER DRIVER PER SESSION
#    We want each driver's personal best in their highest segment
# ─────────────────────────────────────────────────────────────
seg_order = {"Q3": 3, "Q2": 2, "Q1": 1}
df["SegRank"] = df["QualiSegment"].map(seg_order)

# Best (minimum) lap time per driver × event × year
best = (
    df.sort_values("LapTime_sec")
      .groupby(["Driver", "Team", "Year", "Event"], as_index=False)
      .first()
)
print(f" Best-lap table : {len(best):,} rows (driver × event × year)")

# ─────────────────────────────────────────────────────────────
# 5. GAP TO POLE  (target variable)
# ─────────────────────────────────────────────────────────────
pole = (
    best.groupby(["Year", "Event"])["LapTime_sec"]
        .min()
        .reset_index()
        .rename(columns={"LapTime_sec": "PoleLapTime"})
)
best = best.merge(pole, on=["Year", "Event"])
best["GapToPole"] = best["LapTime_sec"] - best["PoleLapTime"]

print(f"\n✓ Target variable 'GapToPole' stats:")
print(best["GapToPole"].describe().round(3).to_string())

# ─────────────────────────────────────────────────────────────
# 6. TEAM PERFORMANCE SCORE
#    Rolling average gap-to-pole per team, per circuit type, per year
#    (captures car pace, independent of individual driver skill)
# ─────────────────────────────────────────────────────────────
team_perf = (
    best.groupby(["Team", "Year", "Event"])["GapToPole"]
        .mean()
        .reset_index()
        .rename(columns={"GapToPole": "TeamAvgGap"})
)

# Year-level team strength (median across events that year)
team_year_strength = (
    team_perf.groupby(["Team", "Year"])["TeamAvgGap"]
             .median()
             .reset_index()
             .rename(columns={"TeamAvgGap": "TeamYearStrength"})
)
best = best.merge(team_year_strength, on=["Team", "Year"], how="left")

# ─────────────────────────────────────────────────────────────
# 7. DRIVER SKILL DELTA
#    How much faster/slower a driver is vs their teammate
#    (positive = slower than teammate)
# ─────────────────────────────────────────────────────────────
teammate_avg = (
    best.groupby(["Team", "Year", "Event"])["LapTime_sec"]
        .transform("mean")
)
best["DriverDelta"] = best["LapTime_sec"] - teammate_avg

# Aggregate: driver's typical delta across full season
driver_skill = (
    best.groupby(["Driver", "Year"])["DriverDelta"]
        .median()
        .reset_index()
        .rename(columns={"DriverDelta": "DriverSkillDelta"})
)
best = best.merge(driver_skill, on=["Driver", "Year"], how="left")

# ─────────────────────────────────────────────────────────────
# 8. TYRE DEGRADATION FEATURE
#    Soft tyres on low tyre-life = fast; older tyres = slower
#    Encode compound + tyre life into a single "TyreAdvantage" score
# ─────────────────────────────────────────────────────────────
COMPOUND_BASE = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8,
                 "INTERMEDIATE": 1.5, "WET": 2.5, "UNKNOWN": 0.4}
best["CompoundPenalty"] = best["Compound"].map(COMPOUND_BASE).fillna(0.4)

# Tyre life penalty: each extra lap on tyre adds ~0.05s (empirical F1 estimate)
best["TyreLifePenalty"] = (best["TyreLife"] - 1).clip(lower=0) * 0.05

# Combined tyre advantage (lower = better grip)
best["TyreConditionScore"] = best["CompoundPenalty"] + best["TyreLifePenalty"]

# ─────────────────────────────────────────────────────────────
# 9. WEATHER COMPOSITE  →  "GripPenalty"
#    Normalise and combine rainfall flag, track temp, humidity
# ─────────────────────────────────────────────────────────────
# TrackTemp: cold = less grip. Optimal ~40°C, penalty below that
best["TrackTempPenalty"] = ((40 - best["TrackTemp"]).clip(lower=0) * 0.02)

# Rainfall: wet = massive penalty
best["RainfallFlag"] = best["Rainfall"].astype(int)
best["RainfallPenalty"] = best["RainfallFlag"] * 3.5

# Humidity: high humidity slightly reduces engine power (tiny effect)
best["HumidityPenalty"] = ((best["Humidity"] - 50).clip(lower=0) * 0.005)

# Combined weather grip penalty
best["WeatherGripPenalty"] = (
    best["TrackTempPenalty"] + best["RainfallPenalty"] + best["HumidityPenalty"]
)
# ─────────────────────────────────────────────────────────────
# 10. CIRCUIT ARCHETYPE ENCODING
#     Combine TrackType + LapSpeedClass + CornerDensity
# ─────────────────────────────────────────────────────────────
# Map speed class to numeric
SPEED_MAP = {"Slow": 0, "Medium": 1, "Fast": 2}
best["SpeedClassNum"] = best["LapSpeedClass"].map(SPEED_MAP).fillna(1)

# TrackType binary
best["IsStreetCircuit"] = (best["TrackType"] == "Street").astype(int)

# Circuit archetype label (useful for display)
def circuit_archetype(row):
    t = "Street" if row["IsStreetCircuit"] else "Permanent"
    s = row["LapSpeedClass"]
    return f"{t}-{s}"

best["CircuitArchetype"] = best.apply(circuit_archetype, axis=1)

# Normalised track characteristics
best["NormTrackLength"]   = (best["TrackLength_m"] - best["TrackLength_m"].mean()) / best["TrackLength_m"].std()
best["NormNumCorners"]    = (best["NumCorners"] - best["NumCorners"].mean()) / best["NumCorners"].std()
best["NormCornerDensity"] = (best["CornerDensity"] - best["CornerDensity"].mean()) / best["CornerDensity"].std()

# ─────────────────────────────────────────────────────────────
# 11. QUALI SEGMENT ENCODING
#     Q3 = drivers have more track time, rubber, optimal conditions
# ─────────────────────────────────────────────────────────────
best["QualiSegmentNum"] = best["QualiSegment"].map({"Q1": 1, "Q2": 2, "Q3": 3})

# ─────────────────────────────────────────────────────────────
# 12. SECTOR TIME FEATURES  (relative sector performance)
# ─────────────────────────────────────────────────────────────
for sec in ["Sector1Time", "Sector2Time", "Sector3Time"]:
    col_sec = sec + "_sec"
    # Parse timedelta → seconds
    best[col_sec] = pd.to_timedelta(best[sec]).dt.total_seconds()

# Sector share of total lap (driver tendency)
total_sec = best["Sector1Time_sec"] + best["Sector2Time_sec"] + best["Sector3Time_sec"]
best["S1_share"] = best["Sector1Time_sec"] / total_sec
best["S2_share"] = best["Sector2Time_sec"] / total_sec
best["S3_share"] = best["Sector3Time_sec"] / total_sec

# ─────────────────────────────────────────────────────────────
# 13. HISTORICAL DRIVER–CIRCUIT AFFINITY
#     Some drivers are consistently better at certain circuit types
# ─────────────────────────────────────────────────────────────
driver_circuit_perf = (
    best.groupby(["Driver", "CircuitArchetype"])["GapToPole"]
        .median()
        .reset_index()
        .rename(columns={"GapToPole": "DriverCircuitAffinity"})
)
best = best.merge(driver_circuit_perf, on=["Driver", "CircuitArchetype"], how="left")

# ─────────────────────────────────────────────────────────────
# 14. FINAL FEATURE SET
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Identifiers (not model inputs)
    "Driver", "Team", "Year", "Event", "CircuitName",
    # Raw track stats needed for normalisation in Step 2
    "TrackLength_m", "NumCorners", "CornerDensity",

    # Target
    "GapToPole",

    # Car/team pace
    "TeamYearStrength",

    # Driver skill
    "DriverSkillDelta",
    "DriverCircuitAffinity",

    # Tyre
    "TyreConditionScore",
    "CompoundPenalty",
    "TyreLifePenalty",

    # Weather
    "WeatherGripPenalty",
    "TrackTempPenalty",
    "RainfallPenalty",
    "TrackTemp",
    "Humidity",

    # Circuit
    "IsStreetCircuit",
    "SpeedClassNum",
    "NormTrackLength",
    "NormNumCorners",
    "NormCornerDensity",
    "Altitude_m",
    "DRSZones",
    "CircuitArchetype",

    # Quali context
    "QualiSegmentNum",

    # Sector performance
    "S1_share", "S2_share", "S3_share",

    # Raw lap info (useful for display)
    "LapTime_sec",
    "PoleLapTime",
    "Compound",
    "TyreLife",
    "QualiSegment",
]

final = best[FEATURE_COLS].copy()
print(f"\n✓ Final feature table    : {len(final):,} rows × {len(FEATURE_COLS)} columns")

# ─────────────────────────────────────────────────────────────
# 15. HANDLE NEW 2025 DRIVERS  (no historical data)
#     ANT (Antonelli – Mercedes), HAD (Hadjar – RB), BOR (Bortoleto – Alfa)
#     Use team-level averages as proxy for their expected performance
# ─────────────────────────────────────────────────────────────
NEW_DRIVER_TEAM_MAP = {
    "ANT": "Mercedes",    # Andrea Kimi Antonelli
    "HAD": "RB",          # Isack Hadjar
    "BOR": "Alfa Romeo",  # Gabriel Bortoleto
}
print(f"\n✓ New 2025 drivers with no history: {list(NEW_DRIVER_TEAM_MAP.keys())}")
print("  → Will use team 2024 average as skill proxy")

# ─────────────────────────────────────────────────────────────
# 16. BASELINE MODEL  (team+year average gap to pole)
#     Used to benchmark ML performance
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("BASELINE: Simple average prediction")
print("─" * 50)

# For each (Driver, Event) in 2025, predict their historical average GapToPole
# at that same event across all prior years
driver_event_hist = (
    final.groupby(["Driver", "Event"])["GapToPole"]
         .mean()
         .reset_index()
         .rename(columns={"GapToPole": "HistAvgGapToPole"})
)

# Fallback: driver average across all events
driver_hist_avg = (
    final.groupby("Driver")["GapToPole"]
         .mean()
         .reset_index()
         .rename(columns={"GapToPole": "DriverOverallAvg"})
)

# Merge into 2025
pred25 = real25.merge(
    driver_event_hist,
    left_on=["driver", "race"], right_on=["Driver", "Event"], how="left"
)
pred25 = pred25.merge(driver_hist_avg, left_on="driver", right_on="Driver", how="left")
pred25["BaselinePredGap"] = pred25["HistAvgGapToPole"].fillna(pred25["DriverOverallAvg"])

# Compute actual gap to pole from 2025 data
pole25 = real25.groupby("race")["real_time_seconds"].min().reset_index().rename(
    columns={"real_time_seconds": "ActualPole2025"}
)
pred25 = pred25.merge(pole25, on="race")
pred25["ActualGapToPole2025"] = pred25["real_time_seconds"] - pred25["ActualPole2025"]
pred25["BaselineError"] = abs(pred25["BaselinePredGap"] - pred25["ActualGapToPole2025"])

baseline_mae = pred25["BaselineError"].mean()
print(f"Baseline MAE (gap-to-pole): {baseline_mae:.4f} seconds")
print(f"This is what our ML model must beat.\n")

# ─────────────────────────────────────────────────────────────
# 17. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
final.to_csv("data/engineered_features.csv", index=False)
pred25.to_csv("data/baseline_results.csv", index=False)

# ─────────────────────────────────────────────────────────────
# 18. FEATURE SUMMARY REPORT
# ─────────────────────────────────────────────────────────────
MODEL_FEATURES = [
    "TeamYearStrength", "DriverSkillDelta", "DriverCircuitAffinity",
    "TyreConditionScore", "CompoundPenalty", "TyreLifePenalty",
    "WeatherGripPenalty", "TrackTempPenalty", "RainfallPenalty",
    "TrackTemp", "Humidity",
    "IsStreetCircuit", "SpeedClassNum",
    "NormTrackLength", "NormNumCorners", "NormCornerDensity",
    "Altitude_m", "DRSZones",
    "QualiSegmentNum",
    "S1_share", "S2_share", "S3_share",
]

report = []
report.append("F1 QUALIFYING PREDICTOR — STEP 1 FEATURE REPORT")
report.append("=" * 60)
report.append(f"Total laps (raw)          : {len(df):,}")
report.append(f"Push laps only            : {len(df):,}")
report.append(f"Best-lap rows             : {len(best):,}")
report.append(f"Feature columns           : {len(MODEL_FEATURES)}")
report.append(f"Year range                : 2019–2024")
report.append(f"Unique drivers            : {final['Driver'].nunique()}")
report.append(f"Unique teams              : {final['Team'].nunique()}")
report.append(f"Unique circuits           : {final['Event'].nunique()}")
report.append("")
report.append("FEATURE GROUPS:")
report.append("  Car pace    : TeamYearStrength")
report.append("  Driver skill: DriverSkillDelta, DriverCircuitAffinity")
report.append("  Tyre        : TyreConditionScore, CompoundPenalty, TyreLifePenalty")
report.append("  Weather     : WeatherGripPenalty, TrackTempPenalty, RainfallPenalty, TrackTemp, Humidity")
report.append("  Circuit     : IsStreetCircuit, SpeedClassNum, NormTrackLength,")
report.append("                NormNumCorners, NormCornerDensity, Altitude_m, DRSZones")
report.append("  Quali ctx   : QualiSegmentNum")
report.append("  Sector pace : S1_share, S2_share, S3_share")
report.append("")
report.append("TARGET VARIABLE: GapToPole (seconds behind fastest lap in session)")
report.append("")
report.append("BASELINE MODEL MAE:")
report.append(f"  {baseline_mae:.4f} seconds (team+year historical average)")
report.append("")
report.append("NEW 2025 DRIVERS (no history — team proxy used):")
for drv, team in NEW_DRIVER_TEAM_MAP.items():
    report.append(f"  {drv} → {team}")
report.append("")
report.append("FEATURE STATISTICS (MODEL INPUTS):")
stats = final[MODEL_FEATURES].describe().round(4)
report.append(stats.to_string())

report_text = "\n".join(report)
print(report_text)

with open("data/feature_summary.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("Outputs saved:")
print("  → data/engineered_features.csv")
print("  → data/claude/baseline_results.csv")
print("  → data/claude/feature_summary.txt")
print("=" * 60)