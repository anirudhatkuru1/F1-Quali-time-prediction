from flask import Flask, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

model_xgb     = pickle.load(open("model/model_xgb.pkl", "rb"))
model_lgb     = pickle.load(open("model/model_lgb.pkl", "rb"))
results_df    = pd.read_csv("model/model_results.csv")
importance_df = pd.read_csv("model/feature_importance.csv")
real25        = pd.read_csv("data/real_lap_time_2025.csv")
tracks        = pd.read_csv("data/tracks.csv")

TEAM_MAP = {
    "Alfa Romeo Racing": "Alfa Romeo", "AlphaTauri": "RB", "Toro Rosso": "RB",
    "Racing Point": "Aston Martin",   "Renault": "Alpine",
    "Haas F1 Team": "Haas",           "Kick Sauber": "Alfa Romeo",
}

DRIVER_TEAM_2025 = {
    "VER": "Red Bull Racing", "LAW": "Red Bull Racing",
    "NOR": "McLaren",         "PIA": "McLaren",
    "HAM": "Ferrari",         "LEC": "Ferrari",
    "RUS": "Mercedes",        "ANT": "Mercedes",
    "ALO": "Aston Martin",    "STR": "Aston Martin",
    "GAS": "Alpine",          "COL": "Alpine",  "DOO": "Alpine",
    "TSU": "Racing Bulls",    "HAD": "Racing Bulls",
    "HUL": "Alfa Romeo",      "BOR": "Alfa Romeo",
    "OCO": "Haas",            "BEA": "Haas",
    "SAI": "Williams",        "ALB": "Williams",
}

DRIVER_FULL_NAMES = {
    "VER": "Max Verstappen",    "LAW": "Liam Lawson",
    "NOR": "Lando Norris",      "PIA": "Oscar Piastri",
    "HAM": "Lewis Hamilton",    "LEC": "Charles Leclerc",
    "RUS": "George Russell",    "ANT": "Kimi Antonelli",
    "ALO": "Fernando Alonso",   "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",      "COL": "Franco Colapinto",
    "DOO": "Jack Doohan",       "TSU": "Yuki Tsunoda",
    "HAD": "Isack Hadjar",      "HUL": "Nico Hulkenberg",
    "BOR": "Gabriel Bortoleto", "OCO": "Esteban Ocon",
    "BEA": "Oliver Bearman",    "SAI": "Carlos Sainz",
    "ALB": "Alexander Albon",
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "McLaren":         "#FF8000",
    "Ferrari":         "#E8002D",
    "Mercedes":        "#27F4D2",
    "Aston Martin":    "#229971",
    "Alpine":          "#FF87BC",
    "Racing Bulls":    "#6692FF",
    "Alfa Romeo":      "#C92D4B",
    "Haas":            "#B6BABD",
    "Williams":        "#64C4FF",
}

# Inaccuracy reasons per race (manually curated from known 2025 events)
RACE_NOTES = {
    "Australian Grand Prix":      "Early season — car development data limited",
    "Bahrain Grand Prix":         "Night race, track temp drops rapidly",
    "Chinese Grand Prix":         "Sprint weekend, limited practice data",
    "Japanese Grand Prix":        "Cool conditions affected tyre warm-up",
    "Saudi Arabian Grand Prix":   "Street circuit, high wall risk",
    "Miami Grand Prix":           "Sprint weekend format",
    "Emilia Romagna Grand Prix":  "Stable conditions, model performs well",
    "Monaco Grand Prix":          "Extreme street circuit, outlier behaviour",
    "Canadian Grand Prix":        "Semi-street, mixed conditions",
    "Spanish Grand Prix":         "Stable — model performs well here",
    "Austrian Grand Prix":        "Short lap, small margins amplified",
    "British Grand Prix":         "Variable UK weather",
    "Hungarian Grand Prix":       "High degradation circuit",
    "Belgian Grand Prix":         "High altitude, weather variable",
    "Dutch Grand Prix":           "Banked corners, unique aero loads",
    "Italian Grand Prix":         "Low downforce setup, high variance",
    "Azerbaijan Grand Prix":      "Street circuit, safety car likely",
    "Singapore Grand Prix":       "Longest street circuit, night race",
    "United States Grand Prix":   "COTA bumps affect tyre behaviour",
    "Mexico City Grand Prix":     "High altitude — significant power unit effect",
    "São Paulo Grand Prix":       "Weather notoriously unpredictable",
    "Las Vegas Grand Prix":       "Night street circuit, cold track temp",
    "Qatar Grand Prix":           "Extreme heat, tyre degradation spike",
    "Abu Dhabi Grand Prix":       "Season finale, development freeze",
}

pole25 = real25.groupby("race")["real_time_seconds"].min().reset_index().rename(columns={"real_time_seconds": "ActualPole"})
real25 = real25.merge(pole25, on="race")
real25["ActualGapToPole"] = real25["real_time_seconds"] - real25["ActualPole"]
real25["ActualPosition"]  = real25.groupby("race")["ActualGapToPole"].rank(method="min").astype(int)
real25["Team"]            = real25["driver"].map(DRIVER_TEAM_2025)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/teams")
def get_teams():
    teams = []
    for team, color in TEAM_COLORS.items():
        drivers = [d for d, t in DRIVER_TEAM_2025.items() if t == team]
        teams.append({
            "name":    team,
            "color":   color,
            "drivers": [{"code": d, "name": DRIVER_FULL_NAMES.get(d, d)} for d in drivers],
        })
    return jsonify(teams)


@app.route("/api/team/<team_name>")
def get_team_prediction(team_name):
    team_name = team_name.replace("-", " ")
    drivers   = [d for d, t in DRIVER_TEAM_2025.items() if t == team_name]
    color     = TEAM_COLORS.get(team_name, "#ffffff")

    driver_data = []
    for drv in drivers:
        drv_results = results_df[results_df["driver"] == drv].copy()
        if drv_results.empty:
            continue

        avg_pred       = drv_results["PredictedGapToPole"].mean()
        avg_actual     = drv_results["ActualGapToPole"].mean()
        avg_error      = drv_results["Error"].mean()
        avg_pos_pred   = drv_results["PredPosition"].mean()
        avg_pos_actual = drv_results["ActualPosition"].mean()

        # Street vs permanent breakdown
        drv_with_track = drv_results.merge(tracks[["Event","TrackType"]], left_on="race", right_on="Event", how="left")
        street_gap = drv_with_track[drv_with_track["TrackType"]=="Street"]["ActualGapToPole"].mean()
        perm_gap   = drv_with_track[drv_with_track["TrackType"]!="Street"]["ActualGapToPole"].mean()

        races = []
        for _, row in drv_results.sort_values("race").iterrows():
            races.append({
                "race":       row["race"],
                "predicted":  round(row["PredictedGapToPole"], 3),
                "actual":     round(row["ActualGapToPole"], 3),
                "error":      round(row["Error"], 3),
                "pred_pos":   int(row["PredPosition"]),
                "actual_pos": int(row["ActualPosition"]),
                "note":       RACE_NOTES.get(row["race"], ""),
            })

        driver_data.append({
            "code":           drv,
            "name":           DRIVER_FULL_NAMES.get(drv, drv),
            "avg_predicted":  round(avg_pred, 3),
            "avg_actual":     round(avg_actual, 3),
            "avg_error":      round(avg_error, 3),
            "avg_pred_pos":   round(avg_pos_pred, 1),
            "avg_actual_pos": round(avg_pos_actual, 1),
            "street_gap":     round(float(street_gap), 3) if not np.isnan(street_gap) else None,
            "perm_gap":       round(float(perm_gap), 3) if not np.isnan(perm_gap) else None,
            "races":          races,
        })

    # Teammate comparison
    teammate_comparison = None
    if len(driver_data) == 2:
        d1, d2 = driver_data[0], driver_data[1]
        faster = d1["code"] if d1["avg_actual"] < d2["avg_actual"] else d2["code"]
        delta  = abs(d1["avg_actual"] - d2["avg_actual"])
        teammate_comparison = {
            "faster": faster,
            "delta":  round(delta, 3),
            "d1_street": d1["street_gap"],
            "d2_street": d2["street_gap"],
            "d1_perm":   d1["perm_gap"],
            "d2_perm":   d2["perm_gap"],
            "d1_code":   d1["code"],
            "d2_code":   d2["code"],
            "d1_name":   d1["name"],
            "d2_name":   d2["name"],
        }

    return jsonify({"team": team_name, "color": color, "drivers": driver_data, "teammate": teammate_comparison})


@app.route("/api/grid/<team_name>")
def get_grid_comparison(team_name):
    team_name = team_name.replace("-", " ")

    avg_gaps = (
        results_df.groupby("driver")[["PredictedGapToPole","ActualGapToPole","Error","PredPosition","ActualPosition"]]
        .mean().reset_index()
    )
    avg_gaps["team"]  = avg_gaps["driver"].map(DRIVER_TEAM_2025)
    avg_gaps["name"]  = avg_gaps["driver"].map(DRIVER_FULL_NAMES)
    avg_gaps["color"] = avg_gaps["team"].map(TEAM_COLORS)
    avg_gaps = avg_gaps.sort_values("ActualGapToPole").reset_index(drop=True)

    grid = []
    for _, row in avg_gaps.iterrows():
        grid.append({
            "driver":      row["driver"],
            "name":        row["name"],
            "team":        row["team"],
            "color":       row["color"],
            "predicted":   round(row["PredictedGapToPole"], 3),
            "actual":      round(row["ActualGapToPole"], 3),
            "error":       round(row["Error"], 3),
            "pred_pos":    round(row["PredPosition"], 1),
            "actual_pos":  round(row["ActualPosition"], 1),
            "is_selected": row["team"] == team_name,
        })

    team_avg = avg_gaps.groupby("team").agg(
        actual=("ActualGapToPole","mean"),
        predicted=("PredictedGapToPole","mean"),
        error=("Error","mean")
    ).reset_index()
    team_avg["color"]       = team_avg["team"].map(TEAM_COLORS)
    team_avg["is_selected"] = team_avg["team"] == team_name
    team_avg = team_avg.sort_values("actual")

    teams_chart = team_avg.to_dict(orient="records")

    # Highest error races across all drivers (inaccuracy analysis)
    all_errors = results_df.copy()
    all_errors["note"] = all_errors["race"].map(RACE_NOTES)
    worst_races = (
        all_errors.groupby("race")["Error"].mean()
        .reset_index().sort_values("Error", ascending=False).head(6)
    )
    worst_races["note"] = worst_races["race"].map(RACE_NOTES)

    inaccuracy_reasons = [
        {"icon": "🌧️", "title": "Live Weather Unknown", "detail": "Wet races and temperature swings aren't predicted in advance. Rainfall adds up to 3.5s penalty but the model assumes dry conditions for future races."},
        {"icon": "🆕", "title": "New Driver History", "detail": "ANT, HAD, BOR have zero historical lap data. Their predictions use team averages as a proxy, introducing ±0.4s additional uncertainty."},
        {"icon": "🔧", "title": "Mid-Season Upgrades", "detail": "Teams bring new parts every 2–3 races. A major aero update can shift performance by 0.2–0.5s overnight — invisible to the model."},
        {"icon": "🏙️", "title": "Street Circuit Variance", "detail": "Monaco, Singapore and Baku have high wall-proximity risk and safety car probability, creating lap time distributions the model underestimates."},
    ]

    # Team-specific R&D suggestions
    team_drivers = [d for d, t in DRIVER_TEAM_2025.items() if t == team_name]
    team_results = results_df[results_df["driver"].isin(team_drivers)]
    avg_team_gap = team_results["ActualGapToPole"].mean()

    team_with_track = team_results.merge(tracks[["Event","TrackType"]], left_on="race", right_on="Event", how="left")
    street_avg = team_with_track[team_with_track["TrackType"]=="Street"]["ActualGapToPole"].mean()
    perm_avg   = team_with_track[team_with_track["TrackType"]!="Street"]["ActualGapToPole"].mean()

    suggestions = []
    if avg_team_gap > 1.5:
        suggestions.append({"priority":"Critical","area":"Overall Car Pace","detail":"Fundamental deficit to leaders. Priority investment in aerodynamic concept and power unit upgrade programme."})
    elif avg_team_gap > 0.8:
        suggestions.append({"priority":"High","area":"Car Pace","detail":"Midfield deficit. Target DRS efficiency gains and front-axle aero refinement for corner entry stability."})
    else:
        suggestions.append({"priority":"Medium","area":"Fine Tuning","detail":"Competitive pace. Marginal gains from aero balance optimisation on specific circuit archetypes."})

    if not np.isnan(street_avg) and not np.isnan(perm_avg):
        delta = street_avg - perm_avg
        if delta > 0.3:
            suggestions.append({"priority":"High","area":"Street Circuit Package","detail":f"Performance drops {delta:.2f}s on street circuits vs permanent tracks. Invest in mechanical grip and low-speed downforce."})
        elif delta < -0.3:
            suggestions.append({"priority":"Medium","area":"Permanent Track Setup","detail":f"Stronger on streets by {abs(delta):.2f}s. High-speed aero efficiency needs attention on permanent circuits."})
        else:
            suggestions.append({"priority":"Low","area":"Circuit Balance","detail":"Well-balanced across circuit types. Minor optimisation available in slow-speed corner packages."})

    suggestions.append({"priority":"Medium","area":"Tyre Strategy","detail":"Fresh soft compound in Q3 is the single biggest performance lever. Optimise warm-up laps and tyre prep protocol."})
    suggestions.append({"priority":"Low","area":"Driver Delta","detail":"Analyse sector-by-sector teammate gaps to identify setup directions. Consistent intra-team delta suggests car balance issue, not driver issue."})

    return jsonify({
        "grid":               grid,
        "teams_chart":        teams_chart,
        "inaccuracy_reasons": inaccuracy_reasons,
        "worst_races":        worst_races.to_dict(orient="records"),
        "suggestions":        suggestions,
        "selected_team":      team_name,
        "selected_color":     TEAM_COLORS.get(team_name, "#fff"),
    })

@app.route("/api/season/<team_name>")
def get_season_data(team_name):
    team_name = team_name.replace("-", " ")
    drivers   = [d for d, t in DRIVER_TEAM_2025.items() if t == team_name]
    color     = TEAM_COLORS.get(team_name, "#ffffff")

    # Race order by calendar
    race_order = [
        "Australian Grand Prix","Chinese Grand Prix","Japanese Grand Prix",
        "Bahrain Grand Prix","Saudi Arabian Grand Prix","Miami Grand Prix",
        "Emilia Romagna Grand Prix","Monaco Grand Prix","Canadian Grand Prix",
        "Spanish Grand Prix","Austrian Grand Prix","British Grand Prix",
        "Belgian Grand Prix","Hungarian Grand Prix","Dutch Grand Prix",
        "Italian Grand Prix","Azerbaijan Grand Prix","Singapore Grand Prix",
        "United States Grand Prix","Mexico City Grand Prix","São Paulo Grand Prix",
        "Las Vegas Grand Prix","Qatar Grand Prix","Abu Dhabi Grand Prix",
    ]

    season_lines = []
    for drv in drivers:
        drv_results = results_df[results_df["driver"] == drv].copy()
        if drv_results.empty:
            continue
        drv_results["race_order"] = drv_results["race"].apply(
            lambda r: race_order.index(r) if r in race_order else 99
        )
        drv_results = drv_results.sort_values("race_order")
        season_lines.append({
            "driver":    drv,
            "name":      DRIVER_FULL_NAMES.get(drv, drv),
            "predicted": [{"race": r["race"], "val": round(r["PredictedGapToPole"],3)} for _,r in drv_results.iterrows()],
            "actual":    [{"race": r["race"], "val": round(r["ActualGapToPole"],3)}    for _,r in drv_results.iterrows()],
            "labels":    [r["race"].replace(" Grand Prix","") for _,r in drv_results.iterrows()],
        })

    # Radar: driver performance by circuit archetype
    radar_data = []
    archetypes = ["Street-Slow","Street-Medium","Street-Fast","Permanent-Slow","Permanent-Medium","Permanent-Fast"]
    for drv in drivers:
        drv_results = results_df[results_df["driver"] == drv].copy()
        drv_with_track = drv_results.merge(
            tracks[["Event","TrackType","LapSpeedClass"]], left_on="race", right_on="Event", how="left"
        )
        drv_with_track["archetype"] = drv_with_track["TrackType"].apply(
            lambda x: "Street" if x=="Street" else "Permanent"
        ) + "-" + drv_with_track["LapSpeedClass"].fillna("Medium")

        arch_gaps = {}
        for arch in archetypes:
            subset = drv_with_track[drv_with_track["archetype"]==arch]["ActualGapToPole"]
            arch_gaps[arch] = round(float(subset.mean()), 3) if not subset.empty else None

        # Normalise: lower gap = better = higher score (invert, scale 0-100)
        all_vals = [v for v in arch_gaps.values() if v is not None]
        max_gap  = max(all_vals) if all_vals else 1
        scores   = {}
        for arch in archetypes:
            if arch_gaps[arch] is not None:
                scores[arch] = round(max(0, 100 - (arch_gaps[arch]/max_gap)*100), 1)
            else:
                scores[arch] = None

        radar_data.append({
            "driver": drv,
            "name":   DRIVER_FULL_NAMES.get(drv, drv),
            "scores": scores,
            "gaps":   arch_gaps,
        })

    return jsonify({
        "team":        team_name,
        "color":       color,
        "season_lines": season_lines,
        "radar":       radar_data,
        "archetypes":  archetypes,
    })


@app.route("/api/scatter/<team_name>")
def get_scatter_data(team_name):
    team_name = team_name.replace("-", " ")

    scatter_points = []
    for _, row in results_df.iterrows():
        scatter_points.append({
            "driver":      row["driver"],
            "team":        DRIVER_TEAM_2025.get(row["driver"], "Unknown"),
            "color":       TEAM_COLORS.get(DRIVER_TEAM_2025.get(row["driver"],""), "#888"),
            "race":        row["race"],
            "pred_pos":    int(row["PredPosition"]),
            "actual_pos":  int(row["ActualPosition"]),
            "is_selected": DRIVER_TEAM_2025.get(row["driver"]) == team_name,
        })

    return jsonify({"points": scatter_points})


@app.route("/api/confidence/<team_name>")
def get_confidence(team_name):
    team_name = team_name.replace("-", " ")

    # Team confidence
    team_errors = results_df.copy()
    team_errors["team"] = team_errors["driver"].map(DRIVER_TEAM_2025)
    team_avg_error = team_errors.groupby("team")["Error"].mean().reset_index()

    max_err = team_avg_error["Error"].max()
    min_err = team_avg_error["Error"].min()
    team_avg_error["confidence"] = (
        100 - ((team_avg_error["Error"] - min_err) / (max_err - min_err + 0.001)) * 100
    ).round(1)
    team_avg_error["color"]       = team_avg_error["team"].map(TEAM_COLORS)
    team_avg_error["is_selected"] = team_avg_error["team"] == team_name
    team_avg_error = team_avg_error.sort_values("confidence", ascending=False)

    # Circuit confidence
    race_order = [
        "Australian Grand Prix","Chinese Grand Prix","Japanese Grand Prix",
        "Bahrain Grand Prix","Saudi Arabian Grand Prix","Miami Grand Prix",
        "Emilia Romagna Grand Prix","Monaco Grand Prix","Canadian Grand Prix",
        "Spanish Grand Prix","Austrian Grand Prix","British Grand Prix",
        "Belgian Grand Prix","Hungarian Grand Prix","Dutch Grand Prix",
        "Italian Grand Prix","Azerbaijan Grand Prix","Singapore Grand Prix",
        "United States Grand Prix","Mexico City Grand Prix","São Paulo Grand Prix",
        "Las Vegas Grand Prix","Qatar Grand Prix","Abu Dhabi Grand Prix",
    ]
    circuit_errors = results_df.groupby("race")["Error"].mean().reset_index()
    circuit_errors["note"] = circuit_errors["race"].map(RACE_NOTES)

    max_ce = circuit_errors["Error"].max()
    min_ce = circuit_errors["Error"].min()
    circuit_errors["confidence"] = (
        100 - ((circuit_errors["Error"] - min_ce) / (max_ce - min_ce + 0.001)) * 100
    ).round(1)
    circuit_errors["race_order"] = circuit_errors["race"].apply(
        lambda r: race_order.index(r) if r in race_order else 99
    )
    circuit_errors = circuit_errors.sort_values("confidence", ascending=False)

    # Circuit type from tracks
    circuit_with_type = circuit_errors.merge(
        tracks[["Event","TrackType"]], left_on="race", right_on="Event", how="left"
    )
    circuit_with_type["color"] = circuit_with_type["TrackType"].apply(
        lambda x: "#6692FF" if x=="Street" else "#22c55e"
    )

    return jsonify({
        "teams":    team_avg_error[["team","confidence","color","is_selected","Error"]].to_dict(orient="records"),
        "circuits": circuit_with_type[["race","confidence","color","note","TrackType","Error"]].to_dict(orient="records"),
    })

@app.route("/api/race/<team_name>/<path:race_name>")
def get_race_prediction(team_name, race_name):
    team_name = team_name.replace("-", " ")
    race_name = race_name.replace("-", " ")
    drivers   = [d for d, t in DRIVER_TEAM_2025.items() if t == team_name]
    color     = TEAM_COLORS.get(team_name, "#ffffff")

    track_info = tracks[tracks["Event"] == race_name]
    track      = track_info.iloc[0] if not track_info.empty else None

    driver_data = []
    for drv in drivers:
        row = results_df[(results_df["driver"] == drv) & (results_df["race"] == race_name)]
        if row.empty:
            continue
        row = row.iloc[0]
        driver_data.append({
            "code":        drv,
            "name":        DRIVER_FULL_NAMES.get(drv, drv),
            "predicted":   round(row["PredictedGapToPole"], 3),
            "actual":      round(row["ActualGapToPole"], 3),
            "error":       round(row["Error"], 3),
            "pred_pos":    int(row["PredPosition"]),
            "actual_pos":  int(row["ActualPosition"]),
            "pred_time":   round(row["PredTime"], 3),
            "actual_time": round(row["real_time_seconds"], 3) if "real_time_seconds" in row else None,
            "pole_time":   round(row["ActualPole"], 3),
        })

    # All drivers for this race (for grid context)
    race_results = results_df[results_df["race"] == race_name].copy()
    race_results["team"]  = race_results["driver"].map(DRIVER_TEAM_2025)
    race_results["color"] = race_results["team"].map(TEAM_COLORS)
    race_results["name"]  = race_results["driver"].map(DRIVER_FULL_NAMES)
    race_results = race_results.sort_values("ActualGapToPole")

    full_grid = []
    for _, r in race_results.iterrows():
        full_grid.append({
            "driver":      r["driver"],
            "name":        r["name"],
            "team":        r["team"],
            "color":       r["color"],
            "predicted":   round(r["PredictedGapToPole"], 3),
            "actual":      round(r["ActualGapToPole"], 3),
            "error":       round(r["Error"], 3),
            "pred_pos":    int(r["PredPosition"]),
            "actual_pos":  int(r["ActualPosition"]),
            "is_selected": r["team"] == team_name,
        })

    # Teammate comparison for this race
    teammate = None
    if len(driver_data) == 2:
        d1, d2 = driver_data[0], driver_data[1]
        faster = d1["code"] if d1["actual"] < d2["actual"] else d2["code"]
        teammate = {
            "d1_code": d1["code"], "d1_name": d1["name"],
            "d2_code": d2["code"], "d2_name": d2["name"],
            "d1_gap":  d1["actual"], "d2_gap": d2["actual"],
            "delta":   round(abs(d1["actual"] - d2["actual"]), 3),
            "faster":  faster,
        }

    circuit_info = None
    if track is not None:
        circuit_info = {
            "name":        str(track["CircuitName"]),
            "country":     str(track["Country"]),
            "type":        str(track["TrackType"]),
            "length":      float(track["TrackLength_m"]),
            "corners":     int(track["NumCorners"]),
            "drs":         int(track["DRSZones"]),
            "speed_class": str(track["LapSpeedClass"]),
            "altitude":    float(track["Altitude_m"]),
        }

    return jsonify({
        "team":         team_name,
        "color":        color,
        "race":         race_name,
        "drivers":      driver_data,
        "full_grid":    full_grid,
        "teammate":     teammate,
        "circuit":      circuit_info,
        "note":         RACE_NOTES.get(race_name, ""),
    })

if __name__ == "__main__":
    app.run(debug=True)