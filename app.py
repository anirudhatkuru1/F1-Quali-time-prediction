from flask import Flask, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────
model_xgb = pickle.load(open("model/model_xgb.pkl", "rb"))
model_lgb  = pickle.load(open("model/model_lgb.pkl", "rb"))
results_df = pd.read_csv("model/model_results.csv")
importance_df = pd.read_csv("model/feature_importance.csv")
real25     = pd.read_csv("data/real_lap_time_2025.csv")
tracks     = pd.read_csv("data/tracks.csv")

TEAM_MAP = {
    "Alfa Romeo Racing": "Alfa Romeo", "AlphaTauri": "RB", "Toro Rosso": "RB",
    "Racing Point": "Aston Martin",   "Renault": "Alpine",
    "Haas F1 Team": "Haas",           "Kick Sauber": "Alfa Romeo",
}

# ── 2025 driver → team mapping ────────────────────────────────
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
    "VER": "Max Verstappen",       "LAW": "Liam Lawson",
    "NOR": "Lando Norris",         "PIA": "Oscar Piastri",
    "HAM": "Lewis Hamilton",       "LEC": "Charles Leclerc",
    "RUS": "George Russell",       "ANT": "Kimi Antonelli",
    "ALO": "Fernando Alonso",      "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",         "COL": "Franco Colapinto", "DOO": "Jack Doohan",
    "TSU": "Yuki Tsunoda",         "HAD": "Isack Hadjar",
    "HUL": "Nico Hulkenberg",      "BOR": "Gabriel Bortoleto",
    "OCO": "Esteban Ocon",         "BEA": "Oliver Bearman",
    "SAI": "Carlos Sainz",         "ALB": "Alexander Albon",
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

TEAMS = list(TEAM_COLORS.keys())

# Compute actual gaps for 2025
pole25 = real25.groupby("race")["real_time_seconds"].min().reset_index().rename(columns={"real_time_seconds": "ActualPole"})
real25 = real25.merge(pole25, on="race")
real25["ActualGapToPole"] = real25["real_time_seconds"] - real25["ActualPole"]
real25["ActualPosition"]  = real25.groupby("race")["ActualGapToPole"].rank(method="min").astype(int)
real25["Team"] = real25["driver"].map(DRIVER_TEAM_2025)


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

    # Season-average stats per driver
    driver_data = []
    for drv in drivers:
        drv_results = results_df[results_df["driver"] == drv].copy()
        if drv_results.empty:
            continue

        avg_pred   = drv_results["PredictedGapToPole"].mean()
        avg_actual = drv_results["ActualGapToPole"].mean()
        avg_error  = drv_results["Error"].mean()
        avg_pos_pred   = drv_results["PredPosition"].mean()
        avg_pos_actual = drv_results["ActualPosition"].mean()

        # Race-by-race breakdown
        races = []
        for _, row in drv_results.sort_values("race").iterrows():
            races.append({
                "race":        row["race"],
                "predicted":   round(row["PredictedGapToPole"], 3),
                "actual":      round(row["ActualGapToPole"], 3),
                "error":       round(row["Error"], 3),
                "pred_pos":    int(row["PredPosition"]),
                "actual_pos":  int(row["ActualPosition"]),
            })

        driver_data.append({
            "code":           drv,
            "name":           DRIVER_FULL_NAMES.get(drv, drv),
            "avg_predicted":  round(avg_pred, 3),
            "avg_actual":     round(avg_actual, 3),
            "avg_error":      round(avg_error, 3),
            "avg_pred_pos":   round(avg_pos_pred, 1),
            "avg_actual_pos": round(avg_pos_actual, 1),
            "races":          races,
        })

    return jsonify({"team": team_name, "color": color, "drivers": driver_data})


@app.route("/api/grid/<team_name>")
def get_grid_comparison(team_name):
    team_name = team_name.replace("-", " ")

    # Season average predicted gap per driver
    avg_gaps = (
        results_df.groupby("driver")[["PredictedGapToPole", "ActualGapToPole", "PredPosition", "ActualPosition"]]
        .mean()
        .reset_index()
    )
    avg_gaps["team"]  = avg_gaps["driver"].map(DRIVER_TEAM_2025)
    avg_gaps["name"]  = avg_gaps["driver"].map(DRIVER_FULL_NAMES)
    avg_gaps["color"] = avg_gaps["team"].map(TEAM_COLORS)
    avg_gaps = avg_gaps.sort_values("ActualGapToPole").reset_index(drop=True)
    avg_gaps["grid_rank"] = range(1, len(avg_gaps) + 1)

    grid = []
    for _, row in avg_gaps.iterrows():
        grid.append({
            "driver":     row["driver"],
            "name":       row["name"],
            "team":       row["team"],
            "color":      row["color"],
            "predicted":  round(row["PredictedGapToPole"], 3),
            "actual":     round(row["ActualGapToPole"], 3),
            "pred_pos":   round(row["PredPosition"], 1),
            "actual_pos": round(row["ActualPosition"], 1),
            "is_selected": row["team"] == team_name,
        })

    # Team averages for bar chart
    team_avg = avg_gaps.groupby("team")["ActualGapToPole"].mean().reset_index()
    team_avg["color"] = team_avg["team"].map(TEAM_COLORS)
    team_avg["predicted"] = avg_gaps.groupby("team")["PredictedGapToPole"].mean().values
    team_avg = team_avg.sort_values("ActualGapToPole")

    teams_chart = []
    for _, row in team_avg.iterrows():
        teams_chart.append({
            "team":       row["team"],
            "color":      row["color"],
            "actual":     round(row["ActualGapToPole"], 3),
            "predicted":  round(row["predicted"], 3),
            "is_selected": row["team"] == team_name,
        })

    # Feature importance for R&D
    top_features = importance_df.head(8).to_dict(orient="records")

    # R&D suggestions based on team's weaknesses
    team_drivers = [d for d, t in DRIVER_TEAM_2025.items() if t == team_name]
    team_results = results_df[results_df["driver"].isin(team_drivers)]
    avg_team_gap = team_results["ActualGapToPole"].mean()

    suggestions = []
    if avg_team_gap > 1.5:
        suggestions.append({"area": "Car Pace", "detail": "Significant gap to leaders suggests fundamental aero/power unit deficit. Priority: overall downforce package.", "priority": "Critical"})
    elif avg_team_gap > 0.8:
        suggestions.append({"area": "Car Pace", "detail": "Midfield pace deficit. Focus on DRS efficiency and corner entry stability.", "priority": "High"})
    else:
        suggestions.append({"area": "Car Pace", "detail": "Competitive pace. Fine-tune aero balance for specific circuit types.", "priority": "Medium"})

    # Street vs permanent circuit delta
    street_gap = results_df[results_df["driver"].isin(team_drivers)].merge(
        tracks[["Event","TrackType"]], left_on="race", right_on="Event", how="left"
    )
    if not street_gap.empty and "TrackType" in street_gap.columns:
        street_avg = street_gap[street_gap["TrackType"]=="Street"]["ActualGapToPole"].mean()
        perm_avg   = street_gap[street_gap["TrackType"]!="Street"]["ActualGapToPole"].mean()
        if not np.isnan(street_avg) and not np.isnan(perm_avg):
            if street_avg - perm_avg > 0.3:
                suggestions.append({"area": "Street Circuits", "detail": f"Performance drops {street_avg-perm_avg:.2f}s on street circuits. Focus on mechanical grip and low-speed aero.", "priority": "High"})
            elif perm_avg - street_avg > 0.3:
                suggestions.append({"area": "Permanent Circuits", "detail": f"Stronger on streets than permanent tracks by {perm_avg-street_avg:.2f}s. Investigate high-speed aero efficiency.", "priority": "Medium"})

    suggestions.append({"area": "Tyre Management", "detail": "Optimise soft compound usage in Q3. Fresh tyre advantage is the single biggest performance lever in qualifying.", "priority": "Medium"})
    suggestions.append({"area": "Driver Delta", "detail": "Analyse sector-by-sector gaps between teammates to identify circuit-specific setup opportunities.", "priority": "Low"})

    return jsonify({
        "grid":         grid,
        "teams_chart":  teams_chart,
        "features":     top_features,
        "suggestions":  suggestions,
        "selected_team": team_name,
        "selected_color": TEAM_COLORS.get(team_name, "#fff"),
    })


if __name__ == "__main__":
    app.run(debug=True)