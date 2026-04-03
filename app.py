"""
F1 Qualifying Predictor — Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Qualifying Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

  html, body, [class*="css"] {
    font-family: 'Titillium Web', sans-serif;
  }
  .main { background-color: #0f0f0f; }
  .stApp { background-color: #0f0f0f; }

  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #e10600;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label { color: #888; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
  .metric-value { color: #e10600; font-size: 28px; font-weight: 700; }
  .metric-delta { color: #4CAF50; font-size: 13px; }

  h1 { color: #ffffff !important; font-weight: 900 !important; letter-spacing: 2px; }
  h2, h3 { color: #cccccc !important; }

  .section-header {
    border-left: 4px solid #e10600;
    padding-left: 12px;
    margin: 24px 0 16px 0;
    color: #ffffff;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .stSelectbox label, .stSlider label { color: #aaaaaa !important; }
  div[data-testid="stMetricValue"] { color: #e10600 !important; font-size: 26px !important; }
  div[data-testid="stMetricLabel"] { color: #aaaaaa !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model     = joblib.load("model/xgb_model.pkl")
    le_team   = joblib.load("model/le_team.pkl")
    le_driver = joblib.load("model/le_driver.pkl")
    le_event  = joblib.load("model/le_event.pkl")
    features  = joblib.load("model/features.pkl")
    team_circuit_avg = joblib.load("model/team_circuit_avg.pkl")
    driver_skill     = joblib.load("model/driver_skill.pkl")
    metrics          = joblib.load("model/metrics.pkl")
    return model, le_team, le_driver, le_event, features, team_circuit_avg, driver_skill, metrics

@st.cache_data
def load_data():
    df = pd.read_csv("data/data.csv")
    tracks = pd.read_csv("data/tracks.csv")
    real_2025 = pd.read_csv("data/real_lap_time_2025.csv")
    return df, tracks, real_2025

def seconds_to_laptime(s):
    """Convert seconds float to MM:SS.mmm string."""
    if pd.isna(s) or s <= 0:
        return "N/A"
    m = int(s // 60)
    sec = s % 60
    return f"{m}:{sec:06.3f}"

# ─────────────────────────────────────────────
# CHECK MODEL EXISTS
# ─────────────────────────────────────────────
model_ready = os.path.exists("model/xgb_model.pkl")

if not model_ready:
    st.markdown("# 🏎️ F1 Qualifying Predictor")
    st.error("⚠️ Model not trained yet. Run `python train_model.py` first, then restart the app.")
    st.code("python train_model.py", language="bash")
    st.stop()

model, le_team, le_driver, le_event, features, team_circuit_avg, driver_skill, metrics = load_model()
df, tracks, real_2025 = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ F1 QUALI PREDICTOR")
    st.markdown("---")

    st.markdown("### 🎯 Select Team & Driver")
    all_teams = sorted(df["Team"].unique())
    selected_team = st.selectbox("Team", all_teams)

    team_drivers = sorted(df[df["Team"] == selected_team]["Driver"].unique())
    selected_driver = st.selectbox("Driver", team_drivers)

    st.markdown("---")
    st.markdown("### 🏁 Select Circuit")
    all_events = sorted(df["Event"].unique())
    selected_event = st.selectbox("Grand Prix", all_events)

    # Get track info
    track_info = tracks[tracks["Event"] == selected_event]
    if not track_info.empty:
        ti = track_info.iloc[0]
        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:12px;margin-top:8px'>
          <div style='color:#888;font-size:11px;letter-spacing:1px'>CIRCUIT INFO</div>
          <div style='color:#fff;margin-top:6px'>🏟️ {ti['TrackType']} · {ti['LapSpeedClass']}</div>
          <div style='color:#aaa;font-size:13px'>📏 {ti['TrackLength_m']/1000:.3f} km · {int(ti['NumCorners'])} corners</div>
          <div style='color:#aaa;font-size:13px'>🔵 {int(ti['DRSZones'])} DRS zones · 📍 {int(ti['Altitude_m'])}m alt</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Session Setup")
    quali_segment = st.selectbox("Quali Segment", ["Q3", "Q2", "Q1"])
    compound = st.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])
    tyre_life = st.slider("Tyre Age (laps)", 1, 10, 2)
    fresh_tyre = st.checkbox("Fresh Tyre", value=True)

    st.markdown("---")
    st.markdown("### 🌤️ Weather Conditions")
    air_temp = st.slider("Air Temp (°C)", 10, 45, 25)
    track_temp = st.slider("Track Temp (°C)", 15, 60, 38)
    humidity = st.slider("Humidity (%)", 10, 100, 50)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0)
    rainfall = st.checkbox("Rainfall", value=False)

    st.markdown("---")
    st.markdown(f"**Model MAE:** `{metrics['mae']}s` | **R²:** `{metrics['r2']}`")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.markdown("# 🏎️ F1 QUALIFYING PREDICTOR")
    st.markdown(f"*Predicting qualifying performance for **{selected_driver}** ({selected_team}) at **{selected_event}***")

st.markdown("---")

# ─────────────────────────────────────────────
# BUILD PREDICTION INPUT ROW
# ─────────────────────────────────────────────
def build_input_row(event, team, driver, segment, compound_str, tyre_life_val,
                    fresh, air_t, track_t, hum, wind, rain, year=2024):

    segment_map = {"Q1": 1, "Q2": 2, "Q3": 3}
    compound_map = {"SOFT": 3, "MEDIUM": 2, "HARD": 1, "INTER": 0, "WET": -1}
    speed_map = {"Slow": 1, "Medium": 2, "Fast": 3}

    # Track info
    ti = tracks[tracks["Event"] == event]
    if ti.empty:
        ti_row = {"TrackType": "Permanent", "LapSpeedClass": "Medium", "DRSZones": 2,
                  "Altitude_m": 50, "NumCorners": 15, "CornerDensity": 0.003,
                  "TrackLength_m": 5000, "AvgCornerSpacing_m": 333}
    else:
        ti_row = ti.iloc[0]

    is_street = 1 if ti_row["TrackType"] == "Street" else 0
    speed_class = speed_map.get(ti_row["LapSpeedClass"], 2)
    track_type_str = ti_row["TrackType"]

    # Team circuit avg
    tca = team_circuit_avg[
        (team_circuit_avg["Team"] == team) &
        (team_circuit_avg["TrackType"] == track_type_str)
    ]
    team_circuit_avg_val = tca["TeamCircuitAvg"].values[0] if not tca.empty else df["LapTime_sec"].mean()

    # Driver avg delta
    ds = driver_skill[driver_skill["Driver"] == driver]
    driver_avg_delta_val = ds["DriverAvgDelta"].values[0] if not ds.empty else 1.0

    # Encode categoricals — handle unseen labels gracefully
    def safe_encode(le, val):
        classes = list(le.classes_)
        if val in classes:
            return le.transform([val])[0]
        return 0  # fallback

    # Speed trap estimates by LapSpeedClass
    speed_trap_map = {"Slow": (240, 220, 210, 250), "Medium": (270, 255, 245, 285), "Fast": (300, 285, 275, 315)}
    si1, si2, sfl, sst = speed_trap_map.get(str(ti_row["LapSpeedClass"]), (265, 250, 240, 278))

    row = {
        "Team_enc":    safe_encode(le_team, team),
        "Driver_enc":  safe_encode(le_driver, driver),
        "Event_enc":   safe_encode(le_event, event),
        "Year":        year,
        "QualiSegment_num": segment_map.get(segment, 2),
        "Compound_num":     compound_map.get(compound_str, 3),
        "TyreLife":    tyre_life_val,
        "FreshTyre_int": int(fresh),
        "IsStreet":    is_street,
        "SpeedClass_num": speed_class,
        "DRSZones":    ti_row["DRSZones"],
        "Altitude_m":  ti_row["Altitude_m"],
        "NumCorners":  ti_row["NumCorners"],
        "CornerDensity": ti_row["CornerDensity"],
        "TrackLength_m": ti_row["TrackLength_m"],
        "AvgCornerSpacing_m": ti_row["AvgCornerSpacing_m"],
        "AirTemp":     air_t,
        "TrackTemp":   track_t,
        "Humidity":    hum,
        "Pressure":    1013.0,
        "WindSpeed":   wind,
        "Rainfall_int": int(rain),
        "SpeedI1":     si1,
        "SpeedI2":     si2,
        "SpeedFL":     sfl,
        "SpeedST":     sst,
        "TeamCircuitAvg":  team_circuit_avg_val,
        "DriverAvgDelta":  driver_avg_delta_val,
    }
    return pd.DataFrame([row])[features]

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Qualifying Prediction", "📊 Team Intelligence", "🛠️ R&D Setup Simulator"])

# ════════════════════════════════════════════
# TAB 1 — QUALIFYING PREDICTION
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Qualifying Lap Time Prediction</div>', unsafe_allow_html=True)

    # Predict selected driver
    input_row = build_input_row(
        selected_event, selected_team, selected_driver,
        quali_segment, compound, tyre_life, fresh_tyre,
        air_temp, track_temp, humidity, wind_speed, rainfall
    )
    predicted_time = model.predict(input_row)[0]

    # Predict all drivers for grid comparison
    all_predictions = []
    for team in all_teams:
        drivers_in_team = sorted(df[df["Team"] == team]["Driver"].unique())
        for driver in drivers_in_team:
            row = build_input_row(
                selected_event, team, driver, "Q3", "SOFT", 2, True,
                air_temp, track_temp, humidity, wind_speed, rainfall
            )
            pred = model.predict(row)[0]
            all_predictions.append({"Team": team, "Driver": driver, "PredictedTime": pred})

    grid_df = pd.DataFrame(all_predictions).sort_values("PredictedTime").reset_index(drop=True)
    grid_df["Position"] = range(1, len(grid_df) + 1)
    grid_df["Gap"] = grid_df["PredictedTime"] - grid_df["PredictedTime"].min()
    grid_df["LapTimeStr"] = grid_df["PredictedTime"].apply(seconds_to_laptime)
    grid_df["GapStr"] = grid_df["Gap"].apply(lambda x: "POLE" if x < 0.001 else f"+{x:.3f}s")

    # Find our driver's position
    our_row = grid_df[
        (grid_df["Driver"] == selected_driver) & (grid_df["Team"] == selected_team)
    ]
    our_position = our_row["Position"].values[0] if not our_row.empty else "N/A"
    our_gap = our_row["Gap"].values[0] if not our_row.empty else 0

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Predicted Lap Time", seconds_to_laptime(predicted_time))
    with c2:
        st.metric("Predicted Grid Position", f"P{our_position}")
    with c3:
        pole_time = grid_df["PredictedTime"].min()
        gap_to_pole = predicted_time - pole_time
        st.metric("Gap to Pole", f"+{gap_to_pole:.3f}s")
    with c4:
        st.metric("Circuit Type", f"{tracks[tracks['Event']==selected_event]['TrackType'].values[0] if not tracks[tracks['Event']==selected_event].empty else 'N/A'}")

    st.markdown("---")

    # Grid visualization
    col_grid, col_chart = st.columns([1, 1])

    with col_grid:
        st.markdown("**📋 Predicted Starting Grid (Q3 — SOFT)**")
        display_df = grid_df[["Position", "Driver", "Team", "LapTimeStr", "GapStr"]].head(20)
        display_df.columns = ["Pos", "Driver", "Team", "Lap Time", "Gap"]

        # Highlight our driver
        def highlight_row(row):
            if row["Driver"] == selected_driver:
                return ["background-color: #3d0000; color: #ff6b6b"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_row, axis=1),
            hide_index=True,
            use_container_width=True,
            height=450,
        )

    with col_chart:
        st.markdown("**🏁 Predicted Grid Gap to Pole**")
        top15 = grid_df.head(15).copy()
        colors = ["#e10600" if (d == selected_driver and t == selected_team) else "#444"
                  for d, t in zip(top15["Driver"], top15["Team"])]

        fig = go.Figure(go.Bar(
            x=top15["Gap"],
            y=top15["Driver"] + " (" + top15["Team"] + ")",
            orientation="h",
            marker_color=colors,
            text=top15["GapStr"],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#0f0f0f",
            plot_bgcolor="#0f0f0f",
            font=dict(color="#cccccc", family="Titillium Web"),
            xaxis=dict(title="Gap to Pole (s)", gridcolor="#222"),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            height=480,
            margin=dict(l=0, r=60, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2025 real time comparison if available
    real_event_data = real_2025[real_2025["race"] == selected_event]
    if not real_event_data.empty:
        st.markdown("---")
        st.markdown("**📡 2025 Real Qualifying Times vs Prediction**")
        compare = real_event_data.copy()
        compare["PredictedTime"] = compare.apply(
            lambda r: model.predict(build_input_row(
                selected_event,
                df[df["Driver"] == r["driver"]]["Team"].mode()[0] if r["driver"] in df["Driver"].values else "Unknown",
                r["driver"], "Q3", "SOFT", 2, True, air_temp, track_temp, humidity, wind_speed, rainfall
            ))[0] if r["driver"] in df["Driver"].values else np.nan,
            axis=1
        )
        compare["Error"] = (compare["PredictedTime"] - compare["real_time_seconds"]).round(3)
        compare = compare.dropna(subset=["PredictedTime"])
        compare["RealStr"] = compare["real_time_seconds"].apply(seconds_to_laptime)
        compare["PredStr"] = compare["PredictedTime"].apply(seconds_to_laptime)
        compare_disp = compare[["driver", "RealStr", "PredStr", "Error"]].copy()
        compare_disp.columns = ["Driver", "Real Time", "Predicted", "Error (s)"]
        st.dataframe(compare_disp, hide_index=True, use_container_width=True)


# ════════════════════════════════════════════
# TAB 2 — TEAM INTELLIGENCE
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Team Intelligence Dashboard</div>', unsafe_allow_html=True)

    team_df = df[df["Team"] == selected_team].copy()

    col_a, col_b = st.columns(2)

    # --- Yearly best lap time trend ---
    with col_a:
        st.markdown("**📈 Team Best Qualifying Time by Year**")
        yearly = (
            team_df.groupby(["Year", "Event"])["LapTime_sec"]
            .min()
            .reset_index()
            .groupby("Year")["LapTime_sec"]
            .mean()
            .reset_index()
        )
        field_yearly = (
            df.groupby(["Year", "Event"])["LapTime_sec"]
            .min()
            .reset_index()
            .groupby("Year")["LapTime_sec"]
            .mean()
            .reset_index()
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=yearly["Year"], y=yearly["LapTime_sec"],
            mode="lines+markers", name=selected_team,
            line=dict(color="#e10600", width=3), marker=dict(size=8)
        ))
        fig2.add_trace(go.Scatter(
            x=field_yearly["Year"], y=field_yearly["LapTime_sec"],
            mode="lines+markers", name="Field Average",
            line=dict(color="#555", width=2, dash="dash"), marker=dict(size=6)
        ))
        fig2.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(title="Year", gridcolor="#222"),
            yaxis=dict(title="Avg Best Lap (s)", gridcolor="#222"),
            legend=dict(bgcolor="#1a1a2e"),
            height=300, margin=dict(l=0, r=0, t=10, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Circuit type performance ---
    with col_b:
        st.markdown("**🏁 Performance by Circuit Type**")
        circuit_perf = (
            team_df.groupby("TrackType")["LapTime_sec"].mean().reset_index()
        )
        field_circuit = (
            df.groupby("TrackType")["LapTime_sec"].mean().reset_index()
        )
        circuit_perf["Field"] = circuit_perf["TrackType"].map(
            dict(zip(field_circuit["TrackType"], field_circuit["LapTime_sec"]))
        )
        circuit_perf["Delta"] = circuit_perf["LapTime_sec"] - circuit_perf["Field"]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name=selected_team, x=circuit_perf["TrackType"], y=circuit_perf["LapTime_sec"],
            marker_color="#e10600"
        ))
        fig3.add_trace(go.Bar(
            name="Field Avg", x=circuit_perf["TrackType"], y=circuit_perf["Field"],
            marker_color="#444"
        ))
        fig3.update_layout(
            barmode="group",
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            yaxis=dict(title="Avg Lap Time (s)", gridcolor="#222"),
            legend=dict(bgcolor="#1a1a2e"),
            height=300, margin=dict(l=0, r=0, t=10, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- Lap Speed Class breakdown ---
    st.markdown("**⚡ Performance by Lap Speed Class (Slow / Medium / Fast circuits)**")
    speed_perf = (
        team_df.groupby("LapSpeedClass")["LapTime_sec"].mean().reset_index()
    )
    field_speed = (
        df.groupby("LapSpeedClass")["LapTime_sec"].mean().reset_index()
    )
    speed_perf["Field"] = speed_perf["LapSpeedClass"].map(
        dict(zip(field_speed["LapSpeedClass"], field_speed["LapTime_sec"]))
    )
    speed_perf["DeltaToField"] = (speed_perf["LapTime_sec"] - speed_perf["Field"]).round(3)
    speed_perf["Better"] = speed_perf["DeltaToField"] < 0

    col_c, col_d = st.columns([2, 1])
    with col_c:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=speed_perf["LapSpeedClass"],
            y=speed_perf["DeltaToField"],
            marker_color=["#4CAF50" if b else "#e10600" for b in speed_perf["Better"]],
            text=speed_perf["DeltaToField"].apply(lambda x: f"{x:+.3f}s"),
            textposition="outside",
        ))
        fig4.add_hline(y=0, line_color="#888", line_dash="dash")
        fig4.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(title="Circuit Speed Class"),
            yaxis=dict(title="Delta vs Field (s) — negative = faster", gridcolor="#222"),
            height=280, margin=dict(l=0, r=0, t=10, b=40),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        st.markdown("**🏆 Driver Comparison**")
        driver_best = (
            team_df.groupby("Driver")["LapTime_sec"].min().sort_values().reset_index()
        )
        driver_best["LapTimeStr"] = driver_best["LapTime_sec"].apply(seconds_to_laptime)
        st.dataframe(
            driver_best[["Driver", "LapTimeStr"]].rename(columns={"LapTimeStr": "Best Time"}),
            hide_index=True, use_container_width=True
        )

    # --- Historical event breakdown ---
    st.markdown("**📍 Best Lap Time per Circuit (All Years)**")
    event_best = (
        team_df.groupby("Event")["LapTime_sec"].min().reset_index().sort_values("LapTime_sec")
    )
    event_best["LapTimeStr"] = event_best["LapTime_sec"].apply(seconds_to_laptime)

    fig5 = px.bar(
        event_best, x="Event", y="LapTime_sec",
        color="LapTime_sec",
        color_continuous_scale=["#e10600", "#ff8800", "#ffcc00"],
        labels={"LapTime_sec": "Best Lap (s)", "Event": ""},
    )
    fig5.update_layout(
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc", family="Titillium Web"),
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45, gridcolor="#222"),
        yaxis=dict(gridcolor="#222"),
        height=320, margin=dict(l=0, r=0, t=10, b=100),
    )
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — R&D SETUP SIMULATOR
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">R&D Setup Simulator — Downforce Configuration</div>', unsafe_allow_html=True)

    st.info("""
    **How this works:** Without raw aero telemetry, we infer downforce sensitivity from historical 
    speed trap data and circuit type performance. High downforce = more drag = slower straights, 
    more cornering grip. We simulate this by shifting speed trap values and seeing how the 
    predicted lap time changes.
    """)

    col_rd1, col_rd2 = st.columns(2)

    with col_rd1:
        st.markdown("### ⬆️ High Downforce Setup")
        st.markdown("*More wing angle · Better through corners · Slower straights*")
        st.markdown("Typical circuits: **Monaco, Singapore, Hungary, Zandvoort**")
        downforce_penalty = st.slider("Straight-line speed loss (km/h)", 5, 25, 12,
                                       help="How much top speed you sacrifice for downforce")

    with col_rd2:
        st.markdown("### ⬇️ Low Downforce Setup")
        st.markdown("*Less wing angle · More straight-line speed · Less cornering grip*")
        st.markdown("Typical circuits: **Monza, Baku, Las Vegas, Spa**")
        low_df_gain = st.slider("Straight-line speed gain (km/h)", 5, 25, 12,
                                 help="How much top speed you gain by reducing downforce")

    st.markdown("---")

    # Simulate predictions across all circuits
    sim_results = []

    for _, track_row in tracks.iterrows():
        event = track_row["Event"]
        if event not in df["Event"].values:
            continue

        speed_class = track_row["LapSpeedClass"]
        track_type = track_row["TrackType"]

        # Baseline speed traps
        base_traps = {"Slow": (240, 220, 210, 250), "Medium": (270, 255, 245, 285), "Fast": (300, 285, 275, 315)}
        si1, si2, sfl, sst = base_traps.get(str(speed_class), (265, 250, 240, 278))

        # High downforce: slower straights, better grip (reduce speed trap, improve sector times on corners)
        hi_si1 = si1 - downforce_penalty
        hi_si2 = si2 - downforce_penalty
        hi_sst = sst - downforce_penalty
        hi_sfl = sfl

        # Low downforce: faster straights, less grip
        lo_si1 = si1 + low_df_gain
        lo_si2 = si2 + low_df_gain
        lo_sst = sst + low_df_gain
        lo_sfl = sfl + low_df_gain

        def predict_with_traps(sp1, sp2, spfl, spst):
            row = build_input_row(event, selected_team, selected_driver, "Q3", "SOFT", 2, True,
                                  air_temp, track_temp, humidity, wind_speed, rainfall)
            row["SpeedI1"] = sp1
            row["SpeedI2"] = sp2
            row["SpeedFL"] = spfl
            row["SpeedST"] = spst
            return model.predict(row)[0]

        baseline = predict_with_traps(si1, si2, sfl, sst)
        high_df  = predict_with_traps(hi_si1, hi_si2, hi_sfl, hi_sst)
        low_df   = predict_with_traps(lo_si1, lo_si2, lo_sfl, lo_sst)

        # Best setup recommendation
        best_time = min(baseline, high_df, low_df)
        if best_time == high_df:
            recommendation = "High Downforce"
        elif best_time == low_df:
            recommendation = "Low Downforce"
        else:
            recommendation = "Balanced"

        sim_results.append({
            "Event": event,
            "TrackType": track_type,
            "SpeedClass": speed_class,
            "Baseline (s)": round(baseline, 3),
            "High DF (s)": round(high_df, 3),
            "Low DF (s)": round(low_df, 3),
            "Hi DF Delta": round(high_df - baseline, 3),
            "Lo DF Delta": round(low_df - baseline, 3),
            "Best Setup": recommendation,
        })

    sim_df = pd.DataFrame(sim_results).sort_values("Baseline (s)")

    # Display summary
    col_m1, col_m2, col_m3 = st.columns(3)
    hi_better = (sim_df["Hi DF Delta"] < sim_df["Lo DF Delta"]).sum()
    lo_better = (sim_df["Lo DF Delta"] < sim_df["Hi DF Delta"]).sum()

    with col_m1:
        st.metric("Circuits favouring High DF", hi_better)
    with col_m2:
        st.metric("Circuits favouring Low DF", lo_better)
    with col_m3:
        avg_hi = sim_df["Hi DF Delta"].mean()
        avg_lo = sim_df["Lo DF Delta"].mean()
        overall = "High Downforce" if avg_hi < avg_lo else "Low Downforce"
        st.metric("Overall Season Recommendation", overall)

    st.markdown("---")

    # Chart: delta per circuit
    st.markdown("**🔧 Setup Delta per Circuit (negative = faster than baseline)**")
    fig_rd = go.Figure()
    fig_rd.add_trace(go.Bar(
        name="High Downforce", x=sim_df["Event"], y=sim_df["Hi DF Delta"],
        marker_color="#e10600",
    ))
    fig_rd.add_trace(go.Bar(
        name="Low Downforce", x=sim_df["Event"], y=sim_df["Lo DF Delta"],
        marker_color="#0066cc",
    ))
    fig_rd.add_hline(y=0, line_color="#888", line_dash="dash")
    fig_rd.update_layout(
        barmode="group",
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc", family="Titillium Web"),
        xaxis=dict(tickangle=-45, gridcolor="#222"),
        yaxis=dict(title="Delta vs Baseline (s)", gridcolor="#222"),
        legend=dict(bgcolor="#1a1a2e"),
        height=380, margin=dict(l=0, r=0, t=10, b=120),
    )
    st.plotly_chart(fig_rd, use_container_width=True)

    # Full table
    st.markdown("**📊 Full Setup Simulation Table**")
    display_sim = sim_df[["Event", "TrackType", "SpeedClass", "Hi DF Delta", "Lo DF Delta", "Best Setup"]].copy()
    display_sim["Hi DF Delta"] = display_sim["Hi DF Delta"].apply(lambda x: f"{x:+.3f}s")
    display_sim["Lo DF Delta"] = display_sim["Lo DF Delta"].apply(lambda x: f"{x:+.3f}s")

    def highlight_setup(row):
        if row["Best Setup"] == "High Downforce":
            return ["", "", "", "background-color:#3d0000;color:#ff6b6b", "", "background-color:#3d0000;color:#ff6b6b"]
        elif row["Best Setup"] == "Low Downforce":
            return ["", "", "", "", "background-color:#003d00;color:#6bff6b", "background-color:#003d00;color:#6bff6b"]
        return [""] * len(row)

    st.dataframe(
        display_sim.style.apply(highlight_setup, axis=1),
        hide_index=True,
        use_container_width=True,
        height=400,
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#444;font-size:12px'>F1 Qualifying Predictor · Built with XGBoost + Streamlit · Data: 2019–2024 FastF1</div>",
    unsafe_allow_html=True
)