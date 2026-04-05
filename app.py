"""
F1 Qualifying Predictor — Streamlit Dashboard (2025 Season)
Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
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

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');
  html, body, [class*="css"] { font-family: 'Titillium Web', sans-serif; }
  .stApp { background-color: #0f0f0f; }
  h1 { color: #ffffff !important; font-weight: 900 !important; letter-spacing: 2px; }
  h2, h3 { color: #cccccc !important; }
  .section-header {
    border-left: 4px solid #e10600; padding-left: 12px;
    margin: 20px 0 14px 0; color: #ffffff;
    font-size: 18px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
  }
  .rookie-badge {
    background: #2a1500; border: 1px solid #ff8c00; border-radius: 4px;
    color: #ff8c00; font-size: 11px; padding: 2px 7px; margin-left: 6px;
  }
  div[data-testid="stMetricValue"] { color: #e10600 !important; font-size: 26px !important; }
  div[data-testid="stMetricLabel"] { color: #aaaaaa !important; }
  .stSelectbox label, .stSlider label { color: #aaaaaa !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model            = joblib.load("model/xgb_model.pkl")
    le_team          = joblib.load("model/le_team.pkl")
    le_driver        = joblib.load("model/le_driver.pkl")
    le_event         = joblib.load("model/le_event.pkl")
    features         = joblib.load("model/features.pkl")
    team_circuit_avg = joblib.load("model/team_circuit_avg.pkl")
    driver_skill     = joblib.load("model/driver_skill.pkl")
    team_avg_delta   = joblib.load("model/team_avg_delta.pkl")
    f1_2025_grid     = joblib.load("model/f1_2025_grid.pkl")
    rookies_2025     = joblib.load("model/rookies_2025.pkl")
    metrics          = joblib.load("model/metrics.pkl")
    return (model, le_team, le_driver, le_event, features,
            team_circuit_avg, driver_skill, team_avg_delta,
            f1_2025_grid, rookies_2025, metrics)

@st.cache_data
def load_data():
    df        = pd.read_csv("data/data.csv")
    tracks    = pd.read_csv("data/tracks.csv")
    real_2025 = pd.read_csv("data/real_lap_time_2025.csv")
    return df, tracks, real_2025

def secs_to_str(s):
    if pd.isna(s) or s <= 0:
        return "N/A"
    m = int(s // 60)
    return f"{m}:{s % 60:06.3f}"

# ─────────────────────────────────────────────
# MODEL CHECK
# ─────────────────────────────────────────────
if not os.path.exists("model/xgb_model.pkl"):
    st.markdown("# 🏎️ F1 Qualifying Predictor")
    st.error("Model not trained yet. Run `python train_model.py` first.")
    st.code("python train_model.py", language="bash")
    st.stop()

(model, le_team, le_driver, le_event, features,
 team_circuit_avg, driver_skill, team_avg_delta,
 F1_2025_GRID, ROOKIES_2025, metrics) = load_model()

df, tracks, real_2025 = load_data()

# Convenience lookups
TEAMS_2025   = list(F1_2025_GRID.keys())
ALL_EVENTS   = sorted(real_2025["race"].unique())  # use 2025 calendar

# Driver → team map for 2025
DRIVER_TEAM_2025 = {d: t for t, drivers in F1_2025_GRID.items() for d in drivers}

# ─────────────────────────────────────────────
# HELPER: BUILD PREDICTION INPUT ROW
# ─────────────────────────────────────────────
def build_input(event, team, driver, segment="Q3", compound="SOFT",
                tyre_life=2, fresh=True, air_t=25, track_t=38,
                hum=50, wind=3.0, rain=False, year=2025):

    segment_map  = {"Q1": 1, "Q2": 2, "Q3": 3}
    compound_map = {"SOFT": 3, "MEDIUM": 2, "HARD": 1, "INTER": 0, "WET": -1}
    speed_map    = {"Slow": 1, "Medium": 2, "Fast": 3}

    ti = tracks[tracks["Event"] == event]
    if ti.empty:
        ti_vals = dict(TrackType="Permanent", LapSpeedClass="Medium", DRSZones=2,
                       Altitude_m=50, NumCorners=15, CornerDensity=0.003,
                       TrackLength_m=5000, AvgCornerSpacing_m=333)
    else:
        ti_vals = ti.iloc[0].to_dict()

    is_street  = 1 if ti_vals["TrackType"] == "Street" else 0
    speed_cls  = speed_map.get(str(ti_vals["LapSpeedClass"]), 2)
    track_type = ti_vals["TrackType"]

    # TeamCircuitAvg
    tca = team_circuit_avg[
        (team_circuit_avg["Team"] == team) &
        (team_circuit_avg["TrackType"] == track_type)
    ]
    tc_avg = tca["TeamCircuitAvg"].values[0] if not tca.empty else df["LapTime_sec"].mean()

    # DriverAvgDelta (rookies already in driver_skill from train_model)
    ds = driver_skill[driver_skill["Driver"] == driver]
    if ds.empty:
        # Fallback: use team avg delta
        td = team_avg_delta[team_avg_delta["Team"] == team]
        drv_delta = td["TeamAvgDelta"].values[0] if not td.empty else 1.5
    else:
        drv_delta = ds["DriverAvgDelta"].values[0]

    # Safe label encoding
    def enc(le, val):
        return le.transform([val])[0] if val in le.classes_ else 0

    # Speed trap defaults by circuit speed class
    trap_map = {"Slow": (240, 220, 210, 250), "Medium": (270, 255, 245, 285), "Fast": (300, 285, 275, 315)}
    si1, si2, sfl, sst = trap_map.get(str(ti_vals["LapSpeedClass"]), (265, 250, 240, 278))

    row = {
        "Team_enc":           enc(le_team,   team),
        "Driver_enc":         enc(le_driver, driver),
        "Event_enc":          enc(le_event,  event),
        "Year":               year,
        "QualiSegment_num":   segment_map.get(segment, 2),
        "Compound_num":       compound_map.get(compound, 3),
        "TyreLife":           tyre_life,
        "FreshTyre_int":      int(fresh),
        "IsStreet":           is_street,
        "SpeedClass_num":     speed_cls,
        "DRSZones":           ti_vals["DRSZones"],
        "Altitude_m":         ti_vals["Altitude_m"],
        "NumCorners":         ti_vals["NumCorners"],
        "CornerDensity":      ti_vals["CornerDensity"],
        "TrackLength_m":      ti_vals["TrackLength_m"],
        "AvgCornerSpacing_m": ti_vals["AvgCornerSpacing_m"],
        "AirTemp":            air_t,
        "TrackTemp":          track_t,
        "Humidity":           hum,
        "Pressure":           1013.0,
        "WindSpeed":          wind,
        "Rainfall_int":       int(rain),
        "SpeedI1":            si1,
        "SpeedI2":            si2,
        "SpeedFL":            sfl,
        "SpeedST":            sst,
        "TeamCircuitAvg":     tc_avg,
        "DriverAvgDelta":     drv_delta,
    }
    return pd.DataFrame([row])[features]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ F1 QUALI PREDICTOR")
    st.markdown("*2025 Season*")
    st.markdown("---")

    st.markdown("### 🎯 Team & Driver")
    sel_team   = st.selectbox("Team", TEAMS_2025)
    team_drvs  = F1_2025_GRID[sel_team]
    sel_driver = st.selectbox(
        "Driver",
        team_drvs,
        format_func=lambda d: f"{d} 🆕" if d in ROOKIES_2025 else d
    )
    if sel_driver in ROOKIES_2025:
        st.caption("🆕 Rookie — prediction based on team historical average")

    st.markdown("---")
    st.markdown("### 🏁 Circuit")
    sel_event = st.selectbox("Grand Prix", ALL_EVENTS)

    ti_sidebar = tracks[tracks["Event"] == sel_event]
    if not ti_sidebar.empty:
        t = ti_sidebar.iloc[0]
        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:12px;margin-top:6px'>
          <div style='color:#888;font-size:11px;letter-spacing:1px'>CIRCUIT INFO</div>
          <div style='color:#fff;margin-top:6px'>🏟️ {t['TrackType']} · {t['LapSpeedClass']}</div>
          <div style='color:#aaa;font-size:13px'>📏 {t['TrackLength_m']/1000:.3f} km · {int(t['NumCorners'])} corners</div>
          <div style='color:#aaa;font-size:13px'>🔵 {int(t['DRSZones'])} DRS zones · 📍 {int(t['Altitude_m'])}m alt</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Session Setup")
    quali_seg = st.selectbox("Quali Segment", ["Q3", "Q2", "Q1"])
    compound  = st.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])
    tyre_life = st.slider("Tyre Age (laps)", 1, 10, 2)
    fresh     = st.checkbox("Fresh Tyre", value=True)

    st.markdown("---")
    st.markdown("### 🌤️ Weather")
    air_temp   = st.slider("Air Temp (°C)", 10, 45, 25)
    track_temp = st.slider("Track Temp (°C)", 15, 60, 38)
    humidity   = st.slider("Humidity (%)", 10, 100, 50)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0)
    rainfall   = st.checkbox("Rainfall", value=False)

    st.markdown("---")
    st.markdown(f"**Model MAE:** `{metrics['mae']}s` · **R²:** `{metrics['r2']}`")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🏎️ F1 QUALIFYING PREDICTOR — 2025 SEASON")
st.markdown(f"*{sel_driver} · {sel_team} · {sel_event}*")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📊 Team Intelligence",
    "🛠️ R&D Simulator",
    "✅ 2025 Accuracy",
])

# ════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Qualifying Lap Time Prediction</div>', unsafe_allow_html=True)

    # Selected driver prediction
    inp = build_input(sel_event, sel_team, sel_driver, quali_seg, compound,
                      tyre_life, fresh, air_temp, track_temp, humidity, wind_speed, rainfall)
    pred_time = model.predict(inp)[0]

    # Full 2025 grid predictions (Q3, SOFT, default weather) for comparison
    @st.cache_data
    def predict_full_grid(event, air_t, track_t, hum, wind, rain):
        rows = []
        for team, drivers in F1_2025_GRID.items():
            for drv in drivers:
                row = build_input(event, team, drv, "Q3", "SOFT", 2, True,
                                  air_t, track_t, hum, wind, rain)
                p = model.predict(row)[0]
                rows.append({"Team": team, "Driver": drv, "PredTime": p,
                              "IsRookie": drv in ROOKIES_2025})
        gdf = pd.DataFrame(rows).sort_values("PredTime").reset_index(drop=True)
        gdf["Position"] = range(1, len(gdf) + 1)
        gdf["Gap"]      = gdf["PredTime"] - gdf["PredTime"].min()
        gdf["TimeStr"]  = gdf["PredTime"].apply(secs_to_str)
        gdf["GapStr"]   = gdf["Gap"].apply(lambda x: "POLE" if x < 0.001 else f"+{x:.3f}s")
        return gdf

    grid_df = predict_full_grid(sel_event, air_temp, track_temp, humidity, wind_speed, rainfall)

    our_row = grid_df[(grid_df["Driver"] == sel_driver) & (grid_df["Team"] == sel_team)]
    our_pos = our_row["Position"].values[0] if not our_row.empty else "?"
    our_gap = our_row["Gap"].values[0] if not our_row.empty else 0.0
    pole    = grid_df["PredTime"].min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Lap Time", secs_to_str(pred_time))
    c2.metric("Predicted Grid Position", f"P{our_pos}")
    c3.metric("Gap to Pole", f"+{pred_time - pole:.3f}s")
    ti_info = tracks[tracks["Event"] == sel_event]
    c4.metric("Circuit Type", ti_info["TrackType"].values[0] if not ti_info.empty else "—")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**📋 Predicted Starting Grid (Q3 · SOFT)**")
        disp = grid_df[["Position", "Driver", "Team", "TimeStr", "GapStr", "IsRookie"]].copy()
        disp.columns = ["Pos", "Driver", "Team", "Lap Time", "Gap", "Rookie"]
        disp["Driver"] = disp.apply(
            lambda r: f"{r['Driver']} 🆕" if r["Rookie"] else r["Driver"], axis=1
        )
        disp = disp.drop(columns="Rookie")

        def hl(row):
            if sel_driver in row["Driver"] and row["Team"] == sel_team:
                return ["background-color:#3d0000;color:#ff8888"] * len(row)
            return [""] * len(row)

        st.dataframe(disp.style.apply(hl, axis=1), hide_index=True,
                     use_container_width=True, height=500)

    with col_r:
        st.markdown("**🏁 Gap to Pole — Full 2025 Grid**")
        top = grid_df.copy()
        colors = ["#e10600" if (r["Driver"] == sel_driver and r["Team"] == sel_team)
                  else "#444" for _, r in top.iterrows()]
        fig = go.Figure(go.Bar(
            x=top["Gap"],
            y=top["Driver"] + " (" + top["Team"].str.split().str[-1] + ")",
            orientation="h",
            marker_color=colors,
            text=top["GapStr"],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(title="Gap to Pole (s)", gridcolor="#222"),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            height=540, margin=dict(l=0, r=70, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════
# TAB 2 — TEAM INTELLIGENCE
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Team Intelligence Dashboard</div>', unsafe_allow_html=True)

    # Filter to 2025 team name only (use historical data for that team)
    team_df  = df[df["Team"] == sel_team].copy()
    field_df = df.copy()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📈 Team Avg Qualifying Time by Year vs Field**")
        yr_team  = team_df.groupby("Year")["LapTime_sec"].mean().reset_index()
        yr_field = field_df.groupby("Year")["LapTime_sec"].mean().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=yr_team["Year"],  y=yr_team["LapTime_sec"],
            mode="lines+markers", name=sel_team,
            line=dict(color="#e10600", width=3), marker=dict(size=8)))
        fig2.add_trace(go.Scatter(x=yr_field["Year"], y=yr_field["LapTime_sec"],
            mode="lines+markers", name="Field Average",
            line=dict(color="#555", width=2, dash="dash"), marker=dict(size=6)))
        fig2.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(gridcolor="#222"), yaxis=dict(title="Avg Lap (s)", gridcolor="#222"),
            legend=dict(bgcolor="#1a1a2e"), height=300, margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("**🏁 Performance by Circuit Type vs Field**")
        cperf  = team_df.groupby("TrackType")["LapTime_sec"].mean().reset_index()
        cfield = field_df.groupby("TrackType")["LapTime_sec"].mean().reset_index()
        cperf["Field"] = cperf["TrackType"].map(
            dict(zip(cfield["TrackType"], cfield["LapTime_sec"])))
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name=sel_team, x=cperf["TrackType"], y=cperf["LapTime_sec"],
            marker_color="#e10600"))
        fig3.add_trace(go.Bar(name="Field", x=cperf["TrackType"], y=cperf["Field"],
            marker_color="#444"))
        fig3.update_layout(barmode="group",
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            yaxis=dict(title="Avg Lap (s)", gridcolor="#222"),
            legend=dict(bgcolor="#1a1a2e"), height=300, margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**⚡ Delta vs Field by Speed Class (negative = faster than field)**")
    sperf  = team_df.groupby("LapSpeedClass")["LapTime_sec"].mean().reset_index()
    sfield = field_df.groupby("LapSpeedClass")["LapTime_sec"].mean().reset_index()
    sperf["Field"] = sperf["LapSpeedClass"].map(
        dict(zip(sfield["LapSpeedClass"], sfield["LapTime_sec"])))
    sperf["Delta"] = (sperf["LapTime_sec"] - sperf["Field"]).round(3)

    col_c, col_d = st.columns([3, 1])
    with col_c:
        fig4 = go.Figure(go.Bar(
            x=sperf["LapSpeedClass"], y=sperf["Delta"],
            marker_color=["#4CAF50" if d < 0 else "#e10600" for d in sperf["Delta"]],
            text=sperf["Delta"].apply(lambda x: f"{x:+.3f}s"), textposition="outside"))
        fig4.add_hline(y=0, line_color="#888", line_dash="dash")
        fig4.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(title="Speed Class"),
            yaxis=dict(title="Delta vs Field (s)", gridcolor="#222"),
            height=280, margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        st.markdown("**2025 Drivers**")
        for drv in F1_2025_GRID[sel_team]:
            ds = driver_skill[driver_skill["Driver"] == drv]
            delta = ds["DriverAvgDelta"].values[0] if not ds.empty else 0
            rookie_tag = " 🆕" if drv in ROOKIES_2025 else ""
            st.markdown(f"**{drv}{rookie_tag}** — avg delta `+{delta:.3f}s`")

    st.markdown("**📍 Best Lap per Circuit (Historical)**")
    evbest = team_df.groupby("Event")["LapTime_sec"].min().reset_index().sort_values("LapTime_sec")
    fig5 = px.bar(evbest, x="Event", y="LapTime_sec",
        color="LapTime_sec", color_continuous_scale=["#e10600","#ff8800","#ffcc00"],
        labels={"LapTime_sec": "Best Lap (s)", "Event": ""})
    fig5.update_layout(
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc", family="Titillium Web"),
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45, gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        height=320, margin=dict(l=0,r=0,t=10,b=110))
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — R&D SIMULATOR
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">R&D Setup Simulator — Downforce Configuration</div>', unsafe_allow_html=True)

    st.info("We infer downforce sensitivity from speed trap data and circuit type. "
            "High downforce = slower straights, better cornering. "
            "Simulated by shifting speed trap values per circuit.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⬆️ High Downforce")
        st.markdown("*Monaco · Singapore · Hungary · Zandvoort*")
        df_penalty = st.slider("Straight speed loss (km/h)", 5, 25, 12)
    with c2:
        st.markdown("### ⬇️ Low Downforce")
        st.markdown("*Monza · Baku · Las Vegas · Spa*")
        df_gain = st.slider("Straight speed gain (km/h)", 5, 25, 12)

    st.markdown("---")
    sim = []
    for _, tr in tracks.iterrows():
        ev = tr["Event"]
        if ev not in df["Event"].values:
            continue
        sc = str(tr["LapSpeedClass"])
        trap_map = {"Slow": (240,220,210,250), "Medium": (270,255,245,285), "Fast": (300,285,275,315)}
        si1, si2, sfl, sst = trap_map.get(sc, (265,250,240,278))

        def pred_trap(s1, s2, sf, ss):
            r = build_input(ev, sel_team, sel_driver, "Q3", "SOFT", 2, True,
                            air_temp, track_temp, humidity, wind_speed, rainfall)
            r["SpeedI1"] = s1; r["SpeedI2"] = s2
            r["SpeedFL"] = sf; r["SpeedST"] = ss
            return model.predict(r)[0]

        base = pred_trap(si1, si2, sfl, sst)
        hi   = pred_trap(si1-df_penalty, si2-df_penalty, sfl, sst-df_penalty)
        lo   = pred_trap(si1+df_gain,    si2+df_gain,    sfl+df_gain, sst+df_gain)

        best = min(base, hi, lo)
        rec  = "High DF" if best==hi else ("Low DF" if best==lo else "Balanced")
        sim.append({"Event": ev, "TrackType": tr["TrackType"], "SpeedClass": sc,
                    "Baseline": round(base,3), "High DF": round(hi,3), "Low DF": round(lo,3),
                    "Hi Delta": round(hi-base,3), "Lo Delta": round(lo-base,3), "Best": rec})

    sim_df = pd.DataFrame(sim).sort_values("Baseline")

    m1, m2, m3 = st.columns(3)
    hi_better = (sim_df["Hi Delta"] < sim_df["Lo Delta"]).sum()
    lo_better = (sim_df["Lo Delta"] < sim_df["Hi Delta"]).sum()
    m1.metric("Circuits: High DF wins", hi_better)
    m2.metric("Circuits: Low DF wins",  lo_better)
    m3.metric("Season Recommendation",
              "High DF" if sim_df["Hi Delta"].mean() < sim_df["Lo Delta"].mean() else "Low DF")

    fig_rd = go.Figure()
    fig_rd.add_trace(go.Bar(name="High DF", x=sim_df["Event"], y=sim_df["Hi Delta"], marker_color="#e10600"))
    fig_rd.add_trace(go.Bar(name="Low DF",  x=sim_df["Event"], y=sim_df["Lo Delta"], marker_color="#0066cc"))
    fig_rd.add_hline(y=0, line_color="#888", line_dash="dash")
    fig_rd.update_layout(barmode="group",
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc", family="Titillium Web"),
        xaxis=dict(tickangle=-45, gridcolor="#222"),
        yaxis=dict(title="Delta vs Baseline (s) — negative=faster", gridcolor="#222"),
        legend=dict(bgcolor="#1a1a2e"), height=380, margin=dict(l=0,r=0,t=10,b=120))
    st.plotly_chart(fig_rd, use_container_width=True)

    disp_sim = sim_df[["Event","TrackType","SpeedClass","Hi Delta","Lo Delta","Best"]].copy()
    disp_sim["Hi Delta"] = disp_sim["Hi Delta"].apply(lambda x: f"{x:+.3f}s")
    disp_sim["Lo Delta"] = disp_sim["Lo Delta"].apply(lambda x: f"{x:+.3f}s")
    st.dataframe(disp_sim, hide_index=True, use_container_width=True, height=400)


# ════════════════════════════════════════════
# TAB 4 — 2025 ACCURACY
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">2025 Season Accuracy — Predicted vs Real</div>', unsafe_allow_html=True)

    st.markdown("Compare the model's predicted qualifying times against the real 2025 results for every race and driver.")

    # Build predictions for all 2025 drivers × all 2025 races
    @st.cache_data
    def compute_2025_accuracy():
        rows = []
        for _, real_row in real_2025.iterrows():
            drv  = real_row["driver"]
            race = real_row["race"]
            real_t = real_row["real_time_seconds"]

            team = DRIVER_TEAM_2025.get(drv)
            if team is None:
                continue  # skip drivers not in 2025 grid

            inp = build_input(race, team, drv, "Q3", "SOFT", 2, True,
                              25, 38, 50, 3.0, False)
            pred_t = model.predict(inp)[0]
            error  = pred_t - real_t

            rows.append({
                "Race":        race,
                "Driver":      drv,
                "Team":        team,
                "Real (s)":    round(real_t, 3),
                "Predicted (s)": round(pred_t, 3),
                "Error (s)":   round(error, 3),
                "Abs Error":   round(abs(error), 3),
                "IsRookie":    drv in ROOKIES_2025,
            })
        return pd.DataFrame(rows)

    acc_df = compute_2025_accuracy()

    if acc_df.empty:
        st.warning("No matching 2025 data found.")
    else:
        # ── Summary metrics ──────────────────────────────
        overall_mae = acc_df["Abs Error"].mean()
        overall_r2  = 1 - (
            ((acc_df["Error (s)"]**2).sum()) /
            ((acc_df["Real (s)"] - acc_df["Real (s)"].mean())**2).sum()
        )
        within_1s = (acc_df["Abs Error"] <= 1.0).mean() * 100
        within_2s = (acc_df["Abs Error"] <= 2.0).mean() * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall MAE (2025)", f"{overall_mae:.3f}s")
        m2.metric("R² Score (2025)",    f"{overall_r2:.4f}")
        m3.metric("Within 1s accuracy", f"{within_1s:.1f}%")
        m4.metric("Within 2s accuracy", f"{within_2s:.1f}%")

        st.markdown("---")

        # ── Filters ──────────────────────────────────────
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_race = st.selectbox("Filter by Race",
                ["All Races"] + sorted(acc_df["Race"].unique()))
        with col_f2:
            filter_team = st.selectbox("Filter by Team",
                ["All Teams"] + sorted(acc_df["Team"].unique()))

        filt = acc_df.copy()
        if filter_race != "All Races":
            filt = filt[filt["Race"] == filter_race]
        if filter_team != "All Teams":
            filt = filt[filt["Team"] == filter_team]

        # ── Error distribution chart ──────────────────────
        st.markdown("**📊 Prediction Error Distribution across all 2025 races**")
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=acc_df["Error (s)"], nbinsx=40,
            marker_color="#e10600", opacity=0.8,
            name="Prediction Error"
        ))
        fig_err.add_vline(x=0, line_color="#ffffff", line_dash="dash")
        fig_err.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(title="Error (Predicted − Real, seconds)", gridcolor="#222"),
            yaxis=dict(title="Count", gridcolor="#222"),
            height=280, margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig_err, use_container_width=True)

        # ── MAE per race ──────────────────────────────────
        st.markdown("**🏁 MAE per Race**")
        race_mae = acc_df.groupby("Race")["Abs Error"].mean().reset_index().sort_values("Abs Error")
        fig_race = px.bar(race_mae, x="Race", y="Abs Error",
            color="Abs Error", color_continuous_scale=["#4CAF50","#ffcc00","#e10600"],
            labels={"Abs Error": "MAE (s)", "Race": ""})
        fig_race.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-45, gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            height=320, margin=dict(l=0,r=0,t=10,b=110))
        st.plotly_chart(fig_race, use_container_width=True)

        # ── MAE per team ──────────────────────────────────
        st.markdown("**🏎️ MAE per Team**")
        team_mae = acc_df.groupby("Team")["Abs Error"].mean().reset_index().sort_values("Abs Error")
        fig_team = px.bar(team_mae, x="Team", y="Abs Error",
            color="Abs Error", color_continuous_scale=["#4CAF50","#ffcc00","#e10600"],
            labels={"Abs Error": "MAE (s)", "Team": ""})
        fig_team.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            coloraxis_showscale=False,
            xaxis=dict(tickangle=-20, gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            height=300, margin=dict(l=0,r=0,t=10,b=60))
        st.plotly_chart(fig_team, use_container_width=True)

        # ── Scatter: predicted vs real ─────────────────────
        st.markdown("**🔵 Predicted vs Real Lap Times (all races)**")
        fig_sc = go.Figure()
        # Ideal line
        mn = acc_df["Real (s)"].min(); mx = acc_df["Real (s)"].max()
        fig_sc.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
            line=dict(color="#888", dash="dash"), name="Perfect prediction"))
        # Points coloured by team
        for team in acc_df["Team"].unique():
            td = acc_df[acc_df["Team"] == team]
            fig_sc.add_trace(go.Scatter(
                x=td["Real (s)"], y=td["Predicted (s)"],
                mode="markers", name=team,
                marker=dict(size=6, opacity=0.75),
                text=td["Driver"] + " · " + td["Race"],
                hovertemplate="%{text}<br>Real: %{x:.3f}s<br>Pred: %{y:.3f}s<extra></extra>"
            ))
        fig_sc.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(title="Real Lap Time (s)", gridcolor="#222"),
            yaxis=dict(title="Predicted Lap Time (s)", gridcolor="#222"),
            legend=dict(bgcolor="#1a1a2e", font=dict(size=10)),
            height=450, margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── Full detail table ──────────────────────────────
        st.markdown(f"**📋 Detailed Results — {filter_race} · {filter_team}**")
        show = filt[["Race","Driver","Team","Real (s)","Predicted (s)","Error (s)","Abs Error","IsRookie"]].copy()
        show["Driver"] = show.apply(
            lambda r: f"{r['Driver']} 🆕" if r["IsRookie"] else r["Driver"], axis=1)
        show = show.drop(columns="IsRookie").sort_values("Abs Error")

        def hl_err(row):
            e = abs(row["Error (s)"])
            if e <= 0.5:
                return [""] * 7 + ["background-color:#003d00;color:#6bff6b"]
            if e <= 1.5:
                return [""] * 8
            return [""] * 7 + ["background-color:#3d0000;color:#ff8888"]

        st.dataframe(show.style.apply(hl_err, axis=1),
                     hide_index=True, use_container_width=True, height=450)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#444;font-size:12px'>"
    "F1 Qualifying Predictor · 2025 Season · XGBoost + Streamlit · Data: 2019–2024 FastF1"
    "</div>", unsafe_allow_html=True)