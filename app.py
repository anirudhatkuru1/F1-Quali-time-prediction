"""
F1 Qualifying Predictor — 2025 Season
Single-team perspective dashboard.
Run: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
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
  h1  { color: #ffffff !important; font-weight: 900 !important; letter-spacing: 2px; }
  h2, h3 { color: #cccccc !important; }
  .section-header {
    border-left: 4px solid #e10600; padding-left: 12px;
    margin: 20px 0 14px 0; color: #ffffff;
    font-size: 18px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
  }
  div[data-testid="stMetricValue"] { color: #e10600 !important; font-size: 24px !important; }
  div[data-testid="stMetricLabel"] { color: #aaaaaa !important; }
  .driver-card {
    background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
    padding: 16px; text-align: center; margin-bottom: 8px;
  }
  .driver-name { font-size: 28px; font-weight: 900; color: #ffffff; letter-spacing: 3px; }
  .driver-team { font-size: 13px; color: #888; letter-spacing: 1px; }
  .faster-tag  { color: #4CAF50; font-size: 13px; font-weight: 700; }
  .slower-tag  { color: #e10600; font-size: 13px; font-weight: 700; }
  .rookie-note { color: #ff8c00; font-size: 11px; }
  .wx-note { background:#1a1a2e; border:1px solid #333; border-radius:5px;
             padding:8px 12px; color:#888; font-size:12px; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    return (
        joblib.load("model/xgb_model.pkl"),
        joblib.load("model/le_team.pkl"),
        joblib.load("model/le_driver.pkl"),
        joblib.load("model/le_event.pkl"),
        joblib.load("model/features.pkl"),
        joblib.load("model/team_circuit_avg_delta.pkl"),
        joblib.load("model/team_circuit_avg_abs.pkl"),
        joblib.load("model/driver_skill.pkl"),
        joblib.load("model/team_avg_delta.pkl"),
        joblib.load("model/weather_defaults.pkl"),
        joblib.load("model/f1_2025_grid.pkl"),
        joblib.load("model/rookies_2025.pkl"),
        joblib.load("model/metrics.pkl"),
    )

@st.cache_data
def load_data():
    return (
        pd.read_csv("data/data.csv"),
        pd.read_csv("data/tracks.csv"),
        pd.read_csv("data/real_lap_time_2025.csv"),
    )

# ─────────────────────────────────────────────
# MODEL CHECK
# ─────────────────────────────────────────────
if not os.path.exists("model/xgb_model.pkl"):
    st.markdown("# 🏎️ F1 Qualifying Predictor")
    st.error("Model not trained yet. Run `python train_model.py` first.")
    st.code("python train_model.py", language="bash")
    st.stop()

(model, le_team, le_driver, le_event, features,
 team_circuit_avg_delta, team_circuit_avg_abs, driver_skill,
 team_avg_delta, weather_defaults, F1_2025_GRID, ROOKIES_2025, metrics) = load_model_artifacts()

df, tracks, real_2025 = load_data()

TEAMS_2025       = list(F1_2025_GRID.keys())
ALL_EVENTS       = sorted(real_2025["race"].unique())
DRIVER_TEAM_2025 = {d: t for t, drvs in F1_2025_GRID.items() for d in drvs}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def secs_to_str(s):
    if pd.isna(s) or s <= 0: return "N/A"
    m = int(s // 60)
    return f"{m}:{s % 60:06.3f}"

def get_wx(event):
    row = weather_defaults[weather_defaults["Event"] == event]
    if row.empty:
        return dict(AirTemp=25.0, TrackTemp=38.0, Humidity=50.0,
                    WindSpeed=2.0, Pressure=1013.0, Rainfall=0.0)
    r = row.iloc[0]
    return dict(AirTemp=r.AirTemp, TrackTemp=r.TrackTemp, Humidity=r.Humidity,
                WindSpeed=r.WindSpeed, Pressure=r.Pressure, Rainfall=r.Rainfall)

def safe_enc(le, val):
    return int(le.transform([val])[0]) if val in le.classes_ else 0

TRAP_MAP  = {"Slow":(240,220,210,250),"Medium":(270,255,245,285),"Fast":(300,285,275,315)}
SPEED_MAP = {"Slow":1,"Medium":2,"Fast":3}

def predict_delta(event, team, driver, segment="Q3", compound="SOFT",
                  tyre_life=2, fresh=True, wx=None):
    """Returns predicted DELTA FROM POLE (seconds)."""
    if wx is None:
        wx = get_wx(event)

    ti = tracks[tracks["Event"] == event]
    if ti.empty:
        ti_v = dict(TrackType="Permanent", LapSpeedClass="Medium", DRSZones=2,
                    Altitude_m=50, NumCorners=15, CornerDensity=0.003,
                    TrackLength_m=5000, AvgCornerSpacing_m=333)
    else:
        ti_v = ti.iloc[0].to_dict()

    track_type = str(ti_v.get("TrackType","Permanent"))
    sc         = str(ti_v.get("LapSpeedClass","Medium"))
    si1,si2,sfl,sst = TRAP_MAP.get(sc,(265,250,240,278))

    tca = team_circuit_avg_delta[
        (team_circuit_avg_delta["Team"] == team) &
        (team_circuit_avg_delta["TrackType"] == track_type)
    ]
    tc_avg_delta = tca["TeamCircuitAvgDelta"].values[0] if not tca.empty else 1.0

    ds  = driver_skill[driver_skill["Driver"] == driver]
    if ds.empty:
        td  = team_avg_delta[team_avg_delta["Team"] == team]
        drv_delta = td["TeamAvgDelta"].values[0] if not td.empty else 1.5
    else:
        drv_delta = ds["DriverAvgDelta"].values[0]

    seg_map  = {"Q1":1,"Q2":2,"Q3":3}
    cmp_map  = {"SOFT":3,"MEDIUM":2,"HARD":1,"INTER":0,"WET":-1}

    row = {
        "Team_enc":           safe_enc(le_team,   team),
        "Driver_enc":         safe_enc(le_driver, driver),
        "Event_enc":          safe_enc(le_event,  event),
        "Year":               2025,
        "QualiSegment_num":   seg_map.get(segment, 2),
        "Compound_num":       cmp_map.get(compound, 3),
        "TyreLife":           tyre_life,
        "FreshTyre_int":      int(fresh),
        "IsStreet":           1 if track_type == "Street" else 0,
        "SpeedClass_num":     SPEED_MAP.get(sc, 2),
        "DRSZones":           ti_v.get("DRSZones",2),
        "Altitude_m":         ti_v.get("Altitude_m",50),
        "NumCorners":         ti_v.get("NumCorners",15),
        "CornerDensity":      ti_v.get("CornerDensity",0.003),
        "TrackLength_m":      ti_v.get("TrackLength_m",5000),
        "AvgCornerSpacing_m": ti_v.get("AvgCornerSpacing_m",333),
        "AirTemp":            wx["AirTemp"],
        "TrackTemp":          wx["TrackTemp"],
        "Humidity":           wx["Humidity"],
        "Pressure":           wx["Pressure"],
        "WindSpeed":          wx["WindSpeed"],
        "Rainfall_int":       1 if wx["Rainfall"] > 0.1 else 0,
        "SpeedI1":            si1, "SpeedI2": si2,
        "SpeedFL":            sfl, "SpeedST": sst,
        "TeamCircuitAvgDelta": tc_avg_delta,
        "DriverAvgDelta":      drv_delta,
    }
    delta = float(model.predict(pd.DataFrame([row])[features])[0])
    return max(0.0, delta)   # delta can't be negative

def get_pole_2025(event):
    """Return real 2025 pole time for a circuit, or None."""
    row = real_2025[real_2025["race"] == event]
    if row.empty: return None
    return float(row["real_time_seconds"].min())

def predict_absolute(event, team, driver, segment="Q3", compound="SOFT",
                     tyre_life=2, fresh=True, wx=None):
    """Delta + real 2025 pole = best possible absolute prediction."""
    delta = predict_delta(event, team, driver, segment, compound, tyre_life, fresh, wx)
    pole  = get_pole_2025(event)
    if pole is None:
        # Fallback: use historical pole + delta
        push = df[df["IsPushLap"]==1]
        best_hist = push.groupby("Event")["LapTime_sec"].min()
        pole = best_hist.get(event, 90.0)
    return pole + delta, delta, pole

# ─────────────────────────────────────────────
# SIDEBAR — TEAM SELECTION (single entry point)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ F1 QUALI PREDICTOR")
    st.markdown("*2025 Season — Team View*")
    st.markdown("---")

    st.markdown("### 🏎️ Select Your Team")
    sel_team = st.selectbox("Team", TEAMS_2025, label_visibility="collapsed")
    drv1, drv2 = F1_2025_GRID[sel_team]

    # Show both drivers
    st.markdown(f"""
    <div style='background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:12px;margin:8px 0'>
      <div style='color:#888;font-size:11px;letter-spacing:1px'>2025 DRIVERS</div>
      <div style='color:#e10600;font-size:22px;font-weight:900;margin-top:6px'>{drv1}</div>
      <div style='color:#0066cc;font-size:22px;font-weight:900'>{drv2}{"  🆕" if drv2 in ROOKIES_2025 else ""}</div>
    </div>
    """, unsafe_allow_html=True)
    if drv1 in ROOKIES_2025: st.caption(f"🆕 {drv1}: rookie — estimate from team avg")
    if drv2 in ROOKIES_2025: st.caption(f"🆕 {drv2}: rookie — estimate from team avg")

    st.markdown("---")
    st.markdown("### 🏁 Circuit")
    sel_event = st.selectbox("Grand Prix", ALL_EVENTS)

    ti_s = tracks[tracks["Event"] == sel_event]
    if not ti_s.empty:
        t = ti_s.iloc[0]
        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:10px;margin-top:6px'>
          <div style='color:#888;font-size:11px;letter-spacing:1px'>CIRCUIT</div>
          <div style='color:#fff;margin-top:4px'>🏟️ {t['TrackType']} · {t['LapSpeedClass']}</div>
          <div style='color:#aaa;font-size:12px'>📏 {t['TrackLength_m']/1000:.3f} km · {int(t['NumCorners'])} corners</div>
          <div style='color:#aaa;font-size:12px'>🔵 {int(t['DRSZones'])} DRS zones · 📍 {int(t['Altitude_m'])}m</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Session Setup")
    quali_seg = st.selectbox("Quali Segment", ["Q3","Q2","Q1"])
    compound  = st.selectbox("Tyre Compound", ["SOFT","MEDIUM","HARD"])
    tyre_life = st.slider("Tyre Age (laps)", 1, 10, 2)
    fresh     = st.checkbox("Fresh Tyre", value=True)

    st.markdown("---")
    st.markdown("### 🌤️ Weather")
    hist_wx     = get_wx(sel_event)
    use_hist_wx = st.checkbox("Use historical circuit weather", value=True)
    if use_hist_wx:
        wx = hist_wx
        st.markdown(f"""<div class='wx-note'>
          Historical avg · {sel_event}<br>
          🌡️ Air {hist_wx['AirTemp']:.1f}°C · Track {hist_wx['TrackTemp']:.1f}°C<br>
          💧 {hist_wx['Humidity']:.0f}% humidity · 💨 {hist_wx['WindSpeed']:.1f} m/s
        </div>""", unsafe_allow_html=True)
    else:
        air_t   = st.slider("Air Temp (°C)",   10, 45,  int(hist_wx["AirTemp"]))
        track_t = st.slider("Track Temp (°C)", 15, 60,  int(hist_wx["TrackTemp"]))
        hum     = st.slider("Humidity (%)",    10, 100, int(hist_wx["Humidity"]))
        wind    = st.slider("Wind (m/s)",      0.0, 15.0, float(round(hist_wx["WindSpeed"],1)))
        rain    = st.checkbox("Rainfall", value=hist_wx["Rainfall"] > 0.1)
        wx = dict(AirTemp=air_t, TrackTemp=track_t, Humidity=hum,
                  WindSpeed=wind, Pressure=hist_wx["Pressure"],
                  Rainfall=1.0 if rain else 0.0)

    st.markdown("---")
    st.markdown(f"**MAE (2024 holdout):** `{metrics['mae']}s`")
    st.markdown(f"**R²:** `{metrics['r2']}`")
    st.caption("Model predicts gap-to-pole; adds real 2025 pole for absolute time.")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"# 🏎️ {sel_team.upper()} — 2025 QUALIFYING ANALYSIS")
pole_2025 = get_pole_2025(sel_event)
if pole_2025:
    st.markdown(f"*{sel_event} · 2025 pole: **{secs_to_str(pole_2025)}***")
else:
    st.markdown(f"*{sel_event}*")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Race Prediction",
    "📊 Season Analysis",
    "🛠️ R&D Simulator",
    "✅ 2025 Accuracy",
])

# ════════════════════════════════════════════
# TAB 1 — RACE PREDICTION (two-driver view)
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Predicted Qualifying — Side by Side</div>', unsafe_allow_html=True)

    # Predict both team drivers
    abs1, delta1, pole1 = predict_absolute(sel_event, sel_team, drv1, quali_seg, compound, tyre_life, fresh, wx)
    abs2, delta2, pole2 = predict_absolute(sel_event, sel_team, drv2, quali_seg, compound, tyre_life, fresh, wx)

    # Predict full field for grid position
    @st.cache_data
    def full_grid(event, _wx_key):
        rows = []
        for team, drvs in F1_2025_GRID.items():
            w = get_wx(event)
            for d in drvs:
                dlt  = predict_delta(event, team, d, "Q3","SOFT",2,True,w)
                pole = get_pole_2025(event)
                if pole is None:
                    pole = df[df["IsPushLap"]==1].groupby("Event")["LapTime_sec"].min().get(event,90)
                rows.append({"Team":team,"Driver":d,
                             "Delta":dlt,"AbsTime":pole+dlt,
                             "IsRookie":d in ROOKIES_2025})
        gdf = pd.DataFrame(rows).sort_values("Delta").reset_index(drop=True)
        gdf["Position"] = range(1, len(gdf)+1)
        gdf["TimeStr"]  = gdf["AbsTime"].apply(secs_to_str)
        gdf["GapStr"]   = gdf["Delta"].apply(lambda x: "POLE" if x<0.001 else f"+{x:.3f}s")
        return gdf

    grid_df = full_grid(sel_event, str(sel_event))

    pos1 = grid_df[(grid_df["Driver"]==drv1)&(grid_df["Team"]==sel_team)]["Position"].values
    pos2 = grid_df[(grid_df["Driver"]==drv2)&(grid_df["Team"]==sel_team)]["Position"].values
    pos1 = pos1[0] if len(pos1) else "?"
    pos2 = pos2[0] if len(pos2) else "?"

    # ── Driver cards ──────────────────────────────────────────────
    c1, mid, c2 = st.columns([5, 1, 5])

    with c1:
        faster_tag1 = "⚡ FASTER" if abs1 < abs2 else ""
        st.markdown(f"""
        <div class='driver-card' style='border-color:#e10600'>
          <div class='driver-name' style='color:#e10600'>{drv1}</div>
          <div class='driver-team'>{sel_team}</div>
          {"<div class='rookie-note'>🆕 Rookie</div>" if drv1 in ROOKIES_2025 else ""}
        </div>
        """, unsafe_allow_html=True)
        m1a, m1b, m1c = st.columns(3)
        m1a.metric("Predicted Time", secs_to_str(abs1))
        m1b.metric("Gap to Pole",    f"+{delta1:.3f}s")
        m1c.metric("Grid Position",  f"P{pos1}")

    with mid:
        st.markdown("<div style='text-align:center;color:#555;font-size:24px;margin-top:40px'>VS</div>",
                    unsafe_allow_html=True)

    with c2:
        faster_tag2 = "⚡ FASTER" if abs2 < abs1 else ""
        st.markdown(f"""
        <div class='driver-card' style='border-color:#0066cc'>
          <div class='driver-name' style='color:#0066cc'>{drv2}</div>
          <div class='driver-team'>{sel_team}</div>
          {"<div class='rookie-note'>🆕 Rookie</div>" if drv2 in ROOKIES_2025 else ""}
        </div>
        """, unsafe_allow_html=True)
        m2a, m2b, m2c = st.columns(3)
        m2a.metric("Predicted Time", secs_to_str(abs2))
        m2b.metric("Gap to Pole",    f"+{delta2:.3f}s")
        m2c.metric("Grid Position",  f"P{pos2}")

    # Intra-team gap
    gap_between = abs(abs1 - abs2)
    faster_drv  = drv1 if abs1 < abs2 else drv2
    st.markdown(f"""
    <div style='text-align:center;background:#1a1a2e;border:1px solid #333;border-radius:6px;
                padding:10px;margin:12px 0;color:#fff'>
      Intra-team gap: <b>{gap_between:.3f}s</b> · <span style='color:#4CAF50'>{faster_drv} predicted faster</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Full grid ────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**📋 Full 2025 Grid — Predicted Q3**")
        disp = grid_df[["Position","Driver","Team","TimeStr","GapStr","IsRookie"]].copy()
        disp["Driver"] = disp.apply(lambda r: f"{r['Driver']} 🆕" if r["IsRookie"] else r["Driver"], axis=1)
        disp = disp.drop(columns="IsRookie")
        disp.columns = ["Pos","Driver","Team","Lap Time","Gap"]

        def hl_grid(row):
            if row["Team"] == sel_team:
                c = "#3d0000" if "e10600" else "#001a3d"
                if drv1 in row["Driver"]:
                    return [f"background-color:#3d0000;color:#ff8888"] * len(row)
                elif drv2 in row["Driver"]:
                    return [f"background-color:#001a3d;color:#6699ff"] * len(row)
            return [""] * len(row)

        st.dataframe(disp.style.apply(hl_grid, axis=1),
                     hide_index=True, use_container_width=True, height=540)

    with col_r:
        st.markdown("**🏁 Gap to Pole Visualisation**")
        colors = []
        for _, r in grid_df.iterrows():
            if r["Team"] == sel_team and r["Driver"] == drv1:
                colors.append("#e10600")
            elif r["Team"] == sel_team and r["Driver"] == drv2:
                colors.append("#0066cc")
            else:
                colors.append("#333")

        fig = go.Figure(go.Bar(
            x=grid_df["Delta"],
            y=grid_df["Driver"] + " (" + grid_df["Team"].str.split().str[-1] + ")",
            orientation="h", marker_color=colors,
            text=grid_df["GapStr"], textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(title="Gap to Pole (s)", gridcolor="#222"),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            height=560, margin=dict(l=0, r=80, t=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # ── Real 2025 result if available ───────────────────────────
    real_race = real_2025[real_2025["race"] == sel_event]
    if not real_race.empty:
        st.markdown("---")
        st.markdown("**📡 Real 2025 Result vs Prediction**")
        real_pole = real_race["real_time_seconds"].min()
        ra,rb = st.columns(2)
        for col, drv, pred_abs in [(ra,drv1,abs1),(rb,drv2,abs2)]:
            real_row = real_race[real_race["driver"]==drv]
            with col:
                if not real_row.empty:
                    real_t = float(real_row["real_time_seconds"].values[0])
                    err    = pred_abs - real_t
                    col.metric(f"{drv} — Real",      secs_to_str(real_t))
                    col.metric(f"{drv} — Predicted", secs_to_str(pred_abs))
                    col.metric(f"{drv} — Error",     f"{err:+.3f}s",
                               delta_color="inverse")
                else:
                    col.info(f"{drv} — no 2025 data for this race")


# ════════════════════════════════════════════
# TAB 2 — SEASON ANALYSIS
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Season Analysis — Both Drivers</div>', unsafe_allow_html=True)

    # ── Predicted delta across all 2025 circuits ─────────────────
    season_rows = []
    for ev in ALL_EVENTS:
        w = get_wx(ev)
        for drv, col_tag in [(drv1,"#e10600"),(drv2,"#0066cc")]:
            dlt = predict_delta(ev, sel_team, drv, "Q3","SOFT",2,True,w)
            season_rows.append({"Race":ev,"Driver":drv,"Delta":round(dlt,3)})
    season_df = pd.DataFrame(season_rows)

    # Summary metrics per driver
    s1 = season_df[season_df["Driver"]==drv1]["Delta"]
    s2 = season_df[season_df["Driver"]==drv2]["Delta"]

    st.markdown("**📊 Predicted Gap to Pole — All 2025 Circuits**")

    ma,mb = st.columns(2)
    with ma:
        st.markdown(f"<div style='color:#e10600;font-size:20px;font-weight:900;text-align:center'>{drv1}</div>",
                    unsafe_allow_html=True)
        ca1,ca2,ca3 = st.columns(3)
        ca1.metric("Avg Gap to Pole", f"+{s1.mean():.3f}s")
        ca2.metric("Best Circuit",    season_df[season_df["Driver"]==drv1].nsmallest(1,"Delta")["Race"].values[0].split()[0])
        ca3.metric("Worst Circuit",   season_df[season_df["Driver"]==drv1].nlargest(1,"Delta")["Race"].values[0].split()[0])
    with mb:
        st.markdown(f"<div style='color:#0066cc;font-size:20px;font-weight:900;text-align:center'>{drv2}</div>",
                    unsafe_allow_html=True)
        cb1,cb2,cb3 = st.columns(3)
        cb1.metric("Avg Gap to Pole", f"+{s2.mean():.3f}s")
        cb2.metric("Best Circuit",    season_df[season_df["Driver"]==drv2].nsmallest(1,"Delta")["Race"].values[0].split()[0])
        cb3.metric("Worst Circuit",   season_df[season_df["Driver"]==drv2].nlargest(1,"Delta")["Race"].values[0].split()[0])

    fig_season = go.Figure()
    for drv, color in [(drv1,"#e10600"),(drv2,"#0066cc")]:
        sd = season_df[season_df["Driver"]==drv]
        fig_season.add_trace(go.Scatter(
            x=sd["Race"], y=sd["Delta"], mode="lines+markers",
            name=drv, line=dict(color=color, width=2), marker=dict(size=7)
        ))
    fig_season.update_layout(
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc", family="Titillium Web"),
        xaxis=dict(tickangle=-45, gridcolor="#222"),
        yaxis=dict(title="Predicted Gap to Pole (s)", gridcolor="#222"),
        legend=dict(bgcolor="#1a1a2e"),
        height=360, margin=dict(l=0,r=0,t=10,b=120))
    st.plotly_chart(fig_season, use_container_width=True)

    # ── Head-to-head delta chart ────────────────────────────────
    st.markdown("**⚔️ Head-to-Head Gap per Circuit  *(positive = {} faster)***".format(drv2))
    pivot = season_df.pivot(index="Race", columns="Driver", values="Delta").reset_index()
    if drv1 in pivot.columns and drv2 in pivot.columns:
        pivot["H2H"] = pivot[drv1] - pivot[drv2]
        pivot_s = pivot.sort_values("H2H")
        fig_h2h = go.Figure(go.Bar(
            x=pivot_s["Race"], y=pivot_s["H2H"],
            marker_color=["#0066cc" if v>0 else "#e10600" for v in pivot_s["H2H"]],
            text=pivot_s["H2H"].apply(lambda x: f"{x:+.3f}s"),
            textposition="outside",
        ))
        fig_h2h.add_hline(y=0, line_color="#888", line_dash="dash")
        fig_h2h.update_layout(
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc", family="Titillium Web"),
            xaxis=dict(tickangle=-45, gridcolor="#222"),
            yaxis=dict(title=f"[{drv1}] − [{drv2}] gap (s)", gridcolor="#222"),
            height=320, margin=dict(l=0,r=0,t=10,b=120))
        st.plotly_chart(fig_h2h, use_container_width=True)

        wins1 = (pivot["H2H"] < 0).sum()
        wins2 = (pivot["H2H"] > 0).sum()
        avg   = pivot["H2H"].abs().mean()
        w1,w2,w3 = st.columns(3)
        w1.metric(f"{drv1} faster in",   f"{wins1} / {len(pivot)} races")
        w2.metric(f"{drv2} faster in",   f"{wins2} / {len(pivot)} races")
        w3.metric("Avg intra-team gap",  f"{avg:.3f}s")

    # ── Circuit type breakdown ───────────────────────────────────
    st.markdown("---")
    st.markdown("**🏁 Performance by Circuit Type**")

    ti_merged = tracks[["Event","TrackType","LapSpeedClass"]].copy()
    season_full = season_df.merge(ti_merged, left_on="Race", right_on="Event", how="left")

    ca, cb = st.columns(2)
    for col, drv, color in [(ca, drv1, "#e10600"), (cb, drv2, "#0066cc")]:
        with col:
            st.markdown(f"<div style='color:{color};font-weight:700;font-size:16px'>{drv}</div>",
                        unsafe_allow_html=True)
            sd = season_full[season_full["Driver"]==drv]
            by_type = sd.groupby("TrackType")["Delta"].mean().reset_index()
            by_speed = sd.groupby("LapSpeedClass")["Delta"].mean().reset_index()

            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(x=by_type["TrackType"], y=by_type["Delta"],
                marker_color=color, name="Circuit Type"))
            fig_t.update_layout(
                paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
                font=dict(color="#ccc",family="Titillium Web"),
                yaxis=dict(title="Avg gap to pole (s)",gridcolor="#222"),
                height=220, margin=dict(l=0,r=0,t=10,b=30))
            st.plotly_chart(fig_t, use_container_width=True)

            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(x=by_speed["LapSpeedClass"], y=by_speed["Delta"],
                marker_color=color, name="Speed Class"))
            fig_s.update_layout(
                paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
                font=dict(color="#ccc",family="Titillium Web"),
                yaxis=dict(title="Avg gap to pole (s)",gridcolor="#222"),
                height=220, margin=dict(l=0,r=0,t=10,b=30))
            st.plotly_chart(fig_s, use_container_width=True)

    # ── Historical team vs field ─────────────────────────────────
    st.markdown("---")
    st.markdown("**📈 Team Historical Qualifying Trend vs Field (2019–2024)**")
    push = df[df["IsPushLap"]==1]
    team_hist = push[push["Team"]==sel_team]
    yr_team  = team_hist.groupby("Year")["LapTime_sec"].mean().reset_index()
    yr_field = push.groupby("Year")["LapTime_sec"].mean().reset_index()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=yr_team["Year"], y=yr_team["LapTime_sec"],
        mode="lines+markers", name=sel_team,
        line=dict(color="#e10600",width=3), marker=dict(size=8)))
    fig_hist.add_trace(go.Scatter(x=yr_field["Year"], y=yr_field["LapTime_sec"],
        mode="lines+markers", name="Field Average",
        line=dict(color="#555",width=2,dash="dash"), marker=dict(size=6)))
    fig_hist.update_layout(
        paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc",family="Titillium Web"),
        xaxis=dict(gridcolor="#222"),
        yaxis=dict(title="Avg Lap (s)",gridcolor="#222"),
        legend=dict(bgcolor="#1a1a2e"),
        height=280, margin=dict(l=0,r=0,t=10,b=40))
    st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — R&D SIMULATOR
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">R&D Setup Simulator</div>', unsafe_allow_html=True)
    st.info("Simulates high vs low downforce setups by shifting speed trap values. "
            "Shows predicted gap-to-pole delta for both drivers across all circuits.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⬆️ High Downforce")
        st.markdown("*Monaco · Singapore · Hungary*")
        df_pen  = st.slider("Straight speed loss (km/h)", 5, 25, 12, key="hd")
    with c2:
        st.markdown("### ⬇️ Low Downforce")
        st.markdown("*Monza · Baku · Las Vegas*")
        df_gain = st.slider("Straight speed gain (km/h)", 5, 25, 12, key="ld")

    st.markdown("---")

    sim_rows = []
    for _, tr in tracks.iterrows():
        ev = tr["Event"]
        if ev not in df["Event"].values: continue
        sc = str(tr.get("LapSpeedClass","Medium"))
        si1,si2,sfl,sst = TRAP_MAP.get(sc,(265,250,240,278))
        w = get_wx(ev)

        for drv in [drv1, drv2]:
            def pred_trap(s1,s2,sf,ss,d=drv):
                inp = predict_delta.__wrapped__ if hasattr(predict_delta,'__wrapped__') else None
                seg_map  = {"Q1":1,"Q2":2,"Q3":3}
                cmp_map  = {"SOFT":3,"MEDIUM":2,"HARD":1,"INTER":0,"WET":-1}
                tc = team_circuit_avg_delta[
                    (team_circuit_avg_delta["Team"]==sel_team) &
                    (team_circuit_avg_delta["TrackType"]==str(tr.get("TrackType","Permanent")))
                ]
                tc_avg = tc["TeamCircuitAvgDelta"].values[0] if not tc.empty else 1.0
                ds = driver_skill[driver_skill["Driver"]==d]
                dd = ds["DriverAvgDelta"].values[0] if not ds.empty else 1.5
                row = {
                    "Team_enc":safe_enc(le_team,sel_team),"Driver_enc":safe_enc(le_driver,d),
                    "Event_enc":safe_enc(le_event,ev),"Year":2025,
                    "QualiSegment_num":3,"Compound_num":3,"TyreLife":2,"FreshTyre_int":1,
                    "IsStreet":1 if str(tr.get("TrackType","Permanent"))=="Street" else 0,
                    "SpeedClass_num":SPEED_MAP.get(sc,2),
                    "DRSZones":tr.get("DRSZones",2),"Altitude_m":tr.get("Altitude_m",50),
                    "NumCorners":tr.get("NumCorners",15),"CornerDensity":tr.get("CornerDensity",0.003),
                    "TrackLength_m":tr.get("TrackLength_m",5000),"AvgCornerSpacing_m":tr.get("AvgCornerSpacing_m",333),
                    "AirTemp":w["AirTemp"],"TrackTemp":w["TrackTemp"],"Humidity":w["Humidity"],
                    "Pressure":w["Pressure"],"WindSpeed":w["WindSpeed"],
                    "Rainfall_int":1 if w["Rainfall"]>0.1 else 0,
                    "SpeedI1":s1,"SpeedI2":s2,"SpeedFL":sf,"SpeedST":ss,
                    "TeamCircuitAvgDelta":tc_avg,"DriverAvgDelta":dd,
                }
                return max(0.0, float(model.predict(pd.DataFrame([row])[features])[0]))

            base = pred_trap(si1,si2,sfl,sst)
            hi   = pred_trap(si1-df_pen, si2-df_pen, sfl,         sst-df_pen)
            lo   = pred_trap(si1+df_gain,si2+df_gain,sfl+df_gain, sst+df_gain)
            best_s = min(base,hi,lo)
            rec  = "High DF" if best_s==hi else ("Low DF" if best_s==lo else "Balanced")
            sim_rows.append({"Circuit":ev,"Driver":drv,
                             "TrackType":tr["TrackType"],"SpeedClass":sc,
                             "Base":round(base,3),"Hi":round(hi,3),"Lo":round(lo,3),
                             "Hi Delta":round(hi-base,3),"Lo Delta":round(lo-base,3),
                             "Best Setup":rec})

    sim_df = pd.DataFrame(sim_rows)

    for drv, color in [(drv1,"#e10600"),(drv2,"#0066cc")]:
        sd = sim_df[sim_df["Driver"]==drv]
        st.markdown(f"<div style='color:{color};font-weight:700;font-size:16px;margin-top:12px'>{drv}</div>",
                    unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Hi DF wins", (sd["Hi Delta"]<sd["Lo Delta"]).sum())
        m2.metric("Lo DF wins", (sd["Lo Delta"]<sd["Hi Delta"]).sum())
        m3.metric("Recommendation",
                  "High DF" if sd["Hi Delta"].mean()<sd["Lo Delta"].mean() else "Low DF")

        fig_rd = go.Figure()
        fig_rd.add_trace(go.Bar(name="High DF",x=sd["Circuit"],y=sd["Hi Delta"],marker_color=color,opacity=0.9))
        fig_rd.add_trace(go.Bar(name="Low DF", x=sd["Circuit"],y=sd["Lo Delta"],marker_color="#888",opacity=0.7))
        fig_rd.add_hline(y=0, line_color="#888", line_dash="dash")
        fig_rd.update_layout(barmode="group",
            paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc",family="Titillium Web"),
            xaxis=dict(tickangle=-45,gridcolor="#222"),
            yaxis=dict(title="Delta vs baseline (s)",gridcolor="#222"),
            legend=dict(bgcolor="#1a1a2e"),
            height=300, margin=dict(l=0,r=0,t=10,b=110))
        st.plotly_chart(fig_rd, use_container_width=True)


# ════════════════════════════════════════════
# TAB 4 — 2025 ACCURACY
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">2025 Accuracy — Your Team vs Full Field</div>', unsafe_allow_html=True)

    @st.cache_data
    def compute_accuracy():
        rows = []
        for _, rr in real_2025.iterrows():
            drv  = rr["driver"]
            race = rr["race"]
            team = DRIVER_TEAM_2025.get(drv)
            if not team: continue
            real_t  = float(rr["real_time_seconds"])
            pred_a, delta, pole = predict_absolute(race, team, drv, "Q3","SOFT",2,True)
            rows.append({
                "Race":race,"Driver":drv,"Team":team,
                "Real (s)":round(real_t,3),
                "Predicted (s)":round(pred_a,3),
                "Error (s)":round(pred_a-real_t,3),
                "Abs Error":round(abs(pred_a-real_t),3),
                "IsRookie":drv in ROOKIES_2025,
                "IsYourTeam":team==sel_team,
            })
        return pd.DataFrame(rows)

    acc = compute_accuracy()
    if acc.empty:
        st.warning("No matching 2025 data found.")
        st.stop()

    # ── Overall vs your team metrics ─────────────────────────────
    full_mae  = acc["Abs Error"].mean()
    team_acc  = acc[acc["IsYourTeam"]]
    team_mae  = team_acc["Abs Error"].mean()

    st.markdown("#### Overall vs Your Team")
    col_ov, col_tm = st.columns(2)
    with col_ov:
        st.markdown("**🌍 Full Field**")
        ss_res = ((acc["Error (s)"]**2).sum())
        ss_tot = ((acc["Real (s)"] - acc["Real (s)"].mean())**2).sum()
        r2_2025 = 1 - ss_res/ss_tot
        fa,fb,fc,fd = st.columns(4)
        fa.metric("MAE",         f"{full_mae:.3f}s")
        fb.metric("R²",          f"{r2_2025:.4f}")
        fc.metric("Within 1s",   f"{(acc['Abs Error']<=1.0).mean()*100:.1f}%")
        fd.metric("Within 2s",   f"{(acc['Abs Error']<=2.0).mean()*100:.1f}%")
    with col_tm:
        st.markdown(f"**🏎️ {sel_team}**")
        if not team_acc.empty:
            ta,tb,tc_,td_ = st.columns(4)
            ta.metric("MAE",       f"{team_mae:.3f}s")
            tb.metric("Within 1s", f"{(team_acc['Abs Error']<=1.0).mean()*100:.1f}%")
            tc_.metric(f"{drv1} MAE", f"{team_acc[team_acc['Driver']==drv1]['Abs Error'].mean():.3f}s")
            td_.metric(f"{drv2} MAE", f"{team_acc[team_acc['Driver']==drv2]['Abs Error'].mean():.3f}s")

    bias = acc["Error (s)"].mean()
    if abs(bias) > 0.2:
        st.warning(f"⚠️ Model has a systematic bias of **{bias:+.3f}s** — it tends to {'over' if bias>0 else 'under'}-predict.")
    else:
        st.success(f"✅ Model is well-calibrated. Bias: {bias:+.3f}s")

    st.markdown("---")

    # ── Error distribution ────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Error Distribution**")
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(x=acc["Error (s)"], nbinsx=40,
            marker_color="#e10600", opacity=0.8, name="All drivers"))
        if not team_acc.empty:
            fig_err.add_trace(go.Histogram(x=team_acc["Error (s)"], nbinsx=20,
                marker_color="#0066cc", opacity=0.9, name=sel_team))
        fig_err.add_vline(x=0, line_color="#fff", line_dash="dash")
        fig_err.update_layout(
            barmode="overlay", paper_bgcolor="#0f0f0f", plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc",family="Titillium Web"),
            xaxis=dict(title="Error (s)",gridcolor="#222"),
            yaxis=dict(title="Count",gridcolor="#222"),
            legend=dict(bgcolor="#1a1a2e"),
            height=300, margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig_err, use_container_width=True)

    with col_b:
        st.markdown("**Predicted vs Real**")
        mn=acc["Real (s)"].min(); mx=acc["Real (s)"].max()
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",
            line=dict(color="#555",dash="dash"),name="Perfect"))
        other = acc[~acc["IsYourTeam"]]
        fig_sc.add_trace(go.Scatter(x=other["Real (s)"],y=other["Predicted (s)"],
            mode="markers",marker=dict(size=4,color="#444",opacity=0.6),name="Other teams",
            text=other["Driver"]+" · "+other["Race"],
            hovertemplate="%{text}<br>Real: %{x:.3f}s<br>Pred: %{y:.3f}s<extra></extra>"))
        if not team_acc.empty:
            fig_sc.add_trace(go.Scatter(x=team_acc["Real (s)"],y=team_acc["Predicted (s)"],
                mode="markers",marker=dict(size=8,color="#e10600",opacity=0.9),name=sel_team,
                text=team_acc["Driver"]+" · "+team_acc["Race"],
                hovertemplate="%{text}<br>Real: %{x:.3f}s<br>Pred: %{y:.3f}s<extra></extra>"))
        fig_sc.update_layout(
            paper_bgcolor="#0f0f0f",plot_bgcolor="#0f0f0f",
            font=dict(color="#ccc",family="Titillium Web"),
            xaxis=dict(title="Real (s)",gridcolor="#222"),
            yaxis=dict(title="Predicted (s)",gridcolor="#222"),
            legend=dict(bgcolor="#1a1a2e"),
            height=300, margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── MAE per race ──────────────────────────────────────────────
    st.markdown("**🏁 MAE per Race**")
    race_mae = acc.groupby("Race")["Abs Error"].mean().reset_index().sort_values("Abs Error")
    fig_rm = px.bar(race_mae, x="Race", y="Abs Error",
        color="Abs Error", color_continuous_scale=["#4CAF50","#ffcc00","#e10600"])
    fig_rm.update_layout(paper_bgcolor="#0f0f0f",plot_bgcolor="#0f0f0f",
        font=dict(color="#ccc",family="Titillium Web"),
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45,gridcolor="#222"),yaxis=dict(gridcolor="#222"),
        height=300, margin=dict(l=0,r=0,t=10,b=110))
    st.plotly_chart(fig_rm, use_container_width=True)

    # ── Your team detail table ────────────────────────────────────
    st.markdown(f"**📋 {sel_team} — Detailed Predictions vs Real 2025**")
    if not team_acc.empty:
        show = team_acc[["Race","Driver","Real (s)","Predicted (s)","Error (s)","Abs Error"]].copy()
        show = show.sort_values(["Race","Driver"])

        def hl_err(row):
            styles = [""] * len(row)
            e = abs(row["Abs Error"])
            styles[-1] = ("background-color:#003d00;color:#6bff6b" if e <= 0.5
                          else "background-color:#3d0000;color:#ff8888" if e > 1.5
                          else "")
            return styles

        st.dataframe(show.style.apply(hl_err, axis=1),
                     hide_index=True, use_container_width=True, height=500)
    else:
        st.info("No 2025 accuracy data available for this team.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#444;font-size:12px'>"
    f"F1 Qualifying Predictor · {sel_team} · 2025 Season · "
    "XGBoost trained on 2019–2024 FastF1 best laps · Target: gap to pole"
    "</div>", unsafe_allow_html=True)