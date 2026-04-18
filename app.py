"""
F1 Qualifying Predictor — 2025 Season
Apple-inspired design. Run: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import spearmanr
import os

st.set_page_config(
    page_title="F1 Qualifying Predictor",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Apple design system ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Apple font stack — SF Pro on Apple devices, Helvetica Neue elsewhere */
:root {
  --apple-font: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
  --c-bg:        #f5f5f7;
  --c-surface:   #ffffff;
  --c-surface2:  #f5f5f7;
  --c-text1:     #1d1d1f;
  --c-text2:     #6e6e73;
  --c-text3:     #86868b;
  --c-border:    rgba(0,0,0,0.08);
  --c-blue:      #0071e3;
  --c-blue-h:    #0077ed;
  --c-red:       #ff3b30;
  --c-green:     #34c759;
  --c-divider:   rgba(0,0,0,0.1);
  --r-sm:        8px;
  --r-md:        12px;
  --r-lg:        18px;
  --r-xl:        22px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --c-bg:       #000000;
    --c-surface:  #1c1c1e;
    --c-surface2: #2c2c2e;
    --c-text1:    #f5f5f7;
    --c-text2:    #ababaf;
    --c-text3:    #6e6e73;
    --c-border:   rgba(255,255,255,0.08);
    --c-divider:  rgba(255,255,255,0.1);
  }
}

html, body, [class*="css"], [class*="st-"] {
  font-family: var(--apple-font) !important;
}

.stApp { background: var(--c-bg) !important; }
.stApp > header { display: none !important; }

/* kill streamlit chrome */
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"],
[data-testid="collapsedControl"], section[data-testid="stSidebar"] { display: none !important; }

[data-testid="stAppViewContainer"] { background: var(--c-bg) !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Nav bar ── */
.apple-nav {
  position: sticky; top: 0; z-index: 999;
  background: rgba(245,245,247,0.85);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 0.5px solid var(--c-divider);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px; height: 52px;
}
@media (prefers-color-scheme: dark) {
  .apple-nav { background: rgba(0,0,0,0.85); }
}
.apple-nav-team {
  font-size: 17px; font-weight: 600; color: var(--c-text1);
  letter-spacing: -0.02em;
}
.apple-nav-pills {
  display: flex; gap: 2px;
  background: var(--c-surface2); border-radius: 10px; padding: 3px;
}
.apple-nav-pill {
  font-size: 13px; font-weight: 500; padding: 5px 16px;
  border-radius: 8px; cursor: pointer; border: none;
  color: var(--c-text2); background: transparent;
  transition: all 0.15s ease;
  font-family: var(--apple-font);
}
.apple-nav-pill.active {
  background: var(--c-surface); color: var(--c-text1);
  box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 0.5px 1px rgba(0,0,0,0.06);
}
.apple-nav-change {
  font-size: 13px; color: var(--c-blue); cursor: pointer;
  font-weight: 400; background: none; border: none;
  font-family: var(--apple-font); padding: 0;
}

/* ── Page wrapper ── */
.apple-page { max-width: 980px; margin: 0 auto; padding: 0 22px 80px; }

/* ── Section title ── */
.apple-section-title {
  font-size: 28px; font-weight: 600; color: var(--c-text1);
  letter-spacing: -0.03em; margin: 44px 0 4px;
}
.apple-section-sub {
  font-size: 15px; color: var(--c-text2); margin: 0 0 24px;
  font-weight: 400;
}

/* ── Circuit selector row ── */
.selector-row {
  display: flex; gap: 10px; margin-bottom: 28px; flex-wrap: wrap;
}

/* ── Driver cards (prediction) ── */
.driver-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.driver-card {
  background: var(--c-surface); border-radius: var(--r-xl);
  padding: 24px 24px 20px;
  border: 0.5px solid var(--c-border);
}
.driver-card-eyebrow {
  font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--c-text3); margin-bottom: 6px;
}
.driver-card-name {
  font-size: 40px; font-weight: 700; letter-spacing: -0.04em;
  color: var(--c-text1); margin-bottom: 20px; line-height: 1;
}
.driver-card-time {
  font-size: 28px; font-weight: 600; letter-spacing: -0.03em;
  color: var(--c-text1); margin-bottom: 4px;
}
.driver-card-gap {
  font-size: 14px; color: var(--c-text2); margin-bottom: 16px;
}
.driver-card-stats { display: flex; gap: 20px; }
.driver-stat { }
.driver-stat-label { font-size: 11px; color: var(--c-text3); margin-bottom: 2px; }
.driver-stat-val { font-size: 15px; font-weight: 600; color: var(--c-text1); }

/* ── Gap badge ── */
.gap-badge {
  background: var(--c-surface); border-radius: var(--r-lg);
  border: 0.5px solid var(--c-border);
  padding: 14px 20px; text-align: center; margin-bottom: 28px;
  font-size: 14px; color: var(--c-text2);
}
.gap-badge strong { color: var(--c-text1); }
.gap-badge .faster { color: var(--c-blue); font-weight: 600; }

/* ── Grid table ── */
.grid-section-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--c-text3); margin-bottom: 10px;
}
.grid-table {
  background: var(--c-surface); border-radius: var(--r-lg);
  border: 0.5px solid var(--c-border); overflow: hidden;
}
.grid-header {
  display: grid; grid-template-columns: 36px 10px 52px 1fr 80px;
  padding: 8px 16px; border-bottom: 0.5px solid var(--c-divider);
  font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--c-text3); gap: 8px; align-items: center;
}
.grid-row {
  display: grid; grid-template-columns: 36px 10px 52px 1fr 80px;
  padding: 9px 16px; border-bottom: 0.5px solid var(--c-divider);
  align-items: center; gap: 8px; transition: background 0.1s;
}
.grid-row:last-child { border-bottom: none; }
.grid-row.highlight-d1 { background: rgba(255, 159, 10, 0.06); }
.grid-row.highlight-d2 { background: rgba(0, 113, 227, 0.05); }
.grid-pos { font-size: 13px; font-weight: 500; color: var(--c-text3); }
.grid-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.grid-name { font-size: 14px; font-weight: 600; color: var(--c-text1); letter-spacing: -0.01em; }
.grid-name.d1 { color: #ff9f0a; }
.grid-name.d2 { color: var(--c-blue); }
.grid-name.rookie::after { content: ' N'; font-size: 10px; color: var(--c-text3); vertical-align: super; font-weight: 400; }
.grid-bar-wrap { height: 3px; background: var(--c-surface2); border-radius: 2px; width: 100%; }
.grid-bar { height: 3px; border-radius: 2px; background: var(--c-text3); min-width: 2px; }
.grid-gap { font-size: 13px; color: var(--c-text2); text-align: right; font-variant-numeric: tabular-nums; }
.grid-gap.pole { color: var(--c-text1); font-weight: 600; }

/* ── Metric cards (analyse) ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 24px; }
.metric-card {
  background: var(--c-surface); border-radius: var(--r-lg);
  border: 0.5px solid var(--c-border); padding: 16px 18px;
}
.metric-card-label { font-size: 12px; color: var(--c-text2); margin-bottom: 4px; }
.metric-card-val { font-size: 22px; font-weight: 600; letter-spacing: -0.02em; color: var(--c-text1); }
.metric-card-sub { font-size: 12px; color: var(--c-text3); margin-top: 2px; }

/* ── Chart wrapper ── */
.chart-card {
  background: var(--c-surface); border-radius: var(--r-lg);
  border: 0.5px solid var(--c-border); padding: 20px 20px 10px;
  margin-bottom: 12px;
}
.chart-card-title { font-size: 13px; font-weight: 600; color: var(--c-text1); margin-bottom: 2px; }
.chart-card-sub { font-size: 12px; color: var(--c-text2); margin-bottom: 12px; }

/* ── Analyse sub-nav ── */
.sub-nav {
  display: flex; gap: 0; margin-bottom: 28px;
  border-bottom: 0.5px solid var(--c-divider);
}
.sub-nav-item {
  font-size: 14px; color: var(--c-text2); padding: 10px 0;
  margin-right: 28px; cursor: pointer; border-bottom: 2px solid transparent;
  transition: all 0.15s; font-weight: 400;
}
.sub-nav-item.active { color: var(--c-text1); font-weight: 600; border-bottom-color: var(--c-text1); }

/* ── Accuracy insight box ── */
.insight-card {
  background: var(--c-surface); border-radius: var(--r-lg);
  border: 0.5px solid var(--c-border); padding: 20px 22px;
  margin-bottom: 12px;
}
.insight-title { font-size: 14px; font-weight: 600; color: var(--c-text1); margin-bottom: 8px; }
.insight-body { font-size: 14px; color: var(--c-text2); line-height: 1.6; }
.insight-highlight { color: var(--c-text1); font-weight: 600; }

/* ── Real vs predicted table ── */
.result-table {
  background: var(--c-surface); border-radius: var(--r-lg);
  border: 0.5px solid var(--c-border); overflow: hidden; margin-bottom: 12px;
}
.result-row {
  display: grid; grid-template-columns: 48px 1fr 80px 80px 80px;
  padding: 10px 16px; border-bottom: 0.5px solid var(--c-divider);
  font-size: 13px; align-items: center; gap: 8px;
}
.result-row:last-child { border-bottom: none; }
.result-row.header { background: var(--c-surface2); font-size: 11px; font-weight: 600;
  letter-spacing: 0.04em; text-transform: uppercase; color: var(--c-text3); }
.result-good { color: var(--c-green); }
.result-bad  { color: var(--c-red); }

/* ── Weather toggle ── */
.wx-toggle {
  background: var(--c-surface); border-radius: var(--r-lg);
  border: 0.5px solid var(--c-border); padding: 14px 18px;
  margin-bottom: 22px; cursor: pointer;
  display: flex; align-items: center; justify-content: space-between;
}
.wx-toggle-label { font-size: 14px; color: var(--c-text1); font-weight: 500; }
.wx-toggle-val { font-size: 13px; color: var(--c-text2); }

/* ── Team selector overlay ── */
.team-selector-page { padding: 80px 22px; text-align: center; max-width: 640px; margin: 0 auto; }
.team-selector-headline { font-size: 48px; font-weight: 700; letter-spacing: -0.04em;
  color: var(--c-text1); margin-bottom: 12px; line-height: 1.1; }
.team-selector-sub { font-size: 19px; color: var(--c-text2); margin-bottom: 44px; font-weight: 400; }
.team-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; text-align: left; }
.team-btn {
  background: var(--c-surface); border: 0.5px solid var(--c-border);
  border-radius: var(--r-lg); padding: 16px 20px; cursor: pointer;
  transition: all 0.15s ease; width: 100%;
  font-family: var(--apple-font); text-align: left;
}
.team-btn:hover { border-color: rgba(0,0,0,0.18); box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.team-btn-name { font-size: 15px; font-weight: 600; color: var(--c-text1); margin-bottom: 4px; }
.team-btn-drivers { font-size: 13px; color: var(--c-text2); }

/* ── Streamlit overrides ── */
div[data-testid="stSelectbox"] > div > div {
  background: var(--c-surface) !important;
  border: 0.5px solid var(--c-border) !important;
  border-radius: var(--r-md) !important;
  font-size: 14px !important;
  color: var(--c-text1) !important;
}
div[data-testid="stSlider"] { padding: 0 !important; }
div[data-testid="stCheckbox"] label { font-size: 14px !important; color: var(--c-text1) !important; }
div[data-testid="stMetricValue"] { color: var(--c-text1) !important; }
.stPlotlyChart { border-radius: var(--r-lg) !important; }

/* hide streamlit plot toolbar */
.modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
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

if not os.path.exists("model/xgb_model.pkl"):
    st.markdown("""
    <div style='max-width:480px;margin:120px auto;text-align:center;font-family:var(--apple-font)'>
      <div style='font-size:48px;font-weight:700;color:#1d1d1f;letter-spacing:-0.04em;margin-bottom:12px'>
        F1 Qualifying Predictor
      </div>
      <div style='font-size:17px;color:#6e6e73;margin-bottom:32px'>
        Model not trained yet.
      </div>
      <code style='background:#f5f5f7;border-radius:8px;padding:12px 20px;font-size:14px;color:#1d1d1f'>
        python train_model.py
      </code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

(model, le_team, le_driver, le_event, features,
 team_circuit_avg_delta, team_circuit_avg_abs, driver_skill,
 team_avg_delta, weather_defaults, F1_2025_GRID, ROOKIES_2025, metrics) = load_artifacts()

df, tracks, real_2025 = load_data()

TEAMS_2025       = list(F1_2025_GRID.keys())
ALL_EVENTS       = sorted(real_2025["race"].unique())
DRIVER_TEAM_2025 = {d: t for t, drvs in F1_2025_GRID.items() for d in drvs}

# Team colours (F1 official)
TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "McLaren": "#FF8000", "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2", "Aston Martin": "#229971", "Alpine": "#FF87BC",
    "Williams": "#64C4FF", "Haas F1 Team": "#B6BABD", "RB": "#6692FF",
    "Kick Sauber": "#52E252",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def secs_to_str(s):
    if pd.isna(s) or s <= 0: return "—"
    return f"{int(s//60)}:{s%60:06.3f}"

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
    if wx is None: wx = get_wx(event)
    ti  = tracks[tracks["Event"] == event]
    ti_v = ti.iloc[0].to_dict() if not ti.empty else dict(
        TrackType="Permanent", LapSpeedClass="Medium", DRSZones=2,
        Altitude_m=50, NumCorners=15, CornerDensity=0.003,
        TrackLength_m=5000, AvgCornerSpacing_m=333)
    track_type = str(ti_v.get("TrackType","Permanent"))
    sc         = str(ti_v.get("LapSpeedClass","Medium"))
    si1,si2,sfl,sst = TRAP_MAP.get(sc,(265,250,240,278))
    tca = team_circuit_avg_delta[
        (team_circuit_avg_delta["Team"]==team) &
        (team_circuit_avg_delta["TrackType"]==track_type)]
    tc_avg = tca["TeamCircuitAvgDelta"].values[0] if not tca.empty else 1.0
    ds = driver_skill[driver_skill["Driver"]==driver]
    if ds.empty:
        td = team_avg_delta[team_avg_delta["Team"]==team]
        drv_delta = td["TeamAvgDelta"].values[0] if not td.empty else 1.5
    else:
        drv_delta = ds["DriverAvgDelta"].values[0]
    row = {
        "Team_enc":safe_enc(le_team,team), "Driver_enc":safe_enc(le_driver,driver),
        "Event_enc":safe_enc(le_event,event), "Year":2025,
        "QualiSegment_num":{"Q1":1,"Q2":2,"Q3":3}.get(segment,2),
        "Compound_num":{"SOFT":3,"MEDIUM":2,"HARD":1,"INTER":0,"WET":-1}.get(compound,3),
        "TyreLife":tyre_life, "FreshTyre_int":int(fresh),
        "IsStreet":1 if track_type=="Street" else 0,
        "SpeedClass_num":SPEED_MAP.get(sc,2),
        "DRSZones":ti_v.get("DRSZones",2), "Altitude_m":ti_v.get("Altitude_m",50),
        "NumCorners":ti_v.get("NumCorners",15), "CornerDensity":ti_v.get("CornerDensity",0.003),
        "TrackLength_m":ti_v.get("TrackLength_m",5000),
        "AvgCornerSpacing_m":ti_v.get("AvgCornerSpacing_m",333),
        "AirTemp":wx["AirTemp"], "TrackTemp":wx["TrackTemp"], "Humidity":wx["Humidity"],
        "Pressure":wx["Pressure"], "WindSpeed":wx["WindSpeed"],
        "Rainfall_int":1 if wx["Rainfall"]>0.1 else 0,
        "SpeedI1":si1, "SpeedI2":si2, "SpeedFL":sfl, "SpeedST":sst,
        "TeamCircuitAvgDelta":tc_avg, "DriverAvgDelta":drv_delta,
    }
    return max(0.0, float(model.predict(pd.DataFrame([row])[features])[0]))

def get_pole_2025(event):
    row = real_2025[real_2025["race"]==event]
    return float(row["real_time_seconds"].min()) if not row.empty else None

def predict_absolute(event, team, driver, segment="Q3", compound="SOFT",
                     tyre_life=2, fresh=True, wx=None):
    delta = predict_delta(event, team, driver, segment, compound, tyre_life, fresh, wx)
    pole  = get_pole_2025(event)
    if pole is None:
        push = df[df["IsPushLap"]==1]
        pole = float(push.groupby("Event")["LapTime_sec"].min().get(event,90.0))
    return pole + delta, delta, pole

# Apple-style plotly theme
def apple_layout(fig, height=280):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial",
                  color="#6e6e73", size=12),
        margin=dict(l=0, r=0, t=8, b=0),
        height=height,
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=11, color="#86868b"),
                   linecolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                   zeroline=False, tickfont=dict(size=11, color="#86868b")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(0,0,0,0.1)",
                       font=dict(color="#1d1d1f", size=13)),
    )
    return fig

@st.cache_data
def compute_all_accuracy():
    rows = []
    for _, rr in real_2025.iterrows():
        drv  = rr["driver"]; race = rr["race"]
        team = DRIVER_TEAM_2025.get(drv)
        if not team: continue
        real_t = float(rr["real_time_seconds"])
        pred_a, delta, pole = predict_absolute(race, team, drv)
        rows.append({"Race":race,"Driver":drv,"Team":team,
                     "Real":round(real_t,3),"Predicted":round(pred_a,3),
                     "Error":round(pred_a-real_t,3),
                     "AbsError":round(abs(pred_a-real_t),3),
                     "IsRookie":drv in ROOKIES_2025})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "team" not in st.session_state:    st.session_state.team    = None
if "page" not in st.session_state:    st.session_state.page    = "predict"
if "sub"  not in st.session_state:    st.session_state.sub     = "season"
if "wx_open" not in st.session_state: st.session_state.wx_open = False

# ─────────────────────────────────────────────
# TEAM SELECTION SCREEN
# ─────────────────────────────────────────────
if st.session_state.team is None:
    st.markdown("""
    <div style='max-width:640px;margin:0 auto;padding:80px 22px 40px;text-align:center;font-family:var(--apple-font)'>
      <div style='font-size:13px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;
                  color:#86868b;margin-bottom:12px'>2025 Season</div>
      <div style='font-size:52px;font-weight:700;letter-spacing:-0.04em;color:#1d1d1f;
                  line-height:1.1;margin-bottom:16px'>
        F1 Qualifying<br>Predictor
      </div>
      <div style='font-size:19px;color:#6e6e73;margin-bottom:48px;font-weight:400'>
        Select your team to get started.
      </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, team in enumerate(TEAMS_2025):
        drv1, drv2 = F1_2025_GRID[team]
        col = cols[i % 2]
        tcolor = TEAM_COLORS.get(team, "#86868b")
        with col:
            if st.button(f"**{team}**\n\n{drv1}  ·  {drv2}",
                         key=f"team_{team}", use_container_width=True):
                st.session_state.team = team
                st.rerun()

    st.stop()

# ─────────────────────────────────────────────
# MAIN APP (team selected)
# ─────────────────────────────────────────────
sel_team = st.session_state.team
drv1, drv2 = F1_2025_GRID[sel_team]
tcolor1, tcolor2 = TEAM_COLORS.get(sel_team,"#ff9f0a"), TEAM_COLORS.get(sel_team,"#0071e3")
# Use slightly different shades for two drivers of same team
d1_color = tcolor1
d2_color = "#0071e3"  # always use Apple blue for #2 driver for contrast

page = st.session_state.page

# ── Nav bar ───────────────────────────────────────────────────────────────────
predict_active = "active" if page == "predict" else ""
analyse_active = "active" if page == "analyse" else ""

st.markdown(f"""
<div class='apple-nav'>
  <span class='apple-nav-team'>{sel_team}</span>
  <div class='apple-nav-pills'>
    <button class='apple-nav-pill {predict_active}' id='nav-predict'>Predict</button>
    <button class='apple-nav-pill {analyse_active}' id='nav-analyse'>Analyse</button>
  </div>
  <button class='apple-nav-change' id='nav-change'>Change team</button>
</div>
""", unsafe_allow_html=True)

# Nav routing via hidden selectbox
nav_choice = st.selectbox("nav", ["predict","analyse","change"], label_visibility="collapsed",
                          key="nav_sel", index=["predict","analyse","change"].index(
                              st.session_state.page if st.session_state.page in ["predict","analyse"] else "predict"))

# JS to wire nav buttons → selectbox
st.markdown("""
<script>
(function() {
  function clickOption(val) {
    const sel = window.parent.document.querySelectorAll('[data-testid="stSelectbox"]');
    sel.forEach(s => {
      const opts = s.querySelectorAll('option');
      opts.forEach(o => { if (o.value === val || o.text === val) {
        s.querySelector('select').value = o.value;
        s.querySelector('select').dispatchEvent(new Event('change',{bubbles:true}));
      }});
    });
  }
  setTimeout(() => {
    const p = window.parent.document.getElementById('nav-predict');
    const a = window.parent.document.getElementById('nav-analyse');
    const c = window.parent.document.getElementById('nav-change');
    if (p) p.onclick = () => clickOption('predict');
    if (a) a.onclick = () => clickOption('analyse');
    if (c) c.onclick = () => clickOption('change');
  }, 400);
})();
</script>
""", unsafe_allow_html=True)

if nav_choice == "change":
    st.session_state.team = None
    st.session_state.page = "predict"
    st.rerun()
elif nav_choice != st.session_state.page:
    st.session_state.page = nav_choice
    st.rerun()

# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────
if page == "predict":
    st.markdown("<div class='apple-page'>", unsafe_allow_html=True)

    # ── Selectors ─────────────────────────────
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        sel_event = st.selectbox("Grand Prix", ALL_EVENTS, key="pred_event",
                                  label_visibility="collapsed")
    with c2:
        seg = st.selectbox("Segment", ["Q3","Q2","Q1"], key="pred_seg",
                            label_visibility="collapsed")
    with c3:
        cmp = st.selectbox("Compound", ["SOFT","MEDIUM","HARD"], key="pred_cmp",
                            label_visibility="collapsed")

    # Weather toggle (collapsed by default)
    hist_wx = get_wx(sel_event)
    wx_label = f"🌡 {hist_wx['AirTemp']:.0f}°C · {hist_wx['TrackTemp']:.0f}°C track · {hist_wx['Humidity']:.0f}% humidity"

    with st.expander(f"Weather conditions — {wx_label}", expanded=False):
        wc1,wc2 = st.columns(2)
        with wc1:
            air_t   = st.slider("Air temp (°C)",   10,45, int(hist_wx["AirTemp"]),   key="wx_air")
            track_t = st.slider("Track temp (°C)", 15,60, int(hist_wx["TrackTemp"]), key="wx_trk")
        with wc2:
            hum  = st.slider("Humidity (%)",   10,100, int(hist_wx["Humidity"]),  key="wx_hum")
            wind = st.slider("Wind (m/s)",     0,15,   int(hist_wx["WindSpeed"]), key="wx_wnd")
            rain = st.checkbox("Rainfall",     value=hist_wx["Rainfall"]>0.1,     key="wx_rain")
        wx = dict(AirTemp=air_t, TrackTemp=track_t, Humidity=hum,
                  WindSpeed=wind, Pressure=hist_wx["Pressure"],
                  Rainfall=1.0 if rain else 0.0)
    else:
        wx = hist_wx

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Predictions ───────────────────────────
    abs1,delta1,_ = predict_absolute(sel_event,sel_team,drv1,seg,cmp,2,True,wx)
    abs2,delta2,_ = predict_absolute(sel_event,sel_team,drv2,seg,cmp,2,True,wx)
    faster = drv1 if abs1 < abs2 else drv2
    gap_bt = abs(abs1-abs2)

    pole_2025 = get_pole_2025(sel_event)
    ti_info   = tracks[tracks["Event"]==sel_event]
    circuit_type = ti_info["TrackType"].values[0] if not ti_info.empty else ""
    circuit_speed = ti_info["LapSpeedClass"].values[0] if not ti_info.empty else ""

    st.markdown(f"""
    <div style='margin-bottom:6px'>
      <span style='font-size:13px;color:var(--c-text3)'>{circuit_type} · {circuit_speed}</span>
      {f"<span style='font-size:13px;color:var(--c-text3);margin-left:12px'>2025 pole: <strong style='color:var(--c-text1)'>{secs_to_str(pole_2025)}</strong></span>" if pole_2025 else ""}
    </div>
    """, unsafe_allow_html=True)

    # Driver cards
    r1 = "N" if drv1 in ROOKIES_2025 else ""
    r2 = "N" if drv2 in ROOKIES_2025 else ""
    st.markdown(f"""
    <div class='driver-cards'>
      <div class='driver-card'>
        <div class='driver-card-eyebrow'>{sel_team}</div>
        <div class='driver-card-name' style='color:{d1_color}'>{drv1}{' <span style="font-size:14px;color:var(--c-text3)">Rookie</span>' if drv1 in ROOKIES_2025 else ''}</div>
        <div class='driver-card-time'>{secs_to_str(abs1)}</div>
        <div class='driver-card-gap'>+{delta1:.3f}s to pole</div>
        <div class='driver-card-stats'>
          <div class='driver-stat'>
            <div class='driver-stat-label'>Compound</div>
            <div class='driver-stat-val'>{cmp}</div>
          </div>
          <div class='driver-stat'>
            <div class='driver-stat-label'>Session</div>
            <div class='driver-stat-val'>{seg}</div>
          </div>
        </div>
      </div>
      <div class='driver-card'>
        <div class='driver-card-eyebrow'>{sel_team}</div>
        <div class='driver-card-name' style='color:{d2_color}'>{drv2}{' <span style="font-size:14px;color:var(--c-text3)">Rookie</span>' if drv2 in ROOKIES_2025 else ''}</div>
        <div class='driver-card-time'>{secs_to_str(abs2)}</div>
        <div class='driver-card-gap'>+{delta2:.3f}s to pole</div>
        <div class='driver-card-stats'>
          <div class='driver-stat'>
            <div class='driver-stat-label'>Compound</div>
            <div class='driver-stat-val'>{cmp}</div>
          </div>
          <div class='driver-stat'>
            <div class='driver-stat-label'>Session</div>
            <div class='driver-stat-val'>{seg}</div>
          </div>
        </div>
      </div>
    </div>
    <div class='gap-badge'>
      Intra-team gap <strong>{gap_bt:.3f}s</strong> &nbsp;·&nbsp;
      <span class='faster'>{faster}</span> predicted faster this circuit
    </div>
    """, unsafe_allow_html=True)

    # ── Full grid ─────────────────────────────
    @st.cache_data
    def full_grid(event):
        rows = []
        w = get_wx(event)
        for team, drvs in F1_2025_GRID.items():
            for d in drvs:
                dlt  = predict_delta(event,team,d,"Q3","SOFT",2,True,w)
                pole = get_pole_2025(event) or 90.0
                rows.append({"Team":team,"Driver":d,"Delta":dlt,
                             "AbsTime":pole+dlt,"IsRookie":d in ROOKIES_2025,
                             "Color":TEAM_COLORS.get(team,"#86868b")})
        gdf = pd.DataFrame(rows).sort_values("Delta").reset_index(drop=True)
        gdf["Position"] = range(1,len(gdf)+1)
        gdf["TimeStr"]  = gdf["AbsTime"].apply(secs_to_str)
        gdf["GapStr"]   = gdf["Delta"].apply(lambda x:"POLE" if x<0.001 else f"+{x:.3f}s")
        return gdf

    gdf = full_grid(sel_event)
    max_delta = gdf["Delta"].max()

    st.markdown("<div class='grid-section-label'>Predicted starting grid — Q3 · SOFT</div>",
                unsafe_allow_html=True)

    rows_html = ""
    for _, r in gdf.iterrows():
        d   = r["Driver"]
        is_d1 = (d==drv1 and r["Team"]==sel_team)
        is_d2 = (d==drv2 and r["Team"]==sel_team)
        hl  = "highlight-d1" if is_d1 else ("highlight-d2" if is_d2 else "")
        nc  = ("d1" if is_d1 else ("d2" if is_d2 else ""))
        rk  = " rookie" if r["IsRookie"] else ""
        bar = round(r["Delta"]/max_delta*100) if max_delta>0 else 0
        clr = r["Color"]
        pos_disp = "P1" if r["Position"]==1 else f"P{r['Position']}"
        gap_cls = "pole" if r["GapStr"]=="POLE" else ""
        rows_html += f"""
        <div class='grid-row {hl}'>
          <span class='grid-pos'>{pos_disp}</span>
          <span class='grid-dot' style='background:{clr}'></span>
          <span class='grid-name {nc}{rk}'>{d}</span>
          <div class='grid-bar-wrap'><div class='grid-bar' style='width:{bar}%;background:{clr}'></div></div>
          <span class='grid-gap {gap_cls}'>{r["GapStr"]}</span>
        </div>"""

    st.markdown(f"""
    <div class='grid-table'>
      <div class='grid-header'>
        <span></span><span></span><span>Driver</span>
        <span></span><span style='text-align:right'>Gap</span>
      </div>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)

    # Real 2025 comparison (compact)
    real_race = real_2025[real_2025["race"]==sel_event]
    if not real_race.empty:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='grid-section-label'>Real 2025 result vs prediction</div>",
                    unsafe_allow_html=True)
        r_html = ""
        for drv, pred_abs in [(drv1,abs1),(drv2,abs2)]:
            rr = real_race[real_race["driver"]==drv]
            if not rr.empty:
                rt   = float(rr["real_time_seconds"].values[0])
                err  = pred_abs - rt
                ec   = "result-good" if abs(err)<=0.5 else ("result-bad" if abs(err)>1.5 else "")
                clr  = d1_color if drv==drv1 else d2_color
                r_html += f"""
                <div class='result-row'>
                  <span style='font-size:14px;font-weight:600;color:{clr}'>{drv}</span>
                  <span style='color:var(--c-text1)'>{secs_to_str(rt)}</span>
                  <span style='color:var(--c-text2)'>{secs_to_str(pred_abs)}</span>
                  <span class='{ec}'>{err:+.3f}s</span>
                </div>"""
        st.markdown(f"""
        <div class='result-table'>
          <div class='result-row header'>
            <span>Driver</span><span>Real</span><span>Predicted</span><span>Error</span>
          </div>
          {r_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ANALYSE
# ─────────────────────────────────────────────
elif page == "analyse":
    st.markdown("<div class='apple-page'>", unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Sub-nav
    sub = st.session_state.sub
    sub_choice = st.radio("sub", ["Season","R&D","Accuracy"], horizontal=True,
                           label_visibility="collapsed", key="sub_radio",
                           index=["Season","R&D","Accuracy"].index(
                               sub.capitalize() if sub.capitalize() in ["Season","R&d","Accuracy"]
                               else "Season"
                           ))
    st.session_state.sub = sub_choice.lower()
    sub = st.session_state.sub

    # ────────────────────────────────
    # SUB: SEASON
    # ────────────────────────────────
    if sub == "season":
        st.markdown(f"<div class='apple-section-title'>{sel_team}</div>", unsafe_allow_html=True)
        st.markdown("<div class='apple-section-sub'>Predicted performance across the 2025 calendar.</div>",
                    unsafe_allow_html=True)

        # Build season data
        season_rows = []
        for ev in ALL_EVENTS:
            w = get_wx(ev)
            for drv in [drv1,drv2]:
                dlt = predict_delta(ev,sel_team,drv,"Q3","SOFT",2,True,w)
                season_rows.append({"Race":ev,"Driver":drv,"Delta":round(dlt,3)})
        sdf = pd.DataFrame(season_rows)

        s1 = sdf[sdf["Driver"]==drv1]["Delta"]
        s2 = sdf[sdf["Driver"]==drv2]["Delta"]
        pivot = sdf.pivot(index="Race",columns="Driver",values="Delta").reset_index()
        if drv1 in pivot.columns and drv2 in pivot.columns:
            pivot["H2H"] = pivot[drv1] - pivot[drv2]
            w1 = (pivot["H2H"]<0).sum(); w2 = (pivot["H2H"]>0).sum()
            avg_gap = pivot["H2H"].abs().mean()
        else:
            w1=w2=0; avg_gap=0.0

        # Metric cards
        st.markdown(f"""
        <div class='metric-grid'>
          <div class='metric-card'>
            <div class='metric-card-label' style='color:{d1_color}'>{drv1} avg gap</div>
            <div class='metric-card-val'>+{s1.mean():.3f}s</div>
            <div class='metric-card-sub'>to pole</div>
          </div>
          <div class='metric-card'>
            <div class='metric-card-label' style='color:{d2_color}'>{drv2} avg gap</div>
            <div class='metric-card-val'>+{s2.mean():.3f}s</div>
            <div class='metric-card-sub'>to pole</div>
          </div>
          <div class='metric-card'>
            <div class='metric-card-label'>{drv1} wins</div>
            <div class='metric-card-val'>{w1} / {len(pivot)}</div>
            <div class='metric-card-sub'>circuits</div>
          </div>
          <div class='metric-card'>
            <div class='metric-card-label'>Intra-team avg</div>
            <div class='metric-card-val'>{avg_gap:.3f}s</div>
            <div class='metric-card-sub'>gap</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gap to pole — both drivers
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card-title'>Predicted gap to pole — all circuits</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card-sub'>Q3 · SOFT · historical weather</div>", unsafe_allow_html=True)
        fig_s = go.Figure()
        for drv,col in [(drv1,d1_color),(drv2,d2_color)]:
            sd = sdf[sdf["Driver"]==drv]
            fig_s.add_trace(go.Scatter(
                x=sd["Race"], y=sd["Delta"], mode="lines+markers", name=drv,
                line=dict(color=col,width=2), marker=dict(size=5,color=col),
                hovertemplate="<b>%{x}</b><br>+%{y:.3f}s<extra></extra>"))
        apple_layout(fig_s, 260)
        fig_s.update_layout(xaxis=dict(tickangle=-40, tickfont=dict(size=10)))
        fig_s.update_layout(legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        # H2H delta
        if drv1 in pivot.columns and drv2 in pivot.columns:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-card-title'>Head-to-head gap per circuit</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-card-sub'>Positive = {drv2} faster · Negative = {drv1} faster</div>", unsafe_allow_html=True)
            ps = pivot.sort_values("H2H")
            fig_h = go.Figure(go.Bar(
                x=ps["Race"], y=ps["H2H"],
                marker_color=[d2_color if v>0 else d1_color for v in ps["H2H"]],
                hovertemplate="<b>%{x}</b><br>%{y:+.3f}s<extra></extra>"))
            fig_h.add_hline(y=0, line_color="rgba(0,0,0,0.12)", line_width=1)
            apple_layout(fig_h, 240)
            fig_h.update_layout(xaxis=dict(tickangle=-40,tickfont=dict(size=10)))
            st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

        # Circuit type breakdown — two columns
        ti_m = tracks[["Event","TrackType","LapSpeedClass"]].copy()
        sf   = sdf.merge(ti_m, left_on="Race", right_on="Event", how="left")

        col_a, col_b = st.columns(2)
        for col, drv, color in [(col_a,drv1,d1_color),(col_b,drv2,d2_color)]:
            with col:
                sd = sf[sf["Driver"]==drv]
                for grp, title, sub_t in [
                    ("TrackType","Circuit type","Street vs Permanent"),
                    ("LapSpeedClass","Speed class","Slow / Medium / Fast")
                ]:
                    by = sd.groupby(grp)["Delta"].mean().reset_index()
                    st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chart-card-title' style='color:{color}'>{drv} · {title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chart-card-sub'>{sub_t}</div>", unsafe_allow_html=True)
                    fig_t = go.Figure(go.Bar(
                        x=by[grp], y=by["Delta"], marker_color=color,
                        hovertemplate="%{x}<br>+%{y:.3f}s<extra></extra>"))
                    apple_layout(fig_t, 180)
                    st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar":False})
                    st.markdown("</div>", unsafe_allow_html=True)

        # Historical trend
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card-title'>Historical team trend vs field</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card-sub'>Average qualifying lap time 2019–2024</div>", unsafe_allow_html=True)
        push = df[df["IsPushLap"]==1]
        yt = push[push["Team"]==sel_team].groupby("Year")["LapTime_sec"].mean().reset_index()
        yf = push.groupby("Year")["LapTime_sec"].mean().reset_index()
        fig_ht = go.Figure()
        fig_ht.add_trace(go.Scatter(x=yt["Year"],y=yt["LapTime_sec"],mode="lines+markers",
            name=sel_team,line=dict(color=d1_color,width=2.5),marker=dict(size=6)))
        fig_ht.add_trace(go.Scatter(x=yf["Year"],y=yf["LapTime_sec"],mode="lines+markers",
            name="Field average",line=dict(color="rgba(110,110,115,0.5)",width=1.5,dash="dot"),
            marker=dict(size=5)))
        apple_layout(fig_ht, 220)
        fig_ht.update_layout(legend=dict(orientation="h",y=1.1,x=0))
        st.plotly_chart(fig_ht, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ────────────────────────────────
    # SUB: R&D
    # ────────────────────────────────
    elif sub == "r&d":
        st.markdown("<div class='apple-section-title'>R&D Simulator</div>", unsafe_allow_html=True)
        st.markdown("<div class='apple-section-sub'>Simulate high vs low downforce setups across the calendar.</div>",
                    unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-card'>
          <div class='insight-title'>How this works</div>
          <div class='insight-body'>
            Without raw telemetry, we use speed trap readings as a downforce proxy.
            High downforce = more drag = slower straights. We shift trap speeds and measure
            the resulting predicted gap-to-pole delta for each driver.
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("<div style='font-size:14px;font-weight:600;color:var(--c-text1);margin-bottom:8px'>High downforce</div>", unsafe_allow_html=True)
            st.caption("Monaco · Singapore · Hungary")
            df_pen  = st.slider("Straight speed loss (km/h)", 5,25,12,key="hd")
        with c2:
            st.markdown("<div style='font-size:14px;font-weight:600;color:var(--c-text1);margin-bottom:8px'>Low downforce</div>", unsafe_allow_html=True)
            st.caption("Monza · Baku · Las Vegas")
            df_gain = st.slider("Straight speed gain (km/h)", 5,25,12,key="ld")

        sim_rows = []
        for _, tr in tracks.iterrows():
            ev = tr["Event"]
            if ev not in df["Event"].values: continue
            sc = str(tr.get("LapSpeedClass","Medium"))
            si1,si2,sfl,sst = TRAP_MAP.get(sc,(265,250,240,278))
            w = get_wx(ev)
            for drv in [drv1,drv2]:
                ds   = driver_skill[driver_skill["Driver"]==drv]
                dd   = ds["DriverAvgDelta"].values[0] if not ds.empty else 1.5
                tca2 = team_circuit_avg_delta[
                    (team_circuit_avg_delta["Team"]==sel_team) &
                    (team_circuit_avg_delta["TrackType"]==str(tr.get("TrackType","Permanent")))]
                tc2  = tca2["TeamCircuitAvgDelta"].values[0] if not tca2.empty else 1.0

                def pred_t(s1,s2,sf,ss,_drv=drv,_dd=dd,_tc=tc2,_ev=ev,_tr=tr,_w=w,_sc=sc):
                    row={
                        "Team_enc":safe_enc(le_team,sel_team),"Driver_enc":safe_enc(le_driver,_drv),
                        "Event_enc":safe_enc(le_event,_ev),"Year":2025,
                        "QualiSegment_num":3,"Compound_num":3,"TyreLife":2,"FreshTyre_int":1,
                        "IsStreet":1 if str(_tr.get("TrackType","Permanent"))=="Street" else 0,
                        "SpeedClass_num":SPEED_MAP.get(_sc,2),
                        "DRSZones":_tr.get("DRSZones",2),"Altitude_m":_tr.get("Altitude_m",50),
                        "NumCorners":_tr.get("NumCorners",15),"CornerDensity":_tr.get("CornerDensity",0.003),
                        "TrackLength_m":_tr.get("TrackLength_m",5000),"AvgCornerSpacing_m":_tr.get("AvgCornerSpacing_m",333),
                        "AirTemp":_w["AirTemp"],"TrackTemp":_w["TrackTemp"],"Humidity":_w["Humidity"],
                        "Pressure":_w["Pressure"],"WindSpeed":_w["WindSpeed"],
                        "Rainfall_int":1 if _w["Rainfall"]>0.1 else 0,
                        "SpeedI1":s1,"SpeedI2":s2,"SpeedFL":sf,"SpeedST":ss,
                        "TeamCircuitAvgDelta":_tc,"DriverAvgDelta":_dd,
                    }
                    return max(0.0,float(model.predict(pd.DataFrame([row])[features])[0]))

                base = pred_t(si1,si2,sfl,sst)
                hi   = pred_t(si1-df_pen, si2-df_pen, sfl,          sst-df_pen)
                lo   = pred_t(si1+df_gain,si2+df_gain,sfl+df_gain,  sst+df_gain)
                best_s = min(base,hi,lo)
                sim_rows.append({"Circuit":ev,"Driver":drv,"SpeedClass":sc,
                                 "Hi":round(hi-base,3),"Lo":round(lo-base,3),
                                 "Best":"High DF" if best_s==hi else ("Low DF" if best_s==lo else "Balanced")})

        sim_df = pd.DataFrame(sim_rows)

        for drv,color in [(drv1,d1_color),(drv2,d2_color)]:
            sd = sim_df[sim_df["Driver"]==drv]
            hi_w = (sd["Hi"]<sd["Lo"]).sum(); lo_w = (sd["Lo"]<sd["Hi"]).sum()
            rec  = "High downforce" if sd["Hi"].mean()<sd["Lo"].mean() else "Low downforce"

            st.markdown(f"""
            <div class='metric-grid' style='grid-template-columns:1fr 1fr 1fr;margin-top:16px'>
              <div class='metric-card'>
                <div class='metric-card-label' style='color:{color}'>{drv}</div>
                <div class='metric-card-val'>{rec}</div>
                <div class='metric-card-sub'>season recommendation</div>
              </div>
              <div class='metric-card'>
                <div class='metric-card-label'>High DF wins</div>
                <div class='metric-card-val'>{hi_w}</div>
                <div class='metric-card-sub'>circuits</div>
              </div>
              <div class='metric-card'>
                <div class='metric-card-label'>Low DF wins</div>
                <div class='metric-card-val'>{lo_w}</div>
                <div class='metric-card-sub'>circuits</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='chart-card-title' style='color:{color}'>{drv} — setup delta per circuit</div>", unsafe_allow_html=True)
            st.markdown("<div class='chart-card-sub'>Negative = faster than baseline</div>", unsafe_allow_html=True)
            fig_rd = go.Figure()
            fig_rd.add_trace(go.Bar(name="High DF",x=sd["Circuit"],y=sd["Hi"],
                marker_color=color,opacity=0.85))
            fig_rd.add_trace(go.Bar(name="Low DF",x=sd["Circuit"],y=sd["Lo"],
                marker_color="rgba(110,110,115,0.4)"))
            fig_rd.add_hline(y=0,line_color="rgba(0,0,0,0.12)",line_width=1)
            apple_layout(fig_rd, 220)
            fig_rd.update_layout(barmode="group",
                xaxis=dict(tickangle=-40,tickfont=dict(size=10)),
                legend=dict(orientation="h",y=1.1,x=0))
            st.plotly_chart(fig_rd, use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

    # ────────────────────────────────
    # SUB: ACCURACY
    # ────────────────────────────────
    elif sub == "accuracy":
        st.markdown("<div class='apple-section-title'>Model accuracy</div>", unsafe_allow_html=True)
        st.markdown("<div class='apple-section-sub'>How well predictions matched the real 2025 qualifying results.</div>",
                    unsafe_allow_html=True)

        acc = compute_all_accuracy()
        team_acc = acc[acc["Team"]==sel_team].copy()

        full_mae  = acc["AbsError"].mean()
        team_mae  = team_acc["AbsError"].mean() if not team_acc.empty else 0
        within1   = (acc["AbsError"]<=1.0).mean()*100
        bias      = acc["Error"].mean()

        real_avg_delta = (real_2025
            .assign(Pole=real_2025.groupby("race")["real_time_seconds"].transform("min"))
            .assign(RealDelta=lambda x: x["real_time_seconds"]-x["Pole"])
            .groupby("driver")["RealDelta"].mean().reset_index()
            .rename(columns={"driver":"Driver","RealDelta":"Real2025AvgDelta"}))
        drivers_both = [d for d in real_avg_delta["Driver"] if d in driver_skill["Driver"].values]
        pred_d = driver_skill[driver_skill["Driver"].isin(drivers_both)][["Driver","DriverAvgDelta"]]
        rm = real_avg_delta[real_avg_delta["Driver"].isin(drivers_both)].merge(pred_d,on="Driver")
        spear_r, spear_p = spearmanr(rm["DriverAvgDelta"], rm["Real2025AvgDelta"])

        # Key metric cards
        st.markdown(f"""
        <div class='metric-grid'>
          <div class='metric-card'>
            <div class='metric-card-label'>Ranking accuracy</div>
            <div class='metric-card-val'>{spear_r:.2f}</div>
            <div class='metric-card-sub'>Spearman r · p={spear_p:.3f}</div>
          </div>
          <div class='metric-card'>
            <div class='metric-card-label'>MAE — all drivers</div>
            <div class='metric-card-val'>{full_mae:.2f}s</div>
            <div class='metric-card-sub'>avg absolute error</div>
          </div>
          <div class='metric-card'>
            <div class='metric-card-label'>MAE — {sel_team}</div>
            <div class='metric-card-val'>{team_mae:.2f}s</div>
            <div class='metric-card-sub'>avg absolute error</div>
          </div>
          <div class='metric-card'>
            <div class='metric-card-label'>Within 1 second</div>
            <div class='metric-card-val'>{within1:.0f}%</div>
            <div class='metric-card-sub'>of all predictions</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Interpretation
        rank_word = "strong" if spear_r>0.7 else ("moderate" if spear_r>0.5 else "weak")
        bias_note = (f"over-predicts by {bias:.2f}s on average" if bias>0.2
                     else f"under-predicts by {abs(bias):.2f}s on average" if bias<-0.2
                     else "well-calibrated — no systematic bias")
        st.markdown(f"""
        <div class='insight-card'>
          <div class='insight-title'>How to read these numbers</div>
          <div class='insight-body'>
            The most important metric is <span class='insight-highlight'>ranking accuracy (Spearman r = {spear_r:.2f})</span>.
            This measures whether the model correctly orders drivers relative to each other — not whether absolute times are exact.
            A score of {spear_r:.2f} represents {rank_word} rank correlation; random guessing would score 0.0.<br><br>
            The model is <span class='insight-highlight'>{bias_note}</span>.
            An MAE of {full_mae:.2f}s means predictions land within {full_mae:.2f} seconds of reality on average,
            across {len(acc)} driver–race combinations in the full 2025 season.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Ranking scatter
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card-title'>Ranking accuracy — predicted vs real delta from pole</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card-sub'>Points on the diagonal = perfect prediction. Coloured = your team.</div>", unsafe_allow_html=True)
        rm["Team"] = rm["Driver"].map(DRIVER_TEAM_2025)
        mn2=rm["Real2025AvgDelta"].min(); mx2=rm["Real2025AvgDelta"].max()
        fig_rk = go.Figure()
        fig_rk.add_trace(go.Scatter(x=[mn2,mx2],y=[mn2,mx2],mode="lines",
            line=dict(color="rgba(110,110,115,0.3)",dash="dot",width=1.5),showlegend=False))
        others = rm[rm["Team"]!=sel_team]
        fig_rk.add_trace(go.Scatter(x=others["Real2025AvgDelta"],y=others["DriverAvgDelta"],
            mode="markers+text", text=others["Driver"], textposition="top center",
            textfont=dict(size=10,color="#86868b"),
            marker=dict(size=8,color="rgba(110,110,115,0.25)",line=dict(color="rgba(110,110,115,0.5)",width=0.5)),
            hovertemplate="%{text}<br>Real: %{x:.3f}s<br>Pred: %{y:.3f}s<extra></extra>",showlegend=False))
        team_r = rm[rm["Team"]==sel_team]
        if not team_r.empty:
            fig_rk.add_trace(go.Scatter(x=team_r["Real2025AvgDelta"],y=team_r["DriverAvgDelta"],
                mode="markers+text", text=team_r["Driver"], textposition="top center",
                textfont=dict(size=11,color=d1_color,weight=600),
                marker=dict(size=11,color=d1_color,line=dict(color="white",width=1.5)),
                hovertemplate="%{text}<br>Real: %{x:.3f}s<br>Pred: %{y:.3f}s<extra></extra>",showlegend=False))
        apple_layout(fig_rk, 340)
        fig_rk.update_layout(xaxis=dict(title="Real 2025 avg gap to pole (s)"),
                              yaxis=dict(title="Predicted avg gap (s)"))
        st.plotly_chart(fig_rk, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        # MAE per race
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card-title'>Accuracy per race</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-card-sub'>Average absolute error per Grand Prix</div>", unsafe_allow_html=True)
        race_mae = acc.groupby("Race")["AbsError"].mean().reset_index().sort_values("AbsError")
        fig_rm = go.Figure(go.Bar(
            x=race_mae["Race"],y=race_mae["AbsError"],
            marker_color=["#ff3b30" if v>2 else ("#ff9f0a" if v>1 else "#34c759") for v in race_mae["AbsError"]],
            hovertemplate="%{x}<br>MAE: %{y:.3f}s<extra></extra>"))
        apple_layout(fig_rm, 220)
        fig_rm.update_layout(xaxis=dict(tickangle=-40,tickfont=dict(size=10)))
        st.plotly_chart(fig_rm, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Team detail table
        if not team_acc.empty:
            st.markdown(f"<div class='grid-section-label' style='margin-top:24px'>{sel_team} — all 2025 predictions</div>",
                        unsafe_allow_html=True)
            rows_html = ""
            for _, r in team_acc.sort_values(["Race","Driver"]).iterrows():
                ec = "result-good" if r["AbsError"]<=0.5 else ("result-bad" if r["AbsError"]>1.5 else "")
                clr = d1_color if r["Driver"]==drv1 else d2_color
                rows_html += f"""
                <div class='result-row'>
                  <span style='font-size:13px;font-weight:600;color:{clr}'>{r['Driver']}</span>
                  <span style='color:var(--c-text2);font-size:12px'>{r['Race'].replace(' Grand Prix','')}</span>
                  <span style='color:var(--c-text1);font-variant-numeric:tabular-nums'>{secs_to_str(r['Real'])}</span>
                  <span style='color:var(--c-text2);font-variant-numeric:tabular-nums'>{secs_to_str(r['Predicted'])}</span>
                  <span class='{ec};font-variant-numeric:tabular-nums'>{r['Error']:+.3f}s</span>
                </div>"""
            st.markdown(f"""
            <div class='result-table'>
              <div class='result-row header'>
                <span>Driver</span><span>Race</span><span>Real</span><span>Predicted</span><span>Error</span>
              </div>
              {rows_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)