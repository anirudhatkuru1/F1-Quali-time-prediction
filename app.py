"""
F1 Qualifying Predictor — 2025 Season
Run: python -m streamlit run app.py
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

# ── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & base ── */
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --f: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
  --bg:   #0a0a0a;
  --sur:  #161618;
  --sur2: #1c1c1e;
  --t1:   #f5f5f7;
  --t2:   #ababaf;
  --t3:   #6e6e73;
  --bdr:  rgba(255,255,255,0.08);
  --div:  rgba(255,255,255,0.08);
  --blue: #2997ff;
  --red:  #ff453a;
  --grn:  #32d74b;
  --amber:#ff9f0a;
  --c-divider: rgba(255,255,255,0.08);
  --r-sm: 8px; --r-md: 12px; --r-lg: 18px; --r-xl: 22px;
  color-scheme: dark !important;
}
html, body, [class*="css"], [class*="st-"] {
  font-family: var(--f) !important;
  color-scheme: light !important;
}
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
  background: #0a0a0a !important; color: #f5f5f7 !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
.stApp > header, #MainMenu, footer,
.stDeployButton, [data-testid="stToolbar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

/* ── Landing: full-viewport dark hero ── */
.hero {
  position: relative;
  width: 100vw; height: 100vh; min-height: 600px;
  background: #0a0a0a;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.hero-bg {
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 70%, #1a1a1a 0%, #050505 100%);
}
/* Subtle red glow bottom center — like a car exhaust/atmosphere */
.hero-glow {
  position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
  width: 600px; height: 300px;
  background: radial-gradient(ellipse at center bottom, rgba(220,0,0,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero-content {
  position: relative; z-index: 2;
  display: flex; flex-direction: column;
  height: 100%;
  padding: 0 48px;
}
.hero-top {
  display: flex; align-items: center; justify-content: space-between;
  height: 64px; flex-shrink: 0;
  border-bottom: 0.5px solid rgba(255,255,255,0.07);
}
.hero-brand {
  font-size: 13px; font-weight: 500; letter-spacing: 0.12em;
  text-transform: uppercase; color: rgba(255,255,255,0.5);
}
.hero-year {
  font-size: 13px; color: rgba(255,255,255,0.35);
  letter-spacing: 0.06em;
}
.hero-main {
  flex: 1; display: flex; flex-direction: column;
  justify-content: center; padding-bottom: 40px;
}
.hero-eyebrow {
  font-size: 12px; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: #e10600;
  margin-bottom: 20px;
}
.hero-title {
  font-size: clamp(52px, 8vw, 96px);
  font-weight: 700; line-height: 0.95;
  letter-spacing: -0.04em; color: #ffffff;
  margin-bottom: 8px;
}
.hero-title-italic {
  font-size: clamp(44px, 7vw, 80px);
  font-style: italic; font-weight: 300;
  color: rgba(255,255,255,0.35);
  letter-spacing: -0.03em;
  display: block; margin-bottom: 32px;
}
.hero-sub {
  font-size: 17px; color: rgba(255,255,255,0.45);
  font-weight: 400; max-width: 460px; line-height: 1.5;
  margin-bottom: 0;
}
.hero-bottom {
  display: flex; align-items: center; justify-content: space-between;
  height: 68px; flex-shrink: 0;
  border-top: 0.5px solid rgba(255,255,255,0.07);
}
.hero-credit {
  font-size: 11px; font-weight: 500; letter-spacing: 0.1em;
  text-transform: uppercase; color: rgba(255,255,255,0.3);
}
.hero-cta {
  font-size: 13px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: rgba(255,255,255,0.85);
  display: flex; align-items: center; gap: 10px;
  cursor: pointer; transition: color 0.2s;
}
.hero-cta-arrow {
  width: 32px; height: 32px; border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.25);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; color: rgba(255,255,255,0.7);
  transition: all 0.2s;
}

/* ── Team selector page ── */
.selector-page {
  min-height: 100vh; background: #0a0a0a;
  padding: 0 0 80px;
}
.selector-nav {
  display: flex; align-items: center; justify-content: space-between;
  height: 64px; padding: 0 48px;
  border-bottom: 0.5px solid rgba(255,255,255,0.07);
}
.selector-back {
  font-size: 13px; color: rgba(255,255,255,0.45);
  letter-spacing: 0.04em; cursor: pointer;
  display: flex; align-items: center; gap: 8px;
}
.selector-heading {
  max-width: 1100px; margin: 0 auto;
  padding: 56px 48px 40px;
}
.selector-title {
  font-size: clamp(36px, 5vw, 64px); font-weight: 700;
  letter-spacing: -0.04em; color: #ffffff;
  line-height: 1; margin-bottom: 12px;
}
.selector-sub {
  font-size: 16px; color: rgba(255,255,255,0.4);
  font-weight: 400;
}
.selector-grid {
  max-width: 1100px; margin: 0 auto;
  padding: 0 48px;
  display: grid; grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

/* ── Liquid glass team card ── */
.team-card-wrap button {
  all: unset !important;
  display: block !important;
  width: 100% !important;
  cursor: pointer !important;
}
.glass-card {
  position: relative; overflow: hidden;
  border-radius: 20px; padding: 24px 28px 22px;
  cursor: pointer;
  background: rgba(255,255,255,0.04);
  border: 0.5px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(24px) saturate(120%);
  -webkit-backdrop-filter: blur(24px) saturate(120%);
  transition: background 0.2s, border-color 0.2s, transform 0.15s;
}
.glass-card::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.07) 0%, transparent 60%);
  pointer-events: none;
}
.glass-card:hover {
  background: rgba(255,255,255,0.07);
  border-color: rgba(255,255,255,0.18);
  transform: translateY(-1px);
}
.glass-card-accent {
  position: absolute; top: 0; left: 0; right: 0;
  height: 2px; border-radius: 20px 20px 0 0;
}
.glass-team {
  font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: rgba(255,255,255,0.4);
  margin-bottom: 14px;
}
.glass-drivers {
  display: flex; gap: 0; align-items: baseline;
}
.glass-d1 {
  font-size: 28px; font-weight: 700; letter-spacing: -0.03em;
  color: #ffffff; margin-right: 14px;
}
.glass-d2 {
  font-size: 28px; font-weight: 700; letter-spacing: -0.03em;
  color: rgba(255,255,255,0.38);
}

/* ── Main app nav ── */
.app-nav {
  position: sticky; top: 0; z-index: 999;
  display: flex; align-items: center; justify-content: space-between;
  height: 54px; padding: 0 32px;
  background: rgba(255,255,255,0.88);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border-bottom: 0.5px solid var(--bdr);
}
.app-nav-left { font-size: 15px; font-weight: 600; color: var(--t1); letter-spacing: -0.01em; }
.app-nav-pills {
  display: flex; gap: 2px; background: var(--sur2);
  border-radius: 10px; padding: 3px;
}
.app-nav-pill {
  font-size: 13px; font-weight: 500; padding: 5px 18px;
  border-radius: 8px; cursor: pointer; border: none;
  color: var(--t2); background: transparent;
  font-family: var(--f); transition: all 0.15s;
}
.app-nav-pill.on {
  background: var(--sur); color: var(--t1);
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.app-nav-right {
  font-size: 13px; color: #0071e3; cursor: pointer;
  font-weight: 400; background: none; border: none;
  font-family: var(--f);
}

/* ── App page wrapper ── */
.pg { max-width: 980px; margin: 0 auto; padding: 0 24px 80px; }

/* ── Selector row ── */
.sel-row { display: flex; gap: 10px; margin-bottom: 24px; margin-top: 28px; }

/* ── Driver hero cards ── */
.dc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px; }
.dc {
  background: var(--sur); border: 0.5px solid var(--bdr);
  border-radius: 22px; padding: 28px 28px 22px;
}
.dc-eyebrow { font-size: 11px; font-weight: 600; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--t3); margin-bottom: 8px; }
.dc-name { font-size: 56px; font-weight: 700; letter-spacing: -0.05em;
  line-height: 1; margin-bottom: 20px; }
.dc-time { font-size: 34px; font-weight: 600; letter-spacing: -0.03em;
  color: var(--t1); font-variant-numeric: tabular-nums; margin-bottom: 4px; }
.dc-gap { font-size: 14px; color: var(--t3); margin-bottom: 18px; }
.dc-row { display: flex; gap: 24px; }
.dc-stat-l { font-size: 11px; color: var(--t3); margin-bottom: 3px; }
.dc-stat-v { font-size: 14px; font-weight: 600; color: var(--t1); }

/* ── Gap badge ── */
.gap-badge {
  background: var(--sur); border: 0.5px solid var(--bdr);
  border-radius: 14px; padding: 13px 20px;
  font-size: 14px; color: var(--t2); text-align: center; margin-bottom: 28px;
}
.gap-badge b { color: var(--t1); }
.gap-badge .faster { color: #0071e3; font-weight: 600; }

/* ── Grid table ── */
.g-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--t3); margin-bottom: 10px;
}
.g-wrap {
  background: var(--sur); border: 0.5px solid var(--bdr);
  border-radius: 18px; overflow: hidden;
}
.g-hdr {
  display: grid; grid-template-columns: 38px 10px 52px 1fr 80px;
  padding: 8px 16px; gap: 8px; align-items: center;
  border-bottom: 0.5px solid rgba(0,0,0,0.06);
  font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--t3);
}
.g-row {
  display: grid; grid-template-columns: 38px 10px 52px 1fr 80px;
  padding: 9px 16px; gap: 8px; align-items: center;
  border-bottom: 0.5px solid rgba(0,0,0,0.05);
  transition: background 0.1s;
}
.g-row:last-child { border-bottom: none; }
.g-row.h1 { background: rgba(255,159,10,0.05); }
.g-row.h2 { background: rgba(0,113,227,0.04); }
.g-pos { font-size: 13px; font-weight: 500; color: var(--t3); }
.g-dot { width: 8px; height: 8px; border-radius: 50%; }
.g-name { font-size: 14px; font-weight: 600; color: var(--t1); letter-spacing: -0.01em; }
.g-bar-w { height: 3px; background: var(--sur2); border-radius: 2px; }
.g-bar   { height: 3px; border-radius: 2px; min-width: 2px; }
.g-gap { font-size: 13px; color: var(--t2); text-align: right; font-variant-numeric: tabular-nums; }
.g-gap.pole { color: var(--t1); font-weight: 600; }

/* ── Metric cards ── */
.mc-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 20px; }
.mc {
  background: var(--sur); border: 0.5px solid var(--bdr);
  border-radius: 16px; padding: 16px 18px;
}
.mc-l { font-size: 12px; color: var(--t2); margin-bottom: 4px; }
.mc-v { font-size: 24px; font-weight: 700; letter-spacing: -0.025em; color: var(--t1); }
.mc-s { font-size: 12px; color: var(--t3); margin-top: 2px; }

/* ── Chart card ── */
.cc {
  background: var(--sur); border: 0.5px solid var(--bdr);
  border-radius: 18px; padding: 20px 20px 10px; margin-bottom: 10px;
}
.cc-t { font-size: 13px; font-weight: 600; color: var(--t1); margin-bottom: 2px; }
.cc-s { font-size: 12px; color: var(--t2); margin-bottom: 12px; }

/* ── Insight card ── */
.ic {
  background: var(--sur); border: 0.5px solid var(--bdr);
  border-radius: 18px; padding: 20px 22px; margin-bottom: 10px;
}
.ic-t { font-size: 14px; font-weight: 600; color: var(--t1); margin-bottom: 8px; }
.ic-b { font-size: 14px; color: var(--t2); line-height: 1.6; }
.ic-hl { color: var(--t1); font-weight: 600; }

/* ── Result table ── */
.rt {
  background: var(--sur); border: 0.5px solid var(--bdr);
  border-radius: 18px; overflow: hidden; margin-bottom: 10px;
}
.rt-row {
  display: grid; grid-template-columns: 48px 1fr 80px 80px 80px;
  padding: 10px 16px; border-bottom: 0.5px solid rgba(0,0,0,0.05);
  font-size: 13px; align-items: center; gap: 8px;
}
.rt-row:last-child { border-bottom: none; }
.rt-hdr { background: var(--sur2); font-size: 11px; font-weight: 600;
  letter-spacing: 0.04em; text-transform: uppercase; color: var(--t3); }
.rg { color: #34c759; } .rb { color: #ff3b30; }

/* ── Section title ── */
.pg-title { font-size: 36px; font-weight: 700; letter-spacing: -0.04em;
  color: var(--t1); margin: 44px 0 4px; }
.pg-sub { font-size: 15px; color: var(--t2); margin: 0 0 24px; }

/* ── Sub nav ── */
.sub-nav {
  display: flex; border-bottom: 0.5px solid var(--div);
  margin-bottom: 28px;
}
.sn-item {
  font-size: 14px; color: var(--t2); padding: 10px 0;
  margin-right: 28px; border-bottom: 2px solid transparent;
  transition: all 0.15s; font-weight: 400;
}
.sn-item.on { color: var(--t1); font-weight: 600; border-bottom-color: var(--t1); }

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] > div > div {
  background: #ffffff !important;
  border: 0.5px solid rgba(0,0,0,0.1) !important;
  border-radius: 12px !important;
  color: #1d1d1f !important; font-size: 14px !important;
}
div[data-testid="stSelectbox"] label { display: none !important; }
div[data-testid="stSlider"] { padding: 0 !important; }
div[data-testid="stCheckbox"] label { font-size: 14px !important; color: #1d1d1f !important; }
div[data-testid="stMetricValue"] { color: #1d1d1f !important; }
.stPlotlyChart { border-radius: 18px !important; }
.modebar { display: none !important; }

/* CRITICAL: make Streamlit buttons full-width and invisible on selector page */
[data-testid="stButton"] > button {
  font-family: var(--f) !important;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD ────────────────────────────────────────────────────────────────────
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
    <div style='height:100vh;display:flex;align-items:center;justify-content:center;
                background:#0a0a0a;font-family:-apple-system,sans-serif;flex-direction:column;gap:16px'>
      <div style='font-size:64px;font-weight:700;color:#fff;letter-spacing:-0.04em'>F1 Predictor</div>
      <div style='font-size:17px;color:rgba(255,255,255,0.4)'>Run <code style="background:rgba(255,255,255,0.1);padding:4px 10px;border-radius:6px;color:#fff">python train_model.py</code> first</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

(model, le_team, le_driver, le_event, features,
 team_circuit_avg_delta, team_circuit_avg_abs, driver_skill,
 team_avg_delta, weather_defaults, F1_2025_GRID, ROOKIES_2025, metrics) = load_artifacts()

df, tracks, real_2025 = load_data()

TEAMS_2025       = list(F1_2025_GRID.keys())
ALL_EVENTS       = sorted(real_2025["race"].unique())
DRIVER_TEAM_2025 = {d: t for t, drvs in F1_2025_GRID.items() for d in drvs}
TEAM_COLORS = {
    "Red Bull Racing":"#3671C6","McLaren":"#FF8000","Ferrari":"#E8002D",
    "Mercedes":"#27F4D2","Aston Martin":"#229971","Alpine":"#FF87BC",
    "Williams":"#64C4FF","Haas F1 Team":"#B6BABD","RB":"#6692FF","Kick Sauber":"#52E252",
}

# ── HELPERS ─────────────────────────────────────────────────────────────────
def secs_to_str(s):
    if pd.isna(s) or s <= 0: return "—"
    return f"{int(s//60)}:{s%60:06.3f}"

def get_wx(event):
    row = weather_defaults[weather_defaults["Event"]==event]
    if row.empty:
        return dict(AirTemp=25.0,TrackTemp=38.0,Humidity=50.0,WindSpeed=2.0,Pressure=1013.0,Rainfall=0.0)
    r = row.iloc[0]
    return dict(AirTemp=r.AirTemp,TrackTemp=r.TrackTemp,Humidity=r.Humidity,
                WindSpeed=r.WindSpeed,Pressure=r.Pressure,Rainfall=r.Rainfall)

def safe_enc(le, val):
    return int(le.transform([val])[0]) if val in le.classes_ else 0

TRAP_MAP  = {"Slow":(240,220,210,250),"Medium":(270,255,245,285),"Fast":(300,285,275,315)}
SPEED_MAP = {"Slow":1,"Medium":2,"Fast":3}

def predict_delta(event, team, driver, segment="Q3", compound="SOFT",
                  tyre_life=2, fresh=True, wx=None):
    if wx is None: wx = get_wx(event)
    ti   = tracks[tracks["Event"]==event]
    ti_v = ti.iloc[0].to_dict() if not ti.empty else dict(
        TrackType="Permanent",LapSpeedClass="Medium",DRSZones=2,
        Altitude_m=50,NumCorners=15,CornerDensity=0.003,TrackLength_m=5000,AvgCornerSpacing_m=333)
    track_type = str(ti_v.get("TrackType","Permanent"))
    sc         = str(ti_v.get("LapSpeedClass","Medium"))
    si1,si2,sfl,sst = TRAP_MAP.get(sc,(265,250,240,278))
    tca = team_circuit_avg_delta[
        (team_circuit_avg_delta["Team"]==team) &
        (team_circuit_avg_delta["TrackType"]==track_type)]
    tc_avg = tca["TeamCircuitAvgDelta"].values[0] if not tca.empty else 1.0
    ds = driver_skill[driver_skill["Driver"]==driver]
    drv_delta = ds["DriverAvgDelta"].values[0] if not ds.empty else (
        team_avg_delta[team_avg_delta["Team"]==team]["TeamAvgDelta"].values[0]
        if not team_avg_delta[team_avg_delta["Team"]==team].empty else 1.5)
    row = {
        "Team_enc":safe_enc(le_team,team),"Driver_enc":safe_enc(le_driver,driver),
        "Event_enc":safe_enc(le_event,event),"Year":2025,
        "QualiSegment_num":{"Q1":1,"Q2":2,"Q3":3}.get(segment,2),
        "Compound_num":{"SOFT":3,"MEDIUM":2,"HARD":1,"INTER":0,"WET":-1}.get(compound,3),
        "TyreLife":tyre_life,"FreshTyre_int":int(fresh),
        "IsStreet":1 if track_type=="Street" else 0,
        "SpeedClass_num":SPEED_MAP.get(sc,2),
        "DRSZones":ti_v.get("DRSZones",2),"Altitude_m":ti_v.get("Altitude_m",50),
        "NumCorners":ti_v.get("NumCorners",15),"CornerDensity":ti_v.get("CornerDensity",0.003),
        "TrackLength_m":ti_v.get("TrackLength_m",5000),"AvgCornerSpacing_m":ti_v.get("AvgCornerSpacing_m",333),
        "AirTemp":wx["AirTemp"],"TrackTemp":wx["TrackTemp"],"Humidity":wx["Humidity"],
        "Pressure":wx["Pressure"],"WindSpeed":wx["WindSpeed"],
        "Rainfall_int":1 if wx["Rainfall"]>0.1 else 0,
        "SpeedI1":si1,"SpeedI2":si2,"SpeedFL":sfl,"SpeedST":sst,
        "TeamCircuitAvgDelta":tc_avg,"DriverAvgDelta":drv_delta,
    }
    return max(0.0, float(model.predict(pd.DataFrame([row])[features])[0]))

def get_pole_2025(event):
    row = real_2025[real_2025["race"]==event]
    return float(row["real_time_seconds"].min()) if not row.empty else None

def predict_absolute(event, team, driver, segment="Q3", compound="SOFT",
                     tyre_life=2, fresh=True, wx=None):
    delta = predict_delta(event,team,driver,segment,compound,tyre_life,fresh,wx)
    pole  = get_pole_2025(event)
    if pole is None:
        pole = float(df[df["IsPushLap"]==1].groupby("Event")["LapTime_sec"].min().get(event,90.0))
    return pole+delta, delta, pole

def fig_theme(fig, height=280):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system,'Helvetica Neue',Arial",color="#6e6e73",size=12),
        margin=dict(l=0,r=0,t=8,b=0), height=height,
        xaxis=dict(showgrid=False,zeroline=False,tickfont=dict(size=11,color="#86868b")),
        yaxis=dict(showgrid=True,gridcolor="rgba(0,0,0,0.05)",zeroline=False,
                   tickfont=dict(size=11,color="#86868b")),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=12)),
        hoverlabel=dict(bgcolor="#fff",bordercolor="rgba(0,0,0,0.1)",font=dict(color="#1d1d1f",size=13)),
    )
    return fig

@st.cache_data
def compute_all_accuracy():
    rows = []
    for _, rr in real_2025.iterrows():
        drv=rr["driver"]; race=rr["race"]
        team=DRIVER_TEAM_2025.get(drv)
        if not team: continue
        real_t=float(rr["real_time_seconds"])
        pred_a,delta,pole=predict_absolute(race,team,drv)
        rows.append({"Race":race,"Driver":drv,"Team":team,
                     "Real":round(real_t,3),"Predicted":round(pred_a,3),
                     "Error":round(pred_a-real_t,3),"AbsError":round(abs(pred_a-real_t),3),
                     "IsRookie":drv in ROOKIES_2025})
    return pd.DataFrame(rows)

# ── SESSION STATE ────────────────────────────────────────────────────────────
for k,v in [("screen","hero"),("team",None),("page","predict"),("sub","season")]:
    if k not in st.session_state: st.session_state[k] = v

import streamlit.components.v1 as components
import json as _json

# ── embedded assets ───────────────────────────────────────────────────────
with open('/tmp/car_b64.txt') as _f:
    _CAR = _f.read().strip()

_GRID = [
  {"team":"Red Bull Racing","tc":"#3671C6","d1":"VER","d2":"LAW","r1":"","r2":"",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Max_Verstappen_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Max_Verstappen_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Liam_Lawson%2C_2024_Singapore_GP.jpg/200px-Liam_Lawson%2C_2024_Singapore_GP.jpg"},
  {"team":"McLaren","tc":"#FF8000","d1":"NOR","d2":"PIA","r1":"","r2":"",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Lando_Norris_2023_British_Grand_Prix.jpg/200px-Lando_Norris_2023_British_Grand_Prix.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Oscar_Piastri_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Oscar_Piastri_2023_British_Grand_Prix_%28cropped%29.jpg"},
  {"team":"Ferrari","tc":"#E8002D","d1":"LEC","d2":"HAM","r1":"","r2":"",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Charles_Leclerc_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Charles_Leclerc_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Lewis_Hamilton_2016_Malaysia_2_%28cropped%29.jpg/200px-Lewis_Hamilton_2016_Malaysia_2_%28cropped%29.jpg"},
  {"team":"Mercedes","tc":"#27F4D2","d1":"RUS","d2":"ANT","r1":"","r2":"R",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/George_Russell_2023_British_Grand_Prix_%28cropped%29.jpg/200px-George_Russell_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Kimi_Antonelli_2024_Italian_Grand_Prix_%28cropped%29.jpg/200px-Kimi_Antonelli_2024_Italian_Grand_Prix_%28cropped%29.jpg"},
  {"team":"Aston Martin","tc":"#229971","d1":"ALO","d2":"STR","r1":"","r2":"",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Fernando_Alonso_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Fernando_Alonso_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Lance_Stroll_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Lance_Stroll_2023_British_Grand_Prix_%28cropped%29.jpg"},
  {"team":"Alpine","tc":"#FF87BC","d1":"GAS","d2":"DOO","r1":"","r2":"",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Pierre_Gasly_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Pierre_Gasly_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Jack_Doohan_2023_Macau_Grand_Prix_%28cropped%29.jpg/200px-Jack_Doohan_2023_Macau_Grand_Prix_%28cropped%29.jpg"},
  {"team":"Haas F1 Team","tc":"#B6BABD","d1":"OCO","d2":"BEA","r1":"","r2":"",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Esteban_Ocon_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Esteban_Ocon_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Oliver_Bearman_2024_British_Grand_Prix_%28cropped%29.jpg/200px-Oliver_Bearman_2024_British_Grand_Prix_%28cropped%29.jpg"},
  {"team":"RB","tc":"#6692FF","d1":"TSU","d2":"HAD","r1":"","r2":"R",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Yuki_Tsunoda_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Yuki_Tsunoda_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Isack_Hadjar_2024_Silverstone_%28cropped%29.jpg/200px-Isack_Hadjar_2024_Silverstone_%28cropped%29.jpg"},
  {"team":"Williams","tc":"#64C4FF","d1":"ALB","d2":"SAI","r1":"","r2":"",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Alexander_Albon_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Alexander_Albon_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Carlos_Sainz_Jr_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Carlos_Sainz_Jr_2023_British_Grand_Prix_%28cropped%29.jpg"},
  {"team":"Kick Sauber","tc":"#52E252","d1":"HUL","d2":"BOR","r1":"","r2":"R",
   "img1":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Nico_H%C3%BClkenberg_2023_British_Grand_Prix_%28cropped%29.jpg/200px-Nico_H%C3%BClkenberg_2023_British_Grand_Prix_%28cropped%29.jpg",
   "img2":"https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Gabriel_Bortoleto_2024_Zandvoort_%28cropped%29.jpg/200px-Gabriel_Bortoleto_2024_Zandvoort_%28cropped%29.jpg"},
]
_TEAMS = [g["team"] for g in _GRID]

# ── HERO HTML ─────────────────────────────────────────────────────────────
def _hero_html():
    return (
        """<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@0,800;0,900;1,400&family=Caveat:wght@700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#000}
.w{width:100vw;height:100vh;position:relative;display:flex;flex-direction:column}
.car{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center 55%}
.ov{position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,0.88) 0%,rgba(0,0,0,0.2) 38%,rgba(0,0,0,0.1) 58%,rgba(0,0,0,0.80) 100%)}
.nav{position:relative;z-index:3;flex-shrink:0;height:52px;display:flex;align-items:center;
  justify-content:space-between;padding:0 44px;
  background:rgba(255,255,255,0.93);backdrop-filter:blur(20px);
  border-bottom:0.5px solid rgba(0,0,0,0.08)}
.nb{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;
  font-size:13px;font-weight:600;color:#1d1d1f}
.ny{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;
  font-size:12px;color:#86868b;letter-spacing:0.06em}
.cnt{flex:1;position:relative;z-index:2;
  display:flex;flex-direction:column;justify-content:center;
  padding:0 52px 52px}
.t1{font-family:'Barlow Condensed',sans-serif;
  font-size:clamp(80px,11vw,148px);font-weight:900;
  line-height:0.87;letter-spacing:-0.01em;
  color:#b9bbc4;text-transform:uppercase;display:block}
.t2{font-family:'Caveat',cursive;
  font-size:clamp(60px,8.5vw,118px);font-weight:700;
  color:#cc0000;line-height:1;display:block;margin-top:8px}
.ft{position:relative;z-index:3;flex-shrink:0;height:62px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 44px;border-top:0.5px solid rgba(255,255,255,0.1)}
.cr{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;
  font-size:11px;font-weight:600;letter-spacing:0.12em;
  text-transform:uppercase;color:rgba(255,255,255,0.35)}
.ct{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;
  font-size:12px;font-weight:700;letter-spacing:0.14em;
  text-transform:uppercase;color:rgba(255,255,255,0.85);
  display:flex;align-items:center;gap:12px}
.ar{width:36px;height:36px;border-radius:50%;
  border:1.5px solid rgba(255,255,255,0.3);
  display:flex;align-items:center;justify-content:center;
  font-size:17px;color:rgba(255,255,255,0.7)}
</style></head><body>
<div class="w">
  <img class="car" src="data:image/jpeg;base64,"""
        + _CAR +
        """" alt=""/>
  <div class="ov"></div>
  <div class="nav">
    <span class="nb">F1 Qualifying Predictor</span>
    <span class="ny">2025 SEASON</span>
  </div>
  <div class="cnt">
    <span class="t1">F1 Qualifying</span>
    <span class="t2">Prediction.</span>
  </div>
  <div class="ft">
    <span class="cr">Done by Anirudh Atkuru</span>
    <span class="ct">SELECT YOUR TEAM <div class="ar">&#8594;</div></span>
  </div>
</div>
</body></html>"""
    )

# ── SELECTOR HTML ─────────────────────────────────────────────────────────
def _selector_html():
    gj = _json.dumps(_GRID)
    return (
        """<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800;900&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{background:#0d0d0d;min-height:100%}
body{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
/* ── title ── */
.hdr{text-align:center;padding:44px 24px 32px;background:#0d0d0d}
.hdr h1{font-family:'Barlow Condensed',sans-serif;
  font-size:clamp(52px,8vw,96px);font-weight:900;
  color:#fff;text-transform:uppercase;letter-spacing:0.02em;line-height:1}
/* ── grid container ── */
.container{background:#0d0d0d;padding:0 32px 60px;max-width:1100px;margin:0 auto}
/* Each row = one team: 4-col grid (2 drivers, each with photo+name) */
.team-row{
  display:grid;grid-template-columns:1fr 1fr 1fr 1fr;
  gap:0;margin-bottom:6px;cursor:pointer;
  border-radius:6px;overflow:hidden;
}
.team-row:hover .dc{opacity:0.88}
.team-row:active{transform:scale(0.995)}
/* driver cell */
.dc{
  position:relative;display:flex;flex-direction:column;
  align-items:center;background:#1a1a1a;
  transition:opacity 0.15s;
}
/* team-colour separator between the two pairs */
.sep{width:4px;background:#0d0d0d;flex-shrink:0}
/* photo */
.ph{
  width:100%;aspect-ratio:1/1.2;
  object-fit:cover;object-position:top center;
  display:block;
  background:#222;
}
/* name banner */
.banner{
  width:100%;padding:6px 4px 7px;
  display:flex;align-items:center;justify-content:center;
  gap:4px;
}
.dname{
  font-family:'Barlow Condensed',sans-serif;
  font-size:clamp(13px,1.6vw,18px);font-weight:900;
  letter-spacing:0.08em;text-transform:uppercase;
  color:#fff;
}
.rk{font-size:9px;color:rgba(255,230,100,0.8);vertical-align:super}
/* footer label */
.footer{
  display:flex;align-items:center;justify-content:space-between;
  padding:16px 32px;max-width:1100px;margin:0 auto;
}
.footer span{font-size:12px;color:rgba(255,255,255,0.3);letter-spacing:0.06em}
/* back button */
.back-btn{
  font-family:-apple-system,BlinkMacSystemFont,sans-serif;
  font-size:13px;color:rgba(255,255,255,0.5);cursor:pointer;
  padding:8px 0;display:inline-block;
}
.back-btn:hover{color:#fff}
</style></head><body>
<div class="hdr">
  <div class="back-btn" id="backbtn">&#8592; Back</div>
  <h1>Choose a Team</h1>
</div>
<div class="container" id="grid"></div>
<div class="footer">
  <span>2025 DRIVER LINE-UP</span>
</div>
<script>
const teams = """ + gj + """;
const grid = document.getElementById('grid');

teams.forEach((t, ti) => {
  const row = document.createElement('div');
  row.className = 'team-row';
  row.onclick = () => {
    // click the hidden streamlit button at index ti+1 (0=back, 1..10=teams)
    const btns = window.parent.document.querySelectorAll('[data-testid="stButton"] button');
    if (btns[ti + 1]) btns[ti + 1].click();
  };

  // left driver
  const c1 = document.createElement('div');
  c1.className = 'dc';
  c1.innerHTML = `
    <img class="ph" src="${t.img1}" onerror="this.style.background='#2a2a2a'" alt="${t.d1}">
    <div class="banner" style="background:${t.tc}">
      <span class="dname">${t.d1}<span class="rk">${t.r1}</span></span>
    </div>`;

  // right driver
  const c2 = document.createElement('div');
  c2.className = 'dc';
  c2.innerHTML = `
    <img class="ph" src="${t.img2}" onerror="this.style.background='#2a2a2a'" alt="${t.d2}">
    <div class="banner" style="background:${t.tc}">
      <span class="dname">${t.d2}<span class="rk">${t.r2}</span></span>
    </div>`;

  row.appendChild(c1);
  row.appendChild(c2);
  grid.appendChild(row);

  // Add a small gap between teams (spacer row)
  if (ti % 2 === 1 && ti < teams.length - 1) {
    const sp = document.createElement('div');
    sp.style.cssText = 'height:6px;background:#0d0d0d;grid-column:1/-1';
    // just use margin on row
  }
});

document.getElementById('backbtn').onclick = () => {
  const btns = window.parent.document.querySelectorAll('[data-testid="stButton"] button');
  if (btns[0]) btns[0].click();
};
</script>
</body></html>"""
    )


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 1: HERO
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "hero":
    st.markdown("""<style>
    .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
    .main,.block-container{background:#000 !important;padding:0 !important}
    [data-testid="stVerticalBlock"]{gap:0 !important}
    [data-testid="stButton"]{position:fixed !important;bottom:10px !important;
      right:32px !important;z-index:9999 !important}
    [data-testid="stButton"] > button{
      min-width:220px !important;height:44px !important;
      background:transparent !important;border:none !important;
      color:transparent !important;font-size:1px !important;cursor:pointer !important;
    }
    </style>""", unsafe_allow_html=True)

    components.html(_hero_html(), height=720, scrolling=False)

    if st.button("go", key="hero_go"):
        st.session_state.screen = "selector"
        st.rerun()
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2: TEAM SELECTOR
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "selector":
    st.markdown("""<style>
    .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
    .main,.block-container{background:#0d0d0d !important;padding:0 !important}
    [data-testid="stVerticalBlock"]{gap:0 !important}
    [data-testid="stButton"] > button{
      opacity:0 !important;height:1px !important;min-height:1px !important;
      padding:0 !important;margin:0 !important;font-size:1px !important;
      border:none !important;background:transparent !important;
      overflow:hidden !important;
    }
    </style>""", unsafe_allow_html=True)

    components.html(_selector_html(), height=1200, scrolling=True)

    # Real Streamlit buttons (invisible) — back first, then teams in order
    if st.button("back", key="sb_back"):
        st.session_state.screen = "hero"
        st.rerun()

    for _t in _TEAMS:
        if st.button(_t, key=f"sb_{_t}"):
            st.session_state.team = _t
            st.session_state.screen = "app"
            st.session_state.page   = "predict"
            st.rerun()

    st.stop()





# SCREEN 3: MAIN APP
# ════════════════════════════════════════════════════════════════════════════
sel_team = st.session_state.team
if not sel_team:
    st.session_state.screen = "selector"
    st.rerun()

drv1, drv2  = F1_2025_GRID[sel_team]
d1c = TEAM_COLORS.get(sel_team, "#ff9f0a")
d2c = "#0071e3"
page = st.session_state.page

# ── Nav ──────────────────────────────────────────────────────────────────────
p_on = "on" if page=="predict" else ""
a_on = "on" if page=="analyse" else ""

st.markdown(f"""
<div class='app-nav'>
  <span class='app-nav-left'>{sel_team}</span>
  <div class='app-nav-pills'>
    <button class='app-nav-pill {p_on}' id='np'>Predict</button>
    <button class='app-nav-pill {a_on}' id='na'>Analyse</button>
  </div>
  <button class='app-nav-right' id='nc'>Change team</button>
</div>
""", unsafe_allow_html=True)

# Hidden nav controls
nav = st.selectbox("n", ["predict","analyse","change"],
                   label_visibility="collapsed", key="nav_sel",
                   index=["predict","analyse","change"].index(
                       page if page in ["predict","analyse"] else "predict"))

st.markdown("""<script>
(function(){
  function pick(v){
    const sels=window.parent.document.querySelectorAll('[data-testid="stSelectbox"] select');
    sels.forEach(s=>{for(let o of s.options){if(o.text===v||o.value===v){s.value=o.value;s.dispatchEvent(new Event('change',{bubbles:true}));break;}}});
  }
  setTimeout(()=>{
    const np=window.parent.document.getElementById('np');
    const na=window.parent.document.getElementById('na');
    const nc=window.parent.document.getElementById('nc');
    if(np) np.onclick=()=>pick('predict');
    if(na) na.onclick=()=>pick('analyse');
    if(nc) nc.onclick=()=>pick('change');
  },300);
})();
</script>""", unsafe_allow_html=True)

if nav=="change":
    st.session_state.team=None; st.session_state.screen="selector"; st.rerun()
elif nav!=page:
    st.session_state.page=nav; st.rerun()

# ════════════════════════════════════════
# PREDICT PAGE
# ════════════════════════════════════════
if page == "predict":
    st.markdown("""<style>
    .stApp,[data-testid='stAppViewContainer'],[data-testid='stMain'],.main,.block-container{
      background:#0a0a0a !important;
    }
    div[data-testid='stExpander']{background:var(--sur) !important;border-color:var(--bdr) !important}
    div[data-testid='stExpander'] *{color:var(--t1) !important}
    div[data-testid='stSlider'] label{color:var(--t1) !important}
    div[data-testid='stSlider'] .stMarkdown{color:var(--t1) !important}
    label[data-testid='stWidgetLabel']{color:var(--t1) !important}
    .stSelectbox label, .stSlider label{color:var(--t1) !important}
    </style>""", unsafe_allow_html=True)
    st.markdown("<div class='pg'>", unsafe_allow_html=True)

    # Header with team name + change link
    hc1, hc2 = st.columns([5,1])
    with hc1:
        st.markdown(f"<div style='padding:20px 0 8px;font-size:28px;font-weight:700;"
                    f"letter-spacing:-0.03em;color:var(--t1)'>{sel_team}</div>",
                    unsafe_allow_html=True)
    with hc2:
        if st.button("Change team", key="change_team_predict"):
            st.session_state.team = None
            st.session_state.screen = "selector"
            st.rerun()

    c1,c2,c3 = st.columns([3,1,1])
    with c1: sel_event = st.selectbox("e", ALL_EVENTS, key="pe", label_visibility="collapsed")
    with c2: seg = st.selectbox("s", ["Q3","Q2","Q1"], key="ps", label_visibility="collapsed")
    with c3: cmp = st.selectbox("c", ["SOFT","MEDIUM","HARD"], key="pc", label_visibility="collapsed")

    hist_wx = get_wx(sel_event)
    wx = dict(**hist_wx)
    wx_str = f"🌡 {hist_wx['AirTemp']:.0f}°C air · {hist_wx['TrackTemp']:.0f}°C track · {hist_wx['Humidity']:.0f}% humidity"
    with st.expander(f"Weather — {wx_str}", expanded=False):
        wc1,wc2 = st.columns(2)
        with wc1:
            at = st.slider("Air (°C)",   10,45, int(hist_wx["AirTemp"]),  key="wa")
            tt = st.slider("Track (°C)", 15,60, int(hist_wx["TrackTemp"]),key="wt")
        with wc2:
            hm = st.slider("Humidity",  10,100,int(hist_wx["Humidity"]), key="wh")
            ws = st.slider("Wind m/s",  0,15,  int(hist_wx["WindSpeed"]),key="ww")
            rn = st.checkbox("Rain", value=hist_wx["Rainfall"]>0.1, key="wr")
        wx = dict(AirTemp=at,TrackTemp=tt,Humidity=hm,WindSpeed=ws,
                  Pressure=hist_wx["Pressure"],Rainfall=1.0 if rn else 0.0)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    abs1,delta1,_ = predict_absolute(sel_event,sel_team,drv1,seg,cmp,2,True,wx)
    abs2,delta2,_ = predict_absolute(sel_event,sel_team,drv2,seg,cmp,2,True,wx)
    faster   = drv1 if abs1<abs2 else drv2
    gap_bt   = abs(abs1-abs2)
    pole_t   = get_pole_2025(sel_event)

    ti_row = tracks[tracks["Event"]==sel_event]
    ct = ti_row["TrackType"].values[0] if not ti_row.empty else ""
    cs = ti_row["LapSpeedClass"].values[0] if not ti_row.empty else ""
    pole_str = f"2025 pole: <b>{secs_to_str(pole_t)}</b>" if pole_t else ""

    st.markdown(f"""
    <div style='font-size:13px;color:#86868b;margin-bottom:8px'>
      {ct} · {cs}&nbsp;&nbsp;&nbsp;{pole_str}
    </div>
    <div class='dc-grid'>
      <div class='dc'>
        <div class='dc-eyebrow'>{sel_team}</div>
        <div class='dc-name' style='color:{d1c}'>{drv1}{"&thinsp;<sup style='font-size:14px;color:#86868b;font-weight:400'>R</sup>" if drv1 in ROOKIES_2025 else ""}</div>
        <div class='dc-time'>{secs_to_str(abs1)}</div>
        <div class='dc-gap'>+{delta1:.3f}s to pole</div>
        <div class='dc-row'>
          <div><div class='dc-stat-l'>Compound</div><div class='dc-stat-v'>{cmp}</div></div>
          <div><div class='dc-stat-l'>Session</div><div class='dc-stat-v'>{seg}</div></div>
        </div>
      </div>
      <div class='dc'>
        <div class='dc-eyebrow'>{sel_team}</div>
        <div class='dc-name' style='color:{d2c}'>{drv2}{"&thinsp;<sup style='font-size:14px;color:#86868b;font-weight:400'>R</sup>" if drv2 in ROOKIES_2025 else ""}</div>
        <div class='dc-time'>{secs_to_str(abs2)}</div>
        <div class='dc-gap'>+{delta2:.3f}s to pole</div>
        <div class='dc-row'>
          <div><div class='dc-stat-l'>Compound</div><div class='dc-stat-v'>{cmp}</div></div>
          <div><div class='dc-stat-l'>Session</div><div class='dc-stat-v'>{seg}</div></div>
        </div>
      </div>
    </div>
    <div class='gap-badge'>
      Intra-team gap &nbsp;<b>{gap_bt:.3f}s</b>&nbsp; · &nbsp;
      <span class='faster'>{faster}</span> predicted faster this circuit
    </div>
    """, unsafe_allow_html=True)

    # Full grid
    @st.cache_data
    def full_grid(event):
        w = get_wx(event); p = get_pole_2025(event) or 90.0
        rows = []
        for team,drvs in F1_2025_GRID.items():
            for d in drvs:
                dlt = predict_delta(event,team,d,"Q3","SOFT",2,True,w)
                rows.append({"Team":team,"Driver":d,"Delta":dlt,"AbsTime":p+dlt,
                             "IsRookie":d in ROOKIES_2025,"Color":TEAM_COLORS.get(team,"#86868b")})
        gdf = pd.DataFrame(rows).sort_values("Delta").reset_index(drop=True)
        gdf["Position"] = range(1,len(gdf)+1)
        gdf["TimeStr"]  = gdf["AbsTime"].apply(secs_to_str)
        gdf["GapStr"]   = gdf["Delta"].apply(lambda x:"POLE" if x<0.001 else f"+{x:.3f}s")
        return gdf

    gdf = full_grid(sel_event)
    mx  = gdf["Delta"].max()

    st.markdown("<div class='g-label' style='margin-bottom:10px;font-size:12px'>Predicted grid — Q3 · SOFT · historical weather</div>", unsafe_allow_html=True)
    rows_h = ""
    for _,r in gdf.iterrows():
        d=r["Driver"]; is1=(d==drv1 and r["Team"]==sel_team); is2=(d==drv2 and r["Team"]==sel_team)
        hl="h1" if is1 else ("h2" if is2 else "")
        nc=("style='color:{}'".format(d1c) if is1 else ("style='color:{}'".format(d2c) if is2 else ""))
        bar=round(r["Delta"]/mx*100) if mx>0 else 0
        gc="pole" if r["GapStr"]=="POLE" else ""
        rows_h += f"""<div class='g-row {hl}'>
          <span class='g-pos'>P{r['Position']}</span>
          <span class='g-dot' style='background:{r["Color"]}'></span>
          <span class='g-name' {nc}>{d}{"ᴿ" if r["IsRookie"] else ""}</span>
          <div class='g-bar-w'><div class='g-bar' style='width:{bar}%;background:{r["Color"]}'></div></div>
          <span class='g-gap {gc}'>{r["GapStr"]}</span>
        </div>"""

    st.markdown(f"""<div class='g-wrap'>
      <div class='g-hdr'><span></span><span></span><span>Driver</span><span></span><span style='text-align:right'>Gap</span></div>
      {rows_h}
    </div>""", unsafe_allow_html=True)

    # Real result comparison
    real_race = real_2025[real_2025["race"]==sel_event]
    if not real_race.empty:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='g-label'>Real 2025 result vs prediction</div>", unsafe_allow_html=True)
        rh = ""
        for drv,pred_a in [(drv1,abs1),(drv2,abs2)]:
            rr=real_race[real_race["driver"]==drv]
            if not rr.empty:
                rt=float(rr["real_time_seconds"].values[0]); err=pred_a-rt
                ec_col = "#32d74b" if abs(err)<=0.5 else ("#ff453a" if abs(err)>1.5 else "#ff9f0a")
                clr=d1c if drv==drv1 else d2c
                rh += f"""<div class='rt-row'>
                  <span style='font-size:14px;font-weight:600;color:{clr}'>{drv}</span>
                  <span style='color:var(--t1)'>{secs_to_str(rt)}</span>
                  <span style='color:var(--t2)'>{secs_to_str(pred_a)}</span>
                  <span style='color:{ec_col};font-weight:600'>{err:+.3f}s</span>
                </div>"""
        if rh:
            st.markdown(f"""<div class='rt'>
              <div class='rt-row rt-hdr'><span>Driver</span><span>Real</span><span>Predicted</span><span>Error</span></div>
              {rh}
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════
# ANALYSE PAGE
# ════════════════════════════════════════
elif page == "analyse":
    st.markdown("""<style>
    .stApp,[data-testid='stAppViewContainer'],[data-testid='stMain'],.main,.block-container{
      background:#0a0a0a !important;
    }
    div[data-testid='stExpander']{background:var(--sur) !important;border-color:var(--bdr) !important}
    div[data-testid='stExpander'] *{color:var(--t1) !important}
    div[data-testid='stSlider'] label{color:var(--t1) !important}
    div[data-testid='stSlider'] .stMarkdown{color:var(--t1) !important}
    label[data-testid='stWidgetLabel']{color:var(--t1) !important}
    .stSelectbox label, .stSlider label{color:var(--t1) !important}
    </style>""", unsafe_allow_html=True)
    st.markdown("<div class='pg'>", unsafe_allow_html=True)
    hca, hcb = st.columns([5,1])
    with hca:
        st.markdown(f"<div style='padding:20px 0 8px;font-size:28px;font-weight:700;"
                    f"letter-spacing:-0.03em;color:var(--t1)'>{sel_team}</div>",
                    unsafe_allow_html=True)
    with hcb:
        if st.button("Change team", key="change_team_analyse"):
            st.session_state.team = None
            st.session_state.screen = "selector"
            st.rerun()

    sub = st.session_state.sub
    sub_choice = st.radio("sub", ["Season","R&D","Accuracy"], horizontal=True,
                          label_visibility="collapsed", key="sub_r",
                          index=["Season","R&D","Accuracy"].index(
                              sub.title() if sub.title() in ["Season","Accuracy"] else
                              ("R&D" if sub=="r&d" else "Season")))
    st.session_state.sub = sub_choice.lower()
    sub = st.session_state.sub

    # ── SEASON ──────────────────────────────────
    if sub == "season":
        st.markdown(f"<div class='pg-title'>{sel_team}</div>", unsafe_allow_html=True)
        st.markdown("<div class='pg-sub'>2025 season performance across all circuits.</div>", unsafe_allow_html=True)

        sr = []
        for ev in ALL_EVENTS:
            w = get_wx(ev)
            for drv in [drv1,drv2]:
                sr.append({"Race":ev,"Driver":drv,"Delta":round(predict_delta(ev,sel_team,drv,"Q3","SOFT",2,True,w),3)})
        sdf = pd.DataFrame(sr)
        s1=sdf[sdf["Driver"]==drv1]["Delta"]; s2=sdf[sdf["Driver"]==drv2]["Delta"]
        piv=sdf.pivot(index="Race",columns="Driver",values="Delta").reset_index()
        if drv1 in piv.columns and drv2 in piv.columns:
            piv["H2H"]=piv[drv1]-piv[drv2]
            w1=(piv["H2H"]<0).sum(); w2=(piv["H2H"]>0).sum(); avg_g=piv["H2H"].abs().mean()
        else:
            w1=w2=0; avg_g=0.0

        st.markdown(f"""<div class='mc-grid'>
          <div class='mc'><div class='mc-l' style='color:{d1c}'>{drv1} avg gap</div>
            <div class='mc-v'>+{s1.mean():.3f}s</div><div class='mc-s'>to pole</div></div>
          <div class='mc'><div class='mc-l' style='color:{d2c}'>{drv2} avg gap</div>
            <div class='mc-v'>+{s2.mean():.3f}s</div><div class='mc-s'>to pole</div></div>
          <div class='mc'><div class='mc-l'>{drv1} wins</div>
            <div class='mc-v'>{w1}/{len(piv)}</div><div class='mc-s'>circuits faster</div></div>
          <div class='mc'><div class='mc-l'>Intra-team avg</div>
            <div class='mc-v'>{avg_g:.3f}s</div><div class='mc-s'>gap</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='cc'><div class='cc-t'>Gap to pole — all circuits</div><div class='cc-s'>Q3 · SOFT · historical weather</div>", unsafe_allow_html=True)
        fig_s=go.Figure()
        for drv,col in [(drv1,d1c),(drv2,d2c)]:
            sd=sdf[sdf["Driver"]==drv]
            fig_s.add_trace(go.Scatter(x=sd["Race"],y=sd["Delta"],mode="lines+markers",name=drv,
                line=dict(color=col,width=2),marker=dict(size=5,color=col),
                hovertemplate="<b>%{x}</b><br>+%{y:.3f}s<extra></extra>"))
        fig_theme(fig_s,260)
        fig_s.update_layout(xaxis=dict(tickangle=-40,tickfont=dict(size=10)),legend=dict(orientation="h",y=1.1,x=0))
        st.plotly_chart(fig_s,use_container_width=True,config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        if drv1 in piv.columns and drv2 in piv.columns:
            ps=piv.sort_values("H2H")
            st.markdown(f"<div class='cc'><div class='cc-t'>Head-to-head per circuit</div><div class='cc-s'>Positive = {drv2} faster · Negative = {drv1} faster</div>", unsafe_allow_html=True)
            fig_h=go.Figure(go.Bar(x=ps["Race"],y=ps["H2H"],
                marker_color=[d2c if v>0 else d1c for v in ps["H2H"]],
                hovertemplate="<b>%{x}</b><br>%{y:+.3f}s<extra></extra>"))
            fig_h.add_hline(y=0,line_color="rgba(0,0,0,0.12)",line_width=1)
            fig_theme(fig_h,240); fig_h.update_layout(xaxis=dict(tickangle=-40,tickfont=dict(size=10)))
            st.plotly_chart(fig_h,use_container_width=True,config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

        ti_m=tracks[["Event","TrackType","LapSpeedClass"]].copy()
        sf=sdf.merge(ti_m,left_on="Race",right_on="Event",how="left")
        ca,cb=st.columns(2)
        for col,drv,color in [(ca,drv1,d1c),(cb,drv2,d2c)]:
            with col:
                sd=sf[sf["Driver"]==drv]
                for grp,title,sub_t in [("TrackType","Circuit type","Street vs Permanent"),
                                         ("LapSpeedClass","Speed class","Slow / Medium / Fast")]:
                    by=sd.groupby(grp)["Delta"].mean().reset_index()
                    st.markdown(f"<div class='cc'><div class='cc-t' style='color:{color}'>{drv} · {title}</div><div class='cc-s'>{sub_t}</div>", unsafe_allow_html=True)
                    ft=go.Figure(go.Bar(x=by[grp],y=by["Delta"],marker_color=color,
                        hovertemplate="%{x}<br>+%{y:.3f}s<extra></extra>"))
                    fig_theme(ft,180)
                    st.plotly_chart(ft,use_container_width=True,config={"displayModeBar":False})
                    st.markdown("</div>", unsafe_allow_html=True)

        push=df[df["IsPushLap"]==1]
        yt=push[push["Team"]==sel_team].groupby("Year")["LapTime_sec"].mean().reset_index()
        yf=push.groupby("Year")["LapTime_sec"].mean().reset_index()
        st.markdown("<div class='cc'><div class='cc-t'>Historical trend vs field</div><div class='cc-s'>Average qualifying lap time 2019–2024</div>", unsafe_allow_html=True)
        fh=go.Figure()
        fh.add_trace(go.Scatter(x=yt["Year"],y=yt["LapTime_sec"],mode="lines+markers",name=sel_team,
            line=dict(color=d1c,width=2.5),marker=dict(size=6)))
        fh.add_trace(go.Scatter(x=yf["Year"],y=yf["LapTime_sec"],mode="lines+markers",name="Field avg",
            line=dict(color="rgba(110,110,115,0.5)",width=1.5,dash="dot"),marker=dict(size=5)))
        fig_theme(fh,220); fh.update_layout(legend=dict(orientation="h",y=1.1,x=0))
        st.plotly_chart(fh,use_container_width=True,config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── R&D ─────────────────────────────────────
    elif sub == "r&d":
        st.markdown("<div class='pg-title'>R&D Simulator</div>", unsafe_allow_html=True)
        st.markdown("<div class='pg-sub'>Simulate high vs low downforce setups across the calendar.</div>", unsafe_allow_html=True)

        st.markdown("""<div class='ic'>
          <div class='ic-t'>How this works</div>
          <div class='ic-b'>Speed trap readings act as a downforce proxy.
          High downforce = more drag = slower straights but better cornering.
          We shift trap speeds and measure the predicted gap-to-pole delta per circuit.</div>
        </div>""", unsafe_allow_html=True)

        r1,r2=st.columns(2)
        with r1:
            st.markdown("<div style='font-size:14px;font-weight:600;color:#1d1d1f;margin-bottom:4px'>High downforce</div>", unsafe_allow_html=True)
            st.caption("Monaco · Singapore · Hungary")
            dfp=st.slider("Straight speed loss (km/h)",5,25,12,key="hd")
        with r2:
            st.markdown("<div style='font-size:14px;font-weight:600;color:#1d1d1f;margin-bottom:4px'>Low downforce</div>", unsafe_allow_html=True)
            st.caption("Monza · Baku · Las Vegas")
            dfg=st.slider("Straight speed gain (km/h)",5,25,12,key="ld")

        sim=[]
        for _,tr in tracks.iterrows():
            ev=tr["Event"]
            if ev not in df["Event"].values: continue
            sc=str(tr.get("LapSpeedClass","Medium"))
            si1,si2,sfl,sst=TRAP_MAP.get(sc,(265,250,240,278))
            w=get_wx(ev)
            for drv in [drv1,drv2]:
                ds=driver_skill[driver_skill["Driver"]==drv]
                dd=ds["DriverAvgDelta"].values[0] if not ds.empty else 1.5
                tca2=team_circuit_avg_delta[(team_circuit_avg_delta["Team"]==sel_team)&
                      (team_circuit_avg_delta["TrackType"]==str(tr.get("TrackType","Permanent")))]
                tc2=tca2["TeamCircuitAvgDelta"].values[0] if not tca2.empty else 1.0
                def pt(s1,s2,sf,ss,_d=drv,_dd=dd,_tc=tc2,_ev=ev,_tr=tr,_w=w,_sc=sc):
                    ro={
                        "Team_enc":safe_enc(le_team,sel_team),"Driver_enc":safe_enc(le_driver,_d),
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
                    return max(0.0,float(model.predict(pd.DataFrame([ro])[features])[0]))
                base=pt(si1,si2,sfl,sst); hi=pt(si1-dfp,si2-dfp,sfl,sst-dfp)
                lo=pt(si1+dfg,si2+dfg,sfl+dfg,sst+dfg)
                best=min(base,hi,lo)
                sim.append({"Circuit":ev,"Driver":drv,"SpeedClass":sc,
                            "Hi":round(hi-base,3),"Lo":round(lo-base,3),
                            "Best":"High DF" if best==hi else ("Low DF" if best==lo else "Balanced")})

        sdf2=pd.DataFrame(sim)
        for drv,color in [(drv1,d1c),(drv2,d2c)]:
            sd=sdf2[sdf2["Driver"]==drv]
            hi_w=(sd["Hi"]<sd["Lo"]).sum(); lo_w=(sd["Lo"]<sd["Hi"]).sum()
            rec="High downforce" if sd["Hi"].mean()<sd["Lo"].mean() else "Low downforce"
            st.markdown(f"""<div class='mc-grid' style='grid-template-columns:1fr 1fr 1fr;margin-top:16px'>
              <div class='mc'><div class='mc-l' style='color:{color}'>{drv}</div>
                <div class='mc-v'>{rec}</div><div class='mc-s'>season rec.</div></div>
              <div class='mc'><div class='mc-l'>Hi DF wins</div>
                <div class='mc-v'>{hi_w}</div><div class='mc-s'>circuits</div></div>
              <div class='mc'><div class='mc-l'>Lo DF wins</div>
                <div class='mc-v'>{lo_w}</div><div class='mc-s'>circuits</div></div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"<div class='cc'><div class='cc-t' style='color:{color}'>{drv} — setup delta per circuit</div><div class='cc-s'>Negative = faster than baseline</div>", unsafe_allow_html=True)
            frd=go.Figure()
            frd.add_trace(go.Bar(name="High DF",x=sd["Circuit"],y=sd["Hi"],marker_color=color,opacity=0.85))
            frd.add_trace(go.Bar(name="Low DF",x=sd["Circuit"],y=sd["Lo"],marker_color="rgba(110,110,115,0.4)"))
            frd.add_hline(y=0,line_color="rgba(0,0,0,0.12)",line_width=1)
            fig_theme(frd,220); frd.update_layout(barmode="group",
                xaxis=dict(tickangle=-40,tickfont=dict(size=10)),legend=dict(orientation="h",y=1.1,x=0))
            st.plotly_chart(frd,use_container_width=True,config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

    # ── ACCURACY ─────────────────────────────────
    elif sub == "accuracy":
        st.markdown("<div class='pg-title'>Model accuracy</div>", unsafe_allow_html=True)
        st.markdown("<div class='pg-sub'>How predictions compared against real 2025 qualifying results.</div>", unsafe_allow_html=True)

        acc=compute_all_accuracy()
        team_acc=acc[acc["Team"]==sel_team].copy()
        full_mae=acc["AbsError"].mean(); team_mae=team_acc["AbsError"].mean() if not team_acc.empty else 0
        within1=(acc["AbsError"]<=1.0).mean()*100; bias=acc["Error"].mean()

        rad=real_2025.assign(Pole=real_2025.groupby("race")["real_time_seconds"].transform("min"))
        rad["RealDelta"]=rad["real_time_seconds"]-rad["Pole"]
        rad=rad.groupby("driver")["RealDelta"].mean().reset_index().rename(columns={"driver":"Driver","RealDelta":"Real2025AvgDelta"})
        db=[d for d in rad["Driver"] if d in driver_skill["Driver"].values]
        pd2=driver_skill[driver_skill["Driver"].isin(db)][["Driver","DriverAvgDelta"]]
        rm=rad[rad["Driver"].isin(db)].merge(pd2,on="Driver")
        spear_r,spear_p=spearmanr(rm["DriverAvgDelta"],rm["Real2025AvgDelta"])

        st.markdown(f"""<div class='mc-grid'>
          <div class='mc'><div class='mc-l'>Ranking accuracy</div>
            <div class='mc-v'>{spear_r:.2f}</div><div class='mc-s'>Spearman r · p={spear_p:.3f}</div></div>
          <div class='mc'><div class='mc-l'>MAE — all drivers</div>
            <div class='mc-v'>{full_mae:.2f}s</div><div class='mc-s'>avg absolute error</div></div>
          <div class='mc'><div class='mc-l'>MAE — {sel_team}</div>
            <div class='mc-v'>{team_mae:.2f}s</div><div class='mc-s'>avg absolute error</div></div>
          <div class='mc'><div class='mc-l'>Within 1 second</div>
            <div class='mc-v'>{within1:.0f}%</div><div class='mc-s'>of predictions</div></div>
        </div>""", unsafe_allow_html=True)

        rw="strong" if spear_r>0.7 else ("moderate" if spear_r>0.5 else "weak")
        bn="over-predicts by {:.2f}s".format(bias) if bias>0.2 else ("under-predicts by {:.2f}s".format(abs(bias)) if bias<-0.2 else "well-calibrated")
        st.markdown(f"""<div class='ic'>
          <div class='ic-t'>How to read these numbers</div>
          <div class='ic-b'>
            The most important metric is <span class='ic-hl'>ranking accuracy (Spearman r = {spear_r:.2f})</span> — {rw} rank correlation.
            This measures whether the model correctly orders drivers relative to each other.
            Random guessing = 0.0.<br><br>
            The model is <span class='ic-hl'>{bn}</span>.
            MAE of {full_mae:.2f}s across {len(acc)} driver–race combinations in 2025.
          </div>
        </div>""", unsafe_allow_html=True)

        rm["Team"]=rm["Driver"].map(DRIVER_TEAM_2025)
        mn2=rm["Real2025AvgDelta"].min(); mx2=rm["Real2025AvgDelta"].max()
        st.markdown("<div class='cc'><div class='cc-t'>Ranking accuracy — predicted vs real delta from pole</div><div class='cc-s'>Points on the diagonal = perfect prediction</div>", unsafe_allow_html=True)
        frk=go.Figure()
        frk.add_trace(go.Scatter(x=[mn2,mx2],y=[mn2,mx2],mode="lines",
            line=dict(color="rgba(110,110,115,0.3)",dash="dot",width=1.5),showlegend=False))
        oth=rm[rm["Team"]!=sel_team]
        frk.add_trace(go.Scatter(x=oth["Real2025AvgDelta"],y=oth["DriverAvgDelta"],
            mode="markers+text",text=oth["Driver"],textposition="top center",
            textfont=dict(size=10,color="#86868b"),
            marker=dict(size=7,color="rgba(110,110,115,0.2)",line=dict(color="rgba(110,110,115,0.4)",width=0.5)),
            hovertemplate="%{text}<br>Real:%{x:.3f}s<br>Pred:%{y:.3f}s<extra></extra>",showlegend=False))
        tr2=rm[rm["Team"]==sel_team]
        if not tr2.empty:
            frk.add_trace(go.Scatter(x=tr2["Real2025AvgDelta"],y=tr2["DriverAvgDelta"],
                mode="markers+text",text=tr2["Driver"],textposition="top center",
                textfont=dict(size=11,color=d1c,weight=600),
                marker=dict(size=11,color=d1c,line=dict(color="white",width=1.5)),
                hovertemplate="%{text}<br>Real:%{x:.3f}s<br>Pred:%{y:.3f}s<extra></extra>",showlegend=False))
        fig_theme(frk,340)
        frk.update_layout(xaxis=dict(title="Real 2025 avg gap to pole (s)"),
                          yaxis=dict(title="Predicted avg gap (s)"))
        st.plotly_chart(frk,use_container_width=True,config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='cc'><div class='cc-t'>Accuracy per race</div><div class='cc-s'>Average absolute error per Grand Prix</div>", unsafe_allow_html=True)
        rm2=acc.groupby("Race")["AbsError"].mean().reset_index().sort_values("AbsError")
        frm=go.Figure(go.Bar(x=rm2["Race"],y=rm2["AbsError"],
            marker_color=["#ff3b30" if v>2 else ("#ff9f0a" if v>1 else "#34c759") for v in rm2["AbsError"]],
            hovertemplate="%{x}<br>MAE:%{y:.3f}s<extra></extra>"))
        fig_theme(frm,220); frm.update_layout(xaxis=dict(tickangle=-40,tickfont=dict(size=10)))
        st.plotly_chart(frm,use_container_width=True,config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

        if not team_acc.empty:
            st.markdown(f"<div class='g-label' style='margin-top:24px'>{sel_team} — all 2025 predictions</div>", unsafe_allow_html=True)
            rh2=""
            for _,r in team_acc.sort_values(["Race","Driver"]).iterrows():
                ec="rg" if r["AbsError"]<=0.5 else ("rb" if r["AbsError"]>1.5 else "")
                clr=d1c if r["Driver"]==drv1 else d2c
                rh2+=f"""<div class='rt-row'>
                  <span style='font-size:13px;font-weight:600;color:{clr}'>{r['Driver']}</span>
                  <span style='color:#6e6e73;font-size:12px'>{r['Race'].replace(' Grand Prix','')}</span>
                  <span style='color:#1d1d1f;font-variant-numeric:tabular-nums'>{secs_to_str(r['Real'])}</span>
                  <span style='color:#6e6e73;font-variant-numeric:tabular-nums'>{secs_to_str(r['Predicted'])}</span>
                  <span class='{ec}'>{r['Error']:+.3f}s</span>
                </div>"""
            st.markdown(f"""<div class='rt'>
              <div class='rt-row rt-hdr'><span>Driver</span><span>Race</span><span>Real</span><span>Predicted</span><span>Error</span></div>
              {rh2}
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)