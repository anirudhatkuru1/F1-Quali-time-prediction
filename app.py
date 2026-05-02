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
}
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
  background: #0a0a0a !important; color: #f5f5f7 !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 4px !important; }
.stApp > header, #MainMenu, footer,
.stDeployButton, [data-testid="stToolbar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

/* Kill Streamlit's default top padding that causes colour strips */
[data-testid="stAppViewContainer"] > section,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}

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
  margin-bottom: 8px;
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
.pg { max-width: 980px; margin: 0 auto; padding: 16px 24px 80px; }
/* ── Driver hero cards ── */
.dc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px; }
.dc {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #1a1a1a;
  transition: opacity 0.15s;
  width: 100%;
  min-width: 0;
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
    
page = st.session_state.get("page", "predict")

import streamlit.components.v1 as components
import json as _json

# ── embedded assets ───────────────────────────────────────────────────────
_CAR = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAQABgADASIAAhEBAxEB/8QAHQABAAMBAQEBAQEAAAAAAAAAAAECAwQFBgcICf/EAFQQAAICAQIEAwUEBgcFBQYCCwABAhEDBCEFEjFBUWFxBhMigZEHFDKhQlKTscHRFSMzYnKCkggWQ1PhJDRUY6IlRHOD8PEXlLLSJjVkhLPCJ0V0/8QAFgEBAQEAAAAAAAAAAAAAAAAAAAEC/8QAGxEBAQEBAQEBAQAAAAAAAAAAABEBITESAkH/2gAMAwEAAhEDEQA/AP4yAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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"

_GRID = [
  {"team":"Red Bull Racing","tc":"#3671C6","d1":"VER","d2":"LAW","r1":"","r2":"",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/L/LIALAW01_Liam_Lawson/lialaw01.png"},
  {"team":"McLaren","tc":"#FF8000","d1":"NOR","d2":"PIA","r1":"","r2":"",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png"},
  {"team":"Ferrari","tc":"#E8002D","d1":"LEC","d2":"HAM","r1":"","r2":"",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png"},
  {"team":"Mercedes","tc":"#27F4D2","d1":"RUS","d2":"ANT","r1":"","r2":"R",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/A/ANDANT01_Andrea_Antonelli/andant01.png"},
  {"team":"Aston Martin","tc":"#229971","d1":"ALO","d2":"STR","r1":"","r2":"",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png"},
  {"team":"Alpine","tc":"#FF87BC","d1":"GAS","d2":"DOO","r1":"","r2":"",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/J/JACDOO01_Jack_Doohan/jacdoo01.png"},
  {"team":"Haas F1 Team","tc":"#B6BABD","d1":"OCO","d2":"BEA","r1":"","r2":"",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/O/OLIBEA01_Oliver_Bearman/olibea01.png"},
  {"team":"RB","tc":"#6692FF","d1":"TSU","d2":"HAD","r1":"","r2":"R",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/Y/YUKTSU01_Yuki_Tsunoda/yuktsu01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/I/ISAHAD01_Isack_Hadjar/isahad01.png"},
  {"team":"Williams","tc":"#64C4FF","d1":"ALB","d2":"SAI","r1":"","r2":"",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/A/ALEALB01_Alexander_Albon/alealb01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png"},
  {"team":"Kick Sauber","tc":"#52E252","d1":"HUL","d2":"BOR","r1":"","r2":"R",
   "img1":"https://www.formula1.com/content/dam/fom-website/drivers/N/NICHUL01_Nico_Hulkenberg/nichul01.png",
   "img2":"https://www.formula1.com/content/dam/fom-website/drivers/G/GABBOR01_Gabriel_Bortoleto/gabbor01.png"},
]
_TEAMS = [g["team"] for g in _GRID]

# ── HERO HTML ─────────────────────────────────────────────────────────────
def _hero_html():
    return (
        """<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@900&family=Dancing+Script:wght@700&display=swap');
@font-face{font-family:'Holiday';src:url('data:font/truetype;base64,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ADcAOAA5ADoAOwA8AD0ARABFAEYARwBIAEkASgBLAEwATQBOAE8AUABRAFIAUwBUAFUAVgBXAFgAWQBaAFsAXABdAKwBBAEFCWNvbnRyb2xMRgljb250cm9sQ1IKc29mdGh5cGhlbgZ5LmFsdDEAAAAAAAH//wACAAEAAAAMAAAAFgAAAAIAAQABAEoAAQAEAAAAAgAAAAAAAQAAAAoAMAA+AAJERkxUAA5sYXRuABoABAAAAAD//wABAAAABAAAAAD//wABAAAAAWFhbHQACAAAAAEAAAABAAQAAQAAAAEACAABAAYABAABAAEARgAAAAEAAAAKADAAPgACREZMVAAObGF0bgAaAAQAAAAA//8AAQAAAAQAAAAA//8AAQAAAAFrZXJuAAgAAAABAAAAAQAEAAIAAAABAAgAAQF6AAQAAAAPACgAbgCIALYAwAD2AQgBFgEgASYBOAFKAVABWgFgABEALgA3ADAAMQAxACwAMwASADQAJAA2ADcANwAvADkAEgA8ACoAQABDAEEAEgBCACQAQwAZAEQAMQBFAFwARgAeAEcAOQAGAC//9AAy//oANgAMADcAJAA4//QARwAMAAsALgAxADAAHwAxACoAMgAeADQAJAA2ACQAOwAYADwAGAA+AB8APwAYAEQAQgACADj/9ABHABIADQAuAFQAMQA9ADIAKgA0ACsANgAYADsAPAA8ACoAPQAkAD4AKgA/ACoAQgAkAEQAKgBFACoABAAv//QAMv/oADj/7gA+//QAAwAy//QAOP/uAEcAEgACADj/9ABHAAwAAQAuABgABAAuACoAMQAeADQAGAA2ABgABAAuAB4AMAAeADEAJABEAB4AAQA4//QAAgAuAAwARAAeAAEAOP/6AAYALgAqADAAHgAxABgAMgASADQAJQA2AB4AAQAPABYALgAvADAAMwA0ADUANgA8AD0APgBAAEEAQgBDAAAAAAABAAAAANre18UAAAAA2FmU1QAAAADYWZTV') format('truetype')}
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#0a0a0a}
.hero{position:relative;width:100vw;height:100vh;overflow:hidden;background:#0a0a0a}
.car-bg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center 0%;z-index:0}
/* Title block: pushed higher — top:6% */
.title-block{
  position:absolute;
  top:6%;
  left:0;right:0;
  z-index:2;
  padding:0 17%;
  display:flex;flex-direction:column;align-items:flex-start;
}
/* F1 QUALIFYING — massive condensed */
.t-f1{
  font-family:'Barlow Condensed',sans-serif;
  font-size:clamp(80px,12.8vw,174px);
  font-weight:900;
  margin-left:0.40em;
  line-height:0.90;
  letter-spacing:-0.01em;
  color:#b9bbc4;
  text-transform:uppercase;
  white-space:nowrap;
  display:block;
}
/* Prediction. — overlapping: negative margin pulls it UP into F1 QUALIFYING */
.t-pred{
  font-family:'Holiday','Dancing Script',cursive;
  font-size:clamp(52px,7.5vw,102px);
  font-weight:350;
  color:#cc0000;
  line-height:1;
  display:block;
  margin-top:-0.30em;
  margin-left:2.55em;
  position:relative;
  z-index:3;
}
/* Bottom bar */
.bottom{
  position:absolute;bottom:0;left:0;right:0;
  z-index:3;height:54px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 30px;
}
.credit{
  font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;
  font-size:11.5px;font-weight:700;letter-spacing:0.13em;
  text-transform:uppercase;color:rgba(255,255,255,0.82);
}
.cta{
  font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;
  font-size:13px;font-weight:700;letter-spacing:0.12em;
  text-transform:uppercase;color:rgba(255,255,255,0.88);
  display:flex;align-items:center;gap:10px;
  cursor:pointer;transition:opacity 0.15s;user-select:none;
}
.cta:hover{opacity:0.7}
</style>
</head>
<body>
<div class="hero">
  <img class="car-bg" src="data:image/jpeg;base64,""" + _CAR + """" alt=""/>
  <div class="title-block">
    <span class="t-f1">F1 Qualifying</span>
    <span class="t-pred">Prediction.</span>
  </div>
  <div class="bottom">
    <span class="credit">Done by Anirudh Atkuru</span>
    <span class="cta" id="cta">SELECT YOUR TEAM &nbsp;&#8594;</span>
  </div>
</div>
<script>
// Robust click handler — tries multiple strategies
function triggerNav() {
  var doc = window.parent.document;
  // Strategy 1: find button whose text contains 'go'
  var btns = doc.querySelectorAll('[data-testid="stButton"] button');
  for (var i = 0; i < btns.length; i++) {
    var txt = (btns[i].textContent || btns[i].innerText || '').split(' ').join('');
    if (txt === 'go') { btns[i].click(); return; }
  }
  // Strategy 2: find first button with key hero_go via aria-label or any button
  var heroBtn = doc.querySelector('button[kind="secondary"], [data-testid="stButton"] button');
  if (heroBtn) { heroBtn.click(); return; }
  // Strategy 3: URL query param fallback
  window.parent.location.search = '?nav=selector';
}
document.getElementById('cta').addEventListener('click', triggerNav);
</script>
</body></html>"""
    )


def _selector_html():
    gj = _json.dumps(_GRID)
    return (
        """<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800;900&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
html,body{background:#0d0d0d;min-height:100%;height:100%}
body{font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
/* ── title ── */
.hdr{text-align:center;padding:44px 24px 32px;background:#0d0d0d}
.hdr h1{font-family:'Barlow Condensed',sans-serif;
  font-size:clamp(52px,8vw,96px);font-weight:900;
  color:#fff;text-transform:uppercase;letter-spacing:0.02em;line-height:1}
/* ── grid container ── */
.container {background: #0d0d0d;padding: 0px 32px 60px;max-width: 1100px;margin: 0 auto;display: grid;grid-template-columns: 1fr 1fr;gap: 6px;}
  
/* Each row = one team: 4-col grid (2 drivers, each with photo+name) */
.team-row {display: grid; grid-template-columns: 1fr 1fr; gap: 0; cursor: pointer; border-radius: 6px; overflow: hidden;}
.team-row:hover .dc{opacity:0.88}
.team-row:active{transform:scale(0.995)}
/* driver cell */
.dc {
  position: relative; display: flex;
  flex-direction: column; align-items: center; background: #1a1a1a; transition: opacity 0.15s; width: 100%; min-width: 0;
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
  border-top:0.5px solid rgba(255,255,255,0.08);
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
<div style="padding:20px 32px 8px;position:sticky;top:0;background:rgba(20,20,20,0.95);border-bottom:0.5px solid rgba(255,255,255,0.08);z-index:10;">
  <div id="backbtn" style="font-size:13px;color:rgba(255,255,255,0.5);cursor:pointer;display:inline-flex;align-items:center;gap:6px;padding:6px 0;">&#8592; Back</div>
</div>
<div style="padding:32px 32px 8px 32px;max-width:1100px;margin:0 auto;">
  <div style="font-family:'Barlow Condensed',sans-serif;font-size:clamp(52px,8vw,96px);font-weight:900;color:#b9bbc4;text-transform:uppercase;letter-spacing:-0.01em;line-height:0.95;margin-left:2.55em;margin-top:0.20em">Select a Team</div>
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
    components.html(_hero_html(), height=900, scrolling=False)

    if st.button("go", key="hero_go"):
        st.session_state.screen = "selector"
        st.rerun()
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2: TEAM SELECTOR
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "selector":
    st.markdown("""<style>
    html, body { margin:0 !important; padding:0 !important; background:#0d0d0d !important; }
    .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
    .main,.block-container,[data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main,.element-container,div.stMarkdown {
      background:#0d0d0d !important;
      padding:0 !important;margin:0 !important;gap:0 !important;
      min-height:0 !important;
    }
    header,[data-testid="stHeader"],footer,#MainMenu,
    [data-testid="stToolbar"],[data-testid="stDecoration"],
    [data-testid="stStatusWidget"]{display:none !important;}
    iframe{display:block !important;border:none !important;margin:0 !important;padding:0 !important;}
    [data-testid="stButton"] > button{
      opacity:0 !important;height:1px !important;min-height:1px !important;
      padding:0 !important;margin:0 !important;font-size:1px !important;
      border:none !important;background:transparent !important;
      overflow:hidden !important;
    }
    .app-nav{display:none !important;}
    </style>""", unsafe_allow_html=True)

    components.html(_selector_html(), height=1950, scrolling=False)

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
st.markdown("""<style>
/* Hide all tab buttons from normal flow — we render them via HTML */
div[data-testid="stColumns"] { display:none !important; }
</style>""", unsafe_allow_html=True)

# Invisible Streamlit buttons (used as click targets from HTML)
col_pred, col_anal, col_change = st.columns([1, 1, 1])
with col_pred:
    if st.button("Predict", use_container_width=True, type="primary" if page=="predict" else "secondary", key="tab_predict"):
        st.session_state.page = "predict"; st.rerun()
with col_anal:
    if st.button("Analyse", use_container_width=True, type="primary" if page=="analyse" else "secondary", key="tab_analyse"):
        st.session_state.page = "analyse"; st.rerun()
with col_change:
    if st.button("↩ Change Team", use_container_width=True, key="tab_change"):
        st.session_state.team = None; st.session_state.screen = "selector"; st.rerun()

# Full nav bar rendered as HTML — centered title + subtle side controls
page_label = "Predict" if page == "predict" else "Analyse"
other_label = "Analyse →" if page == "predict" else "← Predict"
other_key   = "tab_analyse" if page == "predict" else "tab_predict"

st.markdown(f"""
<style>
.nav-bar {{
  position: sticky; top: 0; z-index: 999;
  background: rgba(10,10,10,0.96);
  backdrop-filter: blur(20px);
  border-bottom: 0.5px solid rgba(255,255,255,0.07);
  height: 56px;
  display: flex; align-items: center;
  padding: 0 28px;
  margin-bottom: 0;
}}
.nav-center {{
  position: absolute; left: 50%; transform: translateX(-50%);
  font-size: 11px; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: rgba(255,255,255,0.9);
}}
.nav-left {{
  font-size: 12px; font-weight: 500; color: rgba(255,255,255,0.35);
  letter-spacing: 0.08em; text-transform: uppercase; cursor: pointer;
  display: flex; align-items: center; gap: 6px;
}}
.nav-right {{
  margin-left: auto;
  font-size: 12px; font-weight: 500; color: rgba(255,255,255,0.35);
  letter-spacing: 0.08em; text-transform: uppercase; cursor: pointer;
  display: flex; align-items: center; gap: 6px;
}}
.nav-dot {{ 
  width: 6px; height: 6px; border-radius: 50%;
  background: #e10600; display: inline-block;
}}
</style>
<div class='nav-bar'>
  <span class='nav-left' onclick="(function(){{
    var btns = window.parent.document.querySelectorAll('[data-testid=\\'stButton\\'] button');
    btns.forEach(b => {{ if(b.textContent.trim()==='__change__') b.click(); }});
  }})()">↩ Change team</span>
  <span class='nav-center'><span class='nav-dot'></span>&nbsp;&nbsp;{page_label}</span>
  <span class='nav-right' onclick="(function(){{
    var btns = window.parent.document.querySelectorAll('[data-testid=\\'stButton\\'] button');
    var key = '{other_key}';
    btns.forEach(b => {{ if(b.textContent.trim()==='__{('analyse' if page == 'predict' else 'predict')}__') b.click(); }});
  }})()">{other_label}</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# PREDICT PAGE
# ════════════════════════════════════════
if page == "predict":
    st.markdown("""<style>
    .stApp,[data-testid='stAppViewContainer'],[data-testid='stMain'],.main,.block-container{
      background:#0a0a0a !important;
    }
    /* Hide selectbox labels visually but keep layout space intact */
    div[data-testid='stSelectbox'] label {
      visibility: hidden !important;
      height: 0px !important;
      min-height: 0px !important;
      margin: 0 !important;
      padding: 0 !important;
      font-size: 0 !important;
      line-height: 0 !important;
    }
    /* Fix expander arrow overlapping the summary text */
    div[data-testid='stExpander'] details summary {
      display: flex !important;
      align-items: center !important;
      gap: 8px !important;
    }
    div[data-testid='stExpander'] details summary svg {
      flex-shrink: 0 !important;
      position: relative !important;
      top: auto !important;
      left: auto !important;
    }
    div[data-testid='stExpander']{background:var(--sur) !important;border-color:var(--bdr) !important}
    div[data-testid='stExpander'] *{color:var(--t1) !important}
    div[data-testid='stSlider'] label{color:var(--t1) !important}
    div[data-testid='stSlider'] .stMarkdown{color:var(--t1) !important}
    label[data-testid='stWidgetLabel']{color:var(--t1) !important}
    .stSelectbox label, .stSlider label{color:var(--t1) !important}
    </style>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns([3,1,1])
    with c1: sel_event = st.selectbox("Race", ALL_EVENTS, key="pe")
    with c2: seg = st.selectbox("Session", ["Q3","Q2","Q1"], key="ps")
    with c3: cmp = st.selectbox("Compound", ["SOFT","MEDIUM","HARD"], key="pc")
     
    hist_wx = get_wx(sel_event)
    wx = dict(**hist_wx)
    wx_str = f"🌡 {hist_wx['AirTemp']:.0f}°C air · {hist_wx['TrackTemp']:.0f}°C track · {hist_wx['Humidity']:.0f}% humidity"
    edit_wx = st.checkbox("Edit weather", key="wx_toggle")
    if edit_wx:
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
    else:
        wx = dict(**hist_wx)

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
    <div style='font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
      color:rgba(255,255,255,0.25);margin-top:20px;margin-bottom:6px'>{sel_team}</div>
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
    st.markdown(f"<div class='pg-title' style='margin-top:8px'>Analyse</div>", unsafe_allow_html=True)
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
        
    sub = st.session_state.sub
    _sub_map = {"season":"Season","r&d":"R&D","accuracy":"Accuracy"}
    _sub_opts = ["Season","R&D","Accuracy"]
    _sub_idx = _sub_opts.index(_sub_map.get(sub, "Season"))
    sub_choice = st.radio("sub", _sub_opts, horizontal=True,
                          label_visibility="collapsed", key="sub_r",
                          index=_sub_idx)
    st.session_state.sub = sub_choice.lower().replace("r&d","r&d")
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