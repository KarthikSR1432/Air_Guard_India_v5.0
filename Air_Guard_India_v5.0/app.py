from __future__ import annotations

import io
import html
import json
import os
import hashlib
import smtplib
import sqlite3
from email.message import EmailMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

load_dotenv()

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

try:
    from streamlit_geolocation import streamlit_geolocation
    GEOLOCATION_AVAILABLE = True
except ImportError:
    GEOLOCATION_AVAILABLE = False

# ---------------------------------------------------------------------------
# Air-Guard v5.0
# Providers: Open-Meteo Geocoding, Air Quality (CAMS), Weather Forecast,
# and Historical Weather APIs. Provider AQI is US AQI, never relabelled CPCB.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Air-Guard | Air Intelligence", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

IST = ZoneInfo("Asia/Kolkata")
TODAY = datetime.now(IST).date()
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "air_guard_v4.db"

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WX_URL = "https://api.open-meteo.com/v1/forecast"
WX_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AQ_CURRENT = "us_aqi,european_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
AQ_HOURLY = AQ_CURRENT
WX_CURRENT = "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,weather_code"
WX_DAILY = "temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean,precipitation_sum,pressure_msl_mean"

DEFAULT_CITIES = [
    ("Delhi", 28.6139, 77.2090, "India", "Delhi", "Asia/Kolkata"),
    ("Mumbai", 19.0760, 72.8777, "India", "Maharashtra", "Asia/Kolkata"),
    ("Bengaluru", 12.9716, 77.5946, "India", "Karnataka", "Asia/Kolkata"),
    ("Chennai", 13.0827, 80.2707, "India", "Tamil Nadu", "Asia/Kolkata"),
    ("Kolkata", 22.5726, 88.3639, "India", "West Bengal", "Asia/Kolkata"),
    ("Hyderabad", 17.3850, 78.4867, "India", "Telangana", "Asia/Kolkata"),
    ("Pune", 18.5204, 73.8567, "India", "Maharashtra", "Asia/Kolkata"),
    ("Ahmedabad", 23.0225, 72.5714, "India", "Gujarat", "Asia/Kolkata"),
    ("Jaipur", 26.9124, 75.7873, "India", "Rajasthan", "Asia/Kolkata"),
    ("Lucknow", 26.8467, 80.9462, "India", "Uttar Pradesh", "Asia/Kolkata"),
    ("Kochi", 9.9312, 76.2673, "India", "Kerala", "Asia/Kolkata"),
    ("Guwahati", 26.1445, 91.7362, "India", "Assam", "Asia/Kolkata"),
    ("Surat", 21.1702, 72.8311, "India", "Gujarat", "Asia/Kolkata"),
    ("Nagpur", 21.1458, 79.0882, "India", "Maharashtra", "Asia/Kolkata"),
    ("Indore", 22.7196, 75.8577, "India", "Madhya Pradesh", "Asia/Kolkata"),
    ("Visakhapatnam", 17.6868, 83.2185, "India", "Andhra Pradesh", "Asia/Kolkata"),
    ("Varanasi", 25.3176, 82.9739, "India", "Uttar Pradesh", "Asia/Kolkata"),
    ("Amritsar", 31.6340, 74.8723, "India", "Punjab", "Asia/Kolkata"),
    ("Dehradun", 30.3165, 78.0322, "India", "Uttarakhand", "Asia/Kolkata"),
    ("Bhubaneswar", 20.2961, 85.8245, "India", "Odisha", "Asia/Kolkata"),
    ("Raipur", 21.2514, 81.6296, "India", "Chhattisgarh", "Asia/Kolkata"),
    ("Thiruvananthapuram", 8.5241, 76.9366, "India", "Kerala", "Asia/Kolkata"),
    ("Mysuru", 12.2958, 76.6394, "India", "Karnataka", "Asia/Kolkata"),
    ("Jodhpur", 26.2389, 73.0243, "India", "Rajasthan", "Asia/Kolkata"),
    ("Ranchi", 23.3441, 85.3096, "India", "Jharkhand", "Asia/Kolkata"),
]

def loc_key(loc):
    return f"{loc['name']}|{float(loc['latitude']):.4f}|{float(loc['longitude']):.4f}"

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS observations (
            city_key TEXT NOT NULL, city TEXT NOT NULL, timestamp TEXT NOT NULL,
            aqi REAL, pm25 REAL, pm10 REAL, co REAL, no2 REAL, so2 REAL, o3 REAL,
            source TEXT NOT NULL, PRIMARY KEY(city_key,timestamp,source))""")
        con.execute("""CREATE TABLE IF NOT EXISTS current_snapshots (
            city_key TEXT NOT NULL, city TEXT NOT NULL, fetched_at TEXT NOT NULL,
            observation_time TEXT, aqi REAL, pm25 REAL, pm10 REAL, co REAL, no2 REAL,
            so2 REAL, o3 REAL, source TEXT NOT NULL, PRIMARY KEY(city_key,fetched_at))""")
        con.execute("""CREATE TABLE IF NOT EXISTS pollution_alerts (
            alert_id TEXT PRIMARY KEY, dedupe_key TEXT NOT NULL UNIQUE,
            city TEXT NOT NULL, state TEXT, aqi REAL, category TEXT NOT NULL,
            message TEXT NOT NULL, event_time TEXT NOT NULL, alert_type TEXT NOT NULL,
            notification_status TEXT NOT NULL DEFAULT 'In-app',
            is_read INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS alert_preferences (
            id INTEGER PRIMARY KEY CHECK (id=1), settings_json TEXT NOT NULL,
            updated_at TEXT NOT NULL)""")
        con.commit()

def num(x):
    try:
        return None if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return None

def save_history(loc, frame):
    if frame.empty:
        return
    rows = []
    for _, r in frame.iterrows():
        rows.append((loc_key(loc), loc["name"], str(r["time"]), num(r.get("AQI")),
                     num(r.get("PM2.5")), num(r.get("PM10")), num(r.get("CO")),
                     num(r.get("NO₂")), num(r.get("SO₂")), num(r.get("O₃")),
                     str(r.get("Source", "Open-Meteo historical model estimate"))))
    with sqlite3.connect(DB_PATH) as con:
        con.executemany("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()

def load_history(loc, start, end):
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            'SELECT timestamp AS time, aqi AS AQI, pm25 AS "PM2.5", pm10 AS "PM10", '
            'co AS CO, no2 AS "NO₂", so2 AS "SO₂", o3 AS "O₃", source AS Source '
            'FROM observations WHERE city_key=? AND timestamp>=? AND timestamp<? ORDER BY timestamp',
            con, params=(loc_key(loc), start.isoformat(), (end + timedelta(days=1)).isoformat()))
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["City"] = loc["label"]
        df["Data type"] = "Previously retrieved provider history (SQLite)"
    return df.dropna(subset=["time"]) if not df.empty else df

def save_current(loc, row):
    fetched = datetime.now(IST).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR REPLACE INTO current_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (loc_key(loc), loc["label"], fetched, row.get("Observation time"), num(row.get("AQI")),
             num(row.get("PM2.5")), num(row.get("PM10")), num(row.get("CO")),
             num(row.get("NO₂")), num(row.get("SO₂")), num(row.get("O₃")), row.get("Source", "Open-Meteo")))
        con.commit()

def get_json(url, params):
    resp = requests.get(url, params=params, timeout=(8, 35),
                        headers={"User-Agent": "AirGuard/4.0 (Streamlit air quality dashboard)"})
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=600, show_spinner=False)
def geocode(query):
    if len(query.strip()) < 2:
        return []
    data = get_json(GEO_URL, {"name": query.strip(), "count": 10, "language": "en", "format": "json"})
    return data.get("results", [])

@st.cache_data(ttl=600, show_spinner=False)
def fetch_current(lat, lon, tz):
    aq = get_json(AQ_URL, {"latitude": lat, "longitude": lon, "current": AQ_CURRENT,
                           "timezone": tz or "auto", "forecast_days": 1})
    cur = aq.get("current", {})
    # Weather is an enrichment: a weather-provider issue must not hide an AQI reading.
    try:
        wx = get_json(WX_URL, {"latitude": lat, "longitude": lon, "current": WX_CURRENT,
                               "timezone": tz or "auto"})
        weather = wx.get("current", {})
    except Exception:
        weather = {}
    return {
        "AQI": cur.get("us_aqi"), "European AQI": cur.get("european_aqi"),
        "PM2.5": cur.get("pm2_5"), "PM10": cur.get("pm10"),
        "CO": cur.get("carbon_monoxide"), "NO₂": cur.get("nitrogen_dioxide"),
        "SO₂": cur.get("sulphur_dioxide"), "O₃": cur.get("ozone"),
        "Observation time": cur.get("time"), "Weather time": weather.get("time"),
        "Temperature (°C)": weather.get("temperature_2m"),
        "Humidity (%)": weather.get("relative_humidity_2m"),
        "Wind (km/h)": weather.get("wind_speed_10m"),
        "Pressure (hPa)": weather.get("pressure_msl"),
        "Weather code": weather.get("weather_code"),
        "Source": "Open-Meteo CAMS model estimate",
    }

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_air_history(lat, lon, start, end, tz):
    payload = get_json(AQ_URL, {"latitude": lat, "longitude": lon, "hourly": AQ_HOURLY,
        "start_date": start, "end_date": end, "timezone": tz or "auto"})
    h = payload.get("hourly", {})
    times = h.get("time", [])
    fields = {"us_aqi":"AQI", "pm2_5":"PM2.5", "pm10":"PM10",
              "carbon_monoxide":"CO", "nitrogen_dioxide":"NO₂",
              "sulphur_dioxide":"SO₂", "ozone":"O₃"}
    data = {"time": pd.to_datetime(times, errors="coerce")}
    for k, v in fields.items():
        vals = h.get(k, [])
        if len(vals) == len(times):
            data[v] = vals
    out = pd.DataFrame(data).dropna(subset=["time"])
    out["Source"] = "Open-Meteo CAMS historical model estimate"
    out["Data type"] = "Historical provider estimate"
    return out

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_weather_history(lat, lon, start, end, tz):
    p = {"latitude":lat,"longitude":lon,"start_date":start,"end_date":end,
         "daily":WX_DAILY,"timezone":tz or "auto"}
    data = get_json(WX_ARCHIVE_URL, p).get("daily", {})
    return pd.DataFrame({
        "date":pd.to_datetime(data.get("time",[])).date,
        "temperature_2m":data.get("temperature_2m_mean",[]),
        "relative_humidity_2m":data.get("relative_humidity_2m_mean",[]),
        "wind_speed_10m":data.get("wind_speed_10m_mean",[]),
        "precipitation":data.get("precipitation_sum",[]),
        "pressure_msl":data.get("pressure_msl_mean",[]),
    })

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_forecast(lat, lon, tz):
    p = {"latitude":lat,"longitude":lon,"daily":WX_DAILY,"forecast_days":8,"timezone":tz or "auto"}
    data = get_json(WX_URL, p).get("daily", {})
    return pd.DataFrame({
        "date":pd.to_datetime(data.get("time",[])).date,
        "temperature_2m":data.get("temperature_2m_mean",[]),
        "relative_humidity_2m":data.get("relative_humidity_2m_mean",[]),
        "wind_speed_10m":data.get("wind_speed_10m_mean",[]),
        "precipitation":data.get("precipitation_sum",[]),
        "pressure_msl":data.get("pressure_msl_mean",[]),
    })

def aqi_category(value):
    if value is None or pd.isna(value): return "Unavailable", "⚪"
    v = float(value)
    if v <= 50: return "Good", "🟢"
    if v <= 100: return "Moderate", "🟡"
    if v <= 150: return "Unhealthy for sensitive groups", "🟠"
    if v <= 200: return "Unhealthy", "🔴"
    if v <= 300: return "Very unhealthy", "🟣"
    return "Hazardous", "🟤"

def daily_air(hourly):
    if hourly.empty or "AQI" not in hourly:
        return pd.DataFrame(columns=["date","AQI","PM2.5","PM10"])
    x = hourly.dropna(subset=["AQI"]).copy()
    x["date"] = x["time"].dt.date
    aggs = {"AQI":("AQI","mean")}
    for p in ["PM2.5","PM10"]:
        if p in x.columns: aggs[p] = (p,"mean")
    return x.groupby("date",as_index=False).agg(**aggs)

FEATURES = ["lag1","lag2","lag3","roll7","temperature_2m",
            "relative_humidity_2m","wind_speed_10m","precipitation","pressure_msl","doy_sin","doy_cos"]

def add_features(daily):
    x = daily.sort_values("date").copy()
    x["lag1"] = x["AQI"].shift(1)
    x["lag2"] = x["AQI"].shift(2)
    x["lag3"] = x["AQI"].shift(3)
    x["roll7"] = x["AQI"].shift(1).rolling(7,min_periods=4).mean()
    doy = pd.to_datetime(x["date"]).dt.dayofyear
    x["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    x["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    return x

def predict_city(loc, air_hist, weather_hist):
    ah = daily_air(air_hist)
    if ah.empty or weather_hist.empty:
        raise ValueError("Historical AQI or weather data is missing.")
    training = add_features(ah.merge(weather_hist,on="date",how="inner"))
    training = training.dropna(subset=FEATURES+["AQI"])
    if len(training) < 25:
        raise ValueError(f"Only {len(training)} usable daily rows; at least 25 are needed.")
    model = RandomForestRegressor(n_estimators=220,max_depth=7,min_samples_leaf=2,
                                  random_state=42,n_jobs=-1)
    model.fit(training[FEATURES],training["AQI"])
    mae = None
    if len(training) >= 40:
        split = int(len(training)*.8)
        test = training.iloc[split:]
        check = RandomForestRegressor(n_estimators=140,max_depth=7,min_samples_leaf=2,
                                      random_state=42,n_jobs=-1)
        check.fit(training.iloc[:split][FEATURES],training.iloc[:split]["AQI"])
        mae = float(np.mean(np.abs(check.predict(test[FEATURES])-test["AQI"])))
    future = fetch_weather_forecast(loc["latitude"],loc["longitude"],loc.get("timezone","auto"))
    last = max(ah["date"])
    future = future[future["date"] > max(TODAY,last)].sort_values("date").head(7)
    values = list(ah.sort_values("date")["AQI"].dropna().astype(float))
    results = []
    for _, w in future.iterrows():
        doy = pd.Timestamp(w["date"]).dayofyear
        features = {"lag1":values[-1],"lag2":values[-2],"lag3":values[-3],
                    "roll7":float(np.mean(values[-7:])),
                    "temperature_2m":w["temperature_2m"],
                    "relative_humidity_2m":w["relative_humidity_2m"],
                    "wind_speed_10m":w["wind_speed_10m"],
                    "precipitation":w["precipitation"],"pressure_msl":w["pressure_msl"],
                    "doy_sin":np.sin(2*np.pi*doy/365.25),"doy_cos":np.cos(2*np.pi*doy/365.25)}
        pred = float(np.clip(model.predict(pd.DataFrame([features])[FEATURES])[0],0,500))
        results.append({"City":loc["label"],"Date":w["date"],"Predicted AQI":pred,
                        "Category":aqi_category(pred)[0],"Data type":"AI/ML prediction"})
        values.append(pred)
    if not results: raise ValueError("No future weather rows available for the next seven days.")
    return pd.DataFrame(results),len(training),mae

def health_advice(aqi):
    cat,_ = aqi_category(aqi)
    advice = {
      "Good": ("Normal outdoor activity is generally fine.","No special precautions normally needed."),
      "Moderate": ("Unusually sensitive people should monitor symptoms and reduce activity if affected.","Keep indoor spaces ventilated when outdoor air is clean."),
      "Unhealthy for sensitive groups": ("Children, older adults, and people with heart/lung disease should reduce prolonged outdoor exertion.","Keep windows closed during local pollution peaks; use a well-fitted HEPA purifier if available."),
      "Unhealthy": ("Everyone should reduce prolonged or heavy outdoor exertion; sensitive groups should avoid strenuous outdoor activity.","Use recirculation where appropriate and reduce indoor particle sources."),
      "Very unhealthy": ("Avoid prolonged outdoor exertion; sensitive people should remain indoors in cleaner air.","Use a correctly sized HEPA air cleaner and avoid smoking, incense and unnecessary combustion."),
      "Hazardous": ("Avoid outdoor physical activity and follow local public-health alerts.","Stay in a cleaner indoor space; seek medical help for severe breathing difficulty or chest pain."),
      "Unavailable": ("AQI is unavailable; consult local official advisories.","No AQI-based recommendation can be generated.")
    }
    return cat, advice.get(cat,advice["Unavailable"])

def pdf_report(title, frame):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=landscape(A4),rightMargin=24,leftMargin=24,topMargin=24,bottomMargin=24)
    styles = getSampleStyleSheet()
    story = [Paragraph(title,styles["Title"]),Paragraph(f"Generated {datetime.now(IST):%d %b %Y %I:%M %p IST}",styles["Normal"]),Spacer(1,12)]
    safe = frame.copy().replace({np.nan:None})
    data = [list(safe.columns)] + [[str(v)[:45] if v is not None else "—" for v in row] for row in safe.itertuples(index=False,name=None)]
    tbl = Table(data,repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#135d46")),
      ("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.25,colors.grey),
      ("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
      ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef5f1")])]))
    story.append(tbl)
    doc.build(story)
    return buf.getvalue()

# ---------- UI helpers ----------
init_db()
if "watchlist" not in st.session_state:
    st.session_state.watchlist = [
      {"name":n,"latitude":lat,"longitude":lon,"country":co,"admin1":admin,
       "timezone":tz,"label":f"{n}, {co}","region":admin}
      for n,lat,lon,co,admin,tz in DEFAULT_CITIES]
if "search_results" not in st.session_state: st.session_state.search_results = []

st.markdown("""<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem}
.hero{padding:1.3rem 1.5rem;border-radius:18px;background:linear-gradient(110deg,#073b2d,#147650);color:white}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;opacity:.9}
[data-testid="stMetric"]{background:var(--secondary-background-color);padding:12px;border-radius:12px}
@media(max-width:700px){.block-container{padding-left:.8rem;padding-right:.8rem}.hero h1{font-size:1.5rem}}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌿 AIR-GUARD</h1><p>Air Quality Intelligence · Global city search · Historical analytics · AI forecasts · Health advisories</p></div>',unsafe_allow_html=True)
st.caption("India-wide pollution map · Configurable alerts · Global city search · All displayed timestamps follow each location’s time zone · Provider data provenance is shown throughout")

# Keep city-based selectors synchronized with the primary location.
def sync_city_selection():
    primary = st.session_state.get("active_city")
    if primary:
        st.session_state["selected_locations"] = [primary]

# sidebar location management
with st.sidebar:
    st.markdown("## 📍 Location & Watchlist")
    query = st.text_input("Search any city or region",placeholder="e.g. Bengaluru, India or London")
    if query.strip():
        try:
            st.session_state.search_results = geocode(query)
        except Exception as exc:
            st.warning(f"Location search unavailable: {exc}")
            st.session_state.search_results = []
    results = st.session_state.search_results
    if results:
        options = [f"{r['name']}, {r.get('admin1','')}, {r.get('country','')} · {r['latitude']:.3f},{r['longitude']:.3f}" for r in results]
        choice = st.selectbox("Suggestions",options,key="geo_choice")
        if st.button("＋ Add selected city",use_container_width=True):
            item = results[options.index(choice)]
            loc = {"name":item["name"],"latitude":item["latitude"],"longitude":item["longitude"],
                   "country":item.get("country",""),"admin1":item.get("admin1",""),
                   "timezone":item.get("timezone","auto"),
                   "label":", ".join([p for p in [item['name'], item.get('admin1',''), item.get('country','')] if p]),"region":item.get("admin1","")}
            existing = next((x for x in st.session_state.watchlist if loc_key(x)==loc_key(loc)), None)
            if existing is None:
                st.session_state.watchlist.append(loc)
                existing = loc
                st.success(f"Added {loc['label']}")
            else:
                st.info("This city is already monitored; switched the dashboard to it.")
            st.session_state.active_city = existing["label"]
            st.session_state["selected_locations"] = [existing["label"]]
            st.rerun()
    else:
        if query.strip(): st.caption("No matching location found. Try adding a country name.")
    st.divider()
    if GEOLOCATION_AVAILABLE:
        try:
            geo = streamlit_geolocation()
            if isinstance(geo,dict) and geo.get("latitude") is not None and geo.get("longitude") is not None:
                if st.button("＋ Add my current location",use_container_width=True):
                    lat,lon=float(geo["latitude"]),float(geo["longitude"])
                    loc={"name":"My location","latitude":lat,"longitude":lon,"country":"Coordinates",
                         "admin1":"","timezone":"auto","label":f"My location ({lat:.3f}, {lon:.3f})","region":"—"}
                    if not any(loc_key(x)==loc_key(loc) for x in st.session_state.watchlist):
                        st.session_state.watchlist.append(loc); st.session_state.active_city = loc["label"]; st.session_state["selected_locations"] = [loc["label"]]; st.rerun()
        except Exception:
            st.caption("Location permission was not granted or is unavailable.")
    else:
        st.caption("Install streamlit-geolocation to enable browser location permission.")
    labels = [x["label"] for x in st.session_state.watchlist]
    if st.session_state.get("active_city") not in labels:
        st.session_state.active_city = labels[0] if labels else None
    remove = st.selectbox("Remove a monitored city",["—"]+labels,key="remove_city")
    if remove != "—" and st.button("Remove city"):
        st.session_state.watchlist=[x for x in st.session_state.watchlist if x["label"]!=remove]
        if st.session_state.get("active_city") == remove:
            st.session_state.active_city = st.session_state.watchlist[0]["label"] if st.session_state.watchlist else None
            if st.session_state.active_city:
                st.session_state["selected_locations"] = [st.session_state.active_city]
        st.rerun()
    if labels:
        st.selectbox("Primary city (drives all sections)", labels, key="active_city",
                     on_change=sync_city_selection)
    page = st.radio("NAVIGATION",["Overview","Current AQI Monitoring","Historical AQI Analysis",
        "Future AQI Prediction","City Comparison","Health Precautions","Reports & Analytics",
        "India Pollution Map","Pollution Alerts"],
        index=0,key="nav",label_visibility="visible")
    auto_refresh = st.toggle("Auto-refresh current readings (10 min)",value=True)
    if st.button("↻ Refresh data now",use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if auto_refresh and AUTO_REFRESH_AVAILABLE:
    st_autorefresh(interval=600_000,key="airguard-refresh")
elif auto_refresh:
    st.caption("Install streamlit-autorefresh for timed automatic refresh; manual refresh remains available.")

watchlist = st.session_state.watchlist
labels = [x["label"] for x in watchlist]
if not watchlist:
    st.warning("Add at least one city using the search box.")
    st.stop()

@st.cache_data(ttl=600,show_spinner=False)
def current_for_location(lat,lon,tz):
    return fetch_current(lat,lon,tz)

def get_currents(locations):
    rows,errors=[],{}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs={pool.submit(current_for_location,x["latitude"],x["longitude"],x.get("timezone","auto")):x for x in locations}
        for f in as_completed(futs):
            loc=futs[f]
            try:
                row=f.result()
                row.update({"City":loc["label"],"Region":loc.get("region",""),"Latitude":loc["latitude"],"Longitude":loc["longitude"]})
                rows.append(row); save_current(loc,row)
            except Exception as exc: errors[loc["label"]]=str(exc)
    return pd.DataFrame(rows),errors

def get_history(loc,start,end):
    errors=[]
    try:
        frame=fetch_air_history(loc["latitude"],loc["longitude"],start.isoformat(),end.isoformat(),loc.get("timezone","auto"))
        if not frame.empty:
            save_history(loc,frame)
            return frame,"Open-Meteo historical model estimate"
        errors.append("provider returned no rows")
    except Exception as exc:
        errors.append(str(exc))
    # If a long request is rejected, retry the most recent 92-day interval.
    if (end-start).days > 92:
        fallback_start=max(start,end-timedelta(days=91))
        try:
            recent=fetch_air_history(loc["latitude"],loc["longitude"],fallback_start.isoformat(),end.isoformat(),loc.get("timezone","auto"))
            if not recent.empty:
                save_history(loc,recent)
                older=load_history(loc,start,fallback_start-timedelta(days=1))
                combined=pd.concat([older,recent],ignore_index=True) if not older.empty else recent
                return combined,f"Long-range request unavailable; returned recent provider range from {fallback_start}. {errors[0]}"
        except Exception as exc:
            errors.append(f"92-day fallback: {exc}")
    cached=load_history(loc,start,end)
    if not cached.empty:
        cached["Source"]="Previously retrieved SQLite records"
        return cached,"Live history unavailable; showing locally saved provider records. " + " | ".join(errors)
    raise RuntimeError("Historical data unavailable from provider and local database. " + " | ".join(errors))

def history_window(choice,dates=None):
    end=TODAY-timedelta(days=1)
    days={"Last 7 days":7,"Last 30 days":30,"Last 3 months":92,"Last 6 months":183,"Last 1 year":365}
    if choice=="Custom range" and dates and len(dates)==2:
        return dates[0],dates[1]
    return end-timedelta(days=days.get(choice,30)-1),end

def fetch_bundle(loc,start,end):
    air,msg=get_history(loc,start,end)
    wx=get_weather_history(loc["latitude"],loc["longitude"],start.isoformat(),end.isoformat(),loc.get("timezone","auto"))
    return air,wx,msg

@st.cache_data(ttl=21600,show_spinner=False)
def get_weather_history(lat,lon,start,end,tz):
    return fetch_weather_history(lat,lon,start,end,tz)

def select_locations(key="selected_locations",default_all=False):
    primary = st.session_state.get("active_city")
    defaults = labels if default_all else ([primary] if primary in labels else labels[:1])
    return st.multiselect("Choose cities",labels,default=defaults,key=key)

def active_location():
    chosen = st.session_state.get("active_city")
    return next((loc for loc in watchlist if loc["label"] == chosen), watchlist[0])

def weather_description(code):
    if code is None or pd.isna(code): return "Unavailable"
    code = int(code)
    if code == 0: return "Clear sky"
    if code in (1, 2, 3): return "Mainly clear / partly cloudy"
    if code in (45, 48): return "Fog"
    if code in (51, 53, 55, 56, 57): return "Drizzle"
    if code in (61, 63, 65, 66, 67): return "Rain"
    if code in (71, 73, 75, 77): return "Snow"
    if code in (80, 81, 82): return "Rain showers"
    if code in (85, 86): return "Snow showers"
    if code in (95, 96, 99): return "Thunderstorm"
    return f"Weather code {code}"

def map_marker_color(aqi):
    # Colors follow provider US-AQI bands, not CPCB India AQI thresholds.
    if aqi is None or pd.isna(aqi): return "#64748b"
    v = float(aqi)
    if v <= 50: return "#16a34a"
    if v <= 100: return "#eab308"
    if v <= 150: return "#f97316"
    if v <= 200: return "#dc2626"
    if v <= 300: return "#9333ea"
    return "#7f1d1d"

def city_leaflet_map(loc, current_row):
    aqi = current_row.get("AQI") if current_row else None
    category, _ = aqi_category(aqi)
    color = map_marker_color(aqi)
    display_name = html.escape(loc["label"])
    aqi_text = "Unavailable" if aqi is None or pd.isna(aqi) else f"{float(aqi):.0f} (US AQI)"
    observed = html.escape(str(current_row.get("Observation time", "Unavailable"))) if current_row else "Unavailable"
    popup_html = (
        f"<div style='font-family:Arial;min-width:190px'>"
        f"<b>{display_name}</b><hr style='margin:6px 0'>"
        f"<b>AQI:</b> {aqi_text}<br><b>Category:</b> {html.escape(category)}"
        f"<br><b>Observation:</b> {observed}<br>"
        f"<small>Source: Open-Meteo CAMS model estimate</small></div>"
    )
    m = folium.Map(location=[float(loc["latitude"]), float(loc["longitude"])],
                   zoom_start=11, tiles="OpenStreetMap", control_scale=True)
    folium.CircleMarker(
        location=[float(loc["latitude"]), float(loc["longitude"])],
        radius=12, color=color, weight=3, fill=True, fill_color=color, fill_opacity=0.88,
        tooltip=f"{loc['label']} · US AQI {aqi_text}",
        popup=folium.Popup(popup_html, max_width=320)
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m

def load_selected_history(locs,start,end):
    air_map,wx_map,errors={}, {}, {}
    for loc in locs:
        try:
            a,msg=get_history(loc,start,end)
            air_map[loc["label"]]=a
            if "unavailable" in msg.lower() or "fallback" in msg.lower():
                errors[loc["label"]]=msg
        except Exception as e:
            errors[loc["label"]]=str(e)
        # Keep AQI history visible even if the independent weather archive is down.
        try:
            wx_map[loc["label"]]=get_weather_history(
                loc["latitude"],loc["longitude"],start.isoformat(),end.isoformat(),loc.get("timezone","auto"))
        except Exception as e:
            errors[loc["label"]+" weather"]=str(e)
    return air_map,wx_map,errors

def current_dataframe(locs):
    df,errs=get_currents(locs)
    if errs:
        with st.expander(f"Current data errors ({len(errs)})"):
            for city,e in errs.items(): st.write(f"**{city}:** {e}")
    return df


# ---------------- v5.0 map + alert services ----------------
DEFAULT_ALERT_SETTINGS = {
    "cities": [f"{n}, {co}" for n,lat,lon,co,admin,tz in DEFAULT_CITIES],
    "threshold": 151,
    "cooldown_minutes": 60,
    "monitor_current": True,
    "monitor_predicted": True,
    "browser_notifications": True,
    "email_notifications": False,
    "email_to": "",
    "interval_minutes": 10,
    "location_records": [
        {"name":n,"latitude":lat,"longitude":lon,"country":co,"admin1":admin,
         "timezone":tz,"label":f"{n}, {co}","region":admin}
        for n,lat,lon,co,admin,tz in DEFAULT_CITIES
    ],
}

def load_alert_settings():
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT settings_json FROM alert_preferences WHERE id=1").fetchone()
    if not row:
        return DEFAULT_ALERT_SETTINGS.copy()
    try:
        settings = json.loads(row[0])
        merged = DEFAULT_ALERT_SETTINGS.copy()
        merged.update(settings)
        return merged
    except Exception:
        return DEFAULT_ALERT_SETTINGS.copy()

def save_alert_settings(settings):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR REPLACE INTO alert_preferences(id,settings_json,updated_at) VALUES(1,?,?)",
                    (json.dumps(settings), datetime.now(IST).isoformat(timespec="seconds")))
        con.commit()

def get_alert_history(city_filter=None, severity_filter=None, start=None, end=None, unread_only=False):
    query = "SELECT alert_id AS 'Alert ID',city AS City,state AS State,aqi AS AQI,category AS Category,message AS Message,event_time AS 'Date and time',alert_type AS Type,notification_status AS 'Notification status',is_read AS Read,created_at FROM pollution_alerts WHERE 1=1"
    params = []
    if city_filter:
        query += " AND city IN (" + ",".join(["?"]*len(city_filter)) + ")"
        params.extend(city_filter)
    if severity_filter:
        query += " AND category IN (" + ",".join(["?"]*len(severity_filter)) + ")"
        params.extend(severity_filter)
    if start:
        query += " AND created_at>=?"
        params.append(start.isoformat())
    if end:
        query += " AND created_at<?"
        params.append((end+timedelta(days=1)).isoformat())
    if unread_only:
        query += " AND is_read=0"
    query += " ORDER BY created_at DESC LIMIT 1000"
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(query,con,params=params)

def create_pollution_alert(loc, aqi, category, message, alert_type, cooldown_minutes=60, event_time=None):
    if aqi is None or pd.isna(aqi):
        return False
    now = datetime.now(IST)
    when = event_time or now.isoformat(timespec="seconds")
    # Duplicate prevention: same city/type/category suppressed during cooldown.
    cooldown = (now - timedelta(minutes=int(cooldown_minutes))).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as con:
        recent = con.execute(
            "SELECT 1 FROM pollution_alerts WHERE city=? AND alert_type=? AND category=? AND created_at>=? LIMIT 1",
            (loc["label"], alert_type, category, cooldown)).fetchone()
        if recent:
            return False
        alert_id = hashlib.sha256(f"{loc_key(loc)}|{alert_type}|{category}|{when}|{now.isoformat()}".encode()).hexdigest()[:16]
        dedupe = hashlib.sha256(f"{loc_key(loc)}|{alert_type}|{category}|{int(now.timestamp()//(max(1,int(cooldown_minutes))*60))}".encode()).hexdigest()
        try:
            con.execute("""INSERT INTO pollution_alerts
              (alert_id,dedupe_key,city,state,aqi,category,message,event_time,alert_type,notification_status,is_read,created_at)
              VALUES(?,?,?,?,?,?,?,?,?,'In-app',0,?)""",
              (alert_id,dedupe,loc["label"],loc.get("region",""),float(aqi),category,message,when,alert_type,now.isoformat(timespec="seconds")))
            con.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def email_alert(message, subject, to_address):
    host = os.getenv("SMTP_HOST", "")
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_FROM", user)
    port = int(os.getenv("SMTP_PORT", "587"))
    if not all([host,user,password,sender,to_address]):
        return False, "SMTP is not fully configured; stored as an in-app alert."
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_address
    msg.set_content(message)
    try:
        with smtplib.SMTP(host,port,timeout=15) as server:
            server.starttls()
            server.login(user,password)
            server.send_message(msg)
        return True, "Email sent"
    except Exception as exc:
        return False, f"Email delivery failed: {exc}"

def evaluate_current_alerts(frame, locations, settings):
    new_alerts = []
    if not settings.get("monitor_current", True) or frame.empty:
        return new_alerts
    city_lookup = {x["label"]:x for x in locations}
    for _, row in frame.iterrows():
        aqi = row.get("AQI")
        if aqi is None or pd.isna(aqi) or float(aqi) < float(settings.get("threshold",151)):
            continue
        loc = city_lookup.get(row["City"])
        if not loc:
            continue
        cat,_ = aqi_category(aqi)  # Open-Meteo supplies US AQI, so use US category bands here.
        message = (f"{loc['label']}: provider US AQI is {float(aqi):.0f} ({cat}). "
                   f"Reduce prolonged outdoor exertion as appropriate. Observation: {row.get('Observation time')}.")
        made = create_pollution_alert(loc,aqi,cat,message,"Current",
                                      settings.get("cooldown_minutes",60),row.get("Observation time"))
        if made:
            new_alerts.append({"City":loc["label"],"AQI":float(aqi),"Category":cat,
                               "Message":message,"Type":"Current"})
            if settings.get("email_notifications") and settings.get("email_to"):
                ok, note = email_alert(message,f"Air-Guard current AQI alert — {loc['label']}",settings["email_to"])
                with sqlite3.connect(DB_PATH) as con:
                    con.execute("UPDATE pollution_alerts SET notification_status=? WHERE city=? AND alert_type='Current' AND created_at=(SELECT MAX(created_at) FROM pollution_alerts WHERE city=? AND alert_type='Current')",
                                ("Email sent" if ok else f"In-app; {note}",loc["label"],loc["label"]))
                    con.commit()
    return new_alerts

@st.cache_data(ttl=86400,show_spinner=False)
def fetch_india_geojson():
    # Public repository's all-India GeoJSON; if unavailable the OSM basemap still works.
    urls = [
        "https://cdn.jsdelivr.net/gh/udit-001/india-maps-data@2884453/geojson/india.geojson",
        "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson",
    ]
    last_error = None
    for url in urls:
        try:
            response = requests.get(url,timeout=(8,25),headers={"User-Agent":"AirGuard/5.0"})
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"India boundary GeoJSON unavailable: {last_error}")

def india_aqi_color(aqi):
    # Never use CPCB's category breakpoints on US AQI model data.
    if aqi is None or pd.isna(aqi): return "#64748b"
    v=float(aqi)
    if v<=50: return "#16a34a"
    if v<=100: return "#84cc16"
    if v<=150: return "#eab308"
    if v<=200: return "#f97316"
    if v<=300: return "#dc2626"
    return "#7f1d1d"

def build_india_map(city_frame, mode="Markers", geojson=None, focus=None, stations=None, cpcb_stations=None):
    center = [22.8,79.0]
    zoom = 5
    if focus is not None:
        center=[float(focus["latitude"]),float(focus["longitude"])]
        zoom=10
    m = folium.Map(location=center,zoom_start=zoom,tiles="OpenStreetMap",control_scale=True)
    if geojson:
        folium.GeoJson(geojson,name="India state/district boundary overlay",
            style_function=lambda feat: {"fillColor":"#d9efe5","color":"#3c7660","weight":0.65,"fillOpacity":0.06},
            tooltip=folium.GeoJsonTooltip(fields=list(geojson.get("features",[{}])[0].get("properties",{}).keys())[:2])
                    if geojson.get("features") and geojson["features"][0].get("properties") else None).add_to(m)
    if focus is not None:
        m.location=center
    valid = city_frame.dropna(subset=["Latitude","Longitude"])
    if mode=="Heatmap" and not valid.empty:
        heat=[]
        for _,r in valid.dropna(subset=["AQI"]).iterrows():
            heat.append([float(r["Latitude"]),float(r["Longitude"]),max(0.05,min(1.0,float(r["AQI"])/500))])
        if heat: HeatMap(heat,radius=28,blur=22,min_opacity=.25,name="City AQI intensity").add_to(m)
    for _,r in valid.iterrows():
        aqi=r.get("AQI")
        cat,_=aqi_category(aqi)
        color=india_aqi_color(aqi)
        aqi_txt="Not available" if aqi is None or pd.isna(aqi) else f"{float(aqi):.0f} · US AQI"
        popup=(f"<b>{html.escape(str(r['City']))}</b><br>Provider AQI: {aqi_txt}<br>"
               f"Category: {html.escape(cat)}<br>PM2.5: {r.get('PM2.5')} µg/m³<br>"
               f"PM10: {r.get('PM10')} µg/m³<br>NO₂: {r.get('NO₂')}<br>"
               f"SO₂: {r.get('SO₂')}<br>O₃: {r.get('O₃')}<br>"
               f"Observed: {html.escape(str(r.get('Observation time')))}<br>"
               f"<small>Open-Meteo CAMS city-grid estimate; not a station value.</small>")
        if mode=="Markers":
            folium.Marker([r["Latitude"],r["Longitude"]],
                icon=folium.DivIcon(html=f'<div style="background:{color};border:2px solid white;border-radius:18px;padding:4px 7px;color:#fff;font-weight:bold;white-space:nowrap;box-shadow:0 1px 4px #555">{aqi_txt.split(" ")[0] if aqi_txt!="Not available" else "—"}</div>'),
                tooltip=str(r["City"]),popup=folium.Popup(popup,max_width=320)).add_to(m)
        else:
            folium.CircleMarker([r["Latitude"],r["Longitude"]],radius=7,color=color,fill=True,
                fill_color=color,fill_opacity=.85,tooltip=f"{r['City']} · {aqi_txt}",
                popup=folium.Popup(popup,max_width=320)).add_to(m)
    if stations is not None and not stations.empty:
        for _,s in stations.iterrows():
            popup=(f"<b>{html.escape(str(s.get('station','OpenAQ station')))}</b><br>"
                   f"Provider: OpenAQ sensor measurement<br>{html.escape(str(s.get('parameter','')))}: "
                   f"{s.get('value')} {html.escape(str(s.get('unit','')))}<br>"
                   f"Measured: {html.escape(str(s.get('time','')))}<br>"
                   "<small>Raw sensor measurement; no station AQI inferred.</small>")
            folium.CircleMarker([s["latitude"],s["longitude"]],radius=5,color="#2563eb",
                fill=True,fill_color="#2563eb",fill_opacity=.9,
                tooltip=f"{s.get('station')} · {s.get('parameter')}",
                popup=folium.Popup(popup,max_width=300)).add_to(m)
    add_cpcb_station_layer(m,cpcb_stations)
    folium.LayerControl(collapsed=True).add_to(m)
    return m

@st.cache_data(ttl=900,show_spinner=False)
def fetch_openaq_station_measurements(lat,lon,api_key,radius_m=25000,max_locations=8):
    if not api_key:
        return pd.DataFrame()
    headers={"X-API-Key":api_key,"Accept":"application/json"}
    location_response=requests.get("https://api.openaq.org/v3/locations",
        headers=headers,params={"coordinates":f"{lat},{lon}","radius":radius_m,
                                "limit":max_locations,"page":1},timeout=(8,25))
    location_response.raise_for_status()
    loc_data=location_response.json()
    locs=loc_data.get("results",[])
    rows=[]
    for location in locs[:max_locations]:
        coords=location.get("coordinates") or {}
        slat=coords.get("latitude"); slon=coords.get("longitude")
        if slat is None or slon is None: continue
        loc_id=location.get("id")
        if loc_id is None: continue
        response=requests.get(f"https://api.openaq.org/v3/locations/{loc_id}/latest",
                              headers=headers,params={"limit":50},timeout=(8,25))
        response.raise_for_status()
        results=response.json().get("results",[])
        for item in results:
            parameter=item.get("parameter") or {}
            dt=item.get("datetime") or {}
            rows.append({"station":location.get("name") or f"OpenAQ location {loc_id}",
                "latitude":float(slat),"longitude":float(slon),
                "parameter":parameter.get("name") or item.get("parameterName","Pollutant"),
                "value":item.get("value"),"unit":item.get("unit") or parameter.get("units",""),
                "time":dt.get("utc") or dt.get("local"),"location_id":loc_id})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300,show_spinner=False)
def fetch_cpcb_ogd_station_aqi(api_key,resource_id,limit=1000):
    if not api_key or not resource_id:
        return pd.DataFrame()
    url=f"https://api.data.gov.in/resource/{resource_id}"
    response=requests.get(url,params={"api-key":api_key,"format":"json","limit":limit},
                          timeout=(8,35),headers={"User-Agent":"AirGuard/5.0"})
    response.raise_for_status()
    payload=response.json()
    records=payload.get("records",[])
    if not isinstance(records,list) or not records:
        return pd.DataFrame()
    def normalized(record):
        return {str(k).strip().lower().replace(" ","_").replace("-","_"):v for k,v in record.items()}
    rows=[]
    aliases={
      "city":["city","city_name","town","district","area"],
      "state":["state","state_name"],
      "station":["station","station_name","location","monitoring_station"],
      "lat":["latitude","lat"],"lon":["longitude","lon","long"],
      "aqi":["aqi","air_quality_index","airqualityindex","national_aqi","aqi_value","index_value"],
      "timestamp":["last_update","last_updated","updated_at","date","datetime","timestamp"]
    }
    for record in records:
        r=normalized(record)
        def pick(key):
            for name in aliases[key]:
                if r.get(name) not in (None,"","NA","NaN","null"):
                    return r.get(name)
            return None
        lat=num(pick("lat")); lon=num(pick("lon")); aqi=num(pick("aqi"))
        if lat is None or lon is None or aqi is None:
            continue
        rows.append({"City":str(pick("city") or "—"),"State":str(pick("state") or "—"),
          "Station":str(pick("station") or "CPCB station"),"Latitude":lat,"Longitude":lon,
          "CPCB AQI":aqi,"Timestamp":str(pick("timestamp") or "Not provided"),
          "Source":"data.gov.in / CPCB resource"})
    return pd.DataFrame(rows)

def cpcb_aqi_category(value):
    if value is None or pd.isna(value): return "Unavailable","#64748b"
    v=float(value)
    if v<=50: return "Good","#16a34a"
    if v<=100: return "Satisfactory","#84cc16"
    if v<=200: return "Moderate","#eab308"
    if v<=300: return "Poor","#f97316"
    if v<=400: return "Very Poor","#dc2626"
    return "Severe","#7f1d1d"

def add_cpcb_station_layer(map_obj, stations):
    if stations is None or stations.empty:
        return
    for _,s in stations.iterrows():
        cat,color=cpcb_aqi_category(s["CPCB AQI"])
        popup=(f"<b>{html.escape(str(s['Station']))}</b><br>{html.escape(str(s['City']))}, "
               f"{html.escape(str(s['State']))}<br><b>CPCB National AQI:</b> {float(s['CPCB AQI']):.0f}"
               f"<br><b>Category:</b> {cat}<br><b>Updated:</b> {html.escape(str(s['Timestamp']))}"
               "<br><small>Official OGD/CPCB resource record; confirm freshness in timestamp.</small>")
        folium.CircleMarker([s["Latitude"],s["Longitude"]],radius=8,color=color,weight=3,
            fill=True,fill_color=color,fill_opacity=.9,tooltip=f"{s['Station']} · CPCB AQI {float(s['CPCB AQI']):.0f}",
            popup=folium.Popup(popup,max_width=320)).add_to(map_obj)

def browser_notification_panel(alerts, enabled=True):
    safe_alerts = json.dumps(alerts,ensure_ascii=True)
    html_doc = f"""
    <div style="font-family:Arial,sans-serif;padding:8px;border:1px solid #d1d5db;border-radius:10px">
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <button id="enable" style="padding:9px 12px;border:0;border-radius:8px;background:#166534;color:white;cursor:pointer">Enable browser notifications</button>
        <button id="sound" style="padding:9px 12px;border:0;border-radius:8px;background:#334155;color:white;cursor:pointer">Test alert sound</button>
        <span id="state" style="font-size:12px;color:#64748b">Browser permission not requested</span>
      </div>
      <script>
      const alerts={safe_alerts};
      const state=document.getElementById('state');
      function notifyAll(){{
        if(!('Notification' in window)){{state.textContent='Notifications are not supported by this browser';return;}}
        if(Notification.permission!=='granted'){{state.textContent='Allow notifications using the button';return;}}
        alerts.forEach(a=>{{try{{new Notification('Air-Guard · '+a.City,{{body:a.Message,tag:a.id||a.City+'-'+a.Type}});}}catch(e){{}}}});
        state.textContent=alerts.length ? 'Browser alerts sent for current new alerts' : 'Permission enabled; no new alerts in this refresh';
      }}
      document.getElementById('enable').addEventListener('click',async()=>{{
        if(!('Notification' in window)){{state.textContent='Notifications unsupported';return;}}
        try{{
          const p=await Notification.requestPermission();
          state.textContent='Permission: '+p;
          if(p==='granted')notifyAll();
        }}catch(e){{state.textContent='Permission request blocked in this embedded browser context. Try HTTPS in a full browser tab.';}}
      }});
      document.getElementById('sound').addEventListener('click',()=>{{
        try{{
          const C=window.AudioContext||window.webkitAudioContext; const ctx=new C();
          const osc=ctx.createOscillator(),gain=ctx.createGain();
          osc.type='sine';osc.frequency.value=880;gain.gain.value=.08;
          osc.connect(gain);gain.connect(ctx.destination);osc.start();osc.stop(ctx.currentTime+.35);
          state.textContent='Test sound played';
        }}catch(e){{state.textContent='Sound unavailable in this browser';}}
      }});
      if(Notification.permission==='granted' && alerts.length)notifyAll();
      </script>
    </div>"""
    components.html(html_doc,height=82,scrolling=False)


def evaluate_predicted_alerts_for_locations(locations, settings):
    made_alerts=[]
    if not settings.get("monitor_predicted",True):
        return made_alerts, []
    errors=[]
    for loc in locations:
        try:
            end=TODAY-timedelta(days=1); start=end-timedelta(days=91)
            air,wx,msg=fetch_bundle(loc,start,end)
            forecast,n_train,mae=predict_city(loc,air,wx)
            threshold=float(settings.get("threshold",151))
            crossing=forecast[forecast["Predicted AQI"]>=threshold]
            for _,pr in crossing.iterrows():
                category,_=aqi_category(pr["Predicted AQI"])
                message=(f"FORECAST: {loc['label']} may reach provider US AQI {pr['Predicted AQI']:.0f} "
                         f"({category}) on {pr['Date']}. This is a model prediction, not an observation.")
                if create_pollution_alert(loc,pr["Predicted AQI"],category,message,"Predicted",
                         settings.get("cooldown_minutes",60),str(pr["Date"])):
                    item={"City":loc["label"],"AQI":float(pr["Predicted AQI"]),"Category":category,
                          "Date":str(pr["Date"]),"Type":"Predicted","Message":message}
                    made_alerts.append(item)
                    if settings.get("email_notifications") and settings.get("email_to"):
                        ok,note=email_alert(message,f"Air-Guard forecast AQI alert — {loc['label']}",settings["email_to"])
                        if not ok: errors.append(f"{loc['label']} email: {note}")
        except Exception as exc:
            errors.append(f"{loc['label']}: {exc}")
    return made_alerts,errors

def read_alerts(alert_ids):
    if not alert_ids: return
    with sqlite3.connect(DB_PATH) as con:
        con.executemany("UPDATE pollution_alerts SET is_read=1 WHERE alert_id=?",[(x,) for x in alert_ids])
        con.commit()

def clear_old_alerts(before_date):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM pollution_alerts WHERE created_at<?",(before_date.isoformat(),))
        con.commit()



# Periodic in-session alert evaluation. For true always-on monitoring, deploy this
# same evaluator from a scheduled worker; a browser app cannot run while closed.
alert_settings = load_alert_settings()
alert_cities = [loc for loc in st.session_state.watchlist
                if loc["label"] in alert_settings.get("cities", [])]
if alert_settings.get("monitor_current") and alert_cities:
    try:
        _alert_current, _alert_errors = get_currents(alert_cities)
        _new_current_alerts = evaluate_current_alerts(_alert_current, alert_cities, alert_settings)
        st.session_state["new_airguard_alerts"] = _new_current_alerts
        if _new_current_alerts:
            st.warning(f"🚨 {len(_new_current_alerts)} new pollution alert(s) detected. Open **Pollution Alerts** to review details.")
            if alert_settings.get("browser_notifications") and page != "Pollution Alerts":
                _browser_items=[{"City":a["City"],"Message":a["Message"],"Type":a["Type"],
                                 "id":a["City"]+a["Type"]} for a in _new_current_alerts]
                browser_notification_panel(_browser_items)
    except Exception as _alert_exc:
        st.session_state["alert_poll_error"] = str(_alert_exc)

# ---------------- pages ----------------
if page in ["Overview","Current AQI Monitoring"]:
    loc = active_location()
    title = f"City overview · {loc['label']}" if page == "Overview" else f"Current AQI · {loc['label']}"
    st.header(title)
    st.caption(f"Selected coordinates: {loc['latitude']:.4f}, {loc['longitude']:.4f} · {loc.get('timezone','auto')}")

    with st.spinner(f"Retrieving latest air quality for {loc['label']}…"):
        current, current_errors = get_currents([loc])
    row = current.iloc[0].to_dict() if not current.empty else None
    if current_errors:
        st.error(f"Current AQI data is unavailable for {loc['label']}: {current_errors.get(loc['label'])}")

    history = pd.DataFrame()
    wx_history = pd.DataFrame()
    forecast = pd.DataFrame()
    training_rows = None
    forecast_mae = None
    history_error = None
    history_message = None
    if page == "Overview":
        with st.spinner("Loading historical air quality and weather for trend and prediction…"):
            train_end = TODAY - timedelta(days=1)
            train_start = train_end - timedelta(days=91)
            try:
                history, history_message = get_history(loc, train_start, train_end)
                try:
                    wx_history = get_weather_history(
                        loc["latitude"], loc["longitude"], train_start.isoformat(),
                        train_end.isoformat(), loc.get("timezone", "auto"))
                except Exception as wx_exc:
                    history_error = f"Weather history unavailable: {wx_exc}"
                if not wx_history.empty:
                    forecast, training_rows, forecast_mae = predict_city(loc, history, wx_history)
                elif history_error is None:
                    history_error = "Historical weather data is unavailable; prediction cannot be trained."
            except Exception as exc:
                history_error = str(exc)

    if row:
        aqi = row.get("AQI")
        category, emoji = aqi_category(aqi)
        daily = daily_air(history) if not history.empty else pd.DataFrame()
        yesterday = daily[daily["date"] == TODAY - timedelta(days=1)] if not daily.empty else pd.DataFrame()
        previous_aqi = yesterday.iloc[0]["AQI"] if not yesterday.empty else np.nan
        next_prediction = forecast.sort_values("Date").iloc[0]["Predicted AQI"] if not forecast.empty else np.nan
        pm25 = row.get("PM2.5")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current AQI · US", "—" if aqi is None else f"{float(aqi):.0f}", category)
        c2.metric("Previous-day AQI", "—" if pd.isna(previous_aqi) else f"{previous_aqi:.0f}")
        c3.metric("Next-day AI prediction", "—" if pd.isna(next_prediction) else f"{next_prediction:.0f}")
        c4.metric("PM2.5 concentration", "—" if pm25 is None else f"{float(pm25):.1f} µg/m³")
        st.caption(f"{emoji} {category} · Source: Open-Meteo CAMS model estimate · Observation time: {row.get('Observation time')} · Retrieved: {datetime.now(IST):%d %b %Y, %I:%M %p IST}")

        map_col, weather_col = st.columns([1.35, 1])
        with map_col:
            st.subheader("Selected city · interactive map")
            city_map = city_leaflet_map(loc, row)
            st_folium(city_map, use_container_width=True, height=430,
                      returned_objects=[], key=f"airguard-map-{loc_key(loc)}")
            st.caption("OpenStreetMap · Automatically centered on the selected geocoded city. Individual station markers are omitted because verified station coordinates and readings are not integrated.")
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:10px;font-size:12px">'
                '<span style="color:#16a34a">● Good</span><span style="color:#eab308">● Moderate</span>'
                '<span style="color:#f97316">● Unhealthy for sensitive groups</span>'
                '<span style="color:#dc2626">● Unhealthy</span><span style="color:#9333ea">● Very unhealthy</span>'
                '<span style="color:#7f1d1d">● Hazardous</span></div>',
                unsafe_allow_html=True)
            st.caption("Marker colors use provider US AQI bands; CPCB thresholds are not applied to this value.")
        with weather_col:
            st.subheader("Weather conditions")
            w1,w2=st.columns(2)
            w1.metric("Temperature", "—" if row.get("Temperature (°C)") is None else f"{row['Temperature (°C)']} °C")
            w2.metric("Humidity", "—" if row.get("Humidity (%)") is None else f"{row['Humidity (%)']}%")
            w1.metric("Wind speed", "—" if row.get("Wind (km/h)") is None else f"{row['Wind (km/h)']} km/h")
            w2.metric("Pressure", "—" if row.get("Pressure (hPa)") is None else f"{row['Pressure (hPa)']} hPa")
            st.info(f"Conditions: **{weather_description(row.get('Weather code'))}**")
            st.caption(f"Weather timestamp: {row.get('Weather time') or 'Unavailable'}")
            st.subheader("Pollutant concentrations")
            pollutant_data = [
                {"Pollutant":"PM2.5","Value":row.get("PM2.5"),"Unit":"µg/m³"},
                {"Pollutant":"PM10","Value":row.get("PM10"),"Unit":"µg/m³"},
                {"Pollutant":"CO","Value":row.get("CO"),"Unit":"µg/m³"},
                {"Pollutant":"NO₂","Value":row.get("NO₂"),"Unit":"µg/m³"},
                {"Pollutant":"SO₂","Value":row.get("SO₂"),"Unit":"µg/m³"},
                {"Pollutant":"O₃","Value":row.get("O₃"),"Unit":"µg/m³"},
            ]
            st.dataframe(pd.DataFrame(pollutant_data),hide_index=True,use_container_width=True)

        if page == "Overview":
            st.subheader("Historical AQI trend · last 30 days")
            if not daily.empty:
                d30=daily[daily["date"] >= TODAY-timedelta(days=30)]
                st.plotly_chart(px.line(d30,x="date",y="AQI",markers=True,
                    title=f"{loc['label']} · daily mean US AQI"),use_container_width=True)
                if not yesterday.empty:
                    st.caption(f"Previous-day daily mean AQI: {previous_aqi:.1f}. {history_message or ''}")
                else:
                    st.warning("Previous-day AQI is not available in the returned historical records.")
            else:
                st.warning("Historical AQI records are unavailable for this city. No values have been invented.")

            st.subheader("Upcoming 7-day AQI prediction")
            if not forecast.empty:
                fc=forecast.copy()
                fc["Category"]=fc["Predicted AQI"].apply(lambda v:aqi_category(v)[0])
                st.plotly_chart(px.line(fc,x="Date",y="Predicted AQI",markers=True,
                    title="Random Forest model output · prediction, not observation"),use_container_width=True)
                st.dataframe(fc[["Date","Predicted AQI","Category","Data type"]],hide_index=True,use_container_width=True)
                first=float(fc.iloc[0]["Predicted AQI"]); last=float(fc.iloc[-1]["Predicted AQI"])
                movement="increase" if last>first+3 else "decrease" if last<first-3 else "remain broadly stable"
                st.caption(f"Model expects AQI to {movement} over the forecast window. Training rows: {training_rows}. Holdout MAE: {forecast_mae if forecast_mae is not None else 'not calculated'}.")
            else:
                st.info("Seven-day prediction is unavailable. " + (history_error or "Insufficient aligned historical AQI and weather records."))

            st.subheader("Health precautions")
            _, advice=health_advice(aqi)
            st.info(advice[0])
            st.write("**Indoor guidance:** " + advice[1])
            st.caption("General guidance only; not a substitute for professional medical advice. Provider US AQI is not the Indian CPCB AQI scale.")
        else:
            st.caption(f"Data source: {row.get('Source')} · Coordinates from Open-Meteo Geocoding.")
    else:
        st.error("AQI data could not be retrieved for the selected city. The map remains centered on the selected geocoded coordinates; no fallback city is substituted.")
        st_folium(city_leaflet_map(loc, None),use_container_width=True,height=430,
                  returned_objects=[],key=f"airguard-map-unavailable-{loc_key(loc)}")

elif page=="Historical AQI Analysis":
    st.header("Historical AQI analysis")
    cities=select_locations()
    range_choice=st.selectbox("Historical window",["Last 7 days","Last 30 days","Last 3 months","Last 6 months","Last 1 year","Custom range"])
    custom=st.date_input("Custom start and end dates",value=(TODAY-timedelta(days=30),TODAY-timedelta(days=1)),max_value=TODAY-timedelta(days=1)) if range_choice=="Custom range" else None
    start,end=history_window(range_choice,custom)
    st.caption(f"Requested: {start} to {end}. Actual returned coverage can be shorter.")
    locs=[x for x in watchlist if x["label"] in cities]
    if locs:
        amap,_,errs=load_selected_history(locs,start,end)
        frames=[]
        for loc in locs:
            f=amap.get(loc["label"],pd.DataFrame())
            if not f.empty:
                f=f[(f["time"].dt.date>=start)&(f["time"].dt.date<=end)].copy()
                f["City"]=loc["label"]; frames.append(f)
        hist=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        if not hist.empty:
            hist["date"]=hist["time"].dt.date
            daily=hist.groupby(["date","City"],as_index=False)["AQI"].mean()
            st.plotly_chart(px.line(daily,x="date",y="AQI",color="City",markers=True,title="Daily mean AQI history"),use_container_width=True)
            st.plotly_chart(px.bar(daily,x="date",y="AQI",color="City",barmode="group",title="Daily AQI comparison"),use_container_width=True)
            stats=hist.groupby("City")["AQI"].agg(["min","max","mean","count"]).reset_index()
            stats=stats.rename(columns={"min":"Lowest","max":"Highest","mean":"Average","count":"Hourly records"})
            st.dataframe(stats,hide_index=True,use_container_width=True)
            st.dataframe(daily.sort_values("date",ascending=False),hide_index=True,use_container_width=True)
            st.download_button("Download historical CSV",hist.to_csv(index=False).encode(), "air_guard_historical.csv","text/csv")
        else: st.warning("No historical AQI records were returned. Missing values are not filled.")
        for city,msg in errs.items(): st.warning(f"{city}: {msg}")

elif page=="Future AQI Prediction":
    st.header("7-day AI-based AQI prediction")
    cities=select_locations()
    locs=[x for x in watchlist if x["label"] in cities]
    outputs=[]; errors=[]
    if st.button("Train models and predict",type="primary"):
        with st.spinner("Retrieving history, weather and training city models…"):
            for loc in locs:
                train_end=TODAY-timedelta(days=1)
                train_start=train_end-timedelta(days=91)
                try:
                    air,wx,msg=fetch_bundle(loc,train_start,train_end)
                    pred,n,mae=predict_city(loc,air,wx)
                    pred["Training rows"]=n; pred["Holdout MAE"]=mae
                    outputs.append(pred)
                except Exception as e: errors.append(f"{loc['label']}: {e}")
        if outputs: st.session_state["predictions"]=pd.concat(outputs,ignore_index=True)
        st.session_state["prediction_errors"]=errors
    pred=st.session_state.get("predictions",pd.DataFrame())
    for err in st.session_state.get("prediction_errors",[]): st.warning(err)
    if not pred.empty:
        shown=pred[pred["City"].isin(cities)]
        st.plotly_chart(px.line(shown,x="Date",y="Predicted AQI",color="City",markers=True,title="Random Forest forecast · 7 days"),use_container_width=True)
        for city,part in shown.groupby("City"):
            st.write(f"**{city}:** ",end="") if False else None
            baseline=part["Predicted AQI"].mean()
            first=part["Predicted AQI"].iloc[0]; last=part["Predicted AQI"].iloc[-1]
            trend="increase" if last>first+3 else "decrease" if last<first-3 else "remain broadly stable"
            st.caption(f"{city}: expected to {trend} across the forecast window.")
        st.dataframe(shown,hide_index=True,use_container_width=True)
        st.download_button("Download predictions CSV",shown.to_csv(index=False).encode(),"air_guard_predictions.csv","text/csv")
    else: st.info("Select cities and click Train models and predict. A city needs at least 25 usable daily AQI/weather rows.")

elif page=="City Comparison":
    st.header("City comparison")
    cities=select_locations(default_all=False)
    trend_period=st.selectbox("Comparison trend window",["Daily · 7 days","Weekly · 30 days","Monthly · 90 days"])
    view_days={"Daily · 7 days":7,"Weekly · 30 days":30,"Monthly · 90 days":90}[trend_period]
    locs=[x for x in watchlist if x["label"] in cities]
    if len(locs)<2: st.info("Select at least two cities.")
    else:
        cur=current_dataframe(locs)
        # Retrieve up to 92 days to support both the chosen comparison window and model training.
        end=TODAY-timedelta(days=1)
        train_start=end-timedelta(days=91)
        amap,wxmap,errs=load_selected_history(locs,train_start,end)
        rows=[]; daily_frames=[]
        for loc in locs:
            r=cur[cur["City"]==loc["label"]]
            aqi=r.iloc[0]["AQI"] if not r.empty else np.nan
            h=amap.get(loc["label"],pd.DataFrame())
            d=daily_air(h)
            prev=d[d["date"]==TODAY-timedelta(days=1)]["AQI"]
            visible=d[d["date"]>=TODAY-timedelta(days=view_days)]
            pred_mean=np.nan; pred_first=np.nan; direction="Unavailable"
            weather=wxmap.get(loc["label"],pd.DataFrame())
            try:
                pred,n_train,mae=predict_city(loc,h,weather)
                pred_mean=pred["Predicted AQI"].mean()
                pred_first=pred.sort_values("Date").iloc[0]["Predicted AQI"]
                last_actual=d.sort_values("date").tail(3)["AQI"].mean()
                if pred_mean>last_actual+3: direction="Increase"
                elif pred_mean<last_actual-3: direction="Decrease"
                else: direction="Broadly stable"
            except Exception as model_error:
                errs[loc["label"]+" prediction"]=str(model_error)
            rows.append({"City":loc["label"],"Current observation time":r.iloc[0]["Observation time"] if not r.empty else None,
              "Current AQI (US)":aqi,"Previous day AQI":prev.iloc[0] if not prev.empty else np.nan,
              f"{view_days}-day historical mean":visible["AQI"].mean() if not visible.empty else np.nan,
              "Predicted next-day AQI":pred_first,"Predicted 7-day mean":pred_mean,
              "Prediction direction":direction,
              "PM2.5 (µg/m³)":r.iloc[0]["PM2.5"] if not r.empty else np.nan,
              "PM10 (µg/m³)":r.iloc[0]["PM10"] if not r.empty else np.nan,
              "Current category (US AQI)":aqi_category(aqi)[0]})
            if not visible.empty:
                visible=visible.copy(); visible["City"]=loc["label"]; daily_frames.append(visible)
        comp=pd.DataFrame(rows)
        st.dataframe(comp,hide_index=True,use_container_width=True)
        if daily_frames:
            daily=pd.concat(daily_frames,ignore_index=True)
            st.plotly_chart(px.line(daily,x="date",y="AQI",color="City",markers=True,
                title=f"{view_days}-day historical AQI comparison"),use_container_width=True)
            valid=comp.dropna(subset=["Current AQI (US)"])
            if not valid.empty:
                best=valid.loc[valid["Current AQI (US)"].idxmin()]
                worst=valid.loc[valid["Current AQI (US)"].idxmax()]
                x,y=st.columns(2)
                x.success(f"Lowest current provider AQI: {best['City']} ({best['Current AQI (US)']:.0f})")
                y.error(f"Highest current provider AQI: {worst['City']} ({worst['Current AQI (US)']:.0f})")
        st.download_button("Download comparison CSV",comp.to_csv(index=False).encode(),"air_guard_comparison.csv","text/csv")
        for city,msg in errs.items(): st.caption(f"{city}: {msg}")

elif page=="Health Precautions":
    st.header("AQI-based health precautions")
    loc=active_location()
    city=loc["label"]
    st.caption(f"Advisory follows the primary city selected in the sidebar: {city}")
    try:
        r=fetch_current(loc["latitude"],loc["longitude"],loc.get("timezone","auto"))
        aqi=r.get("AQI"); cat,advice=health_advice(aqi)
        st.metric("Latest provider US AQI","—" if aqi is None else f"{aqi:.0f}",cat)
        st.warning("Health recommendations are based on the provider's US AQI category bands. This value is not India's CPCB National AQI.")
        st.subheader("General precautions")
        st.info(advice[0])
        st.markdown("**Children and older adults:** Reduce prolonged outdoor exertion when air quality is unhealthy; caregivers should watch for symptoms.")
        st.markdown("**People with asthma or heart/lung conditions:** Follow your clinician's action plan and reduce exposure during pollution episodes.")
        st.markdown("**Healthy adults:** Reduce strenuous outdoor activity at unhealthy levels; reschedule exercise to cleaner periods.")
        st.markdown("**Indoor safety:** Keep indoor air cleaner, avoid tobacco smoke/incense and other particle sources, and use a suitable HEPA air cleaner where available.")
        st.markdown("**Masks:** When exposure cannot be avoided, a well-fitting N95/FFP2 respirator can reduce particle inhalation. It does not protect against gases such as ozone or NO₂.")
        st.markdown("**Pollutants:** PM2.5/PM10 are particles; CO can impair oxygen delivery; NO₂ and SO₂ irritate airways; ground-level O₃ can worsen respiratory symptoms.")
        st.subheader("CPCB National AQI reference")
        cpcb=pd.DataFrame([
          ["0–50","Good","Normal outdoor activities"],
          ["51–100","Satisfactory","Sensitive people monitor symptoms"],
          ["101–200","Moderate","Sensitive groups limit prolonged exertion"],
          ["201–300","Poor","Reduce prolonged outdoor activity"],
          ["301–400","Very Poor","Avoid prolonged exposure"],
          ["401–500","Severe","Avoid outdoor physical activity and follow advisories"]],
          columns=["CPCB AQI range","Category","General action"])
        st.dataframe(cpcb,hide_index=True,use_container_width=True)
        st.caption("The CPCB table is shown as a reference only. It is not applied to US AQI values. General guidance does not replace professional medical advice.")
    except Exception as e: st.error(f"Current reading unavailable: {e}")

elif page=="Reports & Analytics":
    st.header("Reports & analytics")
    cities=select_locations()
    locs=[x for x in watchlist if x["label"] in cities]
    period=st.selectbox("Report period",["Daily","Weekly","Monthly"])
    days={"Daily":1,"Weekly":7,"Monthly":30}[period]
    start,end=TODAY-timedelta(days=days),TODAY-timedelta(days=1)
    amap,_,errs=load_selected_history(locs,start,end)
    pieces=[]
    for loc in locs:
        f=amap.get(loc["label"],pd.DataFrame())
        if not f.empty:
            f["City"]=loc["label"]; pieces.append(f)
    report=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
    if not report.empty:
        report["Date"]=report["time"].dt.date
        daily=report.groupby(["City","Date"],as_index=False).agg(
            AQI=("AQI","mean"),PM2_5=("PM2.5","mean"),PM10=("PM10","mean"))
        st.plotly_chart(px.bar(daily,x="Date",y="AQI",color="City",barmode="group",title=f"{period} AQI report"),use_container_width=True)
        st.dataframe(daily,hide_index=True,use_container_width=True)
        st.download_button("Export report CSV",daily.to_csv(index=False).encode(),"air_guard_report.csv","text/csv")
        try:
            st.download_button("Export report PDF",pdf_report(f"Air-Guard {period} AQI Report",daily),
                               "air_guard_report.pdf","application/pdf")
        except Exception as e: st.info(f"PDF export unavailable: {e}. CSV export is available.")
    else: st.warning("No historical records exist for this reporting period yet.")
    for city,msg in errs.items(): st.caption(f"{city}: {msg}")


elif page=="India Pollution Map":
    st.header("India-wide air pollution map")
    st.caption("City-grid model estimates are shown separately from optional OpenAQ station pollutant measurements.")
    india_locs=[x for x in watchlist if x.get("country","").lower()=="india"]
    if not india_locs:
        st.warning("No Indian cities in the watchlist. Search and add Indian cities from the sidebar.")
    else:
        city_values=[]
        with st.spinner("Loading latest AQI estimates for monitored Indian cities…"):
            current_map, current_errors = get_currents(india_locs)
        if not current_map.empty:
            state_options=sorted({str(x.get("region","")) for x in india_locs if x.get("region")})
            selected_states=st.multiselect("Filter by state / region",state_options,default=state_options,key="india_map_states")
            filtered_locs=[x for x in india_locs if x.get("region","") in selected_states]
            filter_names=[x["label"] for x in filtered_locs]
            selected_cities=st.multiselect("Filter cities",filter_names,default=filter_names,key="india_map_cities")
            severity_options=["Good","Moderate","Unhealthy for sensitive groups","Unhealthy","Very unhealthy","Hazardous","Unavailable"]
            selected_severities=st.multiselect("Filter AQI severity (provider US AQI)",severity_options,
                                                default=severity_options,key="india_map_severity")
            mode=st.radio("Map layer",["Markers","Heatmap"],horizontal=True,key="india_map_mode")
            focus_choice=st.selectbox("Zoom to city",["Entire India"]+selected_cities,key="india_map_focus")
            cdf=current_map[current_map["City"].isin(selected_cities)].copy()
            cdf["Severity"]=cdf["AQI"].apply(lambda v:aqi_category(v)[0])
            cdf=cdf[cdf["Severity"].isin(selected_severities)]
            focus_loc=next((x for x in india_locs if x["label"]==focus_choice),None) if focus_choice!="Entire India" else None
            cpcb_df=pd.DataFrame()
            with st.expander("Optional official CPCB National AQI station feed (data.gov.in)"):
                st.caption("Use your own data.gov.in API key and the resource ID associated with the CPCB real-time AQI dataset. The API must return station records with AQI and latitude/longitude fields.")
                cpcb_key=st.text_input("data.gov.in API key (optional)",type="password",
                    value=os.getenv("DATA_GOV_IN_API_KEY",""),key="cpcb_ogd_key")
                cpcb_resource=st.text_input("CPCB OGD resource ID",value=os.getenv("CPCB_AQI_RESOURCE_ID",""),
                    key="cpcb_resource_id")
                if st.button("Load CPCB station AQI records"):
                    if not cpcb_key or not cpcb_resource:
                        st.warning("Enter both the data.gov.in API key and CPCB resource ID.")
                    else:
                        try:
                            cpcb_df=fetch_cpcb_ogd_station_aqi(cpcb_key,cpcb_resource)
                            st.session_state["cpcb_station_data"]=cpcb_df
                            st.success(f"Loaded {len(cpcb_df)} station AQI records with usable coordinates.")
                            if cpcb_df.empty:
                                st.warning("The configured resource returned no records with recognizable AQI and coordinate fields. Check the resource schema/ID; no station markers were fabricated.")
                        except Exception as exc:
                            st.error(f"CPCB OGD request failed: {exc}")
                cpcb_df=st.session_state.get("cpcb_station_data",pd.DataFrame())
                if not cpcb_df.empty:
                    st.dataframe(cpcb_df,hide_index=True,use_container_width=True)
            station_df=pd.DataFrame()
            with st.expander("Optional verified station-level sensor markers (OpenAQ)"):
                st.caption("OpenAQ station markers use actual returned station coordinates and raw pollutant values. They do not invent or calculate station AQI.")
                openaq_key=st.text_input("OpenAQ API key (optional; not saved)",type="password",
                    value=os.getenv("OPENAQ_API_KEY",""),key="openaq_map_key")
                station_city=st.selectbox("Station search city",selected_cities or filter_names,key="openaq_city")
                if st.button("Load nearby OpenAQ stations"):
                    station_loc=next(x for x in india_locs if x["label"]==station_city)
                    if not openaq_key:
                        st.warning("Provide an OpenAQ API key. See https://docs.openaq.org/")
                    else:
                        try:
                            station_df=fetch_openaq_station_measurements(station_loc["latitude"],station_loc["longitude"],openaq_key)
                            st.session_state["openaq_stations"]=station_df
                            st.session_state["openaq_station_city"]=station_city
                        except Exception as exc:
                            st.error(f"OpenAQ station request failed: {exc}")
                if st.session_state.get("openaq_station_city")==station_city:
                    station_df=st.session_state.get("openaq_stations",pd.DataFrame())
                    st.write(f"Returned raw station measurements: {len(station_df)}")
                    if not station_df.empty:
                        st.dataframe(station_df,use_container_width=True,hide_index=True)
            try:
                india_boundary=fetch_india_geojson()
            except Exception as exc:
                india_boundary=None
                st.caption(f"State-boundary overlay unavailable; OpenStreetMap base map remains active. {exc}")
            india_map=build_india_map(cdf,mode,india_boundary,focus_loc,station_df,cpcb_df)
            st_folium(india_map,use_container_width=True,height=620,returned_objects=[],
                      key=f"india-air-map-{mode}-{focus_choice}-{len(cdf)}")
            st.caption("City markers: Open-Meteo CAMS grid-cell US AQI estimates. Blue points: raw OpenAQ sensor pollutant measurements. CPCB-colored station points appear only when the configured OGD feed returns AQI plus real coordinates.")
            if not cpcb_df.empty:
                st.markdown(
                    '<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px">'
                    '<span style="color:#16a34a">● CPCB Good 0–50</span>'
                    '<span style="color:#84cc16">● Satisfactory 51–100</span>'
                    '<span style="color:#eab308">● Moderate 101–200</span>'
                    '<span style="color:#f97316">● Poor 201–300</span>'
                    '<span style="color:#dc2626">● Very Poor 301–400</span>'
                    '<span style="color:#7f1d1d">● Severe 401–500</span></div>',
                    unsafe_allow_html=True)
                st.subheader("Official CPCB station records returned by configured OGD resource")
                cpcb_view=cpcb_df.copy()
                cpcb_view["Category"]=cpcb_view["CPCB AQI"].apply(lambda v:cpcb_aqi_category(v)[0])
                st.dataframe(cpcb_view.sort_values("CPCB AQI",ascending=False),hide_index=True,use_container_width=True)
            # Legend explicitly identifies model AQI category bands.
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px">'
                '<span style="color:#16a34a">● Good (US 0–50)</span>'
                '<span style="color:#84cc16">● Moderate (US 51–100)</span>'
                '<span style="color:#eab308">● Sensitive groups (US 101–150)</span>'
                '<span style="color:#f97316">● Unhealthy (US 151–200)</span>'
                '<span style="color:#dc2626">● Very unhealthy (US 201–300)</span>'
                '<span style="color:#7f1d1d">● Hazardous (US 301+)</span></div>',
                unsafe_allow_html=True)
            # Hotspots and cleanest cities based only on valid, same-source current estimates.
            ranked=cdf.dropna(subset=["AQI"]).sort_values("AQI",ascending=False)
            left,right=st.columns(2)
            with left:
                st.subheader("Highest-AQI monitored city estimates")
                if ranked.empty: st.info("No current AQI values available.")
                else: st.dataframe(ranked[["City","AQI","PM2.5","PM10","Observation time"]].head(10),hide_index=True,use_container_width=True)
            with right:
                st.subheader("Lowest-AQI monitored city estimates")
                if ranked.empty: st.info("No current AQI values available.")
                else: st.dataframe(ranked[["City","AQI","PM2.5","PM10","Observation time"]].tail(10).sort_values("AQI"),hide_index=True,use_container_width=True)
            st.subheader("City details")
            if selected_cities:
                detail_city=st.selectbox("Inspect city",selected_cities,key="india_map_detail")
                detail=next(x for x in india_locs if x["label"]==detail_city)
                drow=current_map[current_map["City"]==detail_city]
                if not drow.empty:
                    r=drow.iloc[0]
                    cat,_=aqi_category(r["AQI"])
                    st.write(f"**{detail_city}** · {cat} · observation {r['Observation time']}")
                    st.dataframe(pd.DataFrame([
                        {"Field":"AQI (US provider)","Value":r["AQI"]},
                        {"Field":"PM2.5 (µg/m³)","Value":r["PM2.5"]},
                        {"Field":"PM10 (µg/m³)","Value":r["PM10"]},
                        {"Field":"CO (µg/m³)","Value":r["CO"]},
                        {"Field":"NO₂ (µg/m³)","Value":r["NO₂"]},
                        {"Field":"SO₂ (µg/m³)","Value":r["SO₂"]},
                        {"Field":"O₃ (µg/m³)","Value":r["O₃"]},
                        {"Field":"Temperature °C","Value":r["Temperature (°C)"]},
                        {"Field":"Humidity %","Value":r["Humidity (%)"]},
                        {"Field":"Wind km/h","Value":r["Wind (km/h)"]},
                    ]),hide_index=True,use_container_width=True)
                    _,advice=health_advice(r["AQI"])
                    st.info(advice[0])
                    st.subheader("Previous day and historical trend")
                    try:
                        detail_end=TODAY-timedelta(days=1)
                        detail_start=detail_end-timedelta(days=91)
                        detail_hist,detail_msg=get_history(detail,detail_start,detail_end)
                        detail_daily=daily_air(detail_hist)
                        prev=detail_daily[detail_daily["date"]==detail_end]
                        if not prev.empty:
                            st.metric("Previous-day mean US AQI",f"{prev.iloc[0]['AQI']:.0f}")
                        else:
                            st.caption("Previous-day value was not present in the returned historical data.")
                        if not detail_daily.empty:
                            recent=detail_daily[detail_daily["date"]>=TODAY-timedelta(days=30)]
                            st.plotly_chart(px.line(recent,x="date",y="AQI",markers=True,
                                title=f"{detail['label']} · previous 30 days"),use_container_width=True)
                            st.caption(detail_msg)
                        detail_wx=get_weather_history(detail["latitude"],detail["longitude"],
                            detail_start.isoformat(),detail_end.isoformat(),detail.get("timezone","auto"))
                        detail_forecast,detail_rows,detail_mae=predict_city(detail,detail_hist,detail_wx)
                        st.subheader("Upcoming 7-day prediction")
                        st.plotly_chart(px.line(detail_forecast,x="Date",y="Predicted AQI",markers=True,
                            title="AI prediction · not an observation"),use_container_width=True)
                        st.dataframe(detail_forecast[["Date","Predicted AQI","Category"]],hide_index=True,use_container_width=True)
                    except Exception as detail_exc:
                        st.info(f"Historical trend / prediction unavailable for this city: {detail_exc}")
            if current_errors:
                for city,error in current_errors.items(): st.warning(f"{city}: {error}")
        else:
            st.error("No city AQI records were received. The map will not generate placeholder values.")

elif page=="Pollution Alerts":
    st.header("Pollution alerts & notification center")
    st.caption("The dashboard checks alerts while open. For monitoring while the browser is closed, run the included air_guard_worker.py in a second terminal after saving alert settings.")
    settings=load_alert_settings()
    all_city_labels=[x["label"] for x in watchlist]
    defaults=[x for x in settings.get("cities",[]) if x in all_city_labels]
    with st.form("airguard_alert_settings"):
        st.subheader("Alert preferences")
        alert_cities_ui=st.multiselect("Cities to monitor",all_city_labels,default=defaults)
        threshold_ui=st.select_slider("Alert at or above provider US AQI",
            options=[51,101,151,201,301,401],value=int(settings.get("threshold",151)))
        cooldown_ui=st.selectbox("Duplicate alert cooldown",["15 minutes","30 minutes","1 hour","3 hours","6 hours"],
            index={15:0,30:1,60:2,180:3,360:4}.get(int(settings.get("cooldown_minutes",60)),2))
        current_toggle=st.checkbox("Alert on current AQI",value=bool(settings.get("monitor_current",True)))
        predicted_toggle=st.checkbox("Alert on predicted AQI",value=bool(settings.get("monitor_predicted",True)))
        browser_toggle=st.checkbox("Enable browser notification attempts",value=bool(settings.get("browser_notifications",True)))
        email_toggle=st.checkbox("Send email when SMTP is configured",value=bool(settings.get("email_notifications",False)))
        email_to=st.text_input("Alert email recipient",value=settings.get("email_to",""),placeholder="name@example.com")
        interval_ui=st.selectbox("Background current-AQI polling interval",
            [5,10,15,30],index=[5,10,15,30].index(int(settings.get("interval_minutes",10))),
            format_func=lambda v:f"Every {v} minutes")
        submitted=st.form_submit_button("Save alert settings",type="primary")
        if submitted:
            cooldown_map={"15 minutes":15,"30 minutes":30,"1 hour":60,"3 hours":180,"6 hours":360}
            settings={"cities":alert_cities_ui,"threshold":threshold_ui,
                "cooldown_minutes":cooldown_map[cooldown_ui],"monitor_current":current_toggle,
                "monitor_predicted":predicted_toggle,"browser_notifications":browser_toggle,
                "email_notifications":email_toggle,"email_to":email_to.strip(),
                "interval_minutes":interval_ui,
                "location_records":[x for x in watchlist if x["label"] in alert_cities_ui]}
            save_alert_settings(settings)
            st.success("Alert preferences saved to the local SQLite database.")
            st.rerun()

    new_alerts=st.session_state.get("new_airguard_alerts",[])
    if new_alerts:
        st.subheader("New alerts in this refresh")
        for item in new_alerts:
            st.error(f"{item['City']} · {item['Category']} · AQI {item['AQI']:.0f}: {item['Message']}")
        if settings.get("browser_notifications"):
            browser_notification_panel(new_alerts)
    if st.session_state.get("alert_poll_error"):
        st.warning("Automatic alert check issue: "+st.session_state["alert_poll_error"])
    st.subheader("Active / unread alerts")
    all_alerts=get_alert_history(city_filter=settings.get("cities",[]),unread_only=True)
    if all_alerts.empty:
        st.success("No unread pollution alerts for the saved settings.")
    else:
        st.dataframe(all_alerts.drop(columns=["created_at"]),hide_index=True,use_container_width=True)
        chosen_ids=st.multiselect("Select alerts to mark as read",all_alerts["Alert ID"].tolist())
        if st.button("Mark selected as read"):
            read_alerts(chosen_ids); st.rerun()

    st.subheader("Generate predictive alerts")
    st.caption("Forecast events are clearly stored as Predicted; they are not measurements. Model requires sufficient aligned history/weather records.")
    # Evaluate configured cities every six hours while the alert page is open.
    last_prediction_check=settings.get("last_predicted_check")
    due=True
    if last_prediction_check:
        try:
            last_dt=datetime.fromisoformat(last_prediction_check)
            if last_dt.tzinfo is None: last_dt=last_dt.replace(tzinfo=IST)
            due=(datetime.now(IST)-last_dt)>=timedelta(hours=6)
        except Exception:
            due=True
    if due and settings.get("monitor_predicted",True):
        target_locs=[x for x in watchlist if x["label"] in settings.get("cities",[])]
        with st.spinner("Scheduled six-hour forecast alert scan for configured cities…"):
            _auto_predicted,_auto_errors=evaluate_predicted_alerts_for_locations(target_locs,settings)
        st.session_state["new_predicted_alerts"]=_auto_predicted
        settings["last_predicted_check"]=datetime.now(IST).isoformat(timespec="seconds")
        save_alert_settings(settings)
        for _err in _auto_errors: st.caption(f"Forecast scan note: {_err}")
    if st.button("Check 7-day predictions now"):
        target_locs=[x for x in watchlist if x["label"] in settings.get("cities",[])]
        with st.spinner("Training city models and checking forecast thresholds…"):
            future_alerts,forecast_errors=evaluate_predicted_alerts_for_locations(target_locs,settings)
        st.session_state["new_predicted_alerts"]=future_alerts
        settings["last_predicted_check"]=datetime.now(IST).isoformat(timespec="seconds")
        save_alert_settings(settings)
        if future_alerts: st.success(f"Created {len(future_alerts)} predictive alert(s).")
        else: st.info("No forecast threshold crossings were found, or forecasts were unavailable.")
        for err in forecast_errors: st.caption(err)
    predicted_new=st.session_state.get("new_predicted_alerts",[])
    if predicted_new:
        for a in predicted_new: st.warning(f"Forecast alert · {a['City']} · {a['Date']} · US AQI {a['AQI']:.0f} · {a['Category']}")

    st.subheader("Browser notifications and sound")
    st.caption("Browser notifications require permission and generally require HTTPS or localhost. This is page-open notification delivery, not push delivery when the browser is closed.")
    current_new=[{"City":a["City"],"Message":a["Message"],"Type":a["Type"],"id":a["City"]+a["Type"]} for a in new_alerts]
    predicted_browser=[{"City":a["City"],"Message":a["Message"],"Type":a["Type"],"id":a["City"]+a["Date"]} for a in predicted_new]
    if settings.get("browser_notifications"):
        browser_notification_panel(current_new+predicted_browser)
    st.caption("Use **Test alert sound** in the browser-notification control above; playback requires a user click because browsers block unsolicited audio.")

    st.subheader("Alert history")
    h1,h2,h3=st.columns(3)
    hist_cities=h1.multiselect("Filter city",all_city_labels,default=settings.get("cities",[]),key="alert_history_city")
    hist_severity=h2.multiselect("Filter severity",["Good","Moderate","Unhealthy for sensitive groups","Unhealthy","Very unhealthy","Hazardous"],key="alert_history_severity")
    unread_only=h3.checkbox("Unread only",value=False,key="alert_history_unread")
    alert_dates=st.date_input("Alert date range",value=(TODAY-timedelta(days=30),TODAY),key="alert_history_dates")
    if isinstance(alert_dates,tuple) and len(alert_dates)==2:
        history_start,history_end=alert_dates
    else:
        history_start=history_end=alert_dates
    alert_hist=get_alert_history(hist_cities or None,hist_severity or None,history_start,history_end,unread_only)
    if not alert_hist.empty:
        st.dataframe(alert_hist.drop(columns=["created_at"]),hide_index=True,use_container_width=True)
        ids=st.multiselect("Mark history entries as read",alert_hist["Alert ID"].tolist(),key="alert_read_ids")
        b1,b2=st.columns(2)
        if b1.button("Mark selected read"):
            read_alerts(ids); st.rerun()
        if b2.button("Export alert history CSV"):
            st.download_button("Download CSV",alert_hist.to_csv(index=False).encode(),"air_guard_alert_history.csv","text/csv")
    else:
        st.info("No alerts match the selected filters.")
    with st.expander("Clear old alert records"):
        clear_days=st.selectbox("Delete alerts older than",["90 days","180 days","1 year"])
        cutoff=TODAY-timedelta(days={"90 days":90,"180 days":180,"1 year":365}[clear_days])
        if st.button("Delete old alerts"):
            clear_old_alerts(cutoff); st.success("Old alert records deleted."); st.rerun()


st.divider()
st.caption("Data provenance: Open-Meteo Air Quality API (CAMS city-grid model estimates), optional OpenAQ station-level raw sensor measurements (requires API key), OpenStreetMap base map and public India GeoJSON boundary overlay, Open-Meteo Geocoding API, Weather Forecast API and Historical Weather API. No readings are randomly generated. Missing values remain missing. Provider AQI in this app is US AQI, not CPCB National AQI. Forecasts are experimental Random Forest estimates, not official forecasts. Open-Meteo air-quality documentation describes hourly air-quality data, forecast and past-days limits; longer date-range coverage is provider-dependent. Location search data is based on GeoNames.")
st.markdown("[Air Quality API](https://open-meteo.com/en/docs/air-quality-api) · [Geocoding API](https://open-meteo.com/en/docs/geocoding-api) · [Weather API](https://open-meteo.com/en/docs) · [CPCB India](https://cpcb.nic.in/)")
