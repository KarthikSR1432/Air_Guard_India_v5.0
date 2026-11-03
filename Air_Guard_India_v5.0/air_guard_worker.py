from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import hashlib
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "air_guard_v4.db"
IST = ZoneInfo("Asia/Kolkata")
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

def ensure_db():
    with sqlite3.connect(DB_PATH) as con:
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

def load_settings():
    with sqlite3.connect(DB_PATH) as con:
        row=con.execute("SELECT settings_json FROM alert_preferences WHERE id=1").fetchone()
    if not row:
        return None
    return json.loads(row[0])

def category(aqi):
    aqi=float(aqi)
    if aqi<=50: return "Good"
    if aqi<=100: return "Moderate"
    if aqi<=150: return "Unhealthy for sensitive groups"
    if aqi<=200: return "Unhealthy"
    if aqi<=300: return "Very unhealthy"
    return "Hazardous"

def send_email(settings,message,city):
    if not settings.get("email_notifications") or not settings.get("email_to"):
        return "In-app"
    host=os.getenv("SMTP_HOST","")
    port=int(os.getenv("SMTP_PORT","587"))
    user=os.getenv("SMTP_USER","")
    password=os.getenv("SMTP_PASSWORD","")
    sender=os.getenv("SMTP_FROM",user)
    if not all([host,user,password,sender]):
        return "In-app; SMTP not configured"
    email=EmailMessage()
    email["Subject"]=f"Air-Guard AQI alert — {city}"
    email["From"]=sender
    email["To"]=settings["email_to"]
    email.set_content(message)
    try:
        with smtplib.SMTP(host,port,timeout=15) as server:
            server.starttls()
            server.login(user,password)
            server.send_message(email)
        return "Email sent"
    except Exception as exc:
        return f"In-app; email failed: {exc}"

def poll_one(loc,settings):
    params={"latitude":loc["latitude"],"longitude":loc["longitude"],
            "current":"us_aqi","timezone":loc.get("timezone","auto")}
    response=requests.get(AQ_URL,params=params,timeout=(8,30),
                          headers={"User-Agent":"AirGuardWorker/5.0"})
    response.raise_for_status()
    current=response.json().get("current",{})
    aqi=current.get("us_aqi")
    obs=current.get("time","")
    if aqi is None or float(aqi)<float(settings.get("threshold",151)):
        return "No threshold crossing"
    cat=category(aqi)
    city=loc.get("label",loc.get("name","Unknown"))
    state=loc.get("region",loc.get("admin1",""))
    message=(f"{city}: provider US AQI is {float(aqi):.0f} ({cat}). "
             f"Reduce prolonged outdoor exertion as appropriate. Observation: {obs}. "
             "This value is a city-grid model estimate, not a CPCB station measurement.")
    now=datetime.now(IST)
    cooldown=int(settings.get("cooldown_minutes",60))
    cutoff=(now-timedelta(minutes=cooldown)).isoformat(timespec="seconds")
    dedupe=hashlib.sha256(f"{city}|Current|{cat}|{int(now.timestamp()//(max(1,cooldown)*60))}".encode()).hexdigest()
    alert_id=hashlib.sha256(f"{dedupe}|{now.isoformat()}".encode()).hexdigest()[:16]
    with sqlite3.connect(DB_PATH) as con:
        recent=con.execute("SELECT 1 FROM pollution_alerts WHERE city=? AND alert_type='Current' AND category=? AND created_at>=? LIMIT 1",
                           (city,cat,cutoff)).fetchone()
        if recent:
            return f"Suppressed duplicate alert: {city}"
        status=send_email(settings,message,city)
        try:
            con.execute("""INSERT INTO pollution_alerts
              (alert_id,dedupe_key,city,state,aqi,category,message,event_time,alert_type,
               notification_status,is_read,created_at)
              VALUES(?,?,?,?,?,?,? ,?,'Current',?,0,?)""",
              (alert_id,dedupe,city,state,float(aqi),cat,message,obs,status,now.isoformat(timespec="seconds")))
            con.commit()
        except sqlite3.IntegrityError:
            return f"Suppressed duplicate alert: {city}"
    return f"ALERT {city}: US AQI {float(aqi):.0f} ({cat}) · {status}"

def run_once():
    settings=load_settings()
    if not settings:
        print("No alert settings found. Start app.py, configure Pollution Alerts and save settings first.")
        return
    if not settings.get("monitor_current",True):
        print("Current AQI monitoring is disabled in saved alert preferences.")
        return
    locations=settings.get("location_records",[])
    if not locations:
        print("No alert cities saved. Add cities in the app and save alert settings.")
        return
    print(f"Polling {len(locations)} locations at {datetime.now(IST):%Y-%m-%d %H:%M:%S %Z}")
    for loc in locations:
        try:
            print(poll_one(loc,settings))
        except Exception as exc:
            print(f"DATA ERROR {loc.get('label',loc.get('name','city'))}: {exc}")

def main():
    parser=argparse.ArgumentParser(description="Air-Guard background current-AQI alert worker")
    parser.add_argument("--once",action="store_true",help="Run one polling cycle and exit")
    args=parser.parse_args()
    ensure_db()
    if args.once:
        run_once()
        return
    while True:
        settings=load_settings() or {}
        interval=max(5,int(settings.get("interval_minutes",10)))*60
        run_once()
        print(f"Next poll in {interval//60} minutes. Press Ctrl+C to stop.")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("Air-Guard worker stopped.")
            break

if __name__=="__main__":
    main()
