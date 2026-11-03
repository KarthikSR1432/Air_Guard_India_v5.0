# Air-Guard v4.0 — Global Air Quality Intelligence Dashboard

A responsive Streamlit dashboard for searching global cities, monitoring current air-quality estimates, viewing historical pollution records, training per-city AI forecasts, comparing cities, generating health guidance, and exporting CSV/PDF reports.

## Features
- Global city search and suggestion list powered by Open-Meteo Geocoding (GeoNames).
- Add/remove multiple locations in a session watchlist.
- Optional browser geolocation permission via `streamlit-geolocation`.
- Current AQI and PM2.5, PM10, CO, NO2, SO2, O3 plus current weather fields where provided.
- Auto-refresh every 10 minutes (with manual refresh).
- Historical pollution analysis with 7-day, 30-day, 3-month, 6-month, 1-year and custom windows.
- SQLite persistence for historical records fetched by the application.
- Per-city Random Forest 7-day predictions trained from retrieved daily AQI and historical weather.
- City comparisons with current, previous-day, historical and model-predicted AQI columns; health precautions; CSV exports and PDF reports.
- Separate navigation pages designed for desktop and mobile layouts.

## Install and run (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Data sources and interpretation
- Open-Meteo Air Quality API: https://open-meteo.com/en/docs/air-quality-api
- Open-Meteo Geocoding API: https://open-meteo.com/en/docs/geocoding-api
- Open-Meteo Weather API: https://open-meteo.com/en/docs
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
- CPCB reference: https://cpcb.nic.in/

### Important accuracy notes
1. The default AQI field is **US AQI** from Open-Meteo, not India's CPCB National AQI. The user interface labels it explicitly and does not convert or relabel it as CPCB AQI.
2. Air-quality fields are CAMS model estimates at a grid cell, not direct measurements from a CPCB/CAAQMS station. Do not describe them as regulatory real-time station readings.
3. Historical air quality is requested from the provider for the chosen dates. Availability may vary by date, region, model and API service. The app stores successfully retrieved records in `air_guard_v4.db` and can fall back to those saved records. It does not create synthetic historic values to fill gaps.
4. The Random Forest model needs at least 25 usable daily AQI/weather training rows. When insufficient data, weather history or forecast data are missing, the app explains why prediction cannot be generated. Model forecasts are indicative estimates, not official AQI forecasts.
5. Historical hourly AQI requests for very long windows may fail or return limited coverage depending on provider availability/endpoint constraints. Review the returned date coverage, not just the requested date range.
6. Browser location requires the user to grant permission. If unavailable, search for a city or region manually.
7. Health content is general information, not medical advice. The CPCB AQI ranges are included as a reference; they are not applied to the US AQI number.

## Local data
`air_guard_v4.db` is created automatically beside `app.py` and stores only provider-returned observations and current snapshots. Keep this file to retain locally collected history between runs.

## API and usage
Open-Meteo free APIs are intended for non-commercial use under the terms and attribution described by the provider. Check current terms, fair-use limits and licensing before production/commercial deployment.


## v4.1 — City-focused overview and Leaflet map
- Sidebar **Primary city** selection is the shared location for Overview, Current AQI and Health Precautions. Newly searched-and-added cities become the primary city automatically.
- Overview shows selected-city AQI, previous-day mean where available, next-day prediction where trainable, PM2.5 summary, available pollutant values, current weather, last-30-day AQI trend, seven-day forecast and health guidance.
- Leaflet/OpenStreetMap map automatically centers at selected geocoded coordinates (zoom 11) with a clickable AQI-colored marker and popup containing city, provider AQI category, observation time and source.
- Map colors use the provider's US AQI category bands. CPCB AQI thresholds are not applied to US AQI numbers.
- Individual monitoring-station markers are intentionally omitted because verified station coordinates and station-level readings are not integrated in this project.
- Streamlit interactions rerun the Python script while preserving the primary city in session state. This provides synchronized page state but is not a custom JavaScript single-page application.


## v5.0 — India Pollution Map and Pollution Alerts

### New navigation pages
- **India Pollution Map**: OpenStreetMap/Leaflet India map with zoom/pan, state/region and city filters, severity filters, city markers, AQI value labels, optional heatmap, city detail panel, top-10 highest and lowest monitored-city tables, and optional OpenAQ station-level raw readings.
- **Pollution Alerts**: configurable city watchlist, threshold, duplicate cooldown, current and predicted event preferences, in-app alerts, browser Notification API attempts, user-triggered sound test, optional email via SMTP environment variables, read/unread status, alert history filters and CSV export.

### Data provenance and AQI interpretation
1. Default map city points use Open-Meteo CAMS **city-grid model estimates** (US AQI); these are not CPCB station measurements.
2. Marker colors and alert categories for this provider are therefore labeled as **US AQI categories**. The CPCB India bands (Good 0–50, Satisfactory 51–100, Moderate 101–200, Poor 201–300, Very Poor 301–400, Severe 401–500) must only be used with actual CPCB National AQI values. This app does not silently apply CPCB breakpoints to US AQI.
3. Optional OpenAQ integration retrieves actual location coordinates and latest raw pollutant sensor measurements when a valid API key is entered. These are displayed as blue station markers and pollutant values only; the app does not infer an AQI from a single pollutant value.
4. The Government of India OGD catalog lists the CPCB real-time AQI dataset, but direct programmatic access may require the portal's current API/resource credentials. This build does not claim direct CPCB integration without those credentials. Official catalog: https://www.data.gov.in/catalog/real-time-air-quality-index
5. India boundary overlay is loaded from a public GeoJSON repository at runtime; if it is unavailable, the OpenStreetMap basemap and city markers remain usable. Boundary repository: https://github.com/udit-001/india-maps-data

### Alert delivery behavior
- Current alerts are evaluated when the Streamlit app reruns/refreshes and the upstream current API cache is due for refresh. They are saved in local SQLite and deduplicated by city, alert type, category and cooldown.
- Predictive alerts are generated when the user clicks **Check 7-day predictions for alert cities**. Forecast alerts are labeled Predicted and stored separately from current alerts.
- Browser notifications require permission and a secure context (HTTPS or localhost), and are best-effort while the dashboard is open. This is not background Web Push when the browser/app is closed; true background push requires a deployed service worker and push/VAPID backend.
- Email is optional. Configure `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, and `SMTP_FROM` as environment variables; enter a recipient in alert settings. Never commit SMTP credentials.
- The app itself is not an always-on scheduled backend. For continuous 24/7 alerts, deploy a scheduled worker (e.g. APScheduler/Celery) that calls the same provider fetch and alert evaluation services on a server.

### Optional OpenAQ API key
Set `OPENAQ_API_KEY` in the environment or enter it into the station panel at runtime. OpenAQ v3 documentation: https://docs.openaq.org/. Do not commit the API key to source control.

### Run
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```


### Optional CPCB Open Government Data (OGD) station AQI feed
The India map contains a CPCB OGD connector that accepts:
- `DATA_GOV_IN_API_KEY`
- `CPCB_AQI_RESOURCE_ID`

These can be set as environment variables or entered into the map's CPCB panel. The OGD portal requires an API key and the specific resource must have an API enabled. The connector only plots records when it can read an AQI field plus station latitude/longitude. It does not infer missing AQI or coordinates. Resource schema aliases are handled defensively; if the returned schema differs, no station markers are plotted and a warning is shown.
Official catalog: https://www.data.gov.in/catalog/real-time-air-quality-index
OGD API key and API URL access help: https://www.data.gov.in/help

The CPCB station layer uses the requested India National AQI categories:
- Good 0–50; Satisfactory 51–100; Moderate 101–200
- Poor 201–300; Very Poor 301–400; Severe 401–500

Those bands are applied only to rows returned by the configured CPCB OGD resource. Open-Meteo city-grid US AQI markers and alerts retain their own US AQI labels and category bands.

- Predicted-alert scanning runs every six hours while the Pollution Alerts page is open (and can also be triggered manually). For scans that run while no user has the dashboard open, a separate server scheduler/worker must be deployed.


### Secrets setup
Copy `.env.example` to `.env`, fill in only the credentials you have, and keep `.env` out of source control. `python-dotenv` loads local environment values at startup. No API credentials are included in this ZIP.


### Run background current-AQI alerts while the browser is closed
1. Start the dashboard: `python -m streamlit run app.py`.
2. Open **Pollution Alerts**, choose the cities and preferences, then click **Save alert settings** once. This stores selected city coordinates in SQLite for the worker.
3. Open a second PowerShell terminal in the same extracted folder, activate the same `.venv`, and run:
   ```powershell
   python air_guard_worker.py
   ```
4. The worker polls current provider US AQI at the saved interval (5–30 minutes), stores threshold alerts in the same SQLite database and sends email if SMTP credentials and a recipient are configured.
5. For a one-time connectivity check, run `python air_guard_worker.py --once`.

The standalone worker is for **current AQI threshold alerts**. Predictive alert generation remains in the Streamlit Pollution Alerts page and is scanned every six hours while that page is open. To make predictive alerts fully independent of the browser, schedule/deploy a separate ML worker with the same model dependencies and persistent model/data pipeline.
