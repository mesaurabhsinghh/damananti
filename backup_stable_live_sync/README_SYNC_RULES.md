# 🔒 DAMAN LIVE SYNCHRONIZATION ENGINE (LOCKED PERMANENT STANDARD)

## Purpose
Yeh directory stable reference backup store karti hai taaki live Daman issue synchronization ka logic kabhi bhi overwrite ya break na ho.

## Core Rules:
1. **Live Daman API Endpoint:**
   - URL: `https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json`
   - Headers: `Origin: https://damanworld.world`, `Referer: https://damanworld.world/`
2. **Issue Format (17-Digit Standard):**
   - Format: `YYYYMMDD` + `10005` + `4-digit UTC counter` (e.g. `20260810100051036`)
   - Target Issue: `int(latest_issue) + 1` (e.g. `20260810100051037`)
3. **UTC Reset Alignment:**
   - Daman daily round counter resets at UTC midnight (`05:30 AM IST`), calculating `utc_seconds // 30`.

## Files in this Backup:
- `app_stable.py`: Complete Streamlit web app with 100% verified live sync.
- `dashboard_stable.py`: Mirrored production dashboard application.
- `daman_live_sync_stable.py`: Modular live lottery data fetching engine.
