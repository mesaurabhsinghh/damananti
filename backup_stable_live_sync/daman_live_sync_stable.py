"""
=============================================================================
DAMAN LIVE SYNC & ISSUE CALCULATOR ENGINE (PERMANENT LOCKED MODULE)
=============================================================================
This module is the single source of truth for:
1. Real-time Live Daman / Wingo issue number synchronization.
2. 17-digit issue format generation (YYYYMMDD10005XXXX).
3. Direct lottery API fetching with fallback.
DO NOT MODIFY THIS CORE ENGINE.
"""

import requests
import time
import datetime
import os
import numpy as np
import pandas as pd

LIVE_API_ENDPOINTS = {
    "Win Go 30Sec": "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
    "Win Go 1Min": "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json",
    "Win Go 3Min": "https://draw.ar-lottery01.com/WinGo/WinGo_3M/GetHistoryIssuePage.json",
    "Win Go 5Min": "https://draw.ar-lottery01.com/WinGo/WinGo_5M/GetHistoryIssuePage.json",
}

def helper_get_color(num):
    if num in [1, 3, 7, 9]:
        return "Green"
    elif num in [2, 4, 6, 8]:
        return "Red"
    elif num == 0:
        return "Red"
    elif num == 5:
        return "Green"
    return "Red"

def helper_get_size(num):
    return "Big" if num >= 5 else "Small"

def fetch_live_daman_game_data(game_mode="Win Go 30Sec"):
    """
    Fetches real-time live game issues & results directly from Daman live lottery API.
    """
    url = LIVE_API_ENDPOINTS.get(game_mode, LIVE_API_ENDPOINTS["Win Go 30Sec"])
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Origin": "https://damanworld.world",
        "Referer": "https://damanworld.world/",
        "Accept": "application/json, text/plain, */*"
    }
    try:
        r = requests.get(f"{url}?ts={int(time.time()*1000)}", headers=headers, timeout=3)
        if r.status_code == 200:
            data = r.json()
            items = data.get("data", {}).get("list", [])
            if items:
                records = []
                for it in items:
                    raw_num = it.get("number")
                    if raw_num is not None and str(raw_num).isdigit():
                        n = int(raw_num)
                        iss = int(it.get("issueNumber"))
                        c = helper_get_color(n)
                        s = helper_get_size(n)
                        records.append({
                            "issue": iss,
                            "number": n,
                            "color": c,
                            "size": s
                        })
                if records:
                    records.sort(key=lambda x: x["issue"])
                    return pd.DataFrame(records)
    except Exception:
        pass
    return None

def get_current_daman_calculated_issue_30s():
    """
    Calculates exact live Daman 17-digit issue counter synchronized with UTC reset.
    """
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    utc_seconds = utc_now.hour * 3600 + utc_now.minute * 60 + utc_now.second
    calculated_30s_index = (utc_seconds // 30)
    return int(utc_now.strftime("%Y%m%d")) * 1000000000 + 100050000 + calculated_30s_index
