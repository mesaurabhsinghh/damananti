import requests
import time
import csv
import json
import os

URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

seen = set()

if os.path.exists("history.csv"):
    with open("history.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                seen.add(row[0])

print("Collector started... listening for live issues...")

while True:
    try:
        r = requests.get(
            URL,
            params={"ts": int(time.time() * 1000)},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Origin": "https://damanworld.org",
                "Referer": "https://damanworld.org/"
            },
            timeout=10
        )

        data = r.json()
        rows = data["data"]["list"]

        if rows:
            latest_item = rows[0]
            latest_issue = str(latest_item["issueNumber"])
            latest_num = int(latest_item["number"])
            latest_col = latest_item["color"]
            latest_size = "Big" if latest_num >= 5 else "Small"

            # Save current_issue.json for app.py
            with open("current_issue.json", "w", encoding="utf-8") as f:
                json.dump({
                    "issue": latest_issue,
                    "target_issue": str(int(latest_issue) + 1),
                    "number": latest_num,
                    "color": latest_col,
                    "size": latest_size,
                    "timestamp": time.time()
                }, f, indent=2)

        for item in rows[::-1]:
            issue = str(item["issueNumber"])

            if issue in seen:
                continue

            number = int(item["number"])
            color = item["color"]
            bigsmall = "Big" if number >= 5 else "Small"

            with open("history.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["issue", "number", "color", "size"])

                writer.writerow([issue, number, color, bigsmall])

            seen.add(issue)
            print(f"Collected Issue: {issue} | Num: {number} | Color: {color} | Size: {bigsmall}")

        time.sleep(3)

    except Exception as e:
        print("Collector Error:", e)
        time.sleep(3)
