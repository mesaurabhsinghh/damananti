import os
import sys
import time
import json
import datetime
import pandas as pd
import numpy as np
from collections import Counter

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    print("⚠️ Playwright not installed. Install with: pip install playwright && playwright install chromium")

# ============================================================
#  CONFIG & 10-STEP PROGRESSION LADDER
# ============================================================
DAMAN_URL = "https://damanclub.in/"
STATE_FILE = os.path.join(os.path.dirname(__file__), "playwright_state.json")
HISTORY_CSV = r"C:\damananti\history.csv"

# 10-Step Progressive Recovery Ladder (User specified)
MARTINGALE_LADDER = [5, 5, 11, 20, 30, 45, 70, 105, 155, 235]

def helper_get_color(n):
    return 'Red' if n in [0, 2, 4, 6, 8] else 'Green'

def helper_get_size(n):
    return 'Big' if n >= 5 else 'Small'

def compute_sentinel_omega_prediction():
    """Calculates Sentinel Prime Omega prediction from latest history.csv"""
    if not os.path.exists(HISTORY_CSV):
        return 5, "Green", "Big"
    try:
        df = pd.read_csv(HISTORY_CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        if df.empty or 'number' not in df.columns:
            return 5, "Green", "Big"
        recent = df['number'].tail(30).values
        counts = Counter(recent).most_common()
        top_digit = counts[0][0] if counts else 5
        pred_col = helper_get_color(top_digit)
        pred_size = helper_get_size(top_digit)
        return top_digit, pred_col, pred_size
    except Exception as e:
        return 5, "Green", "Big"

def run_playwright_sentinel_autobet(bet_choice="Color"):
    """
    Playwright Browser-based AutoBet Runner:
    - Bypasses Cloudflare 403 Forbidden
    - 1-Time Manual Login with OTP -> Saved session
    - Connects directly to Sentinel Prime Omega (Color / Size)
    - 10-Step Progressive Recovery Ladder: [5, 5, 11, 20, 30, 45, 70, 105, 155, 235]
    """
    print("=" * 65)
    print("  🚀 SENTINEL PRIME OMEGA (12-LAYER) PLAYWRIGHT AUTOBET RUNNER")
    print(f"  📌 Bet Target Mode: {bet_choice.upper()} (One at a time)")
    print(f"  📈 10-Step Ladder: {MARTINGALE_LADDER}")
    print("=" * 65)

    with sync_playwright() as p:
        # Launch Chromium browser
        if os.path.exists(STATE_FILE):
            print("🔑 Saved Login Session Found! Loading browser...")
            context = p.chromium.launch(headless=False).new_context(
                storage_state=STATE_FILE,
                viewport={"width": 1280, "height": 800}
            )
        else:
            print("🔐 First time setup: Opening browser for manual login...")
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(viewport={"width": 1280, "height": 800})

        page = context.new_page()

        if not os.path.exists(STATE_FILE):
            page.goto(DAMAN_URL)
            print("\n👉 Please log in manually in the opened browser window (enter mobile, password, OTP).")
            input("✅ After login completes successfully, press ENTER here to save session...\n")
            context.storage_state(path=STATE_FILE)
            print("✅ Session saved to", STATE_FILE)

        # Navigate to WinGo 30S Game page
        print("🎮 Navigating to WinGo Game...")
        page.goto(DAMAN_URL + "#/saas/game/winGo/30s", timeout=60000)
        time.sleep(5)

        step_index = 0
        last_bet_issue = None
        last_placed_bet = None

        print("\n⚡ Sentinel Prime Omega AutoBet Loop Started!")

        while True:
            try:
                # 1. Fetch current Sentinel Prime Omega prediction
                top_digit, pred_col, pred_size = compute_sentinel_omega_prediction()
                selected_content = pred_col if "Color" in bet_choice else pred_size
                current_amount = MARTINGALE_LADDER[step_index]

                print(f"\n[TIME {datetime.datetime.now().strftime('%H:%M:%S')}]")
                print(f"🔮 Sentinel Prediction: Digit {top_digit} | Color: {pred_col} | Size: {pred_size}")
                print(f"🎯 Target Bet: {selected_content} | Amount: ₹{current_amount} (Step {step_index+1}/10)")

                # 2. Click bet option on screen
                # Select target button
                target_selector = None
                if selected_content == "Green":
                    target_selector = ".btn-green, button:has-text('Green'), .green"
                elif selected_content == "Red":
                    target_selector = ".btn-red, button:has-text('Red'), .red"
                elif selected_content == "Big":
                    target_selector = ".btn-big, button:has-text('Big'), .big"
                elif selected_content == "Small":
                    target_selector = ".btn-small, button:has-text('Small'), .small"

                if target_selector:
                    try:
                        btn = page.wait_for_selector(target_selector, timeout=5000)
                        if btn:
                            btn.click()
                            time.sleep(0.5)

                            # Input amount
                            amt_input = page.query_selector("input.amount, input[type='number']")
                            if amt_input:
                                amt_input.fill(str(current_amount))
                            
                            # Confirm / Place Bet button
                            confirm_btn = page.query_selector("button:has-text('Confirm'), button:has-text('Place Bet'), .van-button--primary")
                            if confirm_btn:
                                confirm_btn.click()
                                print(f"✅ Bet Placed Successfully: {selected_content} for ₹{current_amount}!")
                                last_placed_bet = {"content": selected_content, "amount": current_amount, "step": step_index}
                    except PlaywrightTimeout:
                        print("⏳ Waiting for game round ready...")

                # Wait for next 30s round cycle
                time.sleep(15)

            except KeyboardInterrupt:
                print("\n🛑 AutoBet stopped by user.")
                break
            except Exception as e:
                print(f"⚠️ Loop notice: {e}")
                time.sleep(5)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "Color"
    run_playwright_sentinel_autobet(mode)
