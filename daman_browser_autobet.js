/**
 * ⚡ SENTINEL PRIME OMEGA (12-LAYER) 100% IN-BROWSER AUTOBET RUNNER
 * -----------------------------------------------------------------
 * Instructions:
 * 1. Open https://damanclub.in/ in your Chrome/Brave browser and log in.
 * 2. Go to WinGo 30S game page.
 * 3. Press F12 -> Go to Console tab.
 * 4. Paste this entire code and press ENTER.
 * 5. AutoBet is now running with 10-step ladder [5, 5, 11, 20, 30, 45, 70, 105, 155, 235]!
 */

(function() {
    // ⚙️ CONFIGURATION
    const BET_MODE = "Color"; // Change to "Size" for Big/Small
    const LADDER = [5, 5, 11, 20, 30, 45, 70, 105, 155, 235];
    
    let stepIndex = 0;
    let lastBetIssue = null;
    let lastBetResult = null;
    let isBetting = false;

    console.clear();
    console.log("%c🚀 SENTINEL PRIME OMEGA AUTOBET ACTIVE 🟢", "color: #22c55e; font-size: 16px; font-weight: bold;");
    console.log(`📌 Bet Mode: ${BET_MODE} | Ladder: [${LADDER.join(", ")}]`);

    function getDamanToken() {
        try {
            const raw = localStorage.getItem("ar_token");
            if (raw) {
                const parsed = JSON.parse(raw);
                if (parsed && parsed.value) return parsed.value;
            }
        } catch(e) {}
        return localStorage.getItem("token") || "";
    }

    function getDamanSign() {
        try {
            const raw = localStorage.getItem("userInfo");
            if (raw) {
                const parsed = JSON.parse(raw);
                if (parsed && parsed.sign) return parsed.sign;
            }
        } catch(e) {}
        return "";
    }

    async function checkAndExecuteAutoBet() {
        if (isBetting) return;
        try {
            const token = getDamanToken();
            if (!token) {
                console.warn("⚠️ Token not found in localStorage. Make sure you are logged in on damanclub.in.");
                return;
            }

            // Fetch live history to check round status and derive Sentinel prediction
            const res = await fetch("https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?ts=" + Date.now(), {
                headers: { "Accept": "application/json, text/plain, */*" }
            });
            if (!res.ok) return;

            const data = await res.json();
            const list = (data && data.data && data.data.list) ? data.data.list : [];
            if (!list || list.length === 0) return;

            const latestResolved = list[0];
            const latestIssue = parseInt(latestResolved.issueNumber);
            const targetIssue = latestIssue + 1;

            // Check previous bet outcome to update 10-step ladder
            if (lastBetResult && lastBetResult.issue === latestIssue) {
                const actNum = parseInt(latestResolved.number);
                const actCol = [0,2,4,6,8].includes(actNum) ? "Red" : "Green";
                const actSize = actNum >= 5 ? "Big" : "Small";

                let won = false;
                if (lastBetResult.betType === "Color") {
                    won = (lastBetResult.choice === actCol);
                } else {
                    won = (lastBetResult.choice === actSize);
                }

                if (won) {
                    console.log(`%c🎉 [ROUND #${latestIssue}] WON! Result: ${actNum} (${actCol} | ${actSize}) -> Resetting to Step 1 (₹${LADDER[0]})`, "color: #22c55e; font-weight: bold;");
                    stepIndex = 0;
                } else {
                    stepIndex = Math.min(stepIndex + 1, LADDER.length - 1);
                    console.log(`%c🔴 [ROUND #${latestIssue}] LOST! Result: ${actNum} (${actCol} | ${actSize}) -> Next Step ${stepIndex+1} (₹${LADDER[stepIndex]})`, "color: #ef4444; font-weight: bold;");
                }
                lastBetResult = null;
            }

            // Don't bet twice on the same target issue
            if (lastBetIssue === targetIssue) return;

            // Derive Sentinel Prime Omega Prediction (Mode analysis on last 30 rounds)
            const recentNums = list.slice(0, 30).map(x => parseInt(x.number));
            const freq = {};
            recentNums.forEach(n => freq[n] = (freq[n] || 0) + 1);
            let topDigit = 5, maxF = 0;
            for (let k in freq) {
                if (freq[k] > maxF) {
                    maxF = freq[k];
                    topDigit = parseInt(k);
                }
            }

            const predCol = [0,2,4,6,8].includes(topDigit) ? "Red" : "Green";
            const predSize = topDigit >= 5 ? "Big" : "Small";
            const betChoice = (BET_MODE === "Color") ? predCol : predSize;
            const currentAmt = LADDER[stepIndex];

            // SelectType values: Green=1, Red=3, Violet=2, Big=1, Small=2
            let selectTypeVal = 1;
            if (betChoice === "Green") selectTypeVal = 1;
            else if (betChoice === "Red") selectTypeVal = 3;
            else if (betChoice === "Violet") selectTypeVal = 2;
            else if (betChoice === "Big") selectTypeVal = 1;
            else if (betChoice === "Small") selectTypeVal = 2;

            const randomNonce = String(Date.now()) + String(Math.floor(1000 + Math.random() * 9000));
            const sign = getDamanSign();

            const payload = {
                typeId: 1,
                gameId: 1,
                gameCode: "WinGo_30S",
                issueNumber: String(targetIssue),
                amount: currentAmt,
                betMultiple: 1,
                betCount: 1,
                betContent: betChoice,
                selectType: selectTypeVal,
                language: "en",
                random: randomNonce,
                timestamp: Date.now()
            };
            if (sign) {
                payload.signature = sign;
                payload.sign = sign;
            }

            isBetting = true;
            console.log(`⏳ [SENDING BET] #${targetIssue} ➔ ${betChoice} | Amount: ₹${currentAmt} (Step ${stepIndex+1}/10)...`);

            const betRes = await fetch("https://api.ar-lottery01.com/api/Lottery/WinGoBet?ts=" + Date.now(), {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + token,
                    "Origin": "https://damanclub.in",
                    "Referer": "https://damanclub.in/"
                },
                body: JSON.stringify(payload)
            });

            const resJson = await betRes.json().catch(() => ({ status: betRes.status }));
            if (betRes.ok && (resJson.code === 0 || resJson.code === 200 || resJson.data)) {
                console.log(`%c✅ [SUCCESS] Bet Placed on Issue #${targetIssue} ➔ ${betChoice} (₹${currentAmt})!`, "color: #22c55e; font-weight: bold; font-size: 14px;", resJson);
            } else {
                console.warn(`⚠️ [BET RESPONSE] Issue #${targetIssue} (HTTP ${betRes.status}):`, resJson);
            }

            lastBetIssue = targetIssue;
            lastBetResult = {
                issue: targetIssue,
                betType: BET_MODE,
                choice: betChoice,
                amount: currentAmt,
                step: stepIndex
            };

        } catch (err) {
            console.error("AutoBet Execution Notice:", err);
        } finally {
            isBetting = false;
        }
    }

    // Run check every 2.5 seconds
    setInterval(checkAndExecuteAutoBet, 2500);
    checkAndExecuteAutoBet();
})();
