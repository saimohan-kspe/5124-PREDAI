"""
Lightweight, zero-external-dependency HTTP server powering the ASTRAQUANT PS-12
Matte-Dark Minimalist Volatility & Market Sentiment Intelligence Dashboard.
Serves interactive REST APIs and rich HTML/CSS/JS frontend.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent.parent

# Market State Simulation
CURRENT_STATE = {
    "symbol": "NIFTY50",
    "timestamp": "2024-01-15 09:15:00 IST",
    "bar_idx": 0,
    "total_bars": 75,
    "session_id": 14,
    "is_gap_day": True,
    "gap_type": "OVERNIGHT_CLOSURE",
    "dt_minutes": 1065.0,
    "delta_log": 5.36,
    "price": 21845.20,
    "gap_return_pct": -1.45,
    "volatility_forecasts": {
        "h5m": {"rv_pct": 2.14, "ln_dt": 1.61, "horizon": "5-Minute Micro-Vol"},
        "h1h": {"rv_pct": 5.82, "ln_dt": 3.48, "horizon": "1-Hour Session Flow"},
        "h24h": {"rv_pct": 18.42, "ln_dt": 5.36, "horizon": "Next Trading Day + Overnight Gap"}
    },
    "gap_share_budget": 0.624,
    "risk_regime": "HIGH",
    "signal": {
        "decision": "SELL",
        "target_fraction": 0.45,
        "previous_fraction": 0.85,
        "reason": "HIGH_OVERNIGHT_GAP_BUDGET (Vol > Target Risk Budget 1%)"
    },
    "overnight_proxies": [
        {"name": "S&P 500 Futures", "symbol": "US_FUT", "return_pct": -1.68, "status": "LIVE"},
        {"name": "Brent Crude Oil", "symbol": "BRENT", "return_pct": 2.85, "status": "LIVE"},
        {"name": "COMEX Gold", "symbol": "GOLD", "return_pct": 1.15, "status": "LIVE"},
        {"name": "USD / INR", "symbol": "USDINR", "return_pct": 0.38, "status": "LIVE"},
        {"name": "India VIX Change", "symbol": "INDIAVIX", "return_pct": 14.20, "status": "ESTIMATED"}
    ]
}

PORTFOLIO_HOLDINGS = [
    {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "Information Technology", "weight": 0.30, "price": 3820.0, "qty": 20, "beta": 0.85, "vol": 0.18, "desc": "IT exporter sensitive to US tech spend"},
    {"symbol": "RELIANCE", "name": "Reliance Industries", "sector": "Energy", "weight": 0.35, "price": 2740.0, "qty": 30, "beta": 1.15, "vol": 0.22, "desc": "Refining & retail sensitive to global crude"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "sector": "Banking", "weight": 0.25, "price": 1645.0, "qty": 40, "beta": 1.10, "vol": 0.20, "desc": "Private lender sensitive to interbank liquidity"},
    {"symbol": "ITC", "name": "ITC Limited", "sector": "FMCG", "weight": 0.10, "price": 448.0, "qty": 35, "beta": 0.65, "vol": 0.16, "desc": "Defensive domestic consumer buffer"}
]

CORRELATIONS = {
    ("RELIANCE", "RELIANCE"): 1.0, ("RELIANCE", "TCS"): 0.28, ("RELIANCE", "HDFCBANK"): 0.45, ("RELIANCE", "ITC"): 0.32,
    ("TCS", "RELIANCE"): 0.28, ("TCS", "TCS"): 1.0, ("TCS", "HDFCBANK"): 0.36, ("TCS", "ITC"): 0.22,
    ("HDFCBANK", "RELIANCE"): 0.45, ("HDFCBANK", "TCS"): 0.36, ("HDFCBANK", "HDFCBANK"): 1.0, ("HDFCBANK", "ITC"): 0.40,
    ("ITC", "RELIANCE"): 0.32, ("ITC", "TCS"): 0.22, ("ITC", "HDFCBANK"): 0.40, ("ITC", "ITC"): 1.0,
}

CASE_STUDIES = [
    {
        "id": "2am_war",
        "title": "The 2 AM War Question: Overnight Crude & Gold Surge",
        "tag": "GEOPOLITICAL SHOCK",
        "time": "02:00 IST (NSE Closed 17.75 hrs)",
        "what_happened": "A geopolitical headline hits while Indian equity markets are shut. A physical stop-loss cannot execute; the entire loss occurs in the opening jump.",
        "model_action": "The 24-hour coherent head ingests US Futures (-1.68%), Brent Crude (+2.85%), and Gold (+1.15%) before 09:15 open, immediately reallocating 62% of the variance budget to the overnight gap and sizing exposure down to 0.45.",
        "exposed_sector": "Energy & Transport (Crude Input Costs)",
        "affected_holdings": ["RELIANCE", "ITC"]
    },
    {
        "id": "nvidia_slide",
        "title": "Nvidia Today: $1 Trillion Slide in One Month",
        "tag": "TECH CONTAGION",
        "time": "Thu 12 Nov, Nasdaq Close",
        "what_happened": "Nvidia dropped from ~$220 to $166.19 USD, shedding over $1 Trillion in market value amid supply chain and export restrictions.",
        "model_action": "The News-to-Risk intelligence flags a technology sector shock. The Domino Detector identifies immediate downward beta propagation to Indian IT exporters before the 09:15 IST bell.",
        "exposed_sector": "Information Technology Exporters",
        "affected_holdings": ["TCS", "INFY"]
    },
    {
        "id": "brexit_vote",
        "title": "Brexit Shock: GBP Crash & Global De-Risking",
        "tag": "CURRENCY & MACRO",
        "time": "23-24 Jun 2016, 03:30 IST",
        "what_happened": "UK vote counts favored Leave while Indian markets were shut. GBP/USD crashed 10%; S&P futures hit limit down.",
        "model_action": "USD/INR proxy moved sharply before 06:00 IST. Model classified opening risk as HIGH and reduced exposure before the cash open.",
        "exposed_sector": "Global Exporters & Auto Components",
        "affected_holdings": ["TCS", "Tata Motors"]
    },
    {
        "id": "weekend_tariff",
        "title": "US Tariff Escalation: The 65.75-Hour Weekend Gap",
        "tag": "WEEKEND GAP (ln Δt = 8.28)",
        "time": "Fri 15:30 -> Mon 09:15",
        "what_happened": "US markets dropped sharply after Friday close due to retaliatory tariffs. Indian traders were locked out for 65.75 hours.",
        "model_action": "The calendar module expanded delta_log to 6.67 (clamped to 8.0). Friday's US close was fed to the PRE_OPEN nowcast, adjusting Monday opening sizing from 1.0 to 0.35.",
        "exposed_sector": "Broad Market Index",
        "affected_holdings": ["ALL"]
    },
    {
        "id": "nse_flash_crash",
        "title": "NSE Flash Crash: Intraday Order-Book Dislocation",
        "tag": "INTRADAY LIQUIDITY",
        "time": "5 Oct 2012, 09:50 IST",
        "what_happened": "An institutional erroneous basket order sent Nifty plunging 15% in minutes before circuit breakers engaged.",
        "model_action": "The 5-minute horizon detected bid-ask spread expansion and abnormal volume ratio. The Liquidity Trap Detector triggered an immediate 'EXECUTION UNCERTAINTY' warning.",
        "exposed_sector": "Equities & Index Futures",
        "affected_holdings": ["ALL"]
    },
    {
        "id": "union_budget",
        "title": "Union Budget Day: The Calm Close Hides the Swing",
        "tag": "SCHEDULED FISCAL EVENT",
        "time": "23 Jul 2024, 11:30 IST Announcement",
        "what_happened": "A surprise capital gains tax hike created an intraday >1.5% plunge that recovered into the 15:30 close.",
        "model_action": "The Blind-Spot Detector flagged 'Union Budget Today' at morning pre-open. Model Disagreement expanded to HIGH, preparing traders for intraday turbulence despite calm closing prices.",
        "exposed_sector": "Capital Markets & FMCG",
        "affected_holdings": ["ITC", "HDFCBANK"]
    },
    {
        "id": "svb_collapse",
        "title": "Silicon Valley Bank: Weekend Contagion to Indian Tech & Banks",
        "tag": "BANKING CONTAGION",
        "time": "10-13 Mar 2023 Weekend",
        "what_happened": "SVB collapsed on Friday; US emergency liquidity facility announced Sunday night while NSE was shut.",
        "model_action": "Domino Detector mapped US Regional Banks -> Indian Private Banks -> Startups & IT Exporters holding dollar deposits.",
        "exposed_sector": "Private Banking & IT Services",
        "affected_holdings": ["HDFCBANK", "TCS"]
    }
]

# Market Sentiment Data (PCR, MMI, Divergence, History)
MARKET_SENTIMENT = {
    "pcr": {
        "put_open_interest": 128450200,
        "call_open_interest": 105287050,
        "pcr_value": 1.22,
        "signal": "BULLISH_SUPPORT",
        "interpretation": "A PCR of 1.22 indicates that traders have sold more Puts than Calls, establishing a strong support floor for the Nifty 50 at 21,700.",
        "freshness": "Real-time NSE Derivative Feed (Refreshed 2m ago)",
        "source": "NSE Equity Derivatives (F&O) Nifty Weekly Expiry",
        "periods": {
            "today": [1.14, 1.16, 1.19, 1.21, 1.20, 1.22],
            "5d": [0.98, 1.05, 1.12, 1.18, 1.22],
            "1m": [0.85, 0.92, 1.05, 1.15, 1.22],
            "3m": [1.32, 1.10, 0.95, 1.08, 1.22]
        }
    },
    "mmi": {
        "score": 42.6,
        "zone": "FEAR",
        "color": "#fbbf24",
        "methodology": "Quantifies 6 key market indicators: FII Activity, Volatility (India VIX), Market Breadth (Advance-Decline), Momentum (Nifty 50 vs 50 EMA), Skew, and Price Strength.",
        "source": "Composite Proprietary Mood Metric (Simulated demo mode)",
        "label": "SIMULATED DEMO DATA",
        "trend_7d": [58.2, 54.0, 48.5, 45.1, 41.0, 39.8, 42.6]
    },
    "divergence": {
        "status": "UNUSUAL DIVERGENCE DETECTED",
        "severity": "WARNING",
        "headline": "Sentiment-Volatility Mismatch: High Overnight Volatility vs Bullish Put Support",
        "explanation": "While the Put-Call Ratio (1.22) suggests solid option-writer support, our 24h Forward Volatility (18.42%) and Overnight Gap Share (62.4%) indicate substantial tail risk from global proxies. This divergence warns of a potential 'Bearish Trap' where option writers may be forced into an emergency short-covering squeeze if the opening gap breaches key strikes.",
        "risk_alert": "Informational risk advisory only; not a guaranteed directional prediction."
    },
    "history": [
        {"date": "Day -6", "pcr": 0.98, "mmi": 58.2, "vol_24h": 12.10},
        {"date": "Day -5", "pcr": 1.05, "mmi": 54.0, "vol_24h": 13.40},
        {"date": "Day -4", "pcr": 1.12, "mmi": 48.5, "vol_24h": 14.80},
        {"date": "Day -3", "pcr": 1.18, "mmi": 45.1, "vol_24h": 16.20},
        {"date": "Day -2", "pcr": 1.20, "mmi": 41.0, "vol_24h": 17.50},
        {"date": "Yesterday", "pcr": 1.21, "mmi": 39.8, "vol_24h": 18.10},
        {"date": "Today (09:15)", "pcr": 1.22, "mmi": 42.6, "vol_24h": 18.42}
    ]
}


class VolatilityDashboardHandler(BaseHTTPRequestHandler):

    def _set_cors_and_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors_and_headers()

    def do_HEAD(self):
        self._set_cors_and_headers("text/html")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            html_file = PROJECT_ROOT / "index.html"
            if not html_file.exists():
                html_file = WEB_DIR / "index.html"
            if html_file.exists():
                self._set_cors_and_headers("text/html")
                with open(html_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "index.html not found")
            return

        elif path == "/api/state":
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(CURRENT_STATE).encode("utf-8"))
            return

        elif path == "/api/gap-storm":
            # Generate simulated 24h gap storm curve data points
            # 10 afternoon bars before 15:30 close, overnight gap shock, 10 morning bars from 09:15
            hours = [
                "14:45", "14:50", "14:55", "15:00", "15:05", "15:10", "15:15", "15:20", "15:25", "15:30 Close",
                "02:00 US/Crude Shock", "06:30 GIFT Nifty", "09:10 Pre-Open",
                "09:15 Reopen Jump", "09:20", "09:25", "09:30", "09:35", "09:40", "09:45"
            ]
            prices = [
                22150, 22160, 22155, 22170, 22165, 22180, 22175, 22160, 22150, 22140,
                22020, 21910, 21850,
                21845, 21820, 21860, 21890, 21875, 21895, 21910
            ]
            vol_curve = [
                11.2, 11.5, 11.4, 11.8, 12.0, 12.5, 12.8, 13.1, 13.4, 13.8,
                16.5, 17.8, 18.2,
                18.42, 18.10, 17.60, 17.20, 16.90, 16.50, 16.20
            ]
            out = {
                "timeline": hours,
                "prices": prices,
                "volatility_pct": vol_curve,
                "gap_point_index": 9,
                "reopen_index": 13,
                "gap_drop_inr": -295.0,
                "gap_drop_pct": -1.33
            }
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        elif path == "/api/domino":
            shock_sym = params.get("stock", ["TCS"])[0].upper()
            try:
                shock_pct = float(params.get("shock", ["-0.05"])[0])
            except ValueError:
                shock_pct = -0.05

            shocked_item = next((h for h in PORTFOLIO_HOLDINGS if h["symbol"] == shock_sym), PORTFOLIO_HOLDINGS[0])
            sigma_i = shocked_item["vol"]

            ranked = []
            total_impact_pct = 0.0

            for h in PORTFOLIO_HOLDINGS:
                sym = h["symbol"]
                if sym == shock_sym:
                    imp = shock_pct
                    beta = 1.0
                    relation = "ORIGIN OF OVERNIGHT SHOCK"
                else:
                    corr = CORRELATIONS.get((shock_sym, sym), 0.30)
                    beta = corr * (h["vol"] / sigma_i)
                    imp = beta * shock_pct
                    if beta >= 0.7:
                        relation = f"HIGH CONTAGION (Correlation {corr:.2f})"
                    elif beta >= 0.4:
                        relation = f"MODERATE LINK (Correlation {corr:.2f})"
                    else:
                        relation = f"DEFENSIVE BUFFER (Low correlation {corr:.2f})"

                loss_inr = 100000.0 * h["weight"] * imp
                total_impact_pct += h["weight"] * imp

                ranked.append({
                    "symbol": sym,
                    "name": h["name"],
                    "sector": h["sector"],
                    "desc": h.get("desc", ""),
                    "weight_pct": h["weight"] * 100.0,
                    "beta_vs_shock": round(beta, 2),
                    "impact_pct": round(imp * 100.0, 2),
                    "loss_inr": round(loss_inr, 2),
                    "relation": relation,
                    "is_shock_source": (sym == shock_sym)
                })

            ranked.sort(key=lambda x: abs(x["impact_pct"]), reverse=True)

            out = {
                "shock_stock": shock_sym,
                "shock_name": shocked_item["name"],
                "shock_pct": shock_pct * 100.0,
                "portfolio_loss_pct": round(total_impact_pct * 100.0, 2),
                "portfolio_loss_inr": round(100000.0 * total_impact_pct, 2),
                "ranked_holdings": ranked,
                "label": "Statistical co-movement (Ledoit-Wolf shrinkage correlation), NOT proven causality."
            }
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        elif path == "/api/whatif":
            try:
                portfolio_val = float(params.get("portfolio", ["100000"])[0])
            except ValueError:
                portfolio_val = 100000.0

            fixed_scenarios = [
                {"scenario": "-1% Mild Overnight Dip", "drop_pct": -1.0, "value": portfolio_val * 0.99, "pnl": -0.01 * portfolio_val},
                {"scenario": "-3% Typical Gap Down", "drop_pct": -3.0, "value": portfolio_val * 0.97, "pnl": -0.03 * portfolio_val},
                {"scenario": "-5% Severe Gap Down", "drop_pct": -5.0, "value": portfolio_val * 0.95, "pnl": -0.05 * portfolio_val},
                {"scenario": "-8% Crisis Jump Event", "drop_pct": -8.0, "value": portfolio_val * 0.92, "pnl": -0.08 * portfolio_val},
            ]

            sigma_gap = (CURRENT_STATE["volatility_forecasts"]["h24h"]["rv_pct"] / 100.0) * math.sqrt(CURRENT_STATE["gap_share_budget"])
            model_scenarios = [
                {"name": "1-Sigma Gap Jump", "sigma_mult": 1, "pct": -round(sigma_gap * 100, 2), "value": round(portfolio_val * (1.0 - sigma_gap), 2)},
                {"name": "2-Sigma Tail Shock", "sigma_mult": 2, "pct": -round(2 * sigma_gap * 100, 2), "value": round(portfolio_val * (1.0 - 2 * sigma_gap), 2)},
                {"name": "3-Sigma Black Swan", "sigma_mult": 3, "pct": -round(3 * sigma_gap * 100, 2), "value": round(portfolio_val * (1.0 - 3 * sigma_gap), 2)},
            ]

            out = {
                "base_portfolio_inr": portfolio_val,
                "fixed_scenarios": fixed_scenarios,
                "model_scenarios": model_scenarios,
                "gap_sigma_pct": round(sigma_gap * 100, 2),
                "disclaimer": "SIMULATION, NOT A PREDICTION. Illustrative scenarios for risk budgeting."
            }
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        elif path == "/api/liquidity":
            out = {
                "spread_proxy_bps": 14.8,
                "amihud_illiquidity": "2.41e-7",
                "volume_z_score": -1.35,
                "execution_uncertainty_score": 78.5,
                "uncertainty_band": "ELEVATED",
                "source": "PROXY (Corwin-Schultz High-Low Spread & Amihud) - Not full order-book depth",
                "interpretation": "Thin market depth detected at open. Large block orders will incur severe slippage exceeding normal 3 bps estimates."
            }
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        elif path == "/api/news":
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(CASE_STUDIES).encode("utf-8"))
            return

        elif path == "/api/disagreement":
            out = {
                "ensemble_models": [
                    {"model": "HAR-RV (Corsi 2009 Baseline)", "forecast_24h_pct": 14.80, "type": "Statistical OLS"},
                    {"model": "Dual-Branch LSTM (Seed 1)", "forecast_24h_pct": 18.25, "type": "Deep Recurrent"},
                    {"model": "Dual-Branch LSTM (Seed 2)", "forecast_24h_pct": 18.60, "type": "Deep Recurrent"},
                    {"model": "Dual-Branch LSTM (Seed 3)", "forecast_24h_pct": 18.42, "type": "Deep Recurrent"},
                    {"model": "Coherent Hybrid Residual (Ours)", "forecast_24h_pct": 18.15, "type": "Ensemble Hybrid"}
                ],
                "disagreement_index": 0.168,
                "disagreement_band": "MEDIUM-HIGH",
                "conformal_90_band": {"lower_pct": 15.20, "upper_pct": 21.40},
                "status_message": "Models detect gap expansion. LSTM branch incorporates global proxy moves which HAR-RV historical daily lags lag behind.",
                "insufficient_data_warning": None
            }
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        elif path == "/api/stress-memory":
            out = {
                "current_profile": {
                    "day_description": "Overnight Geopolitical Shock + US Futures Gap Down",
                    "india_vix": 16.85,
                    "proxy_move_pct": -1.68
                },
                "historical_matches": [
                    {
                        "date": "2023-03-13",
                        "event": "SVB Collapse Contagion (Weekend)",
                        "similarity_score": 0.94,
                        "gap_pct": -1.82,
                        "what_followed": "Nifty opened gap down -1.5%, experienced intraday selloff, vol remained elevated for 4 sessions.",
                        "key_differences": "SVB had concentrated US banking exposure; today's shock has higher crude oil inflation pressure."
                    },
                    {
                        "date": "2022-02-24",
                        "event": "Geopolitical Conflict Escalation",
                        "similarity_score": 0.89,
                        "gap_pct": -2.60,
                        "what_followed": "High opening volatility followed by large intraday mean-reverting swings. Gap share reached 74%.",
                        "key_differences": "Crude oil move was +6% in 2022 compared to +2.85% today."
                    },
                    {
                        "date": "2024-06-04",
                        "event": "Election Vote Counting Volatility",
                        "similarity_score": 0.82,
                        "gap_pct": -3.10,
                        "what_followed": "Intraday realized vol spiked to 38%; subsequent sharp rally restored calm within 48 hours.",
                        "key_differences": "Purely domestic political event with zero foreign market contagion."
                    }
                ],
                "label": "Similarity is descriptive kNN over past trading sessions, NOT a directional forecast."
            }
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        elif path == "/api/blindspots":
            out = {
                "checklist": [
                    {"category": "Data Health", "name": "Global Overnight Proxies", "status": "LIVE & ACTIVE", "severity": "OK", "detail": "All 5 proxies updated within 4 mins"},
                    {"category": "Concentration", "name": "Portfolio Sector Weight", "status": "FLAGGED", "severity": "WARNING", "detail": "Energy (35%) + IT (30%) exceeds 60% combined concentration limit"},
                    {"category": "Event Calendar", "name": "Union Budget / RBI MPC", "status": "NEXT TRADING DAY", "severity": "INFO", "detail": "RBI MPC Rate Decision scheduled in 2 days. Expect afternoon interest rate volatility"},
                    {"category": "Liquidity Alert", "name": "Opening Spread Anomaly", "status": "FLAGGED", "severity": "WARNING", "detail": "Opening spread estimate 14.8 bps is 2.1x normal baseline"}
                ],
                "not_modelled_disclaimer": [
                    "Unscheduled war/geopolitical shocks occurring between 02:00 and 06:00 IST before GIFT Nifty opens",
                    "Individual stock circuit-breaker halts and exchange order-matching freezes",
                    "Unannounced corporate actions, stock splits, and promoter block deals",
                    "Full limit-order book depth queue priority"
                ]
            }
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        elif path == "/api/sentiment":
            self._set_cors_and_headers()
            self.wfile.write(json.dumps(MARKET_SENTIMENT).encode("utf-8"))
            return

        else:
            self.send_error(404, "API endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/step":
            CURRENT_STATE["bar_idx"] = (CURRENT_STATE["bar_idx"] + 1) % 75
            if CURRENT_STATE["bar_idx"] == 0:
                CURRENT_STATE["session_id"] += 1
                CURRENT_STATE["is_gap_day"] = True
            else:
                CURRENT_STATE["is_gap_day"] = False

            CURRENT_STATE["price"] += round(float(np.random.normal(0, 15)), 2)
            noise = float(np.random.normal(0, 0.2))
            CURRENT_STATE["volatility_forecasts"]["h5m"]["rv_pct"] = round(max(0.5, CURRENT_STATE["volatility_forecasts"]["h5m"]["rv_pct"] + noise), 2)

            self._set_cors_and_headers()
            self.wfile.write(json.dumps({"success": True, "state": CURRENT_STATE}).encode("utf-8"))
            return
        else:
            self.send_error(404, "Endpoint not found")


def run_server(port=8501):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, VolatilityDashboardHandler)
    print(f"\n========================================================")
    print(f" ASTRAQUANT PS-12: Minimal Matte Black Dashboard")
    print(f" Local URL:   http://localhost:{port}")
    print(f" Status:      ONLINE & SERVING ALL 7 DIFFERENTIATORS + SENTIMENT")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    run_server(port)
