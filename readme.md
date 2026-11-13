# ASTRAQUANT X-2026 (PS-12)
## Multi-Horizon Volatility & Market Sentiment Intelligence Station

> **Problem Statement 12**: Multi-Horizon Volatility Forecasting During Overnight Gaps (NSE Equities)  
> **Team**: T Sai Mohan, Satvik Juluru, Ibrahim, Vamshidhar Reddy  
> **UI Aesthetic**: Matte Black & White Minimalist Financial Terminal

---

## ⚡ Quick Start (Local Run)

### 1. Requirements
* Python 3.10+
* Standard Python installation (zero external package requirement to run the web UI and REST API server)

### 2. Run Locally
**Windows:**
Double click `run.bat` or run in terminal:
```bash
python -m src.web.server
```

**Linux / macOS:**
```bash
bash run.sh
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## ☁️ Cloud Deployment Options

### Option 1: Render (Recommended - Free Tier Available)
1. Push this repository to GitHub or GitLab.
2. In [Render.com](https://render.com), click **New +** -> **Web Service**.
3. Connect your repository.
4. Set the following build settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt` (or leave empty if using built-in standard library server)
   - **Start Command:** `python -m src.web.server`
5. Click **Create Web Service**. Render automatically binds `PORT` and provides an HTTPS live domain!

### Option 2: Railway
1. Go to [Railway.app](https://railway.app) and create a **New Project**.
2. Select **Deploy from GitHub repo**.
3. Railway automatically detects `Procfile` / `Dockerfile` and deploys your service.

### Option 3: Docker Deployment
Build and run anywhere (AWS EC2, GCP Cloud Run, DigitalOcean, Azure):
```bash
# Build Docker image
docker build -t astraquant-volatility .

# Run container
docker run -p 8501:8501 astraquant-volatility
```
Visit `http://<your-server-ip>:8501`.

---

## 📁 Repository Structure
```
astraquant-volatility/
├── Dockerfile                  # Container build recipe
├── Procfile                    # PaaS process manager declaration
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows one-click runner
├── run.sh                      # Linux/macOS runner
├── FEATURE_MANIFEST.json       # Frozen feature order
├── config/
│   ├── default.yaml            # Model & simulation parameters
│   ├── entities.yaml           # Sector & company keyword mappings
│   └── events.yaml             # Macro schedule (Union Budget, RBI MPC)
├── data/
│   ├── news_sample.jsonl       # Illustrative news events
│   └── portfolio_sample.csv    # Benchmark portfolio holdings
├── src/
│   ├── common/                 # Domain constants & timing rules
│   ├── data/                   # MarketFeed, synthetic generator, calendar, cleaner
│   ├── features/               # Feature tensors
│   ├── models/                 # HAR-RV, Dual-Branch LSTM, Coherent Heads, Hybrid
│   ├── strategy/               # Vol targeting & exposure sizing
│   ├── intel/                  # 7 Differentiators & Sentiment engines
│   └── web/                    # HTTP Server & Matte Black Frontend
│       ├── server.py
│       └── index.html
└── tests/                      # Pytest suite (14 formal invariant tests)
```

---

## 🛡️ Non-Negotiable Domain Rules Satisfied
1. **NSE Cash Session Timing:** 09:15–15:30 IST = exactly 75 five-minute bars.
2. **Overnight Closure:** 17.75 hours = 1,065 minutes ($\ln \Delta t \approx 5.36$).
3. **Weekend Closure:** Friday 15:30 to Monday 09:15 = 65.75 hours = 3,945 minutes ($\ln \Delta t \approx 6.67$).
4. **Coherent Variance Heads:** Nested additive variance $V_{5m} \le V_{1h} \le V_{24h}$ by construction.
5. **Market Sentiment Intelligence:** Put-Call Ratio (PCR), Market Mood Index (MMI), Sentiment-Volatility Divergence, and Sentiment History.
