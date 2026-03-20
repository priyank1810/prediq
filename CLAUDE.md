# PrediQ - Indian Stock Market Tracker & AI Predictor

## Project Overview
Full-stack stock tracker with AI-powered predictions, signals, and analysis for the Indian stock market (NSE/BSE).

## Tech Stack
- **Backend:** Python 3.11, FastAPI, SQLAlchemy (SQLite/PostgreSQL), Redis
- **Frontend:** Vanilla JS, Jinja2 templates, TailwindCSS (CDN), Lightweight Charts
- **AI/ML:** XGBoost, Prophet (Holt-Winters), TensorFlow (meta-learner), FinBERT (sentiment)
- **Deployment:** AWS Lightsail (Ubuntu), nginx, systemd, Let's Encrypt SSL, DuckDNS

## Architecture
```
main.py          → FastAPI app + background tasks (lifespan)
worker.py        → Background job processor (signals, MTF, OI)
app/
  routers/       → API endpoints (stocks, signals, predictions, portfolio, etc.)
  services/      → Business logic (signal_service, prediction_service, stock_learner, etc.)
  ai/            → ML models (xgboost_model, prophet_model, fundamental_model, explainer)
  models.py      → SQLAlchemy ORM models
  database.py    → DB engine + session
  config.py      → All configuration constants
static/          → CSS + JS (vanilla, no build step)
templates/       → Jinja2 HTML (single-page app: index.html)
deploy/          → nginx config, systemd services, setup script
```

## Key Commands
```bash
# Run locally
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Run tests
python3 -m pytest tests/ -v

# Docker
docker compose up --build
```

## Deployment
- **CI/CD via GitHub** — push to `main` branch triggers auto-deploy
- **Never SCP files directly to the server**
- Server: prediq.duckdns.org (Lightsail ap-south-1)
- SSH: `ssh -i ~/Downloads/LightsailDefaultKey-ap-south-1.pem ubuntu@<ip>`

## Key Services
| Service | Purpose |
|---------|---------|
| `signal_service.py` | Multi-component signals (tech + sentiment + global + fundamental + OI) |
| `prediction_service.py` | XGBoost + Prophet ensemble with neural meta-learner |
| `stock_learner.py` | Per-stock AI learning profiles (rebuilds daily at 4:15 PM IST) |
| `ai_summary_service.py` | Plain-English signal/earnings explanations |
| `nl_search_service.py` | Natural language stock search |
| `adaptive_weights.py` | Sector-level adaptive signal weights |
| `screener_service.py` | Technical stock screener |
| `data_fetcher.py` | Multi-source data (Angel One → NSE → Yahoo Finance fallback) |

## Important Notes
- Single-page app — all tabs in `templates/index.html`
- JS modules loaded lazily via `Lazy.loadAndInit()` pattern
- Static files served by nginx directly (no app restart needed for CSS/JS)
- HTML template changes require app restart (Jinja2 caches)
- Mobile-first: always test responsive layouts (stock sub-tabs, grids, forms)
- Market hours: 9:15 AM - 3:30 PM IST (signals use `is_market_open()`)
