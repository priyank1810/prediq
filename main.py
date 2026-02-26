import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers import stocks, predictions, portfolio, alerts, indicators, signals, watchlist
from app.routers.auth import router as auth_router
from app.routers.admin import router as admin_router
from app.routers.options import router as options_router
from app.routers.websocket import router as ws_router, price_streamer, alert_checker, signal_broadcaster, high_confidence_scanner, signal_accuracy_validator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    _migrate_add_user_columns()
    streamer_task = asyncio.create_task(price_streamer())
    alert_task = asyncio.create_task(alert_checker())
    signal_task = asyncio.create_task(signal_broadcaster())
    scanner_task = asyncio.create_task(high_confidence_scanner())
    validator_task = asyncio.create_task(signal_accuracy_validator())
    yield
    # Shutdown
    streamer_task.cancel()
    alert_task.cancel()
    signal_task.cancel()
    scanner_task.cancel()
    validator_task.cancel()


def _migrate_add_user_columns():
    """Add user_id columns to existing tables if they don't exist (SQLite migration)."""
    import sqlite3
    from app.config import DATA_DIR

    db_path = str(DATA_DIR / "stock_tracker.db")
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables_to_migrate = ["portfolio_holdings", "price_alerts", "watchlist_items"]
    for table in tables_to_migrate:
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            if "user_id" not in columns:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER REFERENCES users(id)")
        except Exception:
            pass

    conn.commit()
    conn.close()


app = FastAPI(title="Indian Stock Market Tracker & AI Predictor", lifespan=lifespan)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Auth Routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])

# API Routers
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(indicators.router, prefix="/api/indicators", tags=["indicators"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(watchlist.router, prefix="/api/watchlist", tags=["watchlist"])
app.include_router(options_router, prefix="/api/options", tags=["options"])
app.include_router(ws_router)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
