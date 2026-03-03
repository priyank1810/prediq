import os

# Configure TensorFlow before any TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TF info/warning logs
if os.getenv("RENDER", "").lower() in ("true", "1"):
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers import stocks, predictions, portfolio, alerts, indicators, signals, watchlist
from app.routers.fii_dii import router as fii_dii_router
from app.routers.sectors import router as sectors_router
from app.routers.websocket import router as ws_router, price_streamer, alert_checker, signal_accuracy_validator


async def smart_alert_checker():
    """Background task to check smart alerts. Only runs when active alerts exist."""
    from app.services.alert_service import alert_service
    from app.database import SessionLocal
    from app.models import SmartAlert

    await asyncio.sleep(30)
    while True:
        try:
            from app.utils.helpers import is_market_open
            from app.routers.websocket import manager
            if is_market_open() and manager.active_connections:
                db = SessionLocal()
                try:
                    # Skip entirely if no active smart alerts exist
                    count = db.query(SmartAlert).filter(SmartAlert.is_triggered == False).count()
                    if count > 0:
                        triggered = alert_service.check_smart_alerts(db)
                        if triggered:
                            for alert_data in triggered:
                                await manager.broadcast_to_all("smart_alert_triggered", alert_data)
                finally:
                    db.close()
        except Exception:
            pass
        await asyncio.sleep(300)  # 5 min (was 60s) — smart alerts don't need second-level checks


async def market_mood_broadcaster():
    """Broadcast Market Mood Score every 5 minutes during market hours."""
    from app.config import MARKET_MOOD_REFRESH_INTERVAL

    await asyncio.sleep(60)
    while True:
        try:
            from app.utils.helpers import is_market_open
            from app.routers.websocket import manager
            if is_market_open() and manager.active_connections:
                from app.services.market_mood_service import market_mood_service
                mood = market_mood_service.get_mood()
                await manager.broadcast_to_all("market_mood_update", mood)
        except Exception:
            pass
        await asyncio.sleep(MARKET_MOOD_REFRESH_INTERVAL)


def _migrate_db():
    """Add missing columns to existing tables (SQLite doesn't support IF NOT EXISTS for columns)."""
    import sqlite3
    from app.config import DATABASE_URL

    db_path = DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    migrations = [
        ("prediction_logs", "sector", "TEXT"),
        ("prediction_logs", "regime", "TEXT"),
        ("signal_logs", "sector", "TEXT"),
        ("signal_logs", "regime", "TEXT"),
        ("signal_logs", "oi_score", "REAL"),
    ]

    for table, column, col_type in migrations:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Add indexes for performance (idempotent — IF NOT EXISTS)
    indexes = [
        "CREATE INDEX IF NOT EXISTS ix_portfolio_holdings_user_id ON portfolio_holdings(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_price_alerts_user_id ON price_alerts(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_price_alerts_is_triggered ON price_alerts(is_triggered)",
        "CREATE INDEX IF NOT EXISTS ix_prediction_logs_symbol ON prediction_logs(symbol)",
        "CREATE INDEX IF NOT EXISTS ix_signal_logs_created_at ON signal_logs(created_at)",
        "CREATE INDEX IF NOT EXISTS ix_signal_logs_symbol_created ON signal_logs(symbol, created_at)",
    ]
    for idx_sql in indexes:
        try:
            cursor.execute(idx_sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    _migrate_db()
    streamer_task = asyncio.create_task(price_streamer())
    alert_task = asyncio.create_task(alert_checker())
    validator_task = asyncio.create_task(signal_accuracy_validator())
    smart_alert_task = asyncio.create_task(smart_alert_checker())
    mood_task = asyncio.create_task(market_mood_broadcaster())
    yield
    # Shutdown
    streamer_task.cancel()
    alert_task.cancel()
    validator_task.cancel()
    smart_alert_task.cancel()
    mood_task.cancel()


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

# API Routers
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(indicators.router, prefix="/api/indicators", tags=["indicators"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(watchlist.router, prefix="/api/watchlist", tags=["watchlist"])
app.include_router(fii_dii_router, prefix="/api/fii-dii", tags=["fii-dii"])
app.include_router(sectors_router, prefix="/api/sectors", tags=["sectors"])
app.include_router(ws_router)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
