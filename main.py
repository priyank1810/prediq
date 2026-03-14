import os
import logging

# Configure TensorFlow before any TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TF info/warning logs
if os.getenv("LOW_RESOURCE_MODE", "").lower() in ("true", "1"):
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers import stocks, predictions, portfolio, alerts, indicators, signals, watchlist
from app.routers.fii_dii import router as fii_dii_router
from app.routers.sectors import router as sectors_router
from app.routers.websocket import router as ws_router, price_streamer, alert_checker, signal_accuracy_validator, signal_accuracy_validator_30min, signal_accuracy_validator_1hr
from app.routers.jobs import router as jobs_router
from app.utils.rate_limiter import RateLimiter


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
                def _check_smart():
                    db = SessionLocal()
                    try:
                        count = db.query(SmartAlert).filter(SmartAlert.is_triggered == False).count()
                        if count > 0:
                            return alert_service.check_smart_alerts(db)
                    finally:
                        db.close()
                    return None

                triggered = await asyncio.to_thread(_check_smart)
                if triggered:
                    for alert_data in triggered:
                        await manager.broadcast_to_all("smart_alert_triggered", alert_data)
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
                mood = await asyncio.to_thread(market_mood_service.get_mood)
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
        ("signal_logs", "price_after_30min", "REAL"),
        ("signal_logs", "price_after_1hr", "REAL"),
        ("signal_logs", "was_correct_30min", "BOOLEAN"),
        ("signal_logs", "was_correct_1hr", "BOOLEAN"),
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
        "CREATE INDEX IF NOT EXISTS ix_job_queue_poll ON job_queue(status, priority, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_job_queue_job_type ON job_queue(job_type)",
        "CREATE INDEX IF NOT EXISTS ix_job_queue_status ON job_queue(status)",
    ]
    for idx_sql in indexes:
        try:
            cursor.execute(idx_sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


async def periodic_job_enqueuer():
    """Every 5 min, enqueue background jobs (watchlist_signals, mtf_stream, oi_stream)
    if market is open and no duplicate pending jobs exist."""
    from app.services.job_service import job_service
    from app.utils.helpers import is_market_open
    from app.routers.websocket import manager

    await asyncio.sleep(120)  # Let services warm up
    while True:
        try:
            if is_market_open():
                def _enqueue_jobs():
                    if not job_service.has_pending("watchlist_signals"):
                        job_service.enqueue("watchlist_signals", {}, priority=0)
                    if manager.active_connections:
                        symbols = list(manager.get_all_subscribed_symbols())
                        if symbols:
                            if not job_service.has_pending("mtf_stream"):
                                job_service.enqueue("mtf_stream", {"symbols": symbols}, priority=0)
                            if not job_service.has_pending("oi_stream"):
                                job_service.enqueue("oi_stream", {"symbols": symbols}, priority=0)

                await asyncio.to_thread(_enqueue_jobs)
        except Exception:
            pass
        await asyncio.sleep(300)


async def worker_result_broadcaster():
    """Every 3s, read completed background jobs and broadcast results via WebSocket."""
    from app.services.job_service import job_service
    from app.routers.websocket import manager

    await asyncio.sleep(10)
    while True:
        try:
            if manager.active_connections:
                jobs = await asyncio.to_thread(job_service.get_completed_background_jobs)
                for job in jobs:
                    job_type = job["job_type"]
                    result = job["result"]

                    if job_type == "watchlist_signals":
                        signals_data = result.get("signals", {})
                        for sym, signal_data in signals_data.items():
                            await manager.broadcast_signal(sym, signal_data)

                    elif job_type == "mtf_stream":
                        mtf_data = result.get("mtf_data", {})
                        for sym, data in mtf_data.items():
                            await manager.broadcast_mtf_update(sym, data)

                    elif job_type == "oi_stream":
                        oi_data = result.get("oi_data", {})
                        for sym, data in oi_data.items():
                            await manager.broadcast_oi_update(sym, data)

                    await asyncio.to_thread(job_service.mark_broadcast, job["id"])
        except Exception:
            pass
        await asyncio.sleep(3)


def _setup_audit_logger():
    """Configure audit logger to write to logs/audit.log."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, "audit.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    audit = logging.getLogger("audit")
    audit.setLevel(logging.INFO)
    audit.addHandler(handler)
    audit.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _setup_audit_logger()
    Base.metadata.create_all(bind=engine)
    _migrate_db()
    tasks = [
        asyncio.create_task(price_streamer()),
        asyncio.create_task(alert_checker()),
        asyncio.create_task(signal_accuracy_validator()),
        asyncio.create_task(signal_accuracy_validator_30min()),
        asyncio.create_task(signal_accuracy_validator_1hr()),
        asyncio.create_task(smart_alert_checker()),
        asyncio.create_task(market_mood_broadcaster()),
        asyncio.create_task(periodic_job_enqueuer()),
        asyncio.create_task(worker_result_broadcaster()),
    ]
    yield
    # Shutdown
    for t in tasks:
        t.cancel()


app = FastAPI(title="Indian Stock Market Tracker & AI Predictor", lifespan=lifespan)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_TIMEOUT = 120  # seconds — global safety net for all HTTP requests

# HTTP API rate limiting — configurable via RATE_LIMIT_RPM env var
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
_rate_limiter = RateLimiter(max_requests=RATE_LIMIT_RPM, window_seconds=60)

# Allowed origins for CSRF protection (state-changing requests must have valid Origin)
_ALLOWED_ORIGINS = set(CORS_ORIGINS) | {"null"}  # "null" for same-origin requests without Origin header
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate-limit HTTP API requests per IP (skip WebSocket upgrades)."""
    from fastapi.responses import JSONResponse
    if (request.url.path.startswith("/api/")
            and request.headers.get("upgrade", "").lower() != "websocket"):
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.check_rate_limit(client_ip):
            retry_after = _rate_limiter.time_until_available(client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """Reject state-changing requests from unknown origins."""
    from fastapi.responses import JSONResponse
    if request.method not in _SAFE_METHODS and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        # If Origin header is present, validate it
        if origin and origin not in _ALLOWED_ORIGINS:
            return JSONResponse(status_code=403, content={"detail": "Origin not allowed"})
    return await call_next(request)


audit_logger = logging.getLogger("audit")
_AUDIT_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    """Log state-changing API requests with user identity and outcome."""
    start = time.monotonic()
    response = await call_next(request)

    if request.method in _AUDIT_METHODS and request.url.path.startswith("/api/"):
        duration_ms = round((time.monotonic() - start) * 1000)
        # Extract user from Authorization header (without full decode — just for logging)
        auth_header = request.headers.get("authorization", "")
        user_hint = "anonymous"
        if auth_header.startswith("Bearer "):
            try:
                from jose import jwt as jose_jwt
                from app.config import SECRET_KEY, ALGORITHM
                payload = jose_jwt.decode(auth_header[7:], SECRET_KEY, algorithms=[ALGORITHM])
                user_hint = payload.get("sub", "unknown")
            except Exception:
                user_hint = "invalid-token"
        elif request.headers.get("x-api-key"):
            user_hint = "api-key"

        client_ip = request.client.host if request.client else "unknown"
        audit_logger.info(
            f"{request.method} {request.url.path} | user={user_hint} ip={client_ip} "
            f"status={response.status_code} duration={duration_ms}ms"
        )

    return response


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=504, content={"detail": "Request timed out"})

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
app.include_router(jobs_router, prefix="/api/jobs", tags=["jobs"])
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
