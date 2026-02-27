import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.routers import stocks, predictions, portfolio, alerts, indicators, signals, watchlist
from app.routers.options import router as options_router
from app.routers.fii_dii import router as fii_dii_router
from app.routers.sectors import router as sectors_router
from app.routers.websocket import router as ws_router, price_streamer, alert_checker, signal_broadcaster, high_confidence_scanner, signal_accuracy_validator


async def smart_alert_checker():
    """Background task to check smart alerts every 60 seconds."""
    from app.services.alert_service import alert_service
    from app.database import SessionLocal

    await asyncio.sleep(30)
    while True:
        try:
            from app.utils.helpers import is_market_open
            if is_market_open():
                db = SessionLocal()
                try:
                    triggered = alert_service.check_smart_alerts(db)
                    if triggered:
                        from app.routers.websocket import manager
                        for alert_data in triggered:
                            await manager.broadcast_to_all("smart_alert_triggered", alert_data)
                finally:
                    db.close()
        except Exception:
            pass
        await asyncio.sleep(60)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    streamer_task = asyncio.create_task(price_streamer())
    alert_task = asyncio.create_task(alert_checker())
    signal_task = asyncio.create_task(signal_broadcaster())
    scanner_task = asyncio.create_task(high_confidence_scanner())
    validator_task = asyncio.create_task(signal_accuracy_validator())
    smart_alert_task = asyncio.create_task(smart_alert_checker())
    mood_task = asyncio.create_task(market_mood_broadcaster())
    yield
    # Shutdown
    streamer_task.cancel()
    alert_task.cancel()
    signal_task.cancel()
    scanner_task.cancel()
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
app.include_router(options_router, prefix="/api/options", tags=["options"])
app.include_router(fii_dii_router, prefix="/api/fii-dii", tags=["fii-dii"])
app.include_router(sectors_router, prefix="/api/sectors", tags=["sectors"])
app.include_router(ws_router)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
