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
from app.routers.websocket import router as ws_router, price_streamer, alert_checker, signal_broadcaster, high_confidence_scanner, signal_accuracy_validator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
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
app.include_router(ws_router)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
