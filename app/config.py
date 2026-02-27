import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR}/stock_tracker.db"

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Cache TTL (seconds)
CACHE_TTL_QUOTE = 5
CACHE_TTL_HISTORY = 300
CACHE_TTL_STOCK_LIST = 86400

# LSTM defaults
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 100  # early stopping typically cuts at 20-40
LSTM_BATCH_SIZE = 32
MODEL_FRESHNESS_HOURS = 6
FINE_TUNE_EPOCHS = 12
FINE_TUNE_FRESHNESS_HOURS = 48  # full retrain if older than this

# Early stopping & learning rate scheduling
LSTM_EARLY_STOP_PATIENCE = 8
LSTM_LR_REDUCE_PATIENCE = 4
LSTM_LR_REDUCE_FACTOR = 0.7
LSTM_MIN_LR = 5e-5
LSTM_WALKFORWARD_SPLITS = 5
LSTM_LEARNING_RATE = 0.0005

# Price streaming
PRICE_STREAM_INTERVAL = 5  # seconds
ALERT_CHECK_INTERVAL = 30  # seconds

# --- Intraday Signal Config ---
CACHE_TTL_INTRADAY = 30
CACHE_TTL_SENTIMENT = 300
CACHE_TTL_GLOBAL = 60
CACHE_TTL_OPTION_CHAIN = 120
CACHE_TTL_MARKET_MOVERS = 120
SIGNAL_REFRESH_INTERVAL = 60

SIGNAL_WEIGHT_TECHNICAL = 0.65
SIGNAL_WEIGHT_SENTIMENT = 0.25
SIGNAL_WEIGHT_GLOBAL = 0.10

HIGH_CONFIDENCE_THRESHOLD = 60
HIGH_CONFIDENCE_SCAN_INTERVAL = 300  # seconds (5 min to avoid API rate limiting)

GLOBAL_MARKET_SYMBOLS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Nikkei 225": "^N225",
    "India VIX": "^INDIAVIX",
    "USD/INR": "USDINR=X",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
}

NEWS_RSS_SOURCES = [
    "https://news.google.com/rss/search?q={symbol}+stock+NSE&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q={symbol}+share+market&hl=en-IN&gl=IN&ceid=IN:en",
]

POSITIVE_KEYWORDS = [
    "rally", "surge", "upgrade", "bullish", "profit", "growth", "beat",
    "strong", "buy", "outperform", "gain", "positive", "rise", "high",
    "record", "up", "boost", "recovery", "optimistic", "soar",
]
NEGATIVE_KEYWORDS = [
    "crash", "fall", "downgrade", "bearish", "loss", "weak", "sell",
    "underperform", "decline", "risk", "drop", "negative", "low",
    "plunge", "down", "cut", "warning", "fear", "pessimistic", "slump",
]

# NIFTY 50 symbols (yfinance fallback for market movers)
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
    "ASIANPAINT", "SUNPHARMA", "HCLTECH", "WIPRO", "ULTRACEMCO",
    "NTPC", "POWERGRID", "ONGC", "TATAMOTORS", "TATASTEEL",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "BPCL", "IOC",
    "JSWSTEEL", "TECHM", "INDUSINDBK", "HDFCLIFE", "SBILIFE",
    "BAJAJFINSV", "GRASIM", "DIVISLAB", "DRREDDY", "CIPLA",
    "EICHERMOT", "HEROMOTOCO", "APOLLOHOSP", "TATACONSUM", "NESTLEIND",
    "BRITANNIA", "M&M", "HINDALCO", "UPL", "SHRIRAMFIN",
]

# Popular NSE stocks for quick access
POPULAR_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
    "ASIANPAINT", "SUNPHARMA", "HCLTECH", "WIPRO", "ULTRACEMCO",
]

# Indian Market Indices (symbol -> yfinance ticker)
INDICES = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT",
    "NIFTY NEXT 50": "^NSMIDCP",
    "NIFTY MIDCAP 100": "^NSEMDCP50",
    "NIFTY FINANCIAL": "NIFTY_FIN_SERVICE.NS",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY REALTY": "^CNXREALTY",
    "INDIA VIX": "^INDIAVIX",
}

# Reverse map: yfinance ticker -> display name
INDICES_REVERSE = {v: k for k, v in INDICES.items()}

# Authentication
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-" + "x" * 32)
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Prediction horizons config
PREDICTION_HORIZONS = {
    "15m": {"label": "15 Min", "intraday": True, "bars": 1, "interval": "15m"},
    "1h": {"label": "1 Hour", "intraday": True, "bars": 4, "interval": "15m"},
    "1d": {"label": "1 Day", "intraday": False, "days": 1},
    "1w": {"label": "1 Week", "intraday": False, "days": 5},
    "1mo": {"label": "1 Month", "intraday": False, "days": 22},
    "3mo": {"label": "3 Months", "intraday": False, "days": 66},
    "6mo": {"label": "6 Months", "intraday": False, "days": 126},
    "1y": {"label": "1 Year", "intraday": False, "days": 252},
}

# --- FinBERT Config ---
FINBERT_ENABLED = True  # Set to False to fall back to keyword matching

# --- Finnhub Config ---
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# --- Market Mood Config ---
MARKET_MOOD_REFRESH_INTERVAL = 300  # seconds (5 min)

# --- Sector Map ---
SECTOR_MAP = {
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "EICHERMOT", "HEROMOTOCO"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
    "Energy": ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC", "POWERGRID"],
    "Realty": ["ADANIENT", "ADANIPORTS", "GRASIM", "ULTRACEMCO"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "SHRIRAMFIN"],
}
