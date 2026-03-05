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
CACHE_TTL_QUOTE = 30
CACHE_TTL_HISTORY = 900
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
OI_STREAM_INTERVAL = 300  # 5 minutes
MTF_STREAM_INTERVAL = 300  # 5 minutes
SIGNAL_STREAM_INTERVAL = 300  # 5 minutes

# --- Intraday Signal Config ---
CACHE_TTL_INTRADAY = 120
CACHE_TTL_SENTIMENT = 900
CACHE_TTL_GLOBAL = 300
CACHE_TTL_MARKET_MOVERS = 300
SIGNAL_REFRESH_INTERVAL = 60

SIGNAL_DIRECTION_THRESHOLD = 15

SIGNAL_WEIGHT_TECHNICAL = 0.65
SIGNAL_WEIGHT_SENTIMENT = 0.25
SIGNAL_WEIGHT_GLOBAL = 0.10
SIGNAL_WEIGHT_OI = 0.10

# Adaptive Weights
ADAPTIVE_WEIGHTS_MIN_SIGNALS = 30
ADAPTIVE_WEIGHTS_CACHE_TTL = 600  # 10 minutes
ADAPTIVE_WEIGHTS_DECAY_HALFLIFE_DAYS = 7

# OI Analysis
CACHE_TTL_OI = 300  # 5 minutes

# MTF Confluence Cache
CACHE_TTL_MTF_DAILY = 900  # 15 minutes
CACHE_TTL_MTF_1H = 120  # 2 minutes

# Sector-Relative Strength
CACHE_TTL_SECTOR_STRENGTH = 300  # 5 minutes

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
    "NIFTY FINANCIAL": "^CNXFIN",
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

# Maps event category names to keywords from BIG_EVENT_SCORE
EVENT_CATEGORIES = {
    "war_conflict": [
        "war", "invasion", "airstrike", "missile", "military", "attack",
        "conflict", "escalation", "nuclear", "terrorism", "tensions",
        "sanctions", "embargo", "india pakistan", "india china", "israel",
        "iran", "russia ukraine", "nato", "border", "geopolitical", "defence",
    ],
    "rate_hike": ["rate hike", "fed", "federal reserve", "rbi", "monetary policy", "inflation", "cpi"],
    "rate_cut": ["rate cut", "easing", "stimulus"],
    "oil_shock": ["oil shock", "crude surge", "crude crash", "opec"],
    "trade_war": ["tariff", "trade war"],
    "recession": [
        "recession", "crash", "crisis", "collapse", "default",
        "stagflation", "debt ceiling", "circuit breaker", "capitulation",
    ],
    "pandemic": ["pandemic", "lockdown"],
    "recovery": ["ceasefire", "peace", "deal", "recovery"],
}

# Build reverse lookup: keyword -> category
_KEYWORD_TO_CATEGORY = {}
for _cat, _kws in EVENT_CATEGORIES.items():
    for _kw in _kws:
        _KEYWORD_TO_CATEGORY[_kw] = _cat

# Sector-event modifier: multiplier per (event_category, sector).
# Negative = bad news becomes good for that sector (e.g., war is bullish for metals).
# Values close to 0 = sector barely affected. Values > 1 = sector extra sensitive.
SECTOR_EVENT_MODIFIERS = {
    "war_conflict": {"Metal": -0.6, "Energy": -0.5, "IT": 0.7, "Banking": 0.8, "Pharma": 0.5, "Auto": 0.9, "FMCG": 0.5, "Realty": 1.0, "Finance": 0.8},
    "rate_hike":    {"Metal": 0.8,  "Energy": 0.7,  "IT": 1.2, "Banking": -0.8, "Pharma": 0.6, "Auto": 1.3, "FMCG": 0.5, "Realty": 1.5, "Finance": -0.6},
    "rate_cut":     {"Metal": 1.0,  "Energy": 0.8,  "IT": 1.3, "Banking": 0.5,  "Pharma": 0.8, "Auto": 1.4, "FMCG": 0.8, "Realty": 1.5, "Finance": 0.5},
    "oil_shock":    {"Metal": 0.6,  "Energy": -0.8, "IT": 0.5, "Banking": 0.8,  "Pharma": 0.5, "Auto": 1.3, "FMCG": 0.8, "Realty": 0.7, "Finance": 0.7},
    "trade_war":    {"Metal": 1.0,  "Energy": 0.5,  "IT": 1.5, "Banking": 0.6,  "Pharma": 1.2, "Auto": 1.3, "FMCG": 0.4, "Realty": 0.3, "Finance": 0.5},
    "recession":    {"Metal": 1.2,  "Energy": 1.0,  "IT": 1.2, "Banking": 1.3,  "Pharma": 0.4, "Auto": 1.4, "FMCG": 0.3, "Realty": 1.5, "Finance": 1.3},
    "pandemic":     {"Metal": 0.8,  "Energy": 1.0,  "IT": -0.3, "Banking": 1.0, "Pharma": -1.0, "Auto": 1.3, "FMCG": 0.5, "Realty": 1.3, "Finance": 1.0},
    "recovery":     {"Metal": 1.4,  "Energy": 1.2,  "IT": 0.8, "Banking": 1.2,  "Pharma": 0.6, "Auto": 1.3, "FMCG": 0.8, "Realty": 1.5, "Finance": 1.3},
}
