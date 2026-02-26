from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional


# --- Stock Schemas ---

class StockQuote(BaseModel):
    symbol: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    change: float
    pct_change: float
    timestamp: str


class HistoricalDataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


# --- Portfolio Schemas ---

class PortfolioHoldingCreate(BaseModel):
    symbol: str
    exchange: str = "NSE"
    quantity: int
    buy_price: float
    buy_date: date
    notes: Optional[str] = None


class PortfolioHoldingResponse(BaseModel):
    id: int
    symbol: str
    exchange: str
    quantity: int
    buy_price: float
    buy_date: date
    notes: Optional[str]
    current_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

    class Config:
        from_attributes = True


class PortfolioSummary(BaseModel):
    total_invested: float
    current_value: float
    total_pnl: float
    total_pnl_pct: float
    holdings_count: int


# --- Alert Schemas ---

class PriceAlertCreate(BaseModel):
    symbol: str
    target_price: float
    condition: str  # "above" or "below"


class PriceAlertResponse(BaseModel):
    id: int
    symbol: str
    target_price: float
    condition: str
    is_triggered: bool
    triggered_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# --- Prediction Schemas ---

class PredictionRequest(BaseModel):
    horizon: str = "1d"  # "15m", "1h", "1d", "1mo", "1y"
    models: list[str] = ["lstm", "prophet", "xgboost"]


class ModelPrediction(BaseModel):
    predictions: list[float]
    dates: list[str]
    confidence_score: Optional[float] = None
    confidence_lower: Optional[list[float]] = None
    confidence_upper: Optional[list[float]] = None
    mape: Optional[float] = None


class PredictionResponse(BaseModel):
    symbol: str
    horizon: str
    lstm: Optional[ModelPrediction] = None
    prophet: Optional[ModelPrediction] = None
    xgboost: Optional[ModelPrediction] = None
    ensemble: Optional[ModelPrediction] = None


# --- Indicator Schemas ---

class IndicatorData(BaseModel):
    dates: list[str]
    values: list[Optional[float]]


class IndicatorResponse(BaseModel):
    symbol: str
    rsi: Optional[IndicatorData] = None
    macd_line: Optional[IndicatorData] = None
    macd_signal: Optional[IndicatorData] = None
    macd_histogram: Optional[IndicatorData] = None
    bollinger_upper: Optional[IndicatorData] = None
    bollinger_middle: Optional[IndicatorData] = None
    bollinger_lower: Optional[IndicatorData] = None
    sma_20: Optional[IndicatorData] = None
    sma_50: Optional[IndicatorData] = None
    ema_20: Optional[IndicatorData] = None


# --- Backtest Schemas ---

class BacktestResult(BaseModel):
    symbol: str
    model_type: str
    total_predictions: int
    mae: float
    mape: float
    directional_accuracy: float
    hit_rate_in_band: Optional[float] = None


# --- Watchlist Schemas ---

class WatchlistItemCreate(BaseModel):
    symbol: str
    item_type: str = "stock"


class WatchlistItemResponse(BaseModel):
    id: int
    symbol: str
    item_type: str
    added_at: datetime

    class Config:
        from_attributes = True


# --- Auth Schemas ---

class UserCreate(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    api_key: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    refresh_token: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class UserUpdate(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[str] = None
