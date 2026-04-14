import re
from pydantic import BaseModel, ConfigDict, field_validator
from datetime import date, datetime
from typing import Optional

_SYMBOL_RE = re.compile(r'^[A-Za-z0-9 &^.\-]{1,30}$')


def _check_symbol(v: str) -> str:
    v = v.strip()
    if not v or not _SYMBOL_RE.match(v):
        raise ValueError(f"Invalid symbol: {v!r}")
    return v.upper()


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

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v):
        return _check_symbol(v)


class PortfolioHoldingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class PortfolioSummary(BaseModel):
    total_invested: float
    current_value: float
    total_pnl: float
    total_pnl_pct: float
    holdings_count: int


# --- Trade Journal Schemas ---

class TradeJournalCreate(BaseModel):
    symbol: str
    action: str  # "buy" or "sell"
    price: float
    quantity: int
    notes: Optional[str] = None
    signal_direction: Optional[str] = None
    signal_confidence: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    tags: Optional[str] = None


# --- Alert Schemas ---

class PriceAlertCreate(BaseModel):
    symbol: str
    target_price: float
    condition: str  # "above" or "below"

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v):
        return _check_symbol(v)

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v):
        if v not in ("above", "below"):
            raise ValueError("Condition must be 'above' or 'below'")
        return v


class PriceAlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    target_price: float
    condition: str
    is_triggered: bool
    triggered_at: Optional[datetime]
    created_at: datetime


# --- Prediction Schemas ---

class PredictionRequest(BaseModel):
    horizon: str = "1d"  # "15m", "1h", "1d", "1w", "1mo", "3mo", "6mo", "1y"
    models: Optional[list[str]] = None  # None = auto-select best models for horizon


class SHAPDriver(BaseModel):
    feature: str
    impact_value: float
    direction: str  # "positive" or "negative"


class ModelPrediction(BaseModel):
    predictions: list[float]
    dates: list[str]
    confidence_score: Optional[float] = None
    confidence_lower: Optional[list[float]] = None
    confidence_upper: Optional[list[float]] = None
    mape: Optional[float] = None


class ContributionBreakdown(BaseModel):
    technical: float = 0.0
    seasonal: float = 0.0
    fundamental: float = 0.0
    sentiment: float = 0.0


class PredictionResponse(BaseModel):
    symbol: str
    horizon: str
    prophet: Optional[ModelPrediction] = None
    xgboost: Optional[ModelPrediction] = None
    ensemble: Optional[ModelPrediction] = None
    shap_drivers: Optional[list[SHAPDriver]] = None
    contribution_breakdown: Optional[ContributionBreakdown] = None


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

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v):
        return _check_symbol(v)


class WatchlistItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    item_type: str
    added_at: datetime


# --- Auth Schemas ---

class UserCreate(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    role: str
    is_active: bool
    api_key: Optional[str] = None
    created_at: datetime


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


# --- Smart Alert Schemas ---

class SmartAlertCreate(BaseModel):
    symbol: Optional[str] = None  # nullable for market-wide alerts
    alert_type: str  # prediction_change, sentiment_spike, mood_extreme, confidence_change
    threshold: Optional[float] = None


class SmartAlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: Optional[str]
    alert_type: str
    threshold: Optional[float]
    is_triggered: bool
    triggered_at: Optional[datetime]
    trigger_data: Optional[str]
    created_at: datetime


# --- Chart Pattern Schemas ---

class ChartPattern(BaseModel):
    type: str
    start_date: str
    end_date: str
    confidence: float
    description: str


# --- Broker Order Schemas ---

class OrderCreate(BaseModel):
    symbol: str
    exchange: str = "NSE"
    order_type: str  # "MARKET", "LIMIT", "SL", "SL-M"
    transaction_type: str  # "BUY", "SELL"
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    paper_trade: bool = False
    notes: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v):
        return _check_symbol(v)

    @field_validator("order_type")
    @classmethod
    def validate_order_type(cls, v):
        v = v.upper()
        if v not in ("MARKET", "LIMIT", "SL", "SL-M"):
            raise ValueError("order_type must be MARKET, LIMIT, SL, or SL-M")
        return v

    @field_validator("transaction_type")
    @classmethod
    def validate_transaction_type(cls, v):
        v = v.upper()
        if v not in ("BUY", "SELL"):
            raise ValueError("transaction_type must be BUY or SELL")
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v


# --- Strategy Sharing Schemas ---

class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    symbols: str  # comma-separated
    timeframe: str  # "1d", "1h", "15m", "1w"
    entry_rules: str  # JSON string
    exit_rules: str   # JSON string
    is_public: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        v = v.strip()
        if not v or len(v) > 100:
            raise ValueError("Name must be 1-100 characters")
        return v

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v):
        if v not in ("15m", "1h", "1d", "1w", "1mo"):
            raise ValueError("Timeframe must be 15m, 1h, 1d, 1w, or 1mo")
        return v


class OrderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    symbol: str
    exchange: str
    order_type: str
    transaction_type: str
    quantity: int
    price: Optional[float]
    trigger_price: Optional[float]
    status: str
    broker: str
    broker_order_id: Optional[str]
    paper_trade: bool
    placed_at: Optional[datetime]
    executed_at: Optional[datetime]
    notes: Optional[str]
    created_at: datetime
