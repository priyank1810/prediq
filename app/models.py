from datetime import datetime, date
from sqlalchemy import Column, Index, Integer, String, Float, Boolean, DateTime, Date, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base
from app.utils.helpers import now_ist


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")  # "user" or "admin"
    api_key = Column(String, unique=True, index=True, nullable=True)
    telegram_chat_id = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=now_ist)

    holdings = relationship("PortfolioHolding", back_populates="owner")
    alerts = relationship("PriceAlert", back_populates="owner")
    watchlist = relationship("WatchlistItem", back_populates="owner")
    orders = relationship("Order", back_populates="owner")
    telegram_subscription = relationship("TelegramSubscription", back_populates="owner", uselist=False)


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    exchange = Column(String, default="NSE")
    quantity = Column(Integer, nullable=False)
    buy_price = Column(Float, nullable=False)
    buy_date = Column(Date, nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=now_ist)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    owner = relationship("User", back_populates="holdings")


class PriceAlert(Base):
    __tablename__ = "price_alerts"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    target_price = Column(Float, nullable=False)
    condition = Column(String, nullable=False)  # "above" or "below"
    is_triggered = Column(Boolean, default=False, index=True)
    triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=now_ist)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    owner = relationship("User", back_populates="alerts")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    model_type = Column(String, nullable=False)  # "prophet", "xgboost", "ensemble"
    prediction_date = Column(Date, nullable=False)
    target_date = Column(Date, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    sector = Column(String, nullable=True)
    regime = Column(String, nullable=True)
    created_at = Column(DateTime, default=now_ist)


class SignalLog(Base):
    __tablename__ = "signal_logs"
    __table_args__ = (
        Index("ix_signal_logs_symbol_created", "symbol", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    composite_score = Column(Float, nullable=False)
    technical_score = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    global_score = Column(Float, nullable=False)
    oi_score = Column(Float, nullable=True)
    price_at_signal = Column(Float, nullable=True)
    price_after_15min = Column(Float, nullable=True)
    price_after_30min = Column(Float, nullable=True)
    price_after_1hr = Column(Float, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    was_correct_30min = Column(Boolean, nullable=True)
    was_correct_1hr = Column(Boolean, nullable=True)
    sector = Column(String, nullable=True)
    regime = Column(String, nullable=True)
    created_at = Column(DateTime, default=now_ist, index=True)


class SmartAlert(Base):
    __tablename__ = "smart_alerts"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=True)  # nullable for market-wide alerts
    alert_type = Column(String, nullable=False)  # prediction_change, sentiment_spike, mood_extreme, confidence_change
    threshold = Column(Float, nullable=True)
    is_triggered = Column(Boolean, default=False)
    triggered_at = Column(DateTime, nullable=True)
    trigger_data = Column(String, nullable=True)  # JSON string with details
    created_at = Column(DateTime, default=now_ist)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    __table_args__ = (
        UniqueConstraint("user_id", "symbol", name="uq_watchlist_user_symbol"),
    )

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    item_type = Column(String, default="stock")
    added_at = Column(DateTime, default=now_ist)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    owner = relationship("User", back_populates="watchlist")


class TradeJournal(Base):
    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)  # "buy" or "sell"
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    notes = Column(Text, nullable=True)
    signal_direction = Column(String, nullable=True)  # signal direction at time of trade
    signal_confidence = Column(Float, nullable=True)   # signal confidence at time of trade
    pnl = Column(Float, nullable=True)  # profit/loss (for sell trades)
    pnl_pct = Column(Float, nullable=True)
    tags = Column(String, nullable=True)  # comma-separated tags
    created_at = Column(DateTime, default=now_ist)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_user_status", "user_id", "status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    exchange = Column(String, default="NSE")
    order_type = Column(String, nullable=False)  # "MARKET", "LIMIT", "SL", "SL-M"
    transaction_type = Column(String, nullable=False)  # "BUY", "SELL"
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=True)  # nullable for MARKET orders
    trigger_price = Column(Float, nullable=True)  # for SL / SL-M
    status = Column(String, default="pending")  # pending, placed, executed, cancelled, rejected
    broker = Column(String, default="angel_one")
    broker_order_id = Column(String, nullable=True)
    paper_trade = Column(Boolean, default=False)
    placed_at = Column(DateTime, nullable=True)
    executed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now_ist)

    owner = relationship("User", back_populates="orders")


class SharedStrategy(Base):
    __tablename__ = "shared_strategies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    symbols = Column(String, nullable=False)  # comma-separated
    timeframe = Column(String, nullable=False)
    entry_rules = Column(Text, nullable=False)  # JSON string
    exit_rules = Column(Text, nullable=False)   # JSON string
    is_public = Column(Boolean, default=True)
    upvotes = Column(Integer, default=0)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    avg_return_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    created_at = Column(DateTime, default=now_ist)
    updated_at = Column(DateTime, default=now_ist, onupdate=now_ist)

    followers = relationship("StrategyFollow", back_populates="strategy")


class StrategyFollow(Base):
    __tablename__ = "strategy_follows"
    __table_args__ = (
        UniqueConstraint("user_id", "strategy_id", name="uq_strategy_follow_user_strategy"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    strategy_id = Column(Integer, ForeignKey("shared_strategies.id"), nullable=False, index=True)
    followed_at = Column(DateTime, default=now_ist)

    strategy = relationship("SharedStrategy", back_populates="followers")


class TelegramSubscription(Base):
    __tablename__ = "telegram_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    chat_id = Column(String, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    alert_types = Column(String, default="signals,price_alerts,news,scanner,predictions")
    created_at = Column(DateTime, default=now_ist)

    owner = relationship("User", back_populates="telegram_subscription")


class JobQueue(Base):
    __tablename__ = "job_queue"
    __table_args__ = (
        Index("ix_job_queue_poll", "status", "priority", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String, nullable=False, index=True)
    priority = Column(Integer, default=0)  # 10=user-facing, 0=background
    status = Column(String, default="pending", index=True)  # pending, running, completed, failed, broadcast
    params = Column(Text, nullable=True)  # JSON
    result = Column(Text, nullable=True)  # JSON
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now_ist)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
