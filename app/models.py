from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")  # "user" or "admin"
    api_key = Column(String, unique=True, index=True, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    holdings = relationship("PortfolioHolding", back_populates="owner")
    alerts = relationship("PriceAlert", back_populates="owner")
    watchlist = relationship("WatchlistItem", back_populates="owner")


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    exchange = Column(String, default="NSE")
    quantity = Column(Integer, nullable=False)
    buy_price = Column(Float, nullable=False)
    buy_date = Column(Date, nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    owner = relationship("User", back_populates="holdings")


class PriceAlert(Base):
    __tablename__ = "price_alerts"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    target_price = Column(Float, nullable=False)
    condition = Column(String, nullable=False)  # "above" or "below"
    is_triggered = Column(Boolean, default=False)
    triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    owner = relationship("User", back_populates="alerts")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # "lstm" or "prophet"
    prediction_date = Column(Date, nullable=False)
    target_date = Column(Date, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    sector = Column(String, nullable=True)
    regime = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalLog(Base):
    __tablename__ = "signal_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    composite_score = Column(Float, nullable=False)
    technical_score = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    global_score = Column(Float, nullable=False)
    price_at_signal = Column(Float, nullable=True)
    price_after_15min = Column(Float, nullable=True)
    was_correct = Column(Boolean, nullable=True)
    sector = Column(String, nullable=True)
    regime = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SmartAlert(Base):
    __tablename__ = "smart_alerts"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=True)  # nullable for market-wide alerts
    alert_type = Column(String, nullable=False)  # prediction_change, sentiment_spike, mood_extreme, confidence_change
    threshold = Column(Float, nullable=True)
    is_triggered = Column(Boolean, default=False)
    triggered_at = Column(DateTime, nullable=True)
    trigger_data = Column(String, nullable=True)  # JSON string with details
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    __table_args__ = (
        UniqueConstraint("user_id", "symbol", name="uq_watchlist_user_symbol"),
    )

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    item_type = Column(String, default="stock")
    added_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    owner = relationship("User", back_populates="watchlist")
