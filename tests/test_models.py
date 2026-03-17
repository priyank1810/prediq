"""Unit tests for SQLAlchemy models."""

from datetime import date, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from app.models import (
    User,
    PortfolioHolding,
    TradeJournal,
    Order,
    WatchlistItem,
    PriceAlert,
    SignalLog,
    PredictionLog,
    SmartAlert,
)


class TestUserCreation:
    def test_user_creation(self, db):
        user = User(
            email="model_test@example.com",
            hashed_password="fakehash",
            role="user",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        assert user.id is not None
        assert user.email == "model_test@example.com"
        assert user.role == "user"
        assert user.is_active is True
        assert user.created_at is not None

    def test_user_unique_email(self, db):
        user1 = User(email="unique@example.com", hashed_password="hash1")
        db.add(user1)
        db.commit()

        user2 = User(email="unique@example.com", hashed_password="hash2")
        db.add(user2)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_user_default_role(self, db):
        user = User(email="default@example.com", hashed_password="hash")
        db.add(user)
        db.commit()
        db.refresh(user)
        assert user.role == "user"

    def test_user_nullable_password(self, db):
        """Google auth users have no password."""
        user = User(
            email="google@example.com",
            hashed_password=None,
            google_id="google123",
            auth_provider="google",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        assert user.hashed_password is None
        assert user.auth_provider == "google"


class TestPortfolioHoldingCreation:
    def test_portfolio_holding_creation(self, db):
        user = User(email="holder@example.com", hashed_password="hash")
        db.add(user)
        db.commit()

        holding = PortfolioHolding(
            symbol="RELIANCE",
            exchange="NSE",
            quantity=10,
            buy_price=2500.0,
            buy_date=date(2024, 1, 15),
            user_id=user.id,
        )
        db.add(holding)
        db.commit()
        db.refresh(holding)

        assert holding.id is not None
        assert holding.symbol == "RELIANCE"
        assert holding.quantity == 10
        assert holding.buy_price == 2500.0
        assert holding.user_id == user.id

    def test_holding_relationship(self, db):
        user = User(email="rel@example.com", hashed_password="hash")
        db.add(user)
        db.commit()

        holding = PortfolioHolding(
            symbol="TCS", exchange="NSE", quantity=5,
            buy_price=3500.0, buy_date=date(2024, 3, 1),
            user_id=user.id,
        )
        db.add(holding)
        db.commit()

        # Refresh and access relationship
        db.refresh(user)
        assert len(user.holdings) == 1
        assert user.holdings[0].symbol == "TCS"

    def test_holding_without_user(self, db):
        """Holdings can exist without a user (legacy data)."""
        holding = PortfolioHolding(
            symbol="INFY", exchange="NSE", quantity=20,
            buy_price=1500.0, buy_date=date(2024, 6, 1),
            user_id=None,
        )
        db.add(holding)
        db.commit()
        db.refresh(holding)
        assert holding.user_id is None


class TestTradeJournalCreation:
    def test_trade_journal_creation(self, db):
        trade = TradeJournal(
            symbol="RELIANCE",
            action="buy",
            price=2500.0,
            quantity=10,
            notes="Bought on support level",
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)

        assert trade.id is not None
        assert trade.symbol == "RELIANCE"
        assert trade.action == "buy"
        assert trade.price == 2500.0
        assert trade.quantity == 10
        assert trade.notes == "Bought on support level"

    def test_trade_with_pnl(self, db):
        trade = TradeJournal(
            symbol="TCS",
            action="sell",
            price=3700.0,
            quantity=5,
            pnl=1000.0,
            pnl_pct=5.7,
            tags="momentum,breakout",
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)

        assert trade.pnl == 1000.0
        assert trade.pnl_pct == 5.7
        assert trade.tags == "momentum,breakout"

    def test_trade_with_signal_info(self, db):
        trade = TradeJournal(
            symbol="INFY",
            action="buy",
            price=1500.0,
            quantity=20,
            signal_direction="BULLISH",
            signal_confidence=75.0,
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)

        assert trade.signal_direction == "BULLISH"
        assert trade.signal_confidence == 75.0


class TestOrderCreation:
    def test_order_creation(self, db):
        user = User(email="trader@example.com", hashed_password="hash")
        db.add(user)
        db.commit()

        order = Order(
            user_id=user.id,
            symbol="RELIANCE",
            exchange="NSE",
            order_type="LIMIT",
            transaction_type="BUY",
            quantity=10,
            price=2500.0,
            status="pending",
        )
        db.add(order)
        db.commit()
        db.refresh(order)

        assert order.id is not None
        assert order.symbol == "RELIANCE"
        assert order.order_type == "LIMIT"
        assert order.transaction_type == "BUY"
        assert order.status == "pending"

    def test_order_relationship(self, db):
        user = User(email="orderer@example.com", hashed_password="hash")
        db.add(user)
        db.commit()

        order = Order(
            user_id=user.id, symbol="TCS", order_type="MARKET",
            transaction_type="BUY", quantity=5,
        )
        db.add(order)
        db.commit()

        db.refresh(user)
        assert len(user.orders) == 1
        assert user.orders[0].symbol == "TCS"


class TestWatchlistUniqueConstraint:
    def test_watchlist_unique_constraint(self, db):
        user = User(email="watcher@example.com", hashed_password="hash")
        db.add(user)
        db.commit()

        item1 = WatchlistItem(symbol="RELIANCE", user_id=user.id)
        db.add(item1)
        db.commit()

        item2 = WatchlistItem(symbol="RELIANCE", user_id=user.id)
        db.add(item2)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

    def test_same_symbol_different_users(self, db):
        """Different users can watch the same symbol."""
        user1 = User(email="u1@example.com", hashed_password="hash")
        user2 = User(email="u2@example.com", hashed_password="hash")
        db.add_all([user1, user2])
        db.commit()

        item1 = WatchlistItem(symbol="RELIANCE", user_id=user1.id)
        item2 = WatchlistItem(symbol="RELIANCE", user_id=user2.id)
        db.add_all([item1, item2])
        db.commit()  # Should not raise

        assert item1.id is not None
        assert item2.id is not None
        assert item1.id != item2.id


class TestPriceAlert:
    def test_alert_creation(self, db):
        alert = PriceAlert(
            symbol="RELIANCE",
            target_price=3000.0,
            condition="above",
            is_triggered=False,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        assert alert.id is not None
        assert alert.is_triggered is False

    def test_alert_triggered(self, db):
        alert = PriceAlert(
            symbol="TCS",
            target_price=3500.0,
            condition="below",
            is_triggered=True,
            triggered_at=datetime.now(),
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        assert alert.is_triggered is True
        assert alert.triggered_at is not None


class TestSignalLog:
    def test_signal_log_creation(self, db):
        log = SignalLog(
            symbol="RELIANCE",
            direction="BULLISH",
            confidence=72.5,
            composite_score=65.0,
            technical_score=70.0,
            sentiment_score=60.0,
            global_score=55.0,
            price_at_signal=2500.0,
        )
        db.add(log)
        db.commit()
        db.refresh(log)

        assert log.id is not None
        assert log.direction == "BULLISH"
        assert log.confidence == 72.5


class TestPredictionLog:
    def test_prediction_log_creation(self, db):
        log = PredictionLog(
            symbol="RELIANCE",
            model_type="ensemble",
            prediction_date=date(2025, 3, 17),
            target_date=date(2025, 3, 18),
            predicted_price=2550.0,
            confidence_lower=2480.0,
            confidence_upper=2620.0,
        )
        db.add(log)
        db.commit()
        db.refresh(log)

        assert log.id is not None
        assert log.actual_price is None  # Not yet filled
