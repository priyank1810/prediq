"""Unit tests for app.services.portfolio_service.PortfolioService."""

from datetime import date
from unittest.mock import patch

import pytest

from app.models import PortfolioHolding
from app.services.portfolio_service import PortfolioService


@pytest.fixture()
def svc():
    return PortfolioService()


# ---------------------------------------------------------------------------
# add_holding
# ---------------------------------------------------------------------------

class TestAddHolding:
    def test_adds_holding_to_db(self, db, svc, test_user):
        holding = svc.add_holding(
            db,
            {"symbol": "RELIANCE", "exchange": "NSE", "quantity": 10,
             "buy_price": 2500.0, "buy_date": date(2024, 1, 15)},
            user_id=test_user["id"],
        )
        assert holding.id is not None
        assert holding.symbol == "RELIANCE"
        assert holding.quantity == 10
        assert holding.buy_price == 2500.0
        assert holding.user_id == test_user["id"]

    def test_add_holding_with_notes(self, db, svc, test_user):
        holding = svc.add_holding(
            db,
            {"symbol": "TCS", "exchange": "NSE", "quantity": 5,
             "buy_price": 3500.0, "buy_date": date(2024, 3, 1),
             "notes": "Long-term hold"},
            user_id=test_user["id"],
        )
        assert holding.notes == "Long-term hold"

    def test_add_holding_without_user(self, db, svc):
        holding = svc.add_holding(
            db,
            {"symbol": "INFY", "exchange": "NSE", "quantity": 20,
             "buy_price": 1500.0, "buy_date": date(2024, 6, 1)},
        )
        assert holding.user_id is None


# ---------------------------------------------------------------------------
# get_holdings
# ---------------------------------------------------------------------------

class TestGetHoldings:
    def _seed(self, db, svc, user_id, symbols):
        for sym in symbols:
            svc.add_holding(
                db,
                {"symbol": sym, "exchange": "NSE", "quantity": 1,
                 "buy_price": 100.0, "buy_date": date(2024, 1, 1)},
                user_id=user_id,
            )

    @patch("app.services.portfolio_service.data_fetcher")
    def test_returns_holdings_for_user(self, mock_fetcher, db, svc, test_user):
        mock_fetcher.get_bulk_quotes.return_value = [
            {"symbol": "RELIANCE", "ltp": 2600.0},
        ]
        self._seed(db, svc, test_user["id"], ["RELIANCE"])
        holdings = svc.get_holdings(db, user_id=test_user["id"])
        assert len(holdings) == 1
        assert holdings[0]["symbol"] == "RELIANCE"
        assert holdings[0]["current_price"] == 2600.0

    @patch("app.services.portfolio_service.data_fetcher")
    def test_empty_when_no_holdings(self, mock_fetcher, db, svc, test_user):
        holdings = svc.get_holdings(db, user_id=test_user["id"])
        assert holdings == []
        mock_fetcher.get_bulk_quotes.assert_not_called()

    @patch("app.services.portfolio_service.data_fetcher")
    def test_pnl_calculation(self, mock_fetcher, db, svc, test_user):
        mock_fetcher.get_bulk_quotes.return_value = [
            {"symbol": "TCS", "ltp": 4000.0},
        ]
        svc.add_holding(
            db,
            {"symbol": "TCS", "exchange": "NSE", "quantity": 10,
             "buy_price": 3500.0, "buy_date": date(2024, 1, 1)},
            user_id=test_user["id"],
        )
        holdings = svc.get_holdings(db, user_id=test_user["id"])
        h = holdings[0]
        assert h["pnl"] == 5000.0  # (4000-3500)*10
        assert h["pnl_pct"] == pytest.approx(14.29, abs=0.01)

    @patch("app.services.portfolio_service.data_fetcher")
    def test_user_isolation(self, mock_fetcher, db, svc, test_user, second_user):
        """User A cannot see User B's holdings."""
        mock_fetcher.get_bulk_quotes.return_value = [
            {"symbol": "RELIANCE", "ltp": 2600.0},
            {"symbol": "TCS", "ltp": 4000.0},
        ]
        self._seed(db, svc, test_user["id"], ["RELIANCE"])
        self._seed(db, svc, second_user["id"], ["TCS"])

        a_holdings = svc.get_holdings(db, user_id=test_user["id"])
        b_holdings = svc.get_holdings(db, user_id=second_user["id"])
        assert len(a_holdings) == 1
        assert a_holdings[0]["symbol"] == "RELIANCE"
        assert len(b_holdings) == 1
        assert b_holdings[0]["symbol"] == "TCS"


# ---------------------------------------------------------------------------
# delete_holding
# ---------------------------------------------------------------------------

class TestDeleteHolding:
    def test_delete_own_holding(self, db, svc, test_user):
        holding = svc.add_holding(
            db,
            {"symbol": "INFY", "exchange": "NSE", "quantity": 5,
             "buy_price": 1500.0, "buy_date": date(2024, 1, 1)},
            user_id=test_user["id"],
        )
        assert svc.delete_holding(db, holding.id, user_id=test_user["id"]) is True
        assert db.query(PortfolioHolding).filter_by(id=holding.id).first() is None

    def test_cannot_delete_other_users_holding(self, db, svc, test_user, second_user):
        holding = svc.add_holding(
            db,
            {"symbol": "INFY", "exchange": "NSE", "quantity": 5,
             "buy_price": 1500.0, "buy_date": date(2024, 1, 1)},
            user_id=test_user["id"],
        )
        assert svc.delete_holding(db, holding.id, user_id=second_user["id"]) is False
        assert db.query(PortfolioHolding).filter_by(id=holding.id).first() is not None

    def test_delete_nonexistent_holding(self, db, svc, test_user):
        assert svc.delete_holding(db, 9999, user_id=test_user["id"]) is False


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

class TestGetSummary:
    @patch("app.services.portfolio_service.data_fetcher")
    def test_summary_calculation(self, mock_fetcher, db, svc, test_user):
        mock_fetcher.get_bulk_quotes.return_value = [
            {"symbol": "RELIANCE", "ltp": 2600.0},
            {"symbol": "TCS", "ltp": 4000.0},
        ]
        svc.add_holding(
            db,
            {"symbol": "RELIANCE", "exchange": "NSE", "quantity": 10,
             "buy_price": 2500.0, "buy_date": date(2024, 1, 1)},
            user_id=test_user["id"],
        )
        svc.add_holding(
            db,
            {"symbol": "TCS", "exchange": "NSE", "quantity": 5,
             "buy_price": 3500.0, "buy_date": date(2024, 1, 1)},
            user_id=test_user["id"],
        )
        summary = svc.get_summary(db, user_id=test_user["id"])
        assert summary["holdings_count"] == 2
        assert summary["total_invested"] == 42500.0  # 10*2500 + 5*3500
        assert summary["current_value"] == 46000.0   # 10*2600 + 5*4000
        assert summary["total_pnl"] == 3500.0
        assert summary["total_pnl_pct"] == pytest.approx(8.24, abs=0.01)

    @patch("app.services.portfolio_service.data_fetcher")
    def test_empty_summary(self, mock_fetcher, db, svc, test_user):
        summary = svc.get_summary(db, user_id=test_user["id"])
        assert summary["holdings_count"] == 0
        assert summary["total_invested"] == 0
        assert summary["total_pnl"] == 0
