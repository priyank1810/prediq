"""Unit tests for app.services.alert_service.AlertService."""

from unittest.mock import patch

import pytest

from app.models import PriceAlert
from app.services.alert_service import AlertService


@pytest.fixture()
def svc():
    return AlertService()


# ---------------------------------------------------------------------------
# create_alert
# ---------------------------------------------------------------------------

class TestCreateAlert:
    def test_creates_alert(self, db, svc, test_user):
        alert = svc.create_alert(
            db,
            {"symbol": "RELIANCE", "target_price": 3000.0, "condition": "above"},
            user_id=test_user["id"],
        )
        assert alert.id is not None
        assert alert.symbol == "RELIANCE"
        assert alert.target_price == 3000.0
        assert alert.condition == "above"
        assert alert.is_triggered is False
        assert alert.user_id == test_user["id"]

    def test_creates_below_alert(self, db, svc, test_user):
        alert = svc.create_alert(
            db,
            {"symbol": "TCS", "target_price": 3000.0, "condition": "below"},
            user_id=test_user["id"],
        )
        assert alert.condition == "below"

    def test_create_alert_without_user(self, db, svc):
        alert = svc.create_alert(
            db,
            {"symbol": "INFY", "target_price": 1800.0, "condition": "above"},
        )
        assert alert.user_id is None


# ---------------------------------------------------------------------------
# get_alerts
# ---------------------------------------------------------------------------

class TestGetAlerts:
    def _seed_alerts(self, db, svc, user_id, symbols):
        for sym in symbols:
            svc.create_alert(
                db,
                {"symbol": sym, "target_price": 1000.0, "condition": "above"},
                user_id=user_id,
            )

    def test_returns_alerts_for_user(self, db, svc, test_user):
        self._seed_alerts(db, svc, test_user["id"], ["RELIANCE", "TCS"])
        alerts = svc.get_alerts(db, user_id=test_user["id"])
        assert len(alerts) == 2
        symbols = {a.symbol for a in alerts}
        assert symbols == {"RELIANCE", "TCS"}

    def test_empty_when_no_alerts(self, db, svc, test_user):
        alerts = svc.get_alerts(db, user_id=test_user["id"])
        assert alerts == []

    def test_user_isolation(self, db, svc, test_user, second_user):
        """User A cannot see User B's alerts."""
        self._seed_alerts(db, svc, test_user["id"], ["RELIANCE"])
        self._seed_alerts(db, svc, second_user["id"], ["TCS"])

        a_alerts = svc.get_alerts(db, user_id=test_user["id"])
        b_alerts = svc.get_alerts(db, user_id=second_user["id"])
        assert len(a_alerts) == 1
        assert a_alerts[0].symbol == "RELIANCE"
        assert len(b_alerts) == 1
        assert b_alerts[0].symbol == "TCS"

    def test_respects_limit_and_offset(self, db, svc, test_user):
        self._seed_alerts(db, svc, test_user["id"], ["A", "B", "C", "D", "E"])
        alerts = svc.get_alerts(db, user_id=test_user["id"], limit=2, offset=0)
        assert len(alerts) == 2
        alerts_offset = svc.get_alerts(db, user_id=test_user["id"], limit=2, offset=2)
        assert len(alerts_offset) == 2


# ---------------------------------------------------------------------------
# delete_alert
# ---------------------------------------------------------------------------

class TestDeleteAlert:
    def test_delete_own_alert(self, db, svc, test_user):
        alert = svc.create_alert(
            db,
            {"symbol": "RELIANCE", "target_price": 3000.0, "condition": "above"},
            user_id=test_user["id"],
        )
        assert svc.delete_alert(db, alert.id, user_id=test_user["id"]) is True
        assert db.query(PriceAlert).filter_by(id=alert.id).first() is None

    def test_cannot_delete_other_users_alert(self, db, svc, test_user, second_user):
        alert = svc.create_alert(
            db,
            {"symbol": "RELIANCE", "target_price": 3000.0, "condition": "above"},
            user_id=test_user["id"],
        )
        assert svc.delete_alert(db, alert.id, user_id=second_user["id"]) is False
        assert db.query(PriceAlert).filter_by(id=alert.id).first() is not None

    def test_delete_nonexistent(self, db, svc, test_user):
        assert svc.delete_alert(db, 9999, user_id=test_user["id"]) is False


# ---------------------------------------------------------------------------
# check_alerts (triggering logic)
# ---------------------------------------------------------------------------

class TestCheckAlerts:
    @patch("app.services.alert_service.data_fetcher")
    def test_triggers_above_alert(self, mock_fetcher, db, svc, test_user):
        mock_fetcher.get_live_quote.return_value = {"ltp": 3100.0}
        svc.create_alert(
            db,
            {"symbol": "RELIANCE", "target_price": 3000.0, "condition": "above"},
            user_id=test_user["id"],
        )
        triggered = svc.check_alerts(db)
        assert len(triggered) == 1
        assert triggered[0]["symbol"] == "RELIANCE"
        assert triggered[0]["current_price"] == 3100.0

        # Verify the alert is marked as triggered in DB
        alert = db.query(PriceAlert).first()
        assert alert.is_triggered is True
        assert alert.triggered_at is not None

    @patch("app.services.alert_service.data_fetcher")
    def test_triggers_below_alert(self, mock_fetcher, db, svc, test_user):
        mock_fetcher.get_live_quote.return_value = {"ltp": 2800.0}
        svc.create_alert(
            db,
            {"symbol": "RELIANCE", "target_price": 3000.0, "condition": "below"},
            user_id=test_user["id"],
        )
        triggered = svc.check_alerts(db)
        assert len(triggered) == 1
        assert triggered[0]["condition"] == "below"

    @patch("app.services.alert_service.data_fetcher")
    def test_does_not_trigger_when_condition_not_met(self, mock_fetcher, db, svc, test_user):
        mock_fetcher.get_live_quote.return_value = {"ltp": 2900.0}
        svc.create_alert(
            db,
            {"symbol": "RELIANCE", "target_price": 3000.0, "condition": "above"},
            user_id=test_user["id"],
        )
        triggered = svc.check_alerts(db)
        assert len(triggered) == 0
        alert = db.query(PriceAlert).first()
        assert alert.is_triggered is False

    @patch("app.services.alert_service.data_fetcher")
    def test_already_triggered_alerts_are_skipped(self, mock_fetcher, db, svc, test_user):
        mock_fetcher.get_live_quote.return_value = {"ltp": 3100.0}
        svc.create_alert(
            db,
            {"symbol": "RELIANCE", "target_price": 3000.0, "condition": "above"},
            user_id=test_user["id"],
        )
        # Trigger once
        svc.check_alerts(db)
        # Check again -- should not re-trigger
        mock_fetcher.get_live_quote.reset_mock()
        triggered = svc.check_alerts(db)
        assert len(triggered) == 0
        mock_fetcher.get_live_quote.assert_not_called()


# ---------------------------------------------------------------------------
# check_alerts_batched
# ---------------------------------------------------------------------------

class TestCheckAlertsBatched:
    @patch("app.services.alert_service.data_fetcher")
    def test_batch_trigger(self, mock_fetcher, db, svc, test_user):
        mock_fetcher.get_bulk_quotes.return_value = [
            {"symbol": "RELIANCE", "ltp": 3100.0},
            {"symbol": "TCS", "ltp": 2900.0},
        ]
        svc.create_alert(
            db,
            {"symbol": "RELIANCE", "target_price": 3000.0, "condition": "above"},
            user_id=test_user["id"],
        )
        svc.create_alert(
            db,
            {"symbol": "TCS", "target_price": 3000.0, "condition": "above"},
            user_id=test_user["id"],
        )
        triggered = svc.check_alerts_batched(db)
        # Only RELIANCE should trigger (3100 >= 3000); TCS at 2900 < 3000
        assert len(triggered) == 1
        assert triggered[0]["symbol"] == "RELIANCE"

    @patch("app.services.alert_service.data_fetcher")
    def test_batch_empty_when_no_active_alerts(self, mock_fetcher, db, svc):
        triggered = svc.check_alerts_batched(db)
        assert triggered == []
        mock_fetcher.get_bulk_quotes.assert_not_called()
