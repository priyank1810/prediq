"""Tests for signal endpoints: /api/signals."""

from unittest.mock import patch, MagicMock
from datetime import datetime

import pytest

from app.models import SignalLog


class TestGetSignal:
    @patch("app.utils.helpers.is_market_open", return_value=False)
    def test_get_signal_market_closed_with_history(self, mock_open, client, db):
        """When market is closed and a signal log exists, return last known signal."""
        log = SignalLog(
            symbol="RELIANCE",
            direction="BULLISH",
            confidence=72.5,
            composite_score=65.0,
            technical_score=70.0,
            sentiment_score=60.0,
            global_score=55.0,
            price_at_signal=2500.0,
            created_at=datetime(2025, 3, 17, 14, 30, 0),
        )
        db.add(log)
        db.commit()

        resp = client.get("/api/signals/RELIANCE")
        assert resp.status_code == 200
        data = resp.json()
        assert data["direction"] == "BULLISH"
        assert data["confidence"] == 72.5
        assert data["market_closed"] is True

    @patch("app.utils.helpers.is_market_open", return_value=False)
    def test_get_signal_market_closed_no_history(self, mock_open, client):
        """When market is closed and no log exists, return 404."""
        resp = client.get("/api/signals/RELIANCE")
        assert resp.status_code == 404


class TestSignalHistory:
    def test_signal_history_empty(self, client):
        resp = client.get("/api/signals/RELIANCE/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_signal_history_with_data(self, client, db):
        for i in range(3):
            log = SignalLog(
                symbol="RELIANCE",
                direction="BULLISH" if i % 2 == 0 else "BEARISH",
                confidence=60.0 + i * 5,
                composite_score=55.0 + i,
                technical_score=60.0 + i,
                sentiment_score=50.0 + i,
                global_score=45.0 + i,
                price_at_signal=2500.0 + i * 10,
            )
            db.add(log)
        db.commit()

        resp = client.get("/api/signals/RELIANCE/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_signal_history_limit(self, client, db):
        for i in range(10):
            log = SignalLog(
                symbol="TCS",
                direction="BULLISH",
                confidence=60.0,
                composite_score=55.0,
                technical_score=60.0,
                sentiment_score=50.0,
                global_score=45.0,
            )
            db.add(log)
        db.commit()

        resp = client.get("/api/signals/TCS/history?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) == 5


class TestSignalAccuracy:
    def test_signal_accuracy_empty(self, client):
        resp = client.get("/api/signals/stats/accuracy")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_signal_accuracy_with_data(self, client, db):
        for i in range(5):
            log = SignalLog(
                symbol="RELIANCE",
                direction="BULLISH",
                confidence=70.0,
                composite_score=65.0,
                technical_score=70.0,
                sentiment_score=60.0,
                global_score=55.0,
                was_correct=i < 3,  # 3 correct out of 5
            )
            db.add(log)
        db.commit()

        resp = client.get("/api/signals/stats/accuracy")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "RELIANCE"
        assert data[0]["total"] == 5
        assert data[0]["correct"] == 3
        assert data[0]["accuracy"] == 60.0


class TestMarketMood:
    def test_market_mood(self, client):
        mock_mood = MagicMock()
        mock_mood.get_mood.return_value = {
            "score": 65.0,
            "label": "Mildly Bullish",
            "components": {
                "nifty_trend": 70,
                "advance_decline": 60,
                "vix": 55,
            },
        }
        with patch("app.routers.signals.market_mood_service", mock_mood, create=True):
            resp = client.get("/api/signals/market-mood")
            assert resp.status_code == 200
            data = resp.json()
            assert "score" in data
