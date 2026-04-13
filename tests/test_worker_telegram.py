"""Tests for _fire_telegram_signal and its call sites in the worker scan loop."""

import threading
import asyncio
from unittest.mock import patch, MagicMock, call
import pytest


def _make_sig(confidence=72, direction="BULLISH"):
    return {
        "direction": direction,
        "confidence": confidence,
        "entry": 100.0,
        "target": 105.0,
        "stop_loss": 97.0,
        "regime": "trending",
        "volume_conviction": "high",
        "confidence_trend": "rising",
        "model_confidence": 65.0,
        "risk_reward": 1.67,
    }


def _make_session(symbols=("RELIANCE",)):
    """Build a DB session mock with watchlist rows for the given symbols."""
    items = [MagicMock(symbol=s) for s in symbols]
    session = MagicMock()
    session.query.return_value.distinct.return_value.all.return_value = items
    return session


def test_fire_telegram_signal_spawns_daemon_thread():
    """_fire_telegram_signal starts a daemon thread and calls broadcast."""
    from worker import _fire_telegram_signal

    broadcast_calls = []

    async def fake_broadcast(alert_type, func, data):
        broadcast_calls.append((alert_type, data))

    with patch("app.services.telegram_service.broadcast_to_subscribers", fake_broadcast):
        _fire_telegram_signal({"symbol": "INFY", **_make_sig()})
        # Give the daemon thread a moment to start
        import time; time.sleep(0.1)

    # broadcast was called with "signals" alert type
    assert any(c[0] == "signals" for c in broadcast_calls)


def test_fire_telegram_signal_includes_symbol_and_timeframe():
    """Data passed to broadcast contains symbol and timeframe."""
    from worker import _fire_telegram_signal

    captured = []

    async def fake_broadcast(alert_type, func, data):
        captured.append(data)

    with patch("app.services.telegram_service.broadcast_to_subscribers", fake_broadcast):
        _fire_telegram_signal({"symbol": "HDFC", "timeframe": "intraday_15m", **_make_sig()})
        import time; time.sleep(0.1)

    assert captured
    assert captured[0]["symbol"] == "HDFC"
    assert captured[0]["timeframe"] == "intraday_15m"


def test_scan_fires_telegram_for_bullish_above_threshold():
    """handle_watchlist_trade_scan calls _fire_telegram_signal for BULLISH signals >= 50."""
    from worker import Worker
    import app.config as config_mod

    worker = Worker.__new__(Worker)
    worker._signal_service = None

    mock_signal = _make_sig(confidence=65, direction="BULLISH")
    # Need 2+ bullish TFs to satisfy bullish_count >= 2
    mtf_result = {
        "current_price": 100.0,
        "intraday": {"15m": mock_signal, "1h": mock_signal},
        "short_term": {},
    }

    fired = []
    session = _make_session(["RELIANCE"])

    mock_svc = MagicMock()
    mock_svc.get_multi_timeframe_signals.return_value = mtf_result
    worker._signal_service = mock_svc

    orig_popular = config_mod.POPULAR_STOCKS
    config_mod.POPULAR_STOCKS = []
    try:
        with (
            patch("worker.SessionLocal", return_value=session),
            patch("worker._fire_telegram_signal", side_effect=lambda d: fired.append(d)),
            patch("app.services.trade_tracker.trade_tracker") as mock_tracker,
            patch("app.services.data_fetcher.data_fetcher.get_bulk_quotes", return_value=[]),
            patch("time.sleep"),
        ):
            mock_tracker.log_signal = MagicMock()
            worker.handle_watchlist_trade_scan({"scan_type": "intraday"})
    finally:
        config_mod.POPULAR_STOCKS = orig_popular

    assert any(d.get("direction") == "BULLISH" for d in fired)


def test_scan_does_not_fire_telegram_for_bearish():
    """handle_watchlist_trade_scan does NOT call _fire_telegram_signal for BEARISH signals."""
    from worker import Worker
    import app.config as config_mod

    worker = Worker.__new__(Worker)
    worker._signal_service = None

    bearish_signal = _make_sig(confidence=80, direction="BEARISH")
    mtf_result = {
        "current_price": 100.0,
        "intraday": {"15m": bearish_signal, "1h": bearish_signal},
        "short_term": {},
    }

    fired = []
    session = _make_session(["RELIANCE"])

    mock_svc = MagicMock()
    mock_svc.get_multi_timeframe_signals.return_value = mtf_result
    worker._signal_service = mock_svc

    orig_popular = config_mod.POPULAR_STOCKS
    config_mod.POPULAR_STOCKS = []
    try:
        with (
            patch("worker.SessionLocal", return_value=session),
            patch("worker._fire_telegram_signal", side_effect=lambda d: fired.append(d)),
            patch("app.services.trade_tracker.trade_tracker"),
            patch("app.services.data_fetcher.data_fetcher.get_bulk_quotes", return_value=[]),
            patch("time.sleep"),
        ):
            try:
                worker.handle_watchlist_trade_scan({"scan_type": "intraday"})
            except Exception:
                pass
    finally:
        config_mod.POPULAR_STOCKS = orig_popular

    assert not any(d.get("direction") == "BEARISH" for d in fired)


def test_scan_does_not_fire_telegram_below_50():
    """handle_watchlist_trade_scan does NOT fire Telegram for confidence < 50."""
    from worker import Worker
    import app.config as config_mod

    worker = Worker.__new__(Worker)
    worker._signal_service = None

    # Low-confidence but 2+ bullish TFs so bullish_count >= 2 is satisfied
    low_conf_signal = _make_sig(confidence=45, direction="BULLISH")
    mtf_result = {
        "current_price": 100.0,
        "intraday": {"15m": low_conf_signal, "1h": low_conf_signal},
        "short_term": {},
    }

    fired = []
    session = _make_session(["RELIANCE"])

    mock_svc = MagicMock()
    mock_svc.get_multi_timeframe_signals.return_value = mtf_result
    worker._signal_service = mock_svc

    orig_popular = config_mod.POPULAR_STOCKS
    config_mod.POPULAR_STOCKS = []
    try:
        with (
            patch("worker.SessionLocal", return_value=session),
            patch("worker._fire_telegram_signal", side_effect=lambda d: fired.append(d)),
            patch("app.services.trade_tracker.trade_tracker"),
            patch("app.services.data_fetcher.data_fetcher.get_bulk_quotes", return_value=[]),
            patch("time.sleep"),
        ):
            try:
                worker.handle_watchlist_trade_scan({"scan_type": "intraday"})
            except Exception:
                pass
    finally:
        config_mod.POPULAR_STOCKS = orig_popular

    assert fired == []
