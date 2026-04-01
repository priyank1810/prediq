"""Tests for trade_tracker safeguards — confidence, dedup, market regime,
daily loss limit, SL cooldown, market close cutoff, entry sanity, index filter."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pytest

from app.models import TradeSignalLog
from app.services.trade_tracker import TradeTracker
from app.utils.helpers import now_ist

# Market hours time for tests (11 AM IST)
MARKET_TIME = datetime(2026, 4, 1, 11, 0, 0)


def _make_signal(overrides=None):
    """Create a valid bullish signal dict."""
    sig = {
        "direction": "BULLISH",
        "confidence": 55,
        "entry": 100.0,
        "target": 105.0,
        "stop_loss": 97.0,
        "predicted_price": 104.0,
        "model_used": "v1",
        "risk_reward": 1.5,
    }
    if overrides:
        sig.update(overrides)
    return sig


def _patch_for_logging(tracker, db):
    """Common patches needed for log_signal to reach the DB."""
    return [
        patch("app.services.trade_tracker.SessionLocal", return_value=db),
        patch("app.services.trade_tracker.now_ist", return_value=MARKET_TIME),
        patch.object(tracker, "_check_market_regime", return_value=False),
        patch.object(tracker, "_is_near_earnings", return_value=False),
    ]


@pytest.fixture
def tracker():
    return TradeTracker()


# ── Confidence threshold ──

class TestConfidenceFilter:
    def test_rejects_below_45(self, tracker, db):
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("RELIANCE", "short_1h", _make_signal({"confidence": 44}), 100.0)
        assert db.query(TradeSignalLog).count() == 0
        patch.stopall()

    def test_accepts_at_45(self, tracker, db):
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("RELIANCE", "short_1h", _make_signal({"confidence": 45}), 100.0)
        assert db.query(TradeSignalLog).count() == 1
        patch.stopall()

    def test_accepts_high_confidence(self, tracker, db):
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("RELIANCE", "short_1h", _make_signal({"confidence": 75}), 100.0)
        assert db.query(TradeSignalLog).count() == 1
        patch.stopall()


# ── Direction filter ──

class TestDirectionFilter:
    def test_rejects_bearish(self, tracker, db):
        with patch("app.services.trade_tracker.SessionLocal", return_value=db):
            tracker.log_signal("RELIANCE", "short_1h", _make_signal({"direction": "BEARISH"}), 100.0)
            assert db.query(TradeSignalLog).count() == 0

    def test_rejects_neutral(self, tracker, db):
        with patch("app.services.trade_tracker.SessionLocal", return_value=db):
            tracker.log_signal("RELIANCE", "short_1h", _make_signal({"direction": "NEUTRAL"}), 100.0)
            assert db.query(TradeSignalLog).count() == 0

    def test_rejects_none_signal(self, tracker, db):
        tracker.log_signal("RELIANCE", "short_1h", None, 100.0)
        assert db.query(TradeSignalLog).count() == 0


# ── Entry sanity check ──

class TestEntrySanityCheck:
    def test_rejects_entry_too_far_above(self, tracker, db):
        """Entry 10% above current price should be rejected."""
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("RELIANCE", "short_1h",
                           _make_signal({"entry": 110.0, "target": 115.0}), 100.0)
        assert db.query(TradeSignalLog).count() == 0
        patch.stopall()

    def test_rejects_entry_too_far_below(self, tracker, db):
        """Entry 10% below current price should be rejected."""
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("RELIANCE", "short_1h",
                           _make_signal({"entry": 90.0, "target": 95.0}), 100.0)
        assert db.query(TradeSignalLog).count() == 0
        patch.stopall()

    def test_accepts_entry_within_5pct(self, tracker, db):
        """Entry 2% below current price should pass."""
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("RELIANCE", "short_1h",
                           _make_signal({"entry": 98.0, "target": 103.0}), 100.0)
        assert db.query(TradeSignalLog).count() == 1
        patch.stopall()

    def test_nifty_it_bad_entry_rejected(self, tracker, db):
        """Real bug: NIFTY IT had entry=29249 vs current=20550 — should be rejected."""
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("NIFTY IT", "short_4h",
                           _make_signal({"entry": 29249.0, "target": 30000.0}), 20550.0)
        assert db.query(TradeSignalLog).count() == 0
        patch.stopall()


# ── Cross-timeframe dedup ──

class TestCrossTimeframeDedup:
    def test_allows_first_two_trades(self, tracker, db):
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("SBIN", "short_1h", _make_signal(), 100.0)
        tracker.log_signal("SBIN", "short_4h", _make_signal(), 100.0)
        assert db.query(TradeSignalLog).count() == 2
        patch.stopall()

    def test_blocks_third_trade_same_symbol(self, tracker, db):
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("SBIN", "short_1h", _make_signal(), 100.0)
        tracker.log_signal("SBIN", "short_4h", _make_signal(), 100.0)
        tracker.log_signal("SBIN", "intraday_15m", _make_signal(), 100.0)
        assert db.query(TradeSignalLog).count() == 2
        patch.stopall()

    def test_different_symbols_not_blocked(self, tracker, db):
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("SBIN", "short_1h", _make_signal(), 100.0)
        tracker.log_signal("SBIN", "short_4h", _make_signal(), 100.0)
        tracker.log_signal("RELIANCE", "short_1h", _make_signal(), 100.0)
        assert db.query(TradeSignalLog).count() == 3
        patch.stopall()


# ── Market regime filter ──

class TestMarketRegimeFilter:
    def test_blocks_when_nifty_down(self, tracker, db):
        """Should block signals when NIFTY is down more than threshold."""
        import pandas as pd
        hist_df = pd.DataFrame({"close": [22000, 22100, 22200], "date": ["d1", "d2", "d3"]})
        mock_quotes = [{"symbol": "NIFTY 50", "ltp": 21500}]

        with patch("app.services.data_fetcher.data_fetcher.get_historical_data", return_value=hist_df), \
             patch("app.services.data_fetcher.data_fetcher.get_bulk_quotes", return_value=mock_quotes):
            tracker._market_regime_cache = None
            result = tracker._check_market_regime()
            assert result is True

    def test_allows_when_nifty_up(self, tracker, db):
        """Should allow signals when NIFTY is up."""
        import pandas as pd
        hist_df = pd.DataFrame({"close": [22000, 22100, 22200], "date": ["d1", "d2", "d3"]})
        mock_quotes = [{"symbol": "NIFTY 50", "ltp": 22500}]

        with patch("app.services.data_fetcher.data_fetcher.get_historical_data", return_value=hist_df), \
             patch("app.services.data_fetcher.data_fetcher.get_bulk_quotes", return_value=mock_quotes):
            tracker._market_regime_cache = None
            result = tracker._check_market_regime()
            assert result is False

    def test_stale_angel_close_doesnt_trigger(self, tracker, db):
        """Real bug: Angel One returned March 24 close instead of yesterday.
        NIFTY was +1.5% but appeared -1%. Use prev close + live LTP."""
        import pandas as pd
        # History: last close = 22331 (yesterday)
        hist_df = pd.DataFrame({
            "close": [22500, 22600, 22912, 22331],
            "date": ["2026-03-24", "2026-03-25", "2026-03-27", "2026-03-30"],
        })
        # Live LTP = 22677 (+1.55% from yesterday)
        mock_quotes = [{"symbol": "NIFTY 50", "ltp": 22677, "change_pct": -1.01}]

        with patch("app.services.data_fetcher.data_fetcher.get_historical_data", return_value=hist_df), \
             patch("app.services.data_fetcher.data_fetcher.get_bulk_quotes", return_value=mock_quotes):
            tracker._market_regime_cache = None
            result = tracker._check_market_regime()
            # +1.55% from prev close — should NOT block
            assert result is False


# ── Daily loss limit ──

class TestDailyLossLimit:
    def test_no_trades_no_block(self, tracker, db):
        assert tracker._check_daily_loss_limit(db) is False

    def test_blocks_after_heavy_losses(self, tracker, db):
        """If avg P&L drops below -2%, should block new trades."""
        today = now_ist().replace(tzinfo=None, hour=10, minute=0, second=0, microsecond=0)
        for i in range(5):
            trade = TradeSignalLog(
                symbol=f"STOCK{i}", timeframe="short_1h", direction="BULLISH",
                confidence=50, current_price=100, entry=100, target=105, status="wrong",
                outcome_pct=-3.0, resolved_at=today,
            )
            db.add(trade)
        db.commit()

        tracker._daily_loss_breaker = False
        tracker._breaker_date = None
        assert tracker._check_daily_loss_limit(db) is True

    def test_allows_after_small_losses(self, tracker, db):
        """Avg -0.5% should not trigger the -2% limit."""
        today = now_ist().replace(tzinfo=None, hour=10, minute=0, second=0, microsecond=0)
        for i in range(5):
            trade = TradeSignalLog(
                symbol=f"STOCK{i}", timeframe="short_1h", direction="BULLISH",
                confidence=50, current_price=100, entry=100, target=105, status="wrong",
                outcome_pct=-0.5, resolved_at=today,
            )
            db.add(trade)
        db.commit()

        tracker._daily_loss_breaker = False
        tracker._breaker_date = None
        assert tracker._check_daily_loss_limit(db) is False


# ── SL cooldown ──

class TestSLCooldown:
    def test_blocks_after_recent_sl(self, tracker, db):
        """Stock that hit SL 30 min ago should be blocked (cooldown=120min)."""
        trade = TradeSignalLog(
            symbol="SBIN", timeframe="short_1h", direction="BULLISH",
            confidence=50, current_price=100, entry=100, target=105, status="sl_hit",
            resolved_at=now_ist().replace(tzinfo=None) - timedelta(minutes=30),
        )
        db.add(trade)
        db.commit()
        assert tracker._check_sl_cooldown("SBIN", db) is True

    def test_allows_after_cooldown_expires(self, tracker, db):
        """Stock that hit SL 3 hours ago should be allowed (cooldown=120min)."""
        trade = TradeSignalLog(
            symbol="SBIN", timeframe="short_1h", direction="BULLISH",
            confidence=50, current_price=100, entry=100, target=105, status="sl_hit",
            resolved_at=now_ist().replace(tzinfo=None) - timedelta(minutes=180),
        )
        db.add(trade)
        db.commit()
        assert tracker._check_sl_cooldown("SBIN", db) is False

    def test_different_symbol_not_blocked(self, tracker, db):
        """RELIANCE SL should not block SBIN."""
        trade = TradeSignalLog(
            symbol="RELIANCE", timeframe="short_1h", direction="BULLISH",
            confidence=50, current_price=100, entry=100, target=105, status="sl_hit",
            resolved_at=now_ist().replace(tzinfo=None) - timedelta(minutes=30),
        )
        db.add(trade)
        db.commit()
        assert tracker._check_sl_cooldown("SBIN", db) is False


# ── Market close cutoff ──

class TestMarketCloseCutoff:
    def test_intraday_blocked_after_1510(self, tracker, db):
        """Intraday signals should not be logged after 15:10."""
        late = datetime(2026, 4, 1, 15, 15)
        with patch("app.services.trade_tracker.SessionLocal", return_value=db), \
             patch("app.services.trade_tracker.now_ist", return_value=late), \
             patch.object(tracker, "_check_market_regime", return_value=False), \
             patch.object(tracker, "_is_near_earnings", return_value=False):
            tracker.log_signal("RELIANCE", "intraday_15m", _make_signal(), 100.0)
            assert db.query(TradeSignalLog).count() == 0

    def test_intraday_allowed_before_1510(self, tracker, db):
        """Intraday signals before 15:10 should pass."""
        early = datetime(2026, 4, 1, 14, 30)
        with patch("app.services.trade_tracker.SessionLocal", return_value=db), \
             patch("app.services.trade_tracker.now_ist", return_value=early), \
             patch.object(tracker, "_check_market_regime", return_value=False), \
             patch.object(tracker, "_is_near_earnings", return_value=False):
            tracker.log_signal("RELIANCE", "intraday_15m", _make_signal(), 100.0)
            assert db.query(TradeSignalLog).count() == 1

    def test_shortterm_blocked_after_1525(self, tracker, db):
        """Short-term signals should not be logged after 15:25."""
        late = datetime(2026, 4, 1, 15, 30)
        with patch("app.services.trade_tracker.SessionLocal", return_value=db), \
             patch("app.services.trade_tracker.now_ist", return_value=late), \
             patch.object(tracker, "_check_market_regime", return_value=False), \
             patch.object(tracker, "_is_near_earnings", return_value=False):
            tracker.log_signal("RELIANCE", "short_4h", _make_signal(), 100.0)
            assert db.query(TradeSignalLog).count() == 0

    def test_shortterm_allowed_before_1525(self, tracker, db):
        """Short-term signals before 15:25 should pass."""
        early = datetime(2026, 4, 1, 14, 0)
        with patch("app.services.trade_tracker.SessionLocal", return_value=db), \
             patch("app.services.trade_tracker.now_ist", return_value=early), \
             patch.object(tracker, "_check_market_regime", return_value=False), \
             patch.object(tracker, "_is_near_earnings", return_value=False):
            tracker.log_signal("RELIANCE", "short_4h", _make_signal(), 100.0)
            assert db.query(TradeSignalLog).count() == 1


# ── Same timeframe dedup ──

class TestSameTimeframeDedup:
    def test_blocks_duplicate_within_interval(self, tracker, db):
        """Same symbol + timeframe within min interval should be blocked."""
        for p in _patch_for_logging(tracker, db): p.start()
        tracker.log_signal("RELIANCE", "short_1h", _make_signal(), 100.0)
        tracker.log_signal("RELIANCE", "short_1h", _make_signal(), 100.0)
        # Second should be deduped
        assert db.query(TradeSignalLog).count() == 1
        patch.stopall()
