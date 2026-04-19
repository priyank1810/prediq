# Telegram Signal Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push every BULLISH signal with confidence ≥ 50 to all Telegram subscribers (alert type "signals") in real time during scan runs.

**Architecture:** Two touch points — enhance `send_signal_alert()` in `telegram_service.py` to format the full signal payload, then add a fire-and-forget helper in `worker.py` that calls the broadcast from a daemon thread (worker is synchronous/threading-based, no running event loop).

**Tech Stack:** Python asyncio, aiohttp (existing), threading, pytest + unittest.mock

---

### Task 1: Enhance `send_signal_alert()` format

**Files:**
- Modify: `app/services/telegram_service.py:70-90`
- Test: `tests/test_telegram_service.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_telegram_service.py`:

```python
"""Tests for telegram_service.send_signal_alert formatting."""

import pytest
from unittest.mock import AsyncMock, patch


def _make_signal(**overrides):
    sig = {
        "symbol": "RELIANCE",
        "direction": "BULLISH",
        "confidence": 72,
        "entry": 2450.0,
        "stop_loss": 2390.0,
        "target": 2560.0,
        "timeframe": "intraday_15m",
        "regime": "trending",
        "volume_conviction": "high",
        "confidence_trend": "rising",
        "model_confidence": 68.5,
        "risk_reward": 1.83,
    }
    sig.update(overrides)
    return sig


@pytest.mark.asyncio
async def test_send_signal_alert_full_fields():
    """All optional fields present → all included in message."""
    with patch("app.services.telegram_service.send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        from app.services.telegram_service import send_signal_alert
        await send_signal_alert("12345", _make_signal())

        mock_send.assert_called_once()
        text = mock_send.call_args[0][1]
        assert "RELIANCE" in text
        assert "BULLISH" in text
        assert "72" in text
        assert "2450" in text
        assert "2390" in text
        assert "2560" in text
        assert "trending" in text.lower()
        assert "high" in text.lower()
        assert "rising" in text.lower()
        assert "1.83" in text


@pytest.mark.asyncio
async def test_send_signal_alert_missing_optional_fields():
    """Optional fields absent → no crash, no 'None' in message."""
    with patch("app.services.telegram_service.send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        from app.services.telegram_service import send_signal_alert
        minimal = {
            "symbol": "TCS",
            "direction": "BULLISH",
            "confidence": 55,
            "entry": 3500.0,
            "stop_loss": 3430.0,
            "target": 3620.0,
        }
        await send_signal_alert("99999", minimal)

        text = mock_send.call_args[0][1]
        assert "None" not in text
        assert "TCS" in text


@pytest.mark.asyncio
async def test_send_signal_alert_bearish_arrow():
    """BEARISH direction uses down arrow."""
    with patch("app.services.telegram_service.send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        from app.services.telegram_service import send_signal_alert
        await send_signal_alert("12345", _make_signal(direction="BEARISH"))
        text = mock_send.call_args[0][1]
        assert "⬇" in text
```

- [ ] **Step 2: Install pytest-asyncio if needed and run tests to verify they fail**

```bash
pip install pytest-asyncio 2>/dev/null; python -m pytest tests/test_telegram_service.py -v
```

Expected: 3 failures — `send_signal_alert` exists but message body missing new fields.

- [ ] **Step 3: Rewrite `send_signal_alert()` in `app/services/telegram_service.py`**

Replace lines 70–90 with:

```python
async def send_signal_alert(chat_id: str, signal_data: dict) -> bool:
    """Send a formatted signal alert with full signal details."""
    symbol = signal_data.get("symbol", "?")
    direction = signal_data.get("direction", "?")
    confidence = signal_data.get("confidence", 0)
    entry = signal_data.get("entry") or signal_data.get("price_at_signal") or signal_data.get("ltp", "N/A")
    stop_loss = signal_data.get("stop_loss")
    target = signal_data.get("target")
    timeframe = signal_data.get("timeframe")
    regime = signal_data.get("regime")
    volume_conviction = signal_data.get("volume_conviction")
    confidence_trend = signal_data.get("confidence_trend")
    model_confidence = signal_data.get("model_confidence")
    risk_reward = signal_data.get("risk_reward")

    arrow = "⬆️" if direction.upper() == "BULLISH" else "⬇️" if direction.upper() == "BEARISH" else "➡️"

    lines = [
        f"{arrow} <b>Signal: {symbol}</b>",
        "",
        f"Direction: <b>{direction}</b>",
        f"Confidence: <b>{confidence:.0f}%</b>",
        f"Entry: ₹{entry}",
    ]
    if stop_loss is not None:
        lines.append(f"Stop Loss: ₹{stop_loss}")
    if target is not None:
        lines.append(f"Target: ₹{target}")
    if risk_reward is not None:
        lines.append(f"Risk/Reward: {risk_reward:.2f}")

    extras = []
    if timeframe:
        extras.append(f"TF: {timeframe}")
    if regime:
        extras.append(f"Regime: {regime}")
    if volume_conviction:
        extras.append(f"Volume: {volume_conviction}")
    if confidence_trend:
        extras.append(f"Trend: {confidence_trend}")
    if model_confidence is not None:
        extras.append(f"Model: {model_confidence:.0f}%")
    if extras:
        lines.append(" | ".join(extras))

    return await send_message(chat_id, "\n".join(lines))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_telegram_service.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/services/telegram_service.py tests/test_telegram_service.py
git commit -m "feat: enhance send_signal_alert with full signal details"
```

---

### Task 2: Add `_fire_telegram_signal()` and wire into scan loop

**Files:**
- Modify: `worker.py:120-213`
- Test: `tests/test_worker_telegram.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_worker_telegram.py`:

```python
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


def test_fire_telegram_signal_spawns_daemon_thread():
    """_fire_telegram_signal starts a daemon thread and calls broadcast."""
    from worker import _fire_telegram_signal

    broadcast_calls = []

    async def fake_broadcast(alert_type, func, data):
        broadcast_calls.append((alert_type, data))

    with patch("app.services.telegram_service.broadcast_to_subscribers", fake_broadcast):
        t_before = threading.active_count()
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
    """handle_trade_scan calls _fire_telegram_signal for BULLISH signals ≥ 50."""
    from worker import Worker

    worker = Worker.__new__(Worker)
    worker._signal_service = None
    worker._shutdown = False

    mock_signal = _make_sig(confidence=65, direction="BULLISH")
    mtf_result = {
        "current_price": 100.0,
        "intraday": {"15m": mock_signal},
        "short_term": {},
    }

    fired = []

    with (
        patch.object(Worker, "signal_service", new_callable=lambda: property(lambda self: MagicMock(
            get_multi_timeframe_signals=MagicMock(return_value=mtf_result)
        ))),
        patch("app.services.job_service.job_service") ,
        patch("app.database.SessionLocal") as mock_db,
        patch("app.services.trade_tracker.trade_tracker") as mock_tracker,
        patch("worker._fire_telegram_signal", side_effect=lambda d: fired.append(d)) as mock_fire,
        patch("worker.data_fetcher.get_bulk_quotes", return_value=[]),
        patch("worker._time.sleep"),
    ):
        mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock(
            query=MagicMock(return_value=MagicMock(
                distinct=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
            ))
        ))
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        # Simple DB session for watchlist query
        session = MagicMock()
        session.query.return_value.distinct.return_value.all.return_value = [
            MagicMock(symbol="RELIANCE")
        ]
        mock_db.return_value = session

        mock_tracker.log_signal = MagicMock()

        worker.handle_trade_scan({"scan_type": "intraday"})

    assert any(d.get("direction") == "BULLISH" for d in fired)


def test_scan_does_not_fire_telegram_for_bearish():
    """handle_trade_scan does NOT call _fire_telegram_signal for BEARISH signals."""
    from worker import Worker

    worker = Worker.__new__(Worker)
    worker._signal_service = None

    bearish_signal = _make_sig(confidence=80, direction="BEARISH")
    mtf_result = {
        "current_price": 100.0,
        "intraday": {"15m": bearish_signal},
        "short_term": {},
    }

    fired = []

    with (
        patch.object(Worker, "signal_service", new_callable=lambda: property(lambda self: MagicMock(
            get_multi_timeframe_signals=MagicMock(return_value=mtf_result)
        ))),
        patch("worker._fire_telegram_signal", side_effect=lambda d: fired.append(d)),
        patch("app.database.SessionLocal"),
        patch("app.services.trade_tracker.trade_tracker"),
        patch("worker.data_fetcher.get_bulk_quotes", return_value=[]),
        patch("worker._time.sleep"),
    ):
        try:
            worker.handle_trade_scan({"scan_type": "intraday"})
        except Exception:
            pass

    assert not any(d.get("direction") == "BEARISH" for d in fired)


def test_scan_does_not_fire_telegram_below_50():
    """handle_trade_scan does NOT fire Telegram for confidence < 50."""
    from worker import Worker

    worker = Worker.__new__(Worker)
    worker._signal_service = None

    low_conf_signal = _make_sig(confidence=45, direction="BULLISH")
    mtf_result = {
        "current_price": 100.0,
        "intraday": {"15m": low_conf_signal},
        "short_term": {},
    }

    fired = []

    with (
        patch.object(Worker, "signal_service", new_callable=lambda: property(lambda self: MagicMock(
            get_multi_timeframe_signals=MagicMock(return_value=mtf_result)
        ))),
        patch("worker._fire_telegram_signal", side_effect=lambda d: fired.append(d)),
        patch("app.database.SessionLocal"),
        patch("app.services.trade_tracker.trade_tracker"),
        patch("worker.data_fetcher.get_bulk_quotes", return_value=[]),
        patch("worker._time.sleep"),
    ):
        try:
            worker.handle_trade_scan({"scan_type": "intraday"})
        except Exception:
            pass

    assert fired == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_worker_telegram.py -v
```

Expected: failures — `_fire_telegram_signal` not defined in `worker.py` yet.

- [ ] **Step 3: Add `_fire_telegram_signal()` as a module-level function in `worker.py`**

Add immediately after the imports block (around line 45, after `from app.services.job_service import job_service`):

```python
def _fire_telegram_signal(signal_data: dict) -> None:
    """Fire-and-forget: broadcast a signal alert to Telegram subscribers.

    Runs in a daemon thread because the worker is synchronous — there is no
    running event loop to schedule coroutines onto.
    """
    import asyncio as _asyncio
    import threading as _threading
    from app.services.telegram_service import broadcast_to_subscribers, send_signal_alert

    def _run():
        _asyncio.run(broadcast_to_subscribers("signals", send_signal_alert, signal_data))

    _threading.Thread(target=_run, daemon=True).start()
```

- [ ] **Step 4: Wire the call into the watchlist scan loop (lines ~147-151)**

After each `trade_tracker.log_signal(sym, f"intraday_{tf_key}", sig, live_price)` call in the watchlist loop, add the Telegram fire. The relevant block currently looks like:

```python
if bullish_count >= 2:
    if scan_type in ("intraday", "full"):
        for tf_key, sig in intraday.items():
            if sig and sig.get("direction") == "BULLISH":
                trade_tracker.log_signal(sym, f"intraday_{tf_key}", sig, live_price)
    if scan_type in ("short", "full"):
        for tf_key, sig in short_term.items():
            if sig and sig.get("direction") == "BULLISH":
                trade_tracker.log_signal(sym, f"short_{tf_key}", sig, live_price)
```

Replace with:

```python
if bullish_count >= 2:
    if scan_type in ("intraday", "full"):
        for tf_key, sig in intraday.items():
            if sig and sig.get("direction") == "BULLISH":
                trade_tracker.log_signal(sym, f"intraday_{tf_key}", sig, live_price)
                if (sig.get("confidence") or 0) >= 50:
                    _fire_telegram_signal({**sig, "symbol": sym, "timeframe": f"intraday_{tf_key}"})
    if scan_type in ("short", "full"):
        for tf_key, sig in short_term.items():
            if sig and sig.get("direction") == "BULLISH":
                trade_tracker.log_signal(sym, f"short_{tf_key}", sig, live_price)
                if (sig.get("confidence") or 0) >= 50:
                    _fire_telegram_signal({**sig, "symbol": sym, "timeframe": f"short_{tf_key}"})
```

- [ ] **Step 5: Wire the call into the popular stocks scan loop (lines ~200-208)**

The relevant block currently looks like:

```python
if bullish_count >= 2:
    if scan_type in ("intraday", "full"):
        for tf_key, sig in intraday.items():
            if sig and sig.get("direction") == "BULLISH" and (sig.get("confidence") or 0) >= popular_threshold:
                trade_tracker.log_signal(sym, f"intraday_{tf_key}", sig, live_price)
                popular_logged += 1
    if scan_type in ("short", "full"):
        for tf_key, sig in short_term.items():
            if sig and sig.get("direction") == "BULLISH" and (sig.get("confidence") or 0) >= popular_threshold:
                trade_tracker.log_signal(sym, f"short_{tf_key}", sig, live_price)
                popular_logged += 1
```

Replace with:

```python
if bullish_count >= 2:
    if scan_type in ("intraday", "full"):
        for tf_key, sig in intraday.items():
            if sig and sig.get("direction") == "BULLISH" and (sig.get("confidence") or 0) >= popular_threshold:
                trade_tracker.log_signal(sym, f"intraday_{tf_key}", sig, live_price)
                popular_logged += 1
                if (sig.get("confidence") or 0) >= 50:
                    _fire_telegram_signal({**sig, "symbol": sym, "timeframe": f"intraday_{tf_key}"})
    if scan_type in ("short", "full"):
        for tf_key, sig in short_term.items():
            if sig and sig.get("direction") == "BULLISH" and (sig.get("confidence") or 0) >= popular_threshold:
                trade_tracker.log_signal(sym, f"short_{tf_key}", sig, live_price)
                popular_logged += 1
                if (sig.get("confidence") or 0) >= 50:
                    _fire_telegram_signal({**sig, "symbol": sym, "timeframe": f"short_{tf_key}"})
```

- [ ] **Step 6: Run all tests**

```bash
python -m pytest tests/test_worker_telegram.py tests/test_telegram_service.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Run the full test suite to check for regressions**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add worker.py tests/test_worker_telegram.py
git commit -m "feat: fire Telegram alert for BULLISH signals ≥ 50% confidence during scan"
```

---

## Self-Review

**Spec coverage:**
- ✅ BULLISH only — guarded at both call sites with `direction == "BULLISH"`
- ✅ Confidence ≥ 50 — checked before calling `_fire_telegram_signal`
- ✅ Triggered during scan — wired into both scan loops immediately after `log_signal`
- ✅ All details — `send_signal_alert` now includes timeframe, regime, volume_conviction, confidence_trend, model_confidence, risk_reward
- ✅ Subscribers with "signals" alert type — handled by existing `broadcast_to_subscribers("signals", ...)`
- ✅ Non-blocking — daemon thread

**Placeholder scan:** No TBDs, all code blocks complete.

**Type consistency:** `_fire_telegram_signal(dict)` → `broadcast_to_subscribers("signals", send_signal_alert, dict)` → `send_signal_alert(chat_id, dict)` — consistent across both tasks.
