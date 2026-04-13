# Telegram Signal Alerts — Design Spec

**Date:** 2026-04-13  
**Status:** Approved

---

## Problem

The Telegram infrastructure (`telegram_service.py`, subscriber model, `broadcast_to_subscribers`) is fully built, but signal alerts are never sent. Signals are scanned and logged to DB in `worker.py` with no outbound notification. Users miss live bullish setups.

---

## Goal

During every scan run, push each qualifying bullish signal to all Telegram subscribers (those with `"signals"` in their `alert_types`) as it is found — in real time, with full signal detail.

---

## Trigger Conditions

A signal triggers a Telegram push when ALL of the following are true:
- `direction == "BULLISH"`
- `confidence >= 50`
- The signal was already accepted by `trade_tracker.log_signal()` (existing multi-TF and trend guards still apply)

---

## Architecture

### 1. Enhance `send_signal_alert()` in `telegram_service.py`

The existing function only sends: symbol, direction, confidence, price, stop_loss, target.

Extend it to include all available signal fields:
- **Timeframe** (intraday / short-term, specific TF key)
- **Regime** (trending, ranging, etc.)
- **Volume conviction** (high / medium / low)
- **Confidence trend** (rising / stable / falling)
- **Model confidence** (XGBoost/Prophet model score, if available)
- **Risk/reward ratio**

Format: HTML, concise, mobile-friendly. All optional fields degrade gracefully (omitted if None).

### 2. Add `_fire_telegram_signal()` helper in `worker.py`

A small synchronous helper that launches the async broadcast in a **daemon thread** — fire-and-forget, non-blocking:

```python
def _fire_telegram_signal(signal_data: dict):
    import asyncio
    import threading
    from app.services.telegram_service import broadcast_to_subscribers, send_signal_alert
    def _run():
        asyncio.run(broadcast_to_subscribers("signals", send_signal_alert, signal_data))
    threading.Thread(target=_run, daemon=True).start()
```

### 3. Call site in `worker.py` — `handle_trade_scan()`

After every `trade_tracker.log_signal(sym, tf_key, sig, live_price)` call, check:

```python
if (sig.get("direction") == "BULLISH" and
        (sig.get("confidence") or 0) >= 50):
    _fire_telegram_signal({**sig, "symbol": sym, "timeframe": tf_key})
```

This applies to both the watchlist loop and the popular-stocks loop.

---

## Data Flow

```
worker.handle_trade_scan()
  └─ for each qualifying signal:
       trade_tracker.log_signal(...)        # existing: persist to DB
       _fire_telegram_signal(signal_data)   # new: async broadcast in daemon thread
             └─ broadcast_to_subscribers("signals", send_signal_alert, data)
                   └─ for each active subscriber with "signals" alert type:
                        send_signal_alert(chat_id, data)  # enhanced format
                              └─ telegram_service.send_message(chat_id, html_text)
```

---

## Error Handling

- Telegram failures are already caught and logged inside `send_message()` — no retry, no crash
- Daemon thread: if worker exits, in-flight sends are abandoned (acceptable — best-effort delivery)
- Missing optional fields (regime, model_confidence, etc.) are omitted from the message, not shown as "N/A"

---

## What Does NOT Change

- Scan scheduling and frequency — unchanged
- `trade_tracker.log_signal()` logic and DB persistence — unchanged
- Subscriber management, linking, and preferences — unchanged
- SMS service — unchanged (parity can be added later if needed)

---

## Files Changed

| File | Change |
|------|--------|
| `app/services/telegram_service.py` | Enhance `send_signal_alert()` with full signal fields |
| `worker.py` | Add `_fire_telegram_signal()` helper; call it after each logged signal |

---

## Out of Scope

- BEARISH signals
- Daily digest / summary messages
- Deduplication across scans (same symbol, same TF back-to-back)
- SMS parity
