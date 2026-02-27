import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.data_fetcher import data_fetcher
from app.services.alert_service import alert_service
from app.database import SessionLocal
from app.utils.helpers import is_market_open
from app.config import (
    PRICE_STREAM_INTERVAL, ALERT_CHECK_INTERVAL, SIGNAL_REFRESH_INTERVAL,
    HIGH_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_SCAN_INTERVAL,
)

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.subscriptions: dict[int, set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[id(websocket)] = set()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        self.subscriptions.pop(id(websocket), None)

    def subscribe(self, websocket: WebSocket, symbols: list[str]):
        existing = self.subscriptions.get(id(websocket), set())
        existing.update(s.upper() for s in symbols)
        self.subscriptions[id(websocket)] = existing

    def get_all_subscribed_symbols(self) -> set[str]:
        symbols = set()
        for subs in self.subscriptions.values():
            symbols.update(subs)
        return symbols

    async def broadcast_price(self, symbol: str, data: dict):
        for ws in self.active_connections:
            if symbol in self.subscriptions.get(id(ws), set()):
                try:
                    await ws.send_json({"type": "price_update", "data": data})
                except Exception:
                    pass

    async def broadcast_alert(self, data: dict):
        for ws in self.active_connections:
            try:
                await ws.send_json({"type": "alert_triggered", "data": data})
            except Exception:
                pass

    async def broadcast_signal(self, symbol: str, data: dict):
        for ws in self.active_connections:
            if symbol in self.subscriptions.get(id(ws), set()):
                try:
                    await ws.send_json({"type": "signal_update", "data": data})
                except Exception:
                    pass

    async def broadcast_to_all(self, msg_type: str, data: dict):
        for ws in self.active_connections:
            try:
                await ws.send_json({"type": msg_type, "data": data})
            except Exception:
                pass

    async def broadcast_mood(self, data: dict):
        """Broadcast Market Mood Score to all connected clients."""
        await self.broadcast_to_all("market_mood_update", data)


manager = ConnectionManager()

# Shared signal cache: high_confidence_scanner populates, signal_broadcaster reuses
# Format: {symbol: {"data": signal_dict, "ts": time.time()}}
_recent_signal_cache: dict[str, dict] = {}


@router.websocket("/ws/prices")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if "subscribe" in msg:
                manager.subscribe(websocket, msg["subscribe"])
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def price_streamer():
    """Streams live price updates to connected WebSocket clients.
    Uses Angel One for real-time data when available, falls back to yfinance.
    Feeds ticks into LiveCandleBuilder for real-time candle construction."""
    from app.services.data_fetcher import ANGEL_AVAILABLE
    # Angel One: 5s interval (was 2s — reduced to save API quota)
    interval = PRICE_STREAM_INTERVAL  # 5s for both Angel One and yfinance

    # Import candle builder for live tick accumulation
    candle_builder = None
    if ANGEL_AVAILABLE:
        try:
            from app.services.live_candle_builder import candle_builder as cb
            candle_builder = cb
        except Exception:
            pass

    while True:
        try:
            if is_market_open() and manager.active_connections:
                symbols = list(manager.get_all_subscribed_symbols())
                if symbols:
                    quotes = data_fetcher.get_bulk_quotes(symbols)
                    for quote in quotes:
                        if quote.get("ltp"):
                            await manager.broadcast_price(quote["symbol"], quote)

                            # Feed tick to candle builder when Angel One is active
                            if candle_builder and ANGEL_AVAILABLE:
                                try:
                                    candle_builder.on_tick(
                                        symbol=quote["symbol"],
                                        ltp=float(quote["ltp"]),
                                        volume=int(quote.get("volume", 0)),
                                        bid=float(quote.get("bid", 0)),
                                        ask=float(quote.get("ask", 0)),
                                    )
                                except Exception:
                                    pass
        except Exception:
            pass
        await asyncio.sleep(interval)


async def alert_checker():
    while True:
        try:
            if is_market_open() and manager.active_connections:
                db = SessionLocal()
                try:
                    triggered = alert_service.check_alerts_batched(db)
                    for alert_data in triggered:
                        await manager.broadcast_alert(alert_data)
                finally:
                    db.close()
        except Exception:
            pass
        await asyncio.sleep(ALERT_CHECK_INTERVAL)


async def signal_broadcaster():
    from app.services.signal_service import signal_service

    while True:
        try:
            if is_market_open() and manager.active_connections:
                symbols = manager.get_all_subscribed_symbols()
                import time as _time
                for symbol in symbols:
                    # Reuse signal if high_confidence_scanner fetched it recently
                    cached_sig = _recent_signal_cache.get(symbol)
                    if cached_sig and (_time.time() - cached_sig["ts"]) < SIGNAL_REFRESH_INTERVAL:
                        signal = cached_sig["data"]
                    else:
                        try:
                            signal = signal_service.get_signal(symbol)
                        except Exception:
                            signal = None
                    if signal:
                        await manager.broadcast_signal(symbol, signal)
                    # Stagger API calls: 2s between symbols to avoid burst
                    await asyncio.sleep(2)
        except Exception:
            pass
        await asyncio.sleep(SIGNAL_REFRESH_INTERVAL)


async def signal_accuracy_validator():
    """Background task that checks signals from ~15-20 minutes ago
    and populates price_after_15min and was_correct fields in SignalLog."""
    from datetime import datetime, timedelta
    from app.models import SignalLog

    await asyncio.sleep(30)

    while True:
        try:
            if not is_market_open():
                await asyncio.sleep(180)
                continue

            db = SessionLocal()
            try:
                cutoff_start = datetime.utcnow() - timedelta(minutes=25)
                cutoff_end = datetime.utcnow() - timedelta(minutes=15)

                pending_logs = (
                    db.query(SignalLog)
                    .filter(
                        SignalLog.price_after_15min.is_(None),
                        SignalLog.price_at_signal.isnot(None),
                        SignalLog.created_at >= cutoff_start,
                        SignalLog.created_at <= cutoff_end,
                    )
                    .limit(20)
                    .all()
                )

                if pending_logs:
                    # Batch-fetch all symbols in one API call instead of N individual calls
                    symbols = list({log.symbol for log in pending_logs})
                    quotes = data_fetcher.get_bulk_quotes(symbols)
                    price_map = {}
                    for q in quotes:
                        if q.get("ltp"):
                            price_map[q["symbol"]] = float(q["ltp"])

                    for log in pending_logs:
                        current_price = price_map.get(log.symbol)
                        if current_price:
                            log.price_after_15min = current_price
                            if log.direction == "BULLISH":
                                log.was_correct = current_price > log.price_at_signal
                            elif log.direction == "BEARISH":
                                log.was_correct = current_price < log.price_at_signal
                            else:
                                pct_move = abs(current_price - log.price_at_signal) / log.price_at_signal
                                log.was_correct = pct_move < 0.005

                    db.commit()
            finally:
                db.close()
        except Exception:
            pass

        await asyncio.sleep(180)  # 3 min instead of 2 min


async def high_confidence_scanner():
    """Background task that scans ONLY user-subscribed symbols for high-confidence signals.
    Logs signals to SignalLog and broadcasts high-confidence alerts via WebSocket.
    Does nothing when no users are connected — no point scanning if nobody sees the results."""
    from app.services.signal_service import signal_service
    from app.models import SignalLog

    # Wait a bit on startup to let other things initialize
    await asyncio.sleep(10)

    while True:
        try:
            # Skip entirely if market closed or no users connected
            if not is_market_open() or not manager.active_connections:
                await asyncio.sleep(HIGH_CONFIDENCE_SCAN_INTERVAL)
                continue

            # Only scan symbols users are actually looking at — not all 34
            subscribed = manager.get_all_subscribed_symbols()
            if not subscribed:
                await asyncio.sleep(HIGH_CONFIDENCE_SCAN_INTERVAL)
                continue

            db = SessionLocal()
            try:
                for symbol in subscribed:
                    try:
                        signal = signal_service.get_signal(symbol)
                        if not signal:
                            continue

                        # Cache signal so signal_broadcaster can reuse it
                        import time as _time
                        _recent_signal_cache[symbol] = {
                            "data": signal,
                            "ts": _time.time(),
                        }

                        # Get price from intraday candles
                        price = None
                        candles = signal.get("intraday_candles", [])
                        if candles:
                            price = candles[-1].get("close")

                        # Log to database
                        log = SignalLog(
                            symbol=symbol,
                            direction=signal["direction"],
                            confidence=signal["confidence"],
                            composite_score=signal["composite_score"],
                            technical_score=signal["technical"]["score"],
                            sentiment_score=signal["sentiment"]["score"],
                            global_score=signal["global_market"]["score"],
                            price_at_signal=price,
                        )
                        db.add(log)

                        # Broadcast if high confidence
                        if (signal["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
                                and signal["direction"] != "NEUTRAL"):
                            await manager.broadcast_to_all("high_confidence_alert", {
                                "symbol": symbol,
                                "direction": signal["direction"],
                                "confidence": signal["confidence"],
                                "composite_score": signal["composite_score"],
                                "price": price,
                                "timestamp": signal["timestamp"],
                            })
                    except Exception:
                        pass

                    # 5s between symbols to stay well under rate limit
                    await asyncio.sleep(5)

                db.commit()
            finally:
                db.close()
        except Exception:
            pass
        await asyncio.sleep(HIGH_CONFIDENCE_SCAN_INTERVAL)
