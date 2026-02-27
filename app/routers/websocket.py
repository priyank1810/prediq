import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.data_fetcher import data_fetcher
from app.services.alert_service import alert_service
from app.database import SessionLocal
from app.utils.helpers import is_market_open
from app.config import (
    PRICE_STREAM_INTERVAL, ALERT_CHECK_INTERVAL, SIGNAL_REFRESH_INTERVAL,
    POPULAR_STOCKS, INDICES, HIGH_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_SCAN_INTERVAL,
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
    # Poll faster when we have real-time source (Angel One: 2s, yfinance: 5s)
    interval = 2 if ANGEL_AVAILABLE else PRICE_STREAM_INTERVAL

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
            if is_market_open():
                db = SessionLocal()
                try:
                    triggered = alert_service.check_alerts(db)
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
                for symbol in symbols:
                    try:
                        signal = signal_service.get_signal(symbol)
                        if signal:
                            await manager.broadcast_signal(symbol, signal)
                    except Exception:
                        pass
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

                for log in pending_logs:
                    try:
                        quote = data_fetcher.get_live_quote(log.symbol)
                        if quote and quote.get("ltp"):
                            current_price = float(quote["ltp"])
                            log.price_after_15min = current_price

                            if log.direction == "BULLISH":
                                log.was_correct = current_price > log.price_at_signal
                            elif log.direction == "BEARISH":
                                log.was_correct = current_price < log.price_at_signal
                            else:
                                pct_move = abs(current_price - log.price_at_signal) / log.price_at_signal
                                log.was_correct = pct_move < 0.005
                    except Exception:
                        pass

                    await asyncio.sleep(0.5)

                db.commit()
            finally:
                db.close()
        except Exception:
            pass

        await asyncio.sleep(120)


async def high_confidence_scanner():
    """Background task that scans all popular stocks + indices for high-confidence signals.
    Logs every signal to SignalLog and broadcasts high-confidence alerts via WebSocket."""
    from app.services.signal_service import signal_service
    from app.models import SignalLog

    # Wait a bit on startup to let other things initialize
    await asyncio.sleep(10)

    while True:
        try:
            all_symbols = list(POPULAR_STOCKS) + list(INDICES.keys())
            db = SessionLocal()
            try:
                for symbol in all_symbols:
                    try:
                        signal = signal_service.get_signal(symbol)
                        if not signal:
                            continue

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
                                and signal["direction"] != "NEUTRAL"
                                and manager.active_connections):
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

                    # Delay between symbols to avoid hammering Angel One API
                    await asyncio.sleep(3)

                db.commit()
            finally:
                db.close()
        except Exception:
            pass
        await asyncio.sleep(HIGH_CONFIDENCE_SCAN_INTERVAL)
