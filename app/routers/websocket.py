import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.data_fetcher import data_fetcher
from app.services.alert_service import alert_service
from app.database import SessionLocal
from app.utils.helpers import is_market_open
from app.config import PRICE_STREAM_INTERVAL, ALERT_CHECK_INTERVAL

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

    async def broadcast_oi_update(self, symbol: str, data: dict):
        """Broadcast OI analysis update to subscribers of a symbol."""
        for ws in self.active_connections:
            if symbol in self.subscriptions.get(id(ws), set()):
                try:
                    await ws.send_json({"type": "oi_update", "data": data})
                except Exception:
                    pass

    async def broadcast_mtf_update(self, symbol: str, data: dict):
        """Broadcast MTF confluence update to subscribers of a symbol."""
        for ws in self.active_connections:
            if symbol in self.subscriptions.get(id(ws), set()):
                try:
                    await ws.send_json({"type": "mtf_update", "data": data})
                except Exception:
                    pass


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
                            # Trim to only fields the frontend needs
                            trimmed = {
                                "symbol": quote["symbol"],
                                "ltp": quote["ltp"],
                                "change": quote.get("change", 0),
                                "pct_change": quote.get("pct_change", 0),
                                "high": quote.get("high"),
                                "low": quote.get("low"),
                                "open": quote.get("open"),
                                "volume": quote.get("volume"),
                            }
                            await manager.broadcast_price(trimmed["symbol"], trimmed)

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


async def signal_accuracy_validator():
    """Background task that checks signals from ~15-20 minutes ago
    and populates price_after_15min and was_correct fields in SignalLog.
    Uses yfinance only — this is internal bookkeeping, no need for Angel One."""
    from datetime import timedelta
    from app.models import SignalLog
    import yfinance as yf
    from app.utils.helpers import yfinance_symbol, now_ist

    await asyncio.sleep(30)

    while True:
        try:
            # Also validate shortly after market close to catch late-session signals
            from app.utils.helpers import market_status
            mkt = market_status()
            if mkt in ("closed_weekend", "pre_market"):
                await asyncio.sleep(300)
                continue

            db = SessionLocal()
            try:
                cutoff_start = now_ist() - timedelta(minutes=30)
                cutoff_end = now_ist() - timedelta(minutes=15)

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
                    # Use yfinance directly — no Angel One calls for bookkeeping
                    symbols = list({log.symbol for log in pending_logs})
                    price_map = {}
                    for sym in symbols:
                        try:
                            ticker = yf.Ticker(yfinance_symbol(sym))
                            info = ticker.fast_info
                            if info.last_price:
                                price_map[sym] = round(float(info.last_price), 2)
                        except Exception:
                            pass

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
                                log.was_correct = pct_move < 0.015  # NEUTRAL correct if <1.5% move

                    db.commit()
            finally:
                db.close()
        except Exception:
            pass

        await asyncio.sleep(180)


