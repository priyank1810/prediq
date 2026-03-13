import asyncio
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.data_fetcher import data_fetcher
from app.services.alert_service import alert_service
from app.database import SessionLocal
from app.utils.helpers import is_market_open
from app.config import PRICE_STREAM_INTERVAL, ALERT_CHECK_INTERVAL

router = APIRouter()

# WebSocket limits
WS_MAX_CONNECTIONS = 50          # Total concurrent connections
WS_MAX_PER_IP = 5               # Max connections per IP
WS_MAX_SUBSCRIPTIONS = 50       # Max symbols per client
WS_MSG_RATE_LIMIT = 10          # Max messages per window
WS_MSG_RATE_WINDOW = 10         # Rate window in seconds


def _get_correctness_threshold(symbol: str) -> float:
    from app.config import LOW_VOLATILITY_SYMBOLS, LOW_VOLATILITY_THRESHOLD, ACCURACY_BASE_THRESHOLD
    if symbol in LOW_VOLATILITY_SYMBOLS:
        return LOW_VOLATILITY_THRESHOLD
    return ACCURACY_BASE_THRESHOLD


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.subscriptions: dict[int, set[str]] = {}
        self._ip_counts: dict[str, int] = {}
        self._msg_timestamps: dict[int, list[float]] = {}

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept connection if within limits. Returns False if rejected."""
        client_ip = websocket.client.host if websocket.client else "unknown"

        if len(self.active_connections) >= WS_MAX_CONNECTIONS:
            await websocket.close(code=1013, reason="Server at capacity")
            return False

        if self._ip_counts.get(client_ip, 0) >= WS_MAX_PER_IP:
            await websocket.close(code=1008, reason="Too many connections from this IP")
            return False

        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[id(websocket)] = set()
        self._ip_counts[client_ip] = self._ip_counts.get(client_ip, 0) + 1
        self._msg_timestamps[id(websocket)] = []
        return True

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.subscriptions.pop(id(websocket), None)
        self._msg_timestamps.pop(id(websocket), None)
        client_ip = websocket.client.host if websocket.client else "unknown"
        self._ip_counts[client_ip] = max(0, self._ip_counts.get(client_ip, 0) - 1)
        if self._ip_counts.get(client_ip) == 0:
            self._ip_counts.pop(client_ip, None)

    def check_rate_limit(self, websocket: WebSocket) -> bool:
        """Returns True if message is allowed, False if rate-limited."""
        ws_id = id(websocket)
        now = time.monotonic()
        timestamps = self._msg_timestamps.get(ws_id, [])
        # Prune old timestamps
        timestamps = [t for t in timestamps if now - t < WS_MSG_RATE_WINDOW]
        if len(timestamps) >= WS_MSG_RATE_LIMIT:
            self._msg_timestamps[ws_id] = timestamps
            return False
        timestamps.append(now)
        self._msg_timestamps[ws_id] = timestamps
        return True

    def subscribe(self, websocket: WebSocket, symbols: list[str]):
        existing = self.subscriptions.get(id(websocket), set())
        for s in symbols:
            if len(existing) >= WS_MAX_SUBSCRIPTIONS:
                break
            existing.add(s.upper())
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
    accepted = await manager.connect(websocket)
    if not accepted:
        return
    try:
        while True:
            data = await websocket.receive_text()
            if not manager.check_rate_limit(websocket):
                await websocket.send_json({"type": "error", "detail": "Rate limit exceeded"})
                continue
            msg = json.loads(data)
            if "subscribe" in msg:
                symbols = msg["subscribe"]
                if isinstance(symbols, list):
                    manager.subscribe(websocket, symbols[:WS_MAX_SUBSCRIPTIONS])
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
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
                    quotes = await asyncio.to_thread(data_fetcher.get_bulk_quotes, symbols)
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
                    triggered = await asyncio.to_thread(alert_service.check_alerts_batched, db)
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
    from app.utils.helpers import yfinance_symbol, now_ist
    from app.utils.yahoo_api import yahoo_quote

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
                cutoff_start = now_ist().replace(tzinfo=None) - timedelta(minutes=30)
                cutoff_end = now_ist().replace(tzinfo=None) - timedelta(minutes=15)

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
                    symbols = list({log.symbol for log in pending_logs})

                    def _fetch_prices(syms):
                        pm = {}
                        for sym in syms:
                            try:
                                q = yahoo_quote(yfinance_symbol(sym))
                                if q and q["ltp"]:
                                    pm[sym] = round(float(q["ltp"]), 2)
                            except Exception:
                                pass
                        return pm

                    price_map = await asyncio.to_thread(_fetch_prices, symbols)

                    from app.config import ACCURACY_NEUTRAL_THRESHOLD
                    for log in pending_logs:
                        current_price = price_map.get(log.symbol)
                        if current_price:
                            log.price_after_15min = current_price
                            pct_move = (current_price - log.price_at_signal) / log.price_at_signal
                            threshold = _get_correctness_threshold(log.symbol)
                            if log.direction == "BULLISH":
                                log.was_correct = pct_move >= threshold
                            elif log.direction == "BEARISH":
                                log.was_correct = pct_move <= -threshold
                            else:
                                log.was_correct = abs(pct_move) < ACCURACY_NEUTRAL_THRESHOLD

                    db.commit()
            finally:
                db.close()
        except Exception:
            pass

        await asyncio.sleep(180)


async def signal_accuracy_validator_30min():
    """Background task that checks signals from ~30-60 minutes ago
    and populates price_after_30min and was_correct_30min fields."""
    from datetime import timedelta
    from app.models import SignalLog
    from app.utils.helpers import yfinance_symbol, now_ist
    from app.utils.yahoo_api import yahoo_quote

    await asyncio.sleep(60)

    while True:
        try:
            from app.utils.helpers import market_status
            mkt = market_status()
            if mkt in ("closed_weekend", "pre_market"):
                await asyncio.sleep(300)
                continue

            db = SessionLocal()
            try:
                cutoff_start = now_ist().replace(tzinfo=None) - timedelta(minutes=60)
                cutoff_end = now_ist().replace(tzinfo=None) - timedelta(minutes=35)

                pending_logs = (
                    db.query(SignalLog)
                    .filter(
                        SignalLog.price_after_30min.is_(None),
                        SignalLog.price_at_signal.isnot(None),
                        SignalLog.created_at >= cutoff_start,
                        SignalLog.created_at <= cutoff_end,
                    )
                    .limit(20)
                    .all()
                )

                if pending_logs:
                    symbols = list({log.symbol for log in pending_logs})

                    def _fetch_prices(syms):
                        pm = {}
                        for sym in syms:
                            try:
                                q = yahoo_quote(yfinance_symbol(sym))
                                if q and q["ltp"]:
                                    pm[sym] = round(float(q["ltp"]), 2)
                            except Exception:
                                pass
                        return pm

                    price_map = await asyncio.to_thread(_fetch_prices, symbols)

                    from app.config import ACCURACY_NEUTRAL_THRESHOLD
                    for log in pending_logs:
                        current_price = price_map.get(log.symbol)
                        if current_price:
                            log.price_after_30min = current_price
                            pct_move = (current_price - log.price_at_signal) / log.price_at_signal
                            threshold = _get_correctness_threshold(log.symbol) * 1.5
                            if log.direction == "BULLISH":
                                log.was_correct_30min = pct_move >= threshold
                            elif log.direction == "BEARISH":
                                log.was_correct_30min = pct_move <= -threshold
                            else:
                                log.was_correct_30min = abs(pct_move) < ACCURACY_NEUTRAL_THRESHOLD

                    db.commit()
            finally:
                db.close()
        except Exception:
            pass

        await asyncio.sleep(180)


async def signal_accuracy_validator_1hr():
    """Background task that checks signals from ~65-120 minutes ago
    and populates price_after_1hr and was_correct_1hr fields."""
    from datetime import timedelta
    from app.models import SignalLog
    from app.utils.helpers import yfinance_symbol, now_ist
    from app.utils.yahoo_api import yahoo_quote

    await asyncio.sleep(90)

    while True:
        try:
            from app.utils.helpers import market_status
            mkt = market_status()
            if mkt in ("closed_weekend", "pre_market"):
                await asyncio.sleep(300)
                continue

            db = SessionLocal()
            try:
                cutoff_start = now_ist().replace(tzinfo=None) - timedelta(minutes=120)
                cutoff_end = now_ist().replace(tzinfo=None) - timedelta(minutes=65)

                pending_logs = (
                    db.query(SignalLog)
                    .filter(
                        SignalLog.price_after_1hr.is_(None),
                        SignalLog.price_at_signal.isnot(None),
                        SignalLog.created_at >= cutoff_start,
                        SignalLog.created_at <= cutoff_end,
                    )
                    .limit(20)
                    .all()
                )

                if pending_logs:
                    symbols = list({log.symbol for log in pending_logs})

                    def _fetch_prices(syms):
                        pm = {}
                        for sym in syms:
                            try:
                                q = yahoo_quote(yfinance_symbol(sym))
                                if q and q["ltp"]:
                                    pm[sym] = round(float(q["ltp"]), 2)
                            except Exception:
                                pass
                        return pm

                    price_map = await asyncio.to_thread(_fetch_prices, symbols)

                    from app.config import ACCURACY_NEUTRAL_THRESHOLD
                    for log in pending_logs:
                        current_price = price_map.get(log.symbol)
                        if current_price:
                            log.price_after_1hr = current_price
                            pct_move = (current_price - log.price_at_signal) / log.price_at_signal
                            threshold = _get_correctness_threshold(log.symbol) * 2.0
                            if log.direction == "BULLISH":
                                log.was_correct_1hr = pct_move >= threshold
                            elif log.direction == "BEARISH":
                                log.was_correct_1hr = pct_move <= -threshold
                            else:
                                log.was_correct_1hr = abs(pct_move) < ACCURACY_NEUTRAL_THRESHOLD

                    db.commit()
            finally:
                db.close()
        except Exception:
            pass

        await asyncio.sleep(180)


