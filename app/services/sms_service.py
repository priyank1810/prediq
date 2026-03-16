"""Twilio SMS service for sending alerts to subscribers."""

import asyncio
import logging
import os
import time
from typing import Callable

from app.database import SessionLocal
from app.models import SMSSubscription

logger = logging.getLogger(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")

# Rate limiting: max 1 SMS per symbol per user per 15 minutes
_rate_limit_cache: dict[str, float] = {}
_RATE_LIMIT_SECONDS = 900  # 15 minutes


def _check_rate_limit(user_id: int, symbol: str) -> bool:
    """Return True if sending is allowed, False if rate-limited."""
    key = f"{user_id}:{symbol}"
    now = time.monotonic()
    last_sent = _rate_limit_cache.get(key)
    if last_sent and (now - last_sent) < _RATE_LIMIT_SECONDS:
        return False
    _rate_limit_cache[key] = now
    # Periodically prune stale entries
    if len(_rate_limit_cache) > 5000:
        cutoff = now - _RATE_LIMIT_SECONDS
        stale = [k for k, v in _rate_limit_cache.items() if v < cutoff]
        for k in stale:
            del _rate_limit_cache[k]
    return True


async def send_sms(to_number: str, message: str) -> bool:
    """Send a single SMS via Twilio. Returns True on success."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_FROM_NUMBER:
        logger.debug("Twilio not configured, skipping SMS")
        return False

    def _send():
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            msg = client.messages.create(
                body=message,
                from_=TWILIO_FROM_NUMBER,
                to=to_number,
            )
            logger.info("SMS sent to %s, SID: %s", to_number, msg.sid)
            return True
        except Exception as e:
            logger.error("Failed to send SMS to %s: %s", to_number, e)
            return False

    return await asyncio.to_thread(_send)


async def send_signal_alert(phone: str, signal_data: dict) -> bool:
    """Send a formatted signal alert via SMS."""
    symbol = signal_data.get("symbol", "?")
    direction = signal_data.get("direction", "?").upper()
    confidence = signal_data.get("confidence", 0)
    price = signal_data.get("price_at_signal") or signal_data.get("ltp", "N/A")
    entry = signal_data.get("entry", price)
    stop_loss = signal_data.get("stop_loss", "N/A")
    target = signal_data.get("target", "N/A")

    text = (
        f"\U0001f4ca {symbol}: {direction} ({confidence:.0f}%) "
        f"| Entry \u20b9{entry} | SL \u20b9{stop_loss} | Target \u20b9{target}"
    )
    return await send_sms(phone, text)


async def send_price_alert(phone: str, alert_data: dict) -> bool:
    """Send a price alert notification via SMS."""
    symbol = alert_data.get("symbol", "?")
    condition = alert_data.get("condition", "?")
    target_price = alert_data.get("target_price", "?")

    text = f"\U0001f514 {symbol} crossed \u20b9{target_price} ({condition} target)"
    return await send_sms(phone, text)


async def send_news_alert(phone: str, news_data: dict) -> bool:
    """Send a news/sentiment change alert via SMS."""
    symbol = news_data.get("symbol", "?")
    score = news_data.get("score", 0)
    change = news_data.get("change", 0)

    change_str = f"{'+' if change > 0 else ''}{change:.0f}" if change else ""
    text = f"\U0001f4f0 {symbol} sentiment shifted {change_str} (now {score:.0f})"
    return await send_sms(phone, text)


async def send_scanner_alert(phone: str, scanner_data: dict) -> bool:
    """Send a scanner match notification via SMS."""
    symbol = scanner_data.get("symbol", "?")
    filters = scanner_data.get("matched_filters", [])
    filters_str = ", ".join(filters) if filters else "N/A"

    text = f"\U0001f50d {symbol} matched: {filters_str}"
    return await send_sms(phone, text)


async def broadcast_to_subscribers(alert_type: str, message_func: Callable, data: dict):
    """Send alert to all active SMS subscribers of the given alert_type.

    Args:
        alert_type: One of "signals", "price_alerts", "news", "scanner"
        message_func: Async function(phone, data) -> bool
        data: Data dict to pass to message_func
    """
    if not TWILIO_ACCOUNT_SID:
        return

    symbol = data.get("symbol", "")

    def _get_subscribers():
        db = SessionLocal()
        try:
            subs = (
                db.query(SMSSubscription)
                .filter(
                    SMSSubscription.is_active == True,
                    SMSSubscription.alert_types.contains(alert_type),
                )
                .all()
            )
            return [(s.phone_number, s.user_id) for s in subs]
        finally:
            db.close()

    try:
        subscribers = await asyncio.to_thread(_get_subscribers)
    except Exception as e:
        logger.error("Failed to fetch SMS subscribers: %s", e)
        return

    for phone, user_id in subscribers:
        # Apply per-symbol per-user rate limit
        if symbol and not _check_rate_limit(user_id, symbol):
            logger.debug("SMS rate-limited for user %s symbol %s", user_id, symbol)
            continue
        try:
            await message_func(phone, data)
        except Exception as e:
            logger.debug("SMS broadcast error for %s: %s", phone, e)
