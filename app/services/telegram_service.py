"""Telegram Bot API service for sending alerts to subscribers."""

import asyncio
import logging
import os
import time
from typing import Callable, Optional

import aiohttp

from app.database import SessionLocal
from app.models import TelegramSubscription

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API_BASE = "https://api.telegram.org/bot"

# Rate limiting: Telegram allows ~30 msgs/sec globally
_rate_lock = asyncio.Lock()
_rate_tokens = 30.0
_rate_max = 30.0
_rate_last_refill = 0.0
_rate_refill_interval = 1.0  # refill every 1 second


async def _acquire_rate_token():
    """Token-bucket rate limiter: max 30 messages per second."""
    global _rate_tokens, _rate_last_refill
    async with _rate_lock:
        now = time.monotonic()
        elapsed = now - _rate_last_refill
        if elapsed >= _rate_refill_interval:
            _rate_tokens = min(_rate_max, _rate_tokens + elapsed * _rate_max)
            _rate_last_refill = now
        if _rate_tokens < 1:
            await asyncio.sleep(1.0 / _rate_max)
            _rate_tokens = min(_rate_max, _rate_tokens + 1)
        _rate_tokens -= 1


async def send_message(chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
    """Send a message to a Telegram chat. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN:
        logger.debug("Telegram bot token not configured, skipping message")
        return False

    await _acquire_rate_token()

    url = f"{TELEGRAM_API_BASE}{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                logger.warning("Telegram API error %s: %s", resp.status, body)
                return False
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)
        return False


async def send_signal_alert(chat_id: str, signal_data: dict) -> bool:
    """Send a formatted signal alert with full signal details."""
    symbol = signal_data.get("symbol", "?")
    direction = signal_data.get("direction") or "?"
    confidence = signal_data.get("confidence") or 0
    entry = signal_data.get("entry") or signal_data.get("price_at_signal") or signal_data.get("ltp", "N/A")
    stop_loss = signal_data.get("stop_loss")
    target = signal_data.get("target")
    timeframe = signal_data.get("timeframe")
    regime = signal_data.get("regime")
    volume_conviction = signal_data.get("volume_conviction")
    confidence_trend = signal_data.get("confidence_trend")
    model_confidence = signal_data.get("model_confidence")
    risk_reward = signal_data.get("risk_reward")
    mins_to_close = signal_data.get("mins_to_close")

    arrow = "\u2b06\ufe0f" if direction.upper() == "BULLISH" else "\u2b07\ufe0f" if direction.upper() == "BEARISH" else "\u27a1\ufe0f"

    lines = [
        f"{arrow} <b>Signal: {symbol}</b>",
        "",
        f"Direction: <b>{direction}</b>",
        f"Confidence: <b>{confidence:.0f}%</b>",
        f"Entry: \u20b9{entry}",
    ]
    if mins_to_close is not None:
        lines.append(f"\u23f0 Candle closes in <b>{mins_to_close} min</b> — watch for pullback to support")
    if stop_loss is not None:
        lines.append(f"Stop Loss: \u20b9{stop_loss}")
    if target is not None:
        lines.append(f"Target: \u20b9{target}")
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


async def send_price_alert(chat_id: str, alert_data: dict) -> bool:
    """Send a price alert notification."""
    symbol = alert_data.get("symbol", "?")
    condition = alert_data.get("condition", "?")
    target_price = alert_data.get("target_price", "?")
    current_price = alert_data.get("current_price") or alert_data.get("ltp", "N/A")

    text = (
        f"\U0001f514 <b>Price Alert: {symbol}</b>\n\n"
        f"Condition: Price went <b>{condition}</b> {target_price}\n"
        f"Current Price: {current_price}"
    )
    return await send_message(chat_id, text)


async def send_news_alert(chat_id: str, news_data: dict) -> bool:
    """Send a news/sentiment change alert."""
    symbol = news_data.get("symbol", "?")
    score = news_data.get("score", 0)
    change = news_data.get("change", 0)
    headline = news_data.get("top_headline", "")

    sentiment = "Positive" if score > 0 else "Negative"
    emoji = "\U0001f4c8" if score > 0 else "\U0001f4c9"

    text = (
        f"{emoji} <b>News Alert: {symbol}</b>\n\n"
        f"Sentiment: <b>{sentiment}</b> (score: {score:.1f})"
    )
    if change:
        text += f"\nChange: {'+' if change > 0 else ''}{change:.1f}"
    if headline:
        text += f"\nHeadline: {headline[:200]}"

    return await send_message(chat_id, text)


async def send_scanner_alert(chat_id: str, scanner_data: dict) -> bool:
    """Send a scanner match notification."""
    symbol = scanner_data.get("symbol", "?")
    ltp = scanner_data.get("ltp", "N/A")
    change_pct = scanner_data.get("change_pct", 0)
    filters = scanner_data.get("matched_filters", [])

    sign = "+" if change_pct >= 0 else ""
    text = (
        f"\U0001f50d <b>Scanner Alert: {symbol}</b>\n\n"
        f"Price: {ltp} ({sign}{change_pct:.2f}%)\n"
        f"Matched: {', '.join(filters) if filters else 'N/A'}"
    )
    return await send_message(chat_id, text)


async def broadcast_to_subscribers(alert_type: str, message_func: Callable, data: dict):
    """Send alert to all active subscribers of the given alert_type.

    Args:
        alert_type: One of "signals", "price_alerts", "news", "scanner", "predictions"
        message_func: Async function(chat_id, data) -> bool
        data: Data dict to pass to message_func
    """
    if not TELEGRAM_BOT_TOKEN:
        return

    def _get_subscribers():
        db = SessionLocal()
        try:
            subs = (
                db.query(TelegramSubscription)
                .filter(
                    TelegramSubscription.is_active == True,
                    TelegramSubscription.alert_types.contains(alert_type),
                )
                .all()
            )
            return [(s.chat_id,) for s in subs]
        finally:
            db.close()

    try:
        subscribers = await asyncio.to_thread(_get_subscribers)
    except Exception as e:
        logger.error("Failed to fetch Telegram subscribers: %s", e)
        return

    for (chat_id,) in subscribers:
        try:
            await message_func(chat_id, data)
        except Exception as e:
            logger.debug("Telegram broadcast error for chat %s: %s", chat_id, e)


async def get_webhook_info() -> Optional[dict]:
    """Get current webhook info from Telegram API."""
    if not TELEGRAM_BOT_TOKEN:
        return None

    url = f"{TELEGRAM_API_BASE}{TELEGRAM_BOT_TOKEN}/getWebhookInfo"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None
