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
        assert "68" in text  # model_confidence


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
