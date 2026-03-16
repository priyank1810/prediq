"""Telegram bot integration routes: link/unlink accounts, preferences, webhook."""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from app.auth import get_current_active_user
from app.database import get_db
from app.models import TelegramSubscription, User
from app.services import telegram_service

logger = logging.getLogger(__name__)
router = APIRouter()

_CHAT_ID_RE = re.compile(r"^-?\d{1,20}$")

VALID_ALERT_TYPES = {"signals", "price_alerts", "news", "scanner", "predictions"}


# --- Pydantic schemas ---

class TelegramLinkRequest(BaseModel):
    chat_id: str


class TelegramPreferencesRequest(BaseModel):
    alert_types: list[str]


# --- Routes ---

@router.post("/webhook")
async def telegram_webhook(request: Request, db: Session = Depends(get_db)):
    """Receive updates from Telegram Bot API (webhook mode).

    Handles /start command to display the user's chat ID for linking.
    """
    try:
        body = await request.json()
    except Exception:
        return {"ok": True}

    message = body.get("message", {})
    text = message.get("text", "")
    chat = message.get("chat", {})
    chat_id = str(chat.get("id", ""))

    if not chat_id:
        return {"ok": True}

    if text.strip().startswith("/start"):
        await telegram_service.send_message(
            chat_id,
            (
                "<b>Welcome to Stock Tracker Alerts!</b>\n\n"
                f"Your Chat ID is: <code>{chat_id}</code>\n\n"
                "Copy this Chat ID and paste it in the Telegram Settings "
                "section of the Stock Tracker app to link your account."
            ),
        )

    return {"ok": True}


@router.post("/link")
async def link_telegram(
    data: TelegramLinkRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Link a Telegram chat ID to the current user's account."""
    chat_id = data.chat_id.strip()
    if not _CHAT_ID_RE.match(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat ID format")

    # Check if this chat_id is already linked to another user
    existing = (
        db.query(TelegramSubscription)
        .filter(TelegramSubscription.chat_id == chat_id)
        .first()
    )
    if existing and existing.user_id != user.id:
        raise HTTPException(status_code=400, detail="This Chat ID is already linked to another account")

    # Check if user already has a subscription
    sub = (
        db.query(TelegramSubscription)
        .filter(TelegramSubscription.user_id == user.id)
        .first()
    )

    if sub:
        sub.chat_id = chat_id
        sub.is_active = True
    else:
        sub = TelegramSubscription(
            user_id=user.id,
            chat_id=chat_id,
            is_active=True,
            alert_types="signals,price_alerts,news,scanner,predictions",
        )
        db.add(sub)

    user.telegram_chat_id = chat_id
    db.commit()

    # Send a confirmation message
    await telegram_service.send_message(
        chat_id,
        "\u2705 <b>Account linked successfully!</b>\n\nYou will now receive stock alerts here.",
    )

    return {
        "status": "linked",
        "chat_id": chat_id,
        "alert_types": sub.alert_types.split(",") if sub.alert_types else [],
    }


@router.delete("/unlink")
async def unlink_telegram(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Unlink Telegram from the current user's account."""
    sub = (
        db.query(TelegramSubscription)
        .filter(TelegramSubscription.user_id == user.id)
        .first()
    )

    if sub:
        chat_id = sub.chat_id
        db.delete(sub)
        user.telegram_chat_id = None
        db.commit()

        await telegram_service.send_message(
            chat_id,
            "\u274c <b>Account unlinked.</b>\n\nYou will no longer receive alerts.",
        )
    else:
        user.telegram_chat_id = None
        db.commit()

    return {"status": "unlinked"}


@router.get("/status")
async def telegram_status(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Check if the user has Telegram linked and their preferences."""
    sub = (
        db.query(TelegramSubscription)
        .filter(TelegramSubscription.user_id == user.id)
        .first()
    )

    if not sub:
        return {
            "linked": False,
            "chat_id": None,
            "is_active": False,
            "alert_types": [],
        }

    return {
        "linked": True,
        "chat_id": sub.chat_id,
        "is_active": sub.is_active,
        "alert_types": sub.alert_types.split(",") if sub.alert_types else [],
    }


@router.put("/preferences")
async def update_preferences(
    data: TelegramPreferencesRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update which alert types the user receives on Telegram."""
    sub = (
        db.query(TelegramSubscription)
        .filter(TelegramSubscription.user_id == user.id)
        .first()
    )

    if not sub:
        raise HTTPException(status_code=404, detail="Telegram not linked. Link your account first.")

    # Validate alert types
    invalid = set(data.alert_types) - VALID_ALERT_TYPES
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid alert types: {', '.join(invalid)}. Valid types: {', '.join(sorted(VALID_ALERT_TYPES))}",
        )

    sub.alert_types = ",".join(data.alert_types) if data.alert_types else ""
    db.commit()

    return {
        "alert_types": data.alert_types,
        "message": "Preferences updated",
    }


@router.post("/test")
async def send_test_message(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Send a test message to the user's linked Telegram."""
    sub = (
        db.query(TelegramSubscription)
        .filter(TelegramSubscription.user_id == user.id)
        .first()
    )

    if not sub:
        raise HTTPException(status_code=404, detail="Telegram not linked. Link your account first.")

    success = await telegram_service.send_message(
        sub.chat_id,
        "\U0001f6a8 <b>Test Alert</b>\n\n"
        "This is a test notification from Stock Tracker.\n"
        "If you see this, your Telegram alerts are working!",
    )

    if not success:
        raise HTTPException(status_code=502, detail="Failed to send test message. Check bot token configuration.")

    return {"message": "Test message sent successfully"}
