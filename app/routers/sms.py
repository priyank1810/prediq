"""SMS alert integration routes: link/unlink phone, OTP verification, preferences."""

from __future__ import annotations

import logging
import random
import re
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_active_user
from app.database import get_db
from app.models import SMSSubscription, User
from app.services import sms_service

logger = logging.getLogger(__name__)
router = APIRouter()

_PHONE_RE = re.compile(r"^\+91\d{10}$")

VALID_ALERT_TYPES = {"signals", "price_alerts", "news", "scanner"}

# In-memory OTP store: user_id -> {"code": str, "phone": str, "expires": float}
_otp_store: dict[int, dict] = {}
_OTP_EXPIRY_SECONDS = 300  # 5 minutes


# --- Pydantic schemas ---

class SMSLinkRequest(BaseModel):
    phone_number: str


class SMSVerifyRequest(BaseModel):
    phone_number: str
    otp: str


class SMSPreferencesRequest(BaseModel):
    alert_types: list[str]


# --- Routes ---

@router.post("/link")
async def link_sms(
    data: SMSLinkRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Initiate phone linking by sending an OTP to the given number."""
    phone = data.phone_number.strip()
    if not _PHONE_RE.match(phone):
        raise HTTPException(status_code=400, detail="Invalid phone number. Use +91XXXXXXXXXX format.")

    # Generate 6-digit OTP
    otp_code = f"{random.randint(100000, 999999)}"
    _otp_store[user.id] = {
        "code": otp_code,
        "phone": phone,
        "expires": time.time() + _OTP_EXPIRY_SECONDS,
    }

    # Send OTP via SMS
    success = await sms_service.send_sms(
        phone,
        f"Your Stock Tracker verification code is: {otp_code}. Valid for 5 minutes.",
    )

    if not success:
        raise HTTPException(status_code=502, detail="Failed to send OTP. Check Twilio configuration.")

    return {"message": "OTP sent successfully", "phone_number": phone}


@router.post("/verify")
async def verify_otp(
    data: SMSVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Verify OTP and complete phone linking."""
    phone = data.phone_number.strip()
    otp = data.otp.strip()

    stored = _otp_store.get(user.id)
    if not stored:
        raise HTTPException(status_code=400, detail="No OTP requested. Send OTP first.")

    if time.time() > stored["expires"]:
        del _otp_store[user.id]
        raise HTTPException(status_code=400, detail="OTP expired. Please request a new one.")

    if stored["phone"] != phone:
        raise HTTPException(status_code=400, detail="Phone number does not match OTP request.")

    if stored["code"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP.")

    # OTP verified — clean up
    del _otp_store[user.id]

    # Check if this phone is already linked to another user
    existing = (
        db.query(SMSSubscription)
        .filter(SMSSubscription.phone_number == phone)
        .first()
    )
    if existing and existing.user_id != user.id:
        raise HTTPException(status_code=400, detail="This phone number is already linked to another account.")

    # Create or update subscription
    sub = (
        db.query(SMSSubscription)
        .filter(SMSSubscription.user_id == user.id)
        .first()
    )

    if sub:
        sub.phone_number = phone
        sub.is_active = True
    else:
        sub = SMSSubscription(
            user_id=user.id,
            phone_number=phone,
            is_active=True,
            alert_types="signals,price_alerts,news,scanner",
        )
        db.add(sub)

    user.sms_phone = phone
    db.commit()

    # Send confirmation
    await sms_service.send_sms(phone, "Your phone is now linked to Stock Tracker alerts!")

    return {
        "status": "linked",
        "phone_number": phone,
        "alert_types": sub.alert_types.split(",") if sub.alert_types else [],
    }


@router.delete("/unlink")
async def unlink_sms(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Unlink SMS from the current user's account."""
    sub = (
        db.query(SMSSubscription)
        .filter(SMSSubscription.user_id == user.id)
        .first()
    )

    if sub:
        phone = sub.phone_number
        db.delete(sub)
        user.sms_phone = None
        db.commit()

        await sms_service.send_sms(phone, "Your phone has been unlinked from Stock Tracker. You will no longer receive SMS alerts.")
    else:
        user.sms_phone = None
        db.commit()

    return {"status": "unlinked"}


@router.get("/status")
async def sms_status(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Check if the user has SMS linked and their preferences."""
    sub = (
        db.query(SMSSubscription)
        .filter(SMSSubscription.user_id == user.id)
        .first()
    )

    if not sub:
        return {
            "linked": False,
            "phone_number": None,
            "is_active": False,
            "alert_types": [],
        }

    return {
        "linked": True,
        "phone_number": sub.phone_number,
        "is_active": sub.is_active,
        "alert_types": sub.alert_types.split(",") if sub.alert_types else [],
    }


@router.put("/preferences")
async def update_preferences(
    data: SMSPreferencesRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update which alert types the user receives via SMS."""
    sub = (
        db.query(SMSSubscription)
        .filter(SMSSubscription.user_id == user.id)
        .first()
    )

    if not sub:
        raise HTTPException(status_code=404, detail="SMS not linked. Link your phone first.")

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
async def send_test_sms(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Send a test SMS to the user's linked phone."""
    sub = (
        db.query(SMSSubscription)
        .filter(SMSSubscription.user_id == user.id)
        .first()
    )

    if not sub:
        raise HTTPException(status_code=404, detail="SMS not linked. Link your phone first.")

    success = await sms_service.send_sms(
        sub.phone_number,
        "This is a test alert from Stock Tracker. If you see this, your SMS alerts are working!",
    )

    if not success:
        raise HTTPException(status_code=502, detail="Failed to send test SMS. Check Twilio configuration.")

    return {"message": "Test SMS sent successfully"}
