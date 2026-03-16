from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserResponse, Token, TokenRefresh, PasswordChange
from app.auth import (
    verify_password, get_password_hash, create_access_token,
    create_refresh_token, generate_api_key, get_current_active_user,
)
from app.config import SECRET_KEY, ALGORITHM, GOOGLE_CLIENT_ID
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

router = APIRouter()


class GoogleAuthRequest(BaseModel):
    credential: str


@router.post("/register", response_model=UserResponse)
def register(data: UserCreate, db: Session = Depends(get_db)):
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    existing = db.query(User).filter(User.email == data.email.lower().strip()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # First user becomes admin
    user_count = db.query(User).count()
    role = "admin" if user_count == 0 else "user"

    user = User(
        email=data.email.lower().strip(),
        hashed_password=get_password_hash(data.password),
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Assign orphan data to first admin user
    if role == "admin":
        _assign_orphan_data(db, user.id)

    return user


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username.lower().strip()).first()
    if not user or not user.hashed_password or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    access_token = create_access_token(data={"sub": user.email})
    refresh_token = create_refresh_token(data={"sub": user.email})

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=Token)
def refresh_token(data: TokenRefresh, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(data.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or disabled")

    access_token = create_access_token(data={"sub": user.email})
    new_refresh = create_refresh_token(data={"sub": user.email})

    return Token(access_token=access_token, refresh_token=new_refresh)


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_active_user)):
    return user


@router.post("/api-key")
def regenerate_api_key(
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    new_key = generate_api_key()
    user.api_key = new_key
    db.commit()
    return {"api_key": new_key}


@router.put("/password")
def change_password(
    data: PasswordChange,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    if not user.hashed_password:
        raise HTTPException(status_code=400, detail="Google accounts cannot change password here")
    if not verify_password(data.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    user.hashed_password = get_password_hash(data.new_password)
    db.commit()
    return {"message": "Password updated successfully"}


@router.post("/google", response_model=Token)
def google_auth(data: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticate or register a user via Google ID token."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")

    # Verify the Google ID token
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        id_info = google_id_token.verify_oauth2_token(
            data.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ImportError:
        # Fallback: verify via Google's tokeninfo endpoint using stdlib
        import json
        import urllib.request
        import urllib.parse
        import urllib.error

        token_url = "https://oauth2.googleapis.com/tokeninfo?" + urllib.parse.urlencode(
            {"id_token": data.credential}
        )
        try:
            with urllib.request.urlopen(token_url, timeout=10) as resp:
                id_info = json.loads(resp.read().decode())
        except urllib.error.HTTPError:
            raise HTTPException(status_code=401, detail="Invalid Google token")
        if id_info.get("aud") != GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=401, detail="Token audience mismatch")
    except ValueError as exc:
        logger.warning("Google token verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid Google token")

    google_id = id_info.get("sub")
    email = id_info.get("email", "").lower().strip()
    name = id_info.get("name", "")
    picture = id_info.get("picture", "")

    if not email or not google_id:
        raise HTTPException(status_code=400, detail="Google token missing email or user ID")

    # 1. User with this google_id already exists -> log in
    user = db.query(User).filter(User.google_id == google_id).first()

    if not user:
        # 2. User with same email exists (local account) -> link Google ID
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.google_id = google_id
            user.avatar_url = picture or user.avatar_url
            if user.auth_provider == "local":
                user.auth_provider = "local"  # keep as local since they had a password
            db.commit()
        else:
            # 3. New user -> create account
            user_count = db.query(User).count()
            role = "admin" if user_count == 0 else "user"

            user = User(
                email=email,
                hashed_password=None,
                google_id=google_id,
                avatar_url=picture,
                auth_provider="google",
                role=role,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            if role == "admin":
                _assign_orphan_data(db, user.id)

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    # Update avatar if changed
    if picture and user.avatar_url != picture:
        user.avatar_url = picture
        db.commit()

    access_token = create_access_token(data={"sub": user.email})
    refresh_token = create_refresh_token(data={"sub": user.email})

    return Token(access_token=access_token, refresh_token=refresh_token)


def _assign_orphan_data(db: Session, admin_id: int):
    """Assign existing data with no user_id to the first admin."""
    from app.models import PortfolioHolding, PriceAlert, WatchlistItem

    db.query(PortfolioHolding).filter(
        PortfolioHolding.user_id.is_(None)
    ).update({"user_id": admin_id}, synchronize_session=False)

    db.query(PriceAlert).filter(
        PriceAlert.user_id.is_(None)
    ).update({"user_id": admin_id}, synchronize_session=False)

    db.query(WatchlistItem).filter(
        WatchlistItem.user_id.is_(None)
    ).update({"user_id": admin_id}, synchronize_session=False)

    db.commit()
