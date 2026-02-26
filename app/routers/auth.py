from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserResponse, Token, TokenRefresh, PasswordChange
from app.auth import (
    verify_password, get_password_hash, create_access_token,
    create_refresh_token, generate_api_key, get_current_active_user,
)
from app.config import SECRET_KEY, ALGORITHM
from jose import JWTError, jwt

router = APIRouter()


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
    if not user or not verify_password(form_data.password, user.hashed_password):
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
    if not verify_password(data.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    user.hashed_password = get_password_hash(data.new_password)
    db.commit()
    return {"message": "Password updated successfully"}


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
