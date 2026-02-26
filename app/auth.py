from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
from app.database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db),
):
    from app.models import User

    # Try JWT Bearer token first
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Invalid token type")
            email: str = payload.get("sub")
            if email is None:
                raise HTTPException(status_code=401, detail="Invalid token")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user

    # Fall back to API key
    if api_key:
        user = db.query(User).filter(User.api_key == api_key).first()
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user

    raise HTTPException(status_code=401, detail="Not authenticated")


async def get_current_active_user(current_user=Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user account")
    return current_user


async def get_admin_user(current_user=Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
