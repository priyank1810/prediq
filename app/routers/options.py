"""Options chain API router."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from app.utils.helpers import validate_symbol

router = APIRouter()


@router.get("/{symbol}/chain")
async def get_options_chain(
    symbol: str,
    expiry: Optional[str] = Query(None, description="Expiry date (DD-Mon-YYYY)"),
):
    """Return options chain data for a symbol (calls, puts, strikes, totals)."""
    try:
        sym = validate_symbol(symbol)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol!r}")

    from app.services.options_service import options_service
    data = await asyncio.to_thread(options_service.get_chain, sym, expiry)
    return data


@router.get("/{symbol}/maxpain")
async def get_max_pain(
    symbol: str,
    expiry: Optional[str] = Query(None, description="Expiry date (DD-Mon-YYYY)"),
):
    """Return max pain calculation for a symbol's option chain."""
    try:
        sym = validate_symbol(symbol)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol!r}")

    from app.services.options_service import options_service
    data = await asyncio.to_thread(options_service.get_max_pain, sym, expiry)
    return data
