from fastapi import APIRouter, HTTPException, Query
from app.services.nse_option_chain import nse_option_chain

router = APIRouter()


@router.get("/{symbol}/expiries")
def get_expiries(symbol: str):
    """Get available option expiry dates for a symbol."""
    try:
        dates = nse_option_chain.get_expiry_dates(symbol.upper())
        return dates
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch expiries: {e}")


@router.get("/{symbol}/chain")
def get_option_chain(symbol: str, expiry: str = Query(None)):
    """Get full option chain for a symbol and expiry date."""
    try:
        data = nse_option_chain.get_option_chain(symbol.upper(), expiry)
        return data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch option chain: {e}")
