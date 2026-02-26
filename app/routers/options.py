from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from app.services.nse_option_chain import nse_option_chain, OptionChainError
from app.utils.helpers import market_status

router = APIRouter()


@router.get("/{symbol}/expiries")
def get_expiries(symbol: str):
    """Get available option expiry dates for a symbol."""
    try:
        dates = nse_option_chain.get_expiry_dates(symbol.upper())
        if not dates:
            status = market_status()
            if status in ("post_market", "closed_weekend"):
                msg = "No expiry dates available. Markets are currently closed."
            elif status == "pre_market":
                msg = "No expiry dates available yet. Markets open at 9:15 AM IST."
            else:
                msg = "No expiry dates found. NSE may be temporarily unreachable."
            raise HTTPException(status_code=404, detail=msg)
        return dates
    except HTTPException:
        raise
    except OptionChainError as e:
        return JSONResponse(
            status_code=503,
            content={"detail": str(e), "error_type": e.error_type},
            headers={"X-Error-Type": e.error_type},
        )
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
    except OptionChainError as e:
        return JSONResponse(
            status_code=503,
            content={"detail": str(e), "error_type": e.error_type},
            headers={"X-Error-Type": e.error_type},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch option chain: {e}")
