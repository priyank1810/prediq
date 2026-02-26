from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(IST)


def is_market_open() -> bool:
    now = now_ist()
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def market_status() -> str:
    now = now_ist()
    if now.weekday() >= 5:
        return "closed_weekend"
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now < market_open:
        return "pre_market"
    elif now > market_close:
        return "post_market"
    return "open"


def is_index(symbol: str) -> bool:
    from app.config import INDICES
    return symbol.upper() in INDICES or symbol.startswith("^")


def yfinance_symbol(symbol: str, exchange: str = "NSE") -> str:
    from app.config import INDICES
    upper = symbol.upper()
    # Check if it's an index — return the yfinance ticker directly
    if upper in INDICES:
        return INDICES[upper]
    # Already a yfinance symbol (^NSEI, etc.)
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    return f"{symbol}{suffix}"
