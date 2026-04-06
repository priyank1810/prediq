import math
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sqlalchemy.orm import Session

from app.models import PortfolioHolding
from app.services.data_fetcher import data_fetcher

# Simple sector mapping for common Indian stocks.
# Stocks not found here default to "Other".
SECTOR_MAP = {
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "MPHASIS": "IT", "COFORGE": "IT", "PERSISTENT": "IT",
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "KOTAKBANK": "Banking",
    "SBIN": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BANDHANBNK": "Banking", "FEDERALBNK": "Banking", "PNB": "Banking",
    "BANKBARODA": "Banking", "IDFCFIRSTB": "Banking", "AUBANK": "Banking",
    # NBFC / Financial Services
    "BAJFINANCE": "Financial Services", "BAJAJFINSV": "Financial Services",
    "HDFC": "Financial Services", "SHRIRAMFIN": "Financial Services",
    "MUTHOOTFIN": "Financial Services", "CHOLAFIN": "Financial Services",
    # Energy / Oil & Gas
    "RELIANCE": "Energy", "ONGC": "Energy", "IOC": "Energy", "BPCL": "Energy",
    "GAIL": "Energy", "NTPC": "Energy", "POWERGRID": "Energy",
    "ADANIGREEN": "Energy", "ADANIENT": "Energy", "TATAPOWER": "Energy",
    # Pharma / Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "APOLLOHOSP": "Pharma", "BIOCON": "Pharma",
    "LUPIN": "Pharma", "AUROPHARMA": "Pharma", "TORNTPHARM": "Pharma",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "COLPAL": "FMCG", "GODREJCP": "FMCG", "TATACONSUM": "FMCG",
    # Auto
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "ASHOKLEY": "Auto", "TVSMOTOR": "Auto",
    # Metals & Mining
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "VEDL": "Metals", "COALINDIA": "Metals", "NMDC": "Metals",
    "SAIL": "Metals",
    # Cement / Construction
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "SHREECEM": "Cement",
    "AMBUJACEM": "Cement", "ACC": "Cement", "DALBHARAT": "Cement",
    "RAMCOCEM": "Cement",
    # Telecom
    "BHARTIARTL": "Telecom", "IDEA": "Telecom",
    # Infrastructure
    "LT": "Infrastructure", "ADANIPORTS": "Infrastructure",
    # Insurance
    "SBILIFE": "Insurance", "HDFCLIFE": "Insurance", "ICICIPRULI": "Insurance",
    "LICI": "Insurance",
}

RISK_FREE_RATE = 0.06  # 6% Indian T-bill rate


class PortfolioService:
    def get_holdings(self, db: Session, user_id: int = None, limit: int = 100, offset: int = 0) -> list[dict]:
        query = db.query(PortfolioHolding)
        if user_id is not None:
            query = query.filter(PortfolioHolding.user_id == user_id)
        holdings = query.order_by(PortfolioHolding.buy_date.desc()).offset(offset).limit(limit).all()
        if not holdings:
            return []

        # Batch-fetch all quotes in one call
        symbols = list({h.symbol for h in holdings})
        quotes = data_fetcher.get_bulk_quotes(symbols)
        price_map = {q["symbol"]: q.get("ltp") for q in quotes if q.get("ltp")}

        result = []
        for h in holdings:
            current_price = price_map.get(h.symbol)
            pnl = None
            pnl_pct = None
            if current_price:
                invested = h.quantity * h.buy_price
                current_val = h.quantity * current_price
                pnl = round(current_val - invested, 2)
                pnl_pct = round((pnl / invested) * 100, 2) if invested else 0

            result.append({
                "id": h.id,
                "symbol": h.symbol,
                "exchange": h.exchange,
                "quantity": h.quantity,
                "buy_price": h.buy_price,
                "buy_date": h.buy_date,
                "notes": h.notes,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })
        return result

    def get_summary(self, db: Session, user_id: int = None) -> dict:
        holdings = self.get_holdings(db, user_id=user_id)
        total_invested = sum(h["quantity"] * h["buy_price"] for h in holdings)
        current_value = sum(
            h["quantity"] * h["current_price"] for h in holdings if h["current_price"]
        )
        total_pnl = current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0

        return {
            "total_invested": round(total_invested, 2),
            "current_value": round(current_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "holdings_count": len(holdings),
        }

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_analytics(self, db: Session, user_id: int = None) -> dict:
        """Compute portfolio analytics: sector allocation, CAGR, Sharpe ratio, max drawdown."""
        holdings = self.get_holdings(db, user_id=user_id, limit=500)
        if not holdings:
            return {
                "sector_allocation": [],
                "cagr": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
            }

        # --- Sector allocation ---
        sector_values: dict[str, float] = {}
        total_current = 0.0
        for h in holdings:
            cp = h.get("current_price")
            if not cp:
                continue
            val = h["quantity"] * cp
            sector = SECTOR_MAP.get(h["symbol"], "Other")
            sector_values[sector] = sector_values.get(sector, 0) + val
            total_current += val

        sector_allocation = []
        if total_current > 0:
            for sector, value in sorted(sector_values.items(), key=lambda x: -x[1]):
                sector_allocation.append({
                    "sector": sector,
                    "value": round(value, 2),
                    "percentage": round(value / total_current * 100, 2),
                })

        # --- CAGR ---
        total_invested = sum(h["quantity"] * h["buy_price"] for h in holdings)
        earliest_buy = None
        for h in holdings:
            bd = h.get("buy_date")
            if bd:
                if isinstance(bd, str):
                    bd = datetime.strptime(bd, "%Y-%m-%d").date()
                if earliest_buy is None or bd < earliest_buy:
                    earliest_buy = bd

        cagr = None
        if total_invested > 0 and total_current > 0 and earliest_buy:
            years = (date.today() - earliest_buy).days / 365.25
            if years >= 0.01:  # at least ~4 days
                cagr = round(((total_current / total_invested) ** (1 / years) - 1) * 100, 2)

        # --- Fetch historical data for Sharpe and drawdown ---
        symbols_with_weight: dict[str, float] = {}
        for h in holdings:
            cp = h.get("current_price")
            if cp and total_current > 0:
                weight = (h["quantity"] * cp) / total_current
                symbols_with_weight[h["symbol"]] = (
                    symbols_with_weight.get(h["symbol"], 0) + weight
                )

        # Fetch 1-year daily history for each unique symbol (parallel)
        symbol_returns: dict[str, np.ndarray] = {}

        def _fetch_returns(sym: str):
            try:
                df = data_fetcher.get_historical_data(sym, period="1y")
                if df is not None and len(df) > 1:
                    close = df["Close"].values
                    daily_ret = np.diff(close) / close[:-1]
                    return sym, daily_ret
            except Exception:
                pass
            return sym, None

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_fetch_returns, s): s for s in symbols_with_weight}
            for fut in as_completed(futures):
                sym, ret = fut.result()
                if ret is not None and len(ret) > 0:
                    symbol_returns[sym] = ret

        # --- Compute weighted portfolio daily returns ---
        sharpe_ratio = None
        max_drawdown = None

        if symbol_returns:
            # Find the common length (shortest series)
            min_len = min(len(r) for r in symbol_returns.values())
            if min_len > 5:
                # Build weighted portfolio return series
                portfolio_returns = np.zeros(min_len)
                total_weight_used = 0.0
                for sym, weight in symbols_with_weight.items():
                    if sym in symbol_returns:
                        # Use the most recent min_len days
                        ret = symbol_returns[sym][-min_len:]
                        portfolio_returns += ret * weight
                        total_weight_used += weight

                # Re-normalise if not all symbols had data
                if total_weight_used > 0 and total_weight_used < 0.99:
                    portfolio_returns /= total_weight_used

                # --- Sharpe ratio (annualised) ---
                daily_rf = RISK_FREE_RATE / 252
                excess = portfolio_returns - daily_rf
                std = np.std(excess, ddof=1)
                if std > 0:
                    sharpe_ratio = round(
                        float(np.mean(excess) / std * math.sqrt(252)), 2
                    )

                # --- Max drawdown ---
                cumulative = np.cumprod(1 + portfolio_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = round(float(np.min(drawdowns)) * 100, 2)  # negative %

        return {
            "sector_allocation": sector_allocation,
            "cagr": cagr,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

    def add_holding(self, db: Session, data: dict, user_id: int = None) -> PortfolioHolding:
        holding = PortfolioHolding(**data, user_id=user_id)
        db.add(holding)
        db.commit()
        db.refresh(holding)
        return holding

    def delete_holding(self, db: Session, holding_id: int, user_id: int = None) -> bool:
        query = db.query(PortfolioHolding).filter(PortfolioHolding.id == holding_id)
        if user_id is not None:
            query = query.filter(PortfolioHolding.user_id == user_id)
        holding = query.first()
        if not holding:
            return False
        db.delete(holding)
        db.commit()
        return True


portfolio_service = PortfolioService()
