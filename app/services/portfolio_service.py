from sqlalchemy.orm import Session
from app.models import PortfolioHolding
from app.services.data_fetcher import data_fetcher


class PortfolioService:
    def get_holdings(self, db: Session, user_id: int) -> list[dict]:
        holdings = (
            db.query(PortfolioHolding)
            .filter(PortfolioHolding.user_id == user_id)
            .all()
        )
        result = []
        for h in holdings:
            current_price = None
            pnl = None
            pnl_pct = None
            try:
                quote = data_fetcher.get_live_quote(h.symbol)
                current_price = quote["ltp"]
                invested = h.quantity * h.buy_price
                current_val = h.quantity * current_price
                pnl = round(current_val - invested, 2)
                pnl_pct = round((pnl / invested) * 100, 2) if invested else 0
            except Exception:
                pass

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

    def get_summary(self, db: Session, user_id: int) -> dict:
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

    def add_holding(self, db: Session, data: dict, user_id: int) -> PortfolioHolding:
        holding = PortfolioHolding(**data, user_id=user_id)
        db.add(holding)
        db.commit()
        db.refresh(holding)
        return holding

    def delete_holding(self, db: Session, holding_id: int, user_id: int) -> bool:
        holding = (
            db.query(PortfolioHolding)
            .filter(PortfolioHolding.id == holding_id, PortfolioHolding.user_id == user_id)
            .first()
        )
        if not holding:
            return False
        db.delete(holding)
        db.commit()
        return True


portfolio_service = PortfolioService()
