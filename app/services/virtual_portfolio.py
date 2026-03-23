"""Virtual Portfolio — simulates ₹10,000 investment following AI trade signals.

Reads resolved trades from TradeSignalLog and computes:
- How capital would be allocated across signals
- Daily P&L from resolved trades
- Equity curve over time
- Current open positions
"""

import logging
from datetime import datetime, timedelta
from collections import defaultdict

from app.database import SessionLocal
from app.models import TradeSignalLog
from app.utils.helpers import now_ist

logger = logging.getLogger(__name__)

DEFAULT_CAPITAL = 10000


class VirtualPortfolio:

    def get_portfolio(self, capital: float = DEFAULT_CAPITAL) -> dict:
        """Compute virtual portfolio performance from trade history."""
        db = SessionLocal()
        try:
            # Get all resolved trades
            resolved = (
                db.query(TradeSignalLog)
                .filter(TradeSignalLog.status != "open")
                .order_by(TradeSignalLog.created_at)
                .all()
            )

            # Get open trades
            open_trades = (
                db.query(TradeSignalLog)
                .filter(TradeSignalLog.status == "open")
                .order_by(TradeSignalLog.created_at)
                .all()
            )

            if not resolved and not open_trades:
                return {
                    "initial_capital": capital,
                    "current_value": capital,
                    "total_pnl": 0,
                    "total_pnl_pct": 0,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "daily_pnl": [],
                    "equity_curve": [],
                    "open_positions": [],
                    "recent_trades": [],
                    "best_trade": None,
                    "worst_trade": None,
                }

            # Simulate: equal allocation per trade
            # Each trade gets capital / max_concurrent_trades
            max_concurrent = 5  # max 5 positions at a time
            per_trade = capital / max_concurrent

            # Track equity over time
            equity = capital
            equity_curve = [{"date": None, "value": capital}]
            daily_pnl = defaultdict(float)
            trades_detail = []
            winning = 0
            losing = 0
            best_trade = None
            worst_trade = None

            for trade in resolved:
                if not trade.entry or trade.entry <= 0:
                    continue

                # Only trade bullish signals (long-only)
                if trade.direction != "BULLISH":
                    continue

                # Calculate qty based on per_trade allocation
                qty = int(per_trade / trade.entry)
                if qty <= 0:
                    qty = 1

                invested = qty * trade.entry
                outcome_price = trade.outcome_price or trade.entry
                current_val = qty * outcome_price
                pnl = current_val - invested
                pnl_pct = round((pnl / invested) * 100, 2) if invested > 0 else 0

                equity += pnl

                # Track daily P&L
                trade_date = trade.resolved_at.strftime("%Y-%m-%d") if trade.resolved_at else \
                    trade.created_at.strftime("%Y-%m-%d") if trade.created_at else "unknown"
                daily_pnl[trade_date] += pnl

                # Track equity curve
                equity_curve.append({
                    "date": trade_date,
                    "value": round(equity, 2),
                })

                trade_info = {
                    "symbol": trade.symbol,
                    "timeframe": trade.timeframe,
                    "direction": trade.direction,
                    "qty": qty,
                    "entry": trade.entry,
                    "exit_price": outcome_price,
                    "invested": round(invested, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": pnl_pct,
                    "status": trade.status,
                    "date": trade_date,
                }
                trades_detail.append(trade_info)

                if pnl > 0:
                    winning += 1
                else:
                    losing += 1

                if best_trade is None or pnl > best_trade["pnl"]:
                    best_trade = trade_info
                if worst_trade is None or pnl < worst_trade["pnl"]:
                    worst_trade = trade_info

            # Open positions
            open_positions = []
            for trade in open_trades:
                if trade.direction != "BULLISH" or not trade.entry or trade.entry <= 0:
                    continue
                qty = int(per_trade / trade.entry)
                if qty <= 0:
                    qty = 1
                open_positions.append({
                    "symbol": trade.symbol,
                    "timeframe": trade.timeframe,
                    "qty": qty,
                    "entry": trade.entry,
                    "target": trade.target,
                    "stop_loss": trade.stop_loss,
                    "invested": round(qty * trade.entry, 2),
                    "created_at": trade.created_at.isoformat() if trade.created_at else None,
                })

            total_pnl = round(equity - capital, 2)
            total_trades = winning + losing

            # Daily P&L sorted
            daily_pnl_list = [
                {"date": d, "pnl": round(p, 2)}
                for d, p in sorted(daily_pnl.items())
            ]

            return {
                "initial_capital": capital,
                "current_value": round(equity, 2),
                "total_pnl": total_pnl,
                "total_pnl_pct": round(total_pnl / capital * 100, 2) if capital > 0 else 0,
                "total_trades": total_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": round(winning / total_trades * 100, 1) if total_trades > 0 else 0,
                "per_trade_allocation": round(per_trade, 2),
                "daily_pnl": daily_pnl_list,
                "equity_curve": equity_curve,
                "open_positions": open_positions[:10],
                "recent_trades": trades_detail[-15:],
                "best_trade": best_trade,
                "worst_trade": worst_trade,
            }
        finally:
            db.close()


virtual_portfolio = VirtualPortfolio()
