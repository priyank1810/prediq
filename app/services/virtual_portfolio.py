"""Smart Virtual Portfolio — simulates investment following AI trade signals.

Smart allocation rules:
1. Higher confidence → more capital (1x at 40%, 2x at 80%+)
2. Better timeframe win rate → more capital
3. Stock track record → proven stocks get priority
4. Skip low confidence (<40%)
5. Risk management — max 30% per stock, reduce after losses
"""

import logging
from collections import defaultdict

from app.database import SessionLocal
from app.models import TradeSignalLog
from app.utils.helpers import now_ist

logger = logging.getLogger(__name__)

DEFAULT_CAPITAL = 100000

# Smart allocation config
MIN_CONFIDENCE = 10          # Accept any bullish signal (rare in bear markets)
MAX_STOCK_PCT = 0.30         # Max 30% of capital in one stock
MAX_CONCURRENT = 10          # Max 10 positions at a time
LOSS_STREAK_REDUCE = 0.5     # Halve position after 2 consecutive losses


class VirtualPortfolio:

    def _compute_allocation(self, trade, capital, stock_history, streak):
        """Smart position sizing based on confidence, timeframe, and track record."""
        base = capital / MAX_CONCURRENT

        # 1. Confidence multiplier: 40%→0.5x, 60%→1x, 80%→1.5x, 95%→2x
        conf = trade.confidence or 50
        conf_mult = max(0.5, min(2.0, (conf - 20) / 40))

        # 2. Timeframe multiplier: longer timeframes get slightly more
        tf_mults = {
            "intraday_10m": 0.7, "intraday_30m": 0.85,
            "short_15m": 0.9, "short_1h": 1.1, "short_4h": 1.3,
        }
        tf_mult = tf_mults.get(trade.timeframe, 1.0)

        # 3. Stock track record: if stock has >60% win rate historically, boost
        history = stock_history.get(trade.symbol, {"wins": 0, "total": 0})
        if history["total"] >= 3:
            win_rate = history["wins"] / history["total"]
            track_mult = 0.7 + win_rate * 0.6  # 0% → 0.7x, 50% → 1.0x, 100% → 1.3x
        else:
            track_mult = 0.8  # unknown stock, be conservative

        # 4. Loss streak reduction
        streak_mult = LOSS_STREAK_REDUCE if streak >= 2 else 1.0

        allocation = base * conf_mult * tf_mult * track_mult * streak_mult

        # 5. Cap at MAX_STOCK_PCT of capital
        allocation = min(allocation, capital * MAX_STOCK_PCT)

        # Floor
        allocation = max(allocation, capital * 0.05)  # at least 5%

        return round(allocation, 2)

    def get_portfolio(self, capital: float = DEFAULT_CAPITAL) -> dict:
        db = SessionLocal()
        try:
            resolved = (
                db.query(TradeSignalLog)
                .filter(TradeSignalLog.status != "open")
                .order_by(TradeSignalLog.created_at)
                .all()
            )
            open_trades = (
                db.query(TradeSignalLog)
                .filter(TradeSignalLog.status == "open")
                .order_by(TradeSignalLog.created_at)
                .all()
            )

            empty = {
                "initial_capital": capital, "current_value": capital,
                "total_pnl": 0, "total_pnl_pct": 0, "total_trades": 0,
                "winning_trades": 0, "losing_trades": 0, "win_rate": 0,
                "daily_pnl": [], "equity_curve": [], "open_positions": [],
                "recent_trades": [], "best_trade": None, "worst_trade": None,
                "strategy": "smart", "skipped_signals": 0,
            }

            if not resolved and not open_trades:
                return empty

            # Build per-stock history for track record multiplier
            stock_history = defaultdict(lambda: {"wins": 0, "total": 0})

            equity = capital
            equity_curve = [{"date": "start", "value": capital}]
            daily_pnl = defaultdict(float)
            trades_detail = []
            winning = 0
            losing = 0
            best_trade = None
            worst_trade = None
            skipped = 0
            loss_streak = 0
            stock_exposure = defaultdict(float)  # current exposure per stock

            for trade in resolved:
                if not trade.entry or trade.entry <= 0:
                    continue

                # Only bullish (long-only)
                if trade.direction != "BULLISH":
                    continue

                # Skip low confidence
                if (trade.confidence or 0) < MIN_CONFIDENCE:
                    skipped += 1
                    continue

                # Smart allocation
                allocation = self._compute_allocation(
                    trade, equity, stock_history, loss_streak
                )

                # Check stock exposure cap
                current_exp = stock_exposure.get(trade.symbol, 0)
                if current_exp + allocation > equity * MAX_STOCK_PCT:
                    allocation = max(0, equity * MAX_STOCK_PCT - current_exp)
                    if allocation < equity * 0.03:  # too small, skip
                        skipped += 1
                        continue

                qty = int(allocation / trade.entry)
                if qty <= 0:
                    qty = 1

                invested = qty * trade.entry
                outcome_price = trade.outcome_price or trade.entry
                current_val = qty * outcome_price
                pnl = current_val - invested
                pnl_pct = round((pnl / invested) * 100, 2) if invested > 0 else 0

                equity += pnl

                # Update stock history
                stock_history[trade.symbol]["total"] += 1
                if pnl > 0:
                    stock_history[trade.symbol]["wins"] += 1
                    winning += 1
                    loss_streak = 0
                else:
                    losing += 1
                    loss_streak += 1

                # Track exposure
                stock_exposure[trade.symbol] = stock_exposure.get(trade.symbol, 0) + invested

                trade_date = (trade.resolved_at or trade.created_at).strftime("%Y-%m-%d") \
                    if (trade.resolved_at or trade.created_at) else "unknown"

                daily_pnl[trade_date] += pnl
                equity_curve.append({"date": trade_date, "value": round(equity, 2)})

                trade_info = {
                    "symbol": trade.symbol,
                    "timeframe": trade.timeframe,
                    "direction": trade.direction,
                    "qty": qty,
                    "entry": trade.entry,
                    "exit_price": outcome_price,
                    "invested": round(invested, 2),
                    "allocation": round(allocation, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": pnl_pct,
                    "status": trade.status,
                    "confidence": trade.confidence,
                    "date": trade_date,
                }
                trades_detail.append(trade_info)

                if best_trade is None or pnl > best_trade["pnl"]:
                    best_trade = trade_info
                if worst_trade is None or pnl < worst_trade["pnl"]:
                    worst_trade = trade_info

            # Open positions with smart allocation + live P&L
            open_positions = []
            # Fetch live prices for open positions
            open_symbols = list({t.symbol for t in open_trades if t.direction == "BULLISH"})
            live_quotes = {}
            if open_symbols:
                try:
                    from app.services.data_fetcher import data_fetcher
                    quotes = data_fetcher.get_bulk_quotes(open_symbols)
                    live_quotes = {q["symbol"]: q for q in quotes if q.get("symbol")}
                except Exception:
                    pass

            for trade in open_trades:
                if trade.direction != "BULLISH" or not trade.entry or trade.entry <= 0:
                    continue
                if (trade.confidence or 0) < MIN_CONFIDENCE:
                    continue

                allocation = self._compute_allocation(
                    trade, equity, stock_history, loss_streak
                )
                qty = max(1, int(allocation / trade.entry))

                # Live P&L
                live = live_quotes.get(trade.symbol, {})
                current_ltp = live.get("ltp")
                live_pnl = None
                live_pnl_pct = None
                if current_ltp and trade.entry:
                    live_pnl = round((current_ltp - trade.entry) * qty, 2)
                    live_pnl_pct = round((current_ltp - trade.entry) / trade.entry * 100, 2)

                # Why picked (allocation breakdown)
                conf_mult = max(0.5, min(2.0, ((trade.confidence or 50) - 20) / 40))
                tf_mults = {"intraday_10m": 0.7, "intraday_15m": 0.9, "intraday_30m": 0.85, "short_1h": 1.1, "short_4h": 1.3}
                tf_mult = tf_mults.get(trade.timeframe, 1.0)

                open_positions.append({
                    "symbol": trade.symbol,
                    "timeframe": trade.timeframe,
                    "qty": qty,
                    "entry": trade.entry,
                    "target": trade.target,
                    "stop_loss": trade.stop_loss,
                    "invested": round(qty * trade.entry, 2),
                    "allocation": round(allocation, 2),
                    "confidence": trade.confidence,
                    "current_price": current_ltp,
                    "live_pnl": live_pnl,
                    "live_pnl_pct": live_pnl_pct,
                    "why_picked": f"Conf {(trade.confidence or 0):.0f}% ({conf_mult:.1f}x) × TF {trade.timeframe} ({tf_mult:.1f}x)",
                    "created_at": trade.created_at.isoformat() if trade.created_at else None,
                    "scanned_at": trade.created_at.strftime("%H:%M") if trade.created_at else None,
                })

            total_pnl = round(equity - capital, 2)
            total_trades = winning + losing

            daily_pnl_list = [
                {"date": d, "pnl": round(p, 2)}
                for d, p in sorted(daily_pnl.items())
            ]

            # Per-stock summary
            stock_summary = []
            for sym, hist in stock_history.items():
                if hist["total"] > 0:
                    stock_summary.append({
                        "symbol": sym,
                        "trades": hist["total"],
                        "win_rate": round(hist["wins"] / hist["total"] * 100, 1),
                    })
            stock_summary.sort(key=lambda x: -x["trades"])

            # Scan overview — show all scanned stocks with best confidence
            scan_overview = []
            all_trades = list(resolved) + list(open_trades)
            seen = {}
            for trade in all_trades:
                key = trade.symbol
                conf = trade.confidence or 0
                if key not in seen or conf > seen[key]["confidence"]:
                    picked = trade.direction == "BULLISH" and conf >= MIN_CONFIDENCE
                    seen[key] = {
                        "symbol": trade.symbol,
                        "direction": trade.direction,
                        "confidence": conf,
                        "timeframe": trade.timeframe,
                        "picked_for_portfolio": picked,
                        "entry": trade.entry,
                        "target": trade.target,
                    }
            scan_overview = sorted(seen.values(), key=lambda x: -x["confidence"])

            return {
                "initial_capital": capital,
                "current_value": round(equity, 2),
                "total_pnl": total_pnl,
                "total_pnl_pct": round(total_pnl / capital * 100, 2) if capital > 0 else 0,
                "total_trades": total_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": round(winning / total_trades * 100, 1) if total_trades > 0 else 0,
                "skipped_signals": skipped,
                "daily_pnl": daily_pnl_list,
                "equity_curve": equity_curve,
                "open_positions": open_positions[:10],
                "recent_trades": trades_detail,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "stock_summary": stock_summary[:10],
                "scan_overview": scan_overview[:30],
                "strategy": "smart",
                "rules": {
                    "min_confidence": MIN_CONFIDENCE,
                    "max_stock_pct": MAX_STOCK_PCT * 100,
                    "max_concurrent": MAX_CONCURRENT,
                    "loss_streak_reduce": LOSS_STREAK_REDUCE,
                },
            }
        finally:
            db.close()


virtual_portfolio = VirtualPortfolio()
