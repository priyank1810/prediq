"""Broker order placement service using Angel One SmartAPI.

Supports live order placement via Angel One and paper trading mode.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from app.models import Order, PortfolioHolding
from app.services.angel_provider import angel_provider, KNOWN_TOKENS, EXCHANGE_NSE
from app.utils.helpers import now_ist

logger = logging.getLogger(__name__)


class BrokerService:
    """Broker integration service for order placement and portfolio sync."""

    def _get_angel_client(self):
        """Get authenticated Angel One SmartAPI client, or None."""
        if not angel_provider.is_available:
            return None
        if not angel_provider._ensure_session():
            return None
        return angel_provider._client

    def _resolve_trading_symbol(self, symbol: str) -> dict | None:
        """Resolve a symbol to its Angel One token info."""
        return angel_provider._lookup_token(symbol)

    def place_order(self, db: Session, user_id: int, order_data: dict) -> Order:
        """Place an order via Angel One or log as paper trade.

        Args:
            db: Database session.
            user_id: Authenticated user's ID.
            order_data: Dict with symbol, exchange, order_type, transaction_type,
                        quantity, price, trigger_price, paper_trade, notes.

        Returns:
            The created Order record.

        Raises:
            ValueError: If validation fails.
            RuntimeError: If broker placement fails.
        """
        symbol = order_data["symbol"]
        order_type = order_data["order_type"]
        transaction_type = order_data["transaction_type"]
        quantity = order_data["quantity"]
        price = order_data.get("price")
        trigger_price = order_data.get("trigger_price")
        paper_trade = order_data.get("paper_trade", False)
        exchange = order_data.get("exchange", "NSE")
        notes = order_data.get("notes")

        # Validation
        if order_type in ("LIMIT", "SL") and not price:
            raise ValueError("Price is required for LIMIT and SL orders")
        if order_type in ("SL", "SL-M") and not trigger_price:
            raise ValueError("Trigger price is required for SL and SL-M orders")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        now = now_ist()

        order = Order(
            user_id=user_id,
            symbol=symbol,
            exchange=exchange,
            order_type=order_type,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            status="pending",
            broker="angel_one",
            paper_trade=paper_trade,
            notes=notes,
            created_at=now,
        )
        db.add(order)
        db.flush()  # Get the ID

        if paper_trade:
            order.status = "executed"
            order.placed_at = now
            order.executed_at = now
            order.broker_order_id = f"PAPER-{order.id}"
            logger.info(
                f"Paper trade: {transaction_type} {quantity} {symbol} @ "
                f"{order_type} price={price} trigger={trigger_price}"
            )
            db.commit()
            return order

        # Live order via Angel One
        client = self._get_angel_client()
        if not client:
            order.status = "rejected"
            order.notes = (notes or "") + " [Broker not connected]"
            db.commit()
            raise RuntimeError("Angel One broker is not connected. Check credentials.")

        token_info = self._resolve_trading_symbol(symbol)
        if not token_info:
            order.status = "rejected"
            order.notes = (notes or "") + " [Symbol not found]"
            db.commit()
            raise RuntimeError(f"Could not resolve symbol {symbol} on Angel One")

        # Build Angel One order params
        variety = "NORMAL"
        if order_type in ("SL", "SL-M"):
            variety = "STOPLOSS"

        product_type = "DELIVERY"  # CNC for equity delivery
        duration = "DAY"

        order_params = {
            "variety": variety,
            "tradingsymbol": token_info.get("trading_symbol", symbol),
            "symboltoken": token_info["token"],
            "transactiontype": transaction_type,
            "exchange": token_info.get("exchange", EXCHANGE_NSE),
            "ordertype": order_type,
            "producttype": product_type,
            "duration": duration,
            "quantity": str(quantity),
            "price": str(price or 0),
            "triggerprice": str(trigger_price or 0),
        }

        try:
            angel_provider._throttle()
            response = client.placeOrder(order_params)

            if response and response.get("status"):
                broker_order_id = response.get("data", {}).get("orderid", "")
                order.broker_order_id = str(broker_order_id)
                order.status = "placed"
                order.placed_at = now_ist()
                logger.info(f"Order placed: {broker_order_id} - {transaction_type} {quantity} {symbol}")
            else:
                error_msg = response.get("message", "Unknown error") if response else "No response"
                order.status = "rejected"
                order.notes = (notes or "") + f" [Rejected: {error_msg}]"
                logger.warning(f"Order rejected for {symbol}: {error_msg}")

        except Exception as e:
            order.status = "rejected"
            order.notes = (notes or "") + f" [Error: {str(e)[:200]}]"
            logger.error(f"Order placement error for {symbol}: {e}")

        db.commit()
        return order

    def cancel_order(self, db: Session, user_id: int, order_id: int) -> Order:
        """Cancel a pending/placed order.

        Returns:
            Updated Order record.

        Raises:
            ValueError: If order not found or not cancellable.
        """
        order = (
            db.query(Order)
            .filter(Order.id == order_id, Order.user_id == user_id)
            .first()
        )
        if not order:
            raise ValueError("Order not found")

        if order.status not in ("pending", "placed"):
            raise ValueError(f"Cannot cancel order with status '{order.status}'")

        if order.paper_trade:
            order.status = "cancelled"
            db.commit()
            return order

        if order.broker_order_id:
            client = self._get_angel_client()
            if client:
                try:
                    variety = "STOPLOSS" if order.order_type in ("SL", "SL-M") else "NORMAL"
                    angel_provider._throttle()
                    client.cancelOrder(order.broker_order_id, variety)
                    logger.info(f"Cancelled broker order {order.broker_order_id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel broker order {order.broker_order_id}: {e}")

        order.status = "cancelled"
        db.commit()
        return order

    def get_order_book(self, db: Session, user_id: int) -> list[dict]:
        """Fetch today's orders from local DB and optionally sync with broker.

        Returns list of order dicts.
        """
        today_start = now_ist().replace(hour=0, minute=0, second=0, microsecond=0)
        orders = (
            db.query(Order)
            .filter(Order.user_id == user_id, Order.created_at >= today_start)
            .order_by(Order.created_at.desc())
            .all()
        )
        return orders

    def get_recent_orders(self, db: Session, user_id: int, limit: int = 5) -> list[Order]:
        """Get recent orders for display."""
        return (
            db.query(Order)
            .filter(Order.user_id == user_id)
            .order_by(Order.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_positions(self, user_id: int) -> list[dict]:
        """Fetch live positions from Angel One.

        Returns list of position dicts, or empty list if broker not available.
        """
        client = self._get_angel_client()
        if not client:
            return []

        try:
            angel_provider._throttle()
            response = client.position()
            if not response or not response.get("data"):
                return []

            positions = []
            for pos in response["data"]:
                positions.append({
                    "symbol": pos.get("tradingsymbol", ""),
                    "exchange": pos.get("exchange", ""),
                    "product_type": pos.get("producttype", ""),
                    "quantity": int(pos.get("netqty", 0)),
                    "buy_price": float(pos.get("buyavgprice", 0)),
                    "sell_price": float(pos.get("sellavgprice", 0)),
                    "ltp": float(pos.get("ltp", 0)),
                    "pnl": float(pos.get("pnl", 0)),
                })
            return positions

        except Exception as e:
            logger.warning(f"Failed to fetch positions: {e}")
            return []

    def get_holdings_from_broker(self) -> list[dict]:
        """Fetch holdings (delivery positions) from Angel One."""
        client = self._get_angel_client()
        if not client:
            return []

        try:
            angel_provider._throttle()
            response = client.holding()
            if not response or not response.get("data"):
                return []

            holdings = []
            for h in response["data"]:
                holdings.append({
                    "symbol": h.get("tradingsymbol", "").replace("-EQ", ""),
                    "exchange": h.get("exchange", "NSE"),
                    "quantity": int(h.get("quantity", 0)),
                    "avg_price": float(h.get("averageprice", 0)),
                    "ltp": float(h.get("ltp", 0)),
                    "pnl": float(h.get("pnl", 0)),
                })
            return holdings

        except Exception as e:
            logger.warning(f"Failed to fetch holdings from broker: {e}")
            return []

    def sync_portfolio(self, db: Session, user_id: int) -> dict:
        """Sync broker holdings to local portfolio.

        Returns summary of sync operation.
        """
        broker_holdings = self.get_holdings_from_broker()
        if not broker_holdings:
            return {"synced": 0, "added": 0, "updated": 0, "message": "No holdings from broker"}

        added = 0
        updated = 0
        today = date.today()

        for bh in broker_holdings:
            symbol = bh["symbol"]
            if not symbol or bh["quantity"] <= 0:
                continue

            existing = (
                db.query(PortfolioHolding)
                .filter(
                    PortfolioHolding.user_id == user_id,
                    PortfolioHolding.symbol == symbol,
                )
                .first()
            )

            if existing:
                existing.quantity = bh["quantity"]
                existing.buy_price = bh["avg_price"]
                updated += 1
            else:
                holding = PortfolioHolding(
                    user_id=user_id,
                    symbol=symbol,
                    exchange=bh.get("exchange", "NSE"),
                    quantity=bh["quantity"],
                    buy_price=bh["avg_price"],
                    buy_date=today,
                    notes="Synced from Angel One",
                )
                db.add(holding)
                added += 1

        db.commit()
        total = added + updated
        return {
            "synced": total,
            "added": added,
            "updated": updated,
            "message": f"Synced {total} holdings ({added} new, {updated} updated)",
        }

    def get_broker_status(self) -> dict:
        """Check broker connection status."""
        available = angel_provider.is_available
        connected = False
        client_id = angel_provider.client_id if available else None

        if available:
            connected = angel_provider._ensure_session()

        return {
            "broker": "angel_one",
            "available": available,
            "connected": connected,
            "client_id": client_id,
            "backing_off": angel_provider._is_backing_off(),
        }


broker_service = BrokerService()
