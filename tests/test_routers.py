"""Integration tests for FastAPI routers (stocks, portfolio, alerts, watchlist)."""

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Stocks router  /api/stocks
# ---------------------------------------------------------------------------

class TestStocksRouter:
    @patch("app.routers.stocks.data_fetcher")
    def test_search_returns_results(self, mock_fetcher, client):
        mock_fetcher.search_stocks.return_value = [
            {"symbol": "RELIANCE", "name": "Reliance Industries", "type": "stock"},
        ]
        resp = client.get("/api/stocks/search?q=RELIANCE")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["symbol"] == "RELIANCE"

    @patch("app.routers.stocks.data_fetcher")
    def test_search_empty_query_returns_popular(self, mock_fetcher, client):
        mock_fetcher.get_popular_stocks.return_value = [
            {"symbol": "NIFTY 50", "name": "NIFTY 50", "type": "index"},
        ]
        resp = client.get("/api/stocks/search?q=")
        assert resp.status_code == 200
        mock_fetcher.get_popular_stocks.assert_called_once()


# ---------------------------------------------------------------------------
# Portfolio router  /api/portfolio
# ---------------------------------------------------------------------------

class TestPortfolioRouter:
    def test_get_portfolio_unauthenticated(self, client):
        resp = client.get("/api/portfolio")
        assert resp.status_code == 401

    @patch("app.services.portfolio_service.data_fetcher")
    def test_get_portfolio_authenticated(self, mock_fetcher, client, test_user):
        mock_fetcher.get_bulk_quotes.return_value = []
        resp = client.get("/api/portfolio", headers=test_user["headers"])
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_add_holding(self, client, test_user):
        payload = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "quantity": 10,
            "buy_price": 2500.0,
            "buy_date": "2024-01-15",
        }
        resp = client.post("/api/portfolio", json=payload, headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["message"] == "Holding added successfully"

    def test_add_holding_unauthenticated(self, client):
        payload = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "quantity": 10,
            "buy_price": 2500.0,
            "buy_date": "2024-01-15",
        }
        resp = client.post("/api/portfolio", json=payload)
        assert resp.status_code == 401

    def test_delete_holding(self, client, test_user):
        # First add
        payload = {
            "symbol": "TCS",
            "exchange": "NSE",
            "quantity": 5,
            "buy_price": 3500.0,
            "buy_date": "2024-03-01",
        }
        add_resp = client.post("/api/portfolio", json=payload, headers=test_user["headers"])
        holding_id = add_resp.json()["id"]

        # Then delete
        resp = client.delete(f"/api/portfolio/{holding_id}", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json()["message"] == "Holding deleted"

    def test_delete_nonexistent_holding(self, client, test_user):
        resp = client.delete("/api/portfolio/9999", headers=test_user["headers"])
        assert resp.status_code == 404

    @patch("app.services.portfolio_service.data_fetcher")
    def test_get_summary(self, mock_fetcher, client, test_user):
        mock_fetcher.get_bulk_quotes.return_value = []
        resp = client.get("/api/portfolio/summary", headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert "total_invested" in data
        assert "holdings_count" in data


# ---------------------------------------------------------------------------
# Alerts router  /api/alerts
# ---------------------------------------------------------------------------

class TestAlertsRouter:
    def test_get_alerts_unauthenticated(self, client):
        resp = client.get("/api/alerts")
        assert resp.status_code == 401

    def test_get_alerts_authenticated(self, client, test_user):
        resp = client.get("/api/alerts", headers=test_user["headers"])
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_alert(self, client, test_user):
        payload = {
            "symbol": "RELIANCE",
            "target_price": 3000.0,
            "condition": "above",
        }
        resp = client.post("/api/alerts", json=payload, headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["message"] == "Alert created"

    def test_create_alert_unauthenticated(self, client):
        payload = {
            "symbol": "RELIANCE",
            "target_price": 3000.0,
            "condition": "above",
        }
        resp = client.post("/api/alerts", json=payload)
        assert resp.status_code == 401

    def test_create_alert_invalid_condition(self, client, test_user):
        payload = {
            "symbol": "RELIANCE",
            "target_price": 3000.0,
            "condition": "invalid",
        }
        resp = client.post("/api/alerts", json=payload, headers=test_user["headers"])
        assert resp.status_code == 422  # Pydantic validation error

    def test_delete_alert(self, client, test_user):
        # Create
        payload = {"symbol": "TCS", "target_price": 4000.0, "condition": "below"}
        create_resp = client.post("/api/alerts", json=payload, headers=test_user["headers"])
        alert_id = create_resp.json()["id"]

        # Delete
        resp = client.delete(f"/api/alerts/{alert_id}", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json()["message"] == "Alert deleted"

    def test_delete_nonexistent_alert(self, client, test_user):
        resp = client.delete("/api/alerts/9999", headers=test_user["headers"])
        assert resp.status_code == 404

    def test_user_cannot_delete_other_users_alert(self, client, test_user, second_user):
        # User A creates an alert
        payload = {"symbol": "INFY", "target_price": 2000.0, "condition": "above"}
        create_resp = client.post("/api/alerts", json=payload, headers=test_user["headers"])
        alert_id = create_resp.json()["id"]

        # User B tries to delete it
        resp = client.delete(f"/api/alerts/{alert_id}", headers=second_user["headers"])
        assert resp.status_code == 404  # Not found for this user


# ---------------------------------------------------------------------------
# Watchlist router  /api/watchlist
# ---------------------------------------------------------------------------

class TestWatchlistRouter:
    def test_get_watchlist_unauthenticated(self, client):
        resp = client.get("/api/watchlist")
        assert resp.status_code == 401

    def test_get_watchlist_authenticated(self, client, test_user):
        resp = client.get("/api/watchlist", headers=test_user["headers"])
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_add_to_watchlist(self, client, test_user):
        payload = {"symbol": "RELIANCE", "item_type": "stock"}
        resp = client.post("/api/watchlist", json=payload, headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "RELIANCE"
        assert "id" in data

    def test_add_to_watchlist_unauthenticated(self, client):
        payload = {"symbol": "RELIANCE", "item_type": "stock"}
        resp = client.post("/api/watchlist", json=payload)
        assert resp.status_code == 401

    def test_duplicate_watchlist_entry_rejected(self, client, test_user):
        payload = {"symbol": "RELIANCE", "item_type": "stock"}
        client.post("/api/watchlist", json=payload, headers=test_user["headers"])
        resp = client.post("/api/watchlist", json=payload, headers=test_user["headers"])
        assert resp.status_code == 400
        assert "already in watchlist" in resp.json()["detail"]

    def test_remove_from_watchlist(self, client, test_user):
        # Add first
        payload = {"symbol": "TCS", "item_type": "stock"}
        client.post("/api/watchlist", json=payload, headers=test_user["headers"])

        # Remove
        resp = client.delete("/api/watchlist/TCS", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_remove_nonexistent_from_watchlist(self, client, test_user):
        resp = client.delete("/api/watchlist/DOESNOTEXIST", headers=test_user["headers"])
        assert resp.status_code == 404

    def test_user_isolation_watchlist(self, client, test_user, second_user):
        # User A adds RELIANCE
        client.post(
            "/api/watchlist",
            json={"symbol": "RELIANCE", "item_type": "stock"},
            headers=test_user["headers"],
        )
        # User B adds TCS
        client.post(
            "/api/watchlist",
            json={"symbol": "TCS", "item_type": "stock"},
            headers=second_user["headers"],
        )

        a_list = client.get("/api/watchlist", headers=test_user["headers"]).json()
        b_list = client.get("/api/watchlist", headers=second_user["headers"]).json()

        a_symbols = {item["symbol"] for item in a_list}
        b_symbols = {item["symbol"] for item in b_list}
        assert a_symbols == {"RELIANCE"}
        assert b_symbols == {"TCS"}
