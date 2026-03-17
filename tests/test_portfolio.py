"""Tests for portfolio endpoints: /api/portfolio."""

from unittest.mock import patch

import pytest


class TestAddHolding:
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

    def test_add_holding_invalid_symbol(self, client, test_user):
        payload = {
            "symbol": "!!!",
            "exchange": "NSE",
            "quantity": 10,
            "buy_price": 2500.0,
            "buy_date": "2024-01-15",
        }
        resp = client.post("/api/portfolio", json=payload, headers=test_user["headers"])
        assert resp.status_code == 422

    def test_add_holding_missing_fields(self, client, test_user):
        payload = {"symbol": "RELIANCE"}
        resp = client.post("/api/portfolio", json=payload, headers=test_user["headers"])
        assert resp.status_code == 422


class TestListHoldings:
    @patch("app.services.portfolio_service.data_fetcher")
    def test_list_holdings_empty(self, mock_fetcher, client, test_user):
        mock_fetcher.get_bulk_quotes.return_value = []
        resp = client.get("/api/portfolio", headers=test_user["headers"])
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 0

    @patch("app.services.portfolio_service.data_fetcher")
    def test_list_holdings_with_data(self, mock_fetcher, client, test_user):
        mock_fetcher.get_bulk_quotes.return_value = [
            {"symbol": "RELIANCE", "ltp": 2600.0},
        ]
        # Add a holding first
        client.post("/api/portfolio", json={
            "symbol": "RELIANCE", "exchange": "NSE",
            "quantity": 10, "buy_price": 2500.0, "buy_date": "2024-01-15",
        }, headers=test_user["headers"])

        resp = client.get("/api/portfolio", headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "RELIANCE"


class TestDeleteHolding:
    def test_delete_holding(self, client, test_user):
        # Add
        add_resp = client.post("/api/portfolio", json={
            "symbol": "TCS", "exchange": "NSE",
            "quantity": 5, "buy_price": 3500.0, "buy_date": "2024-03-01",
        }, headers=test_user["headers"])
        holding_id = add_resp.json()["id"]

        # Delete
        resp = client.delete(f"/api/portfolio/{holding_id}", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json()["message"] == "Holding deleted"

    def test_delete_nonexistent_holding(self, client, test_user):
        resp = client.delete("/api/portfolio/9999", headers=test_user["headers"])
        assert resp.status_code == 404


class TestPortfolioSummary:
    @patch("app.services.portfolio_service.data_fetcher")
    def test_portfolio_summary(self, mock_fetcher, client, test_user):
        mock_fetcher.get_bulk_quotes.return_value = []
        resp = client.get("/api/portfolio/summary", headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert "total_invested" in data
        assert "current_value" in data
        assert "total_pnl" in data
        assert "holdings_count" in data

    @patch("app.services.portfolio_service.data_fetcher")
    def test_portfolio_summary_with_holdings(self, mock_fetcher, client, test_user):
        mock_fetcher.get_bulk_quotes.return_value = [
            {"symbol": "RELIANCE", "ltp": 2600.0},
        ]
        client.post("/api/portfolio", json={
            "symbol": "RELIANCE", "exchange": "NSE",
            "quantity": 10, "buy_price": 2500.0, "buy_date": "2024-01-15",
        }, headers=test_user["headers"])

        resp = client.get("/api/portfolio/summary", headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert data["holdings_count"] == 1
        assert data["total_invested"] == 25000.0


class TestPortfolioAuth:
    def test_portfolio_requires_no_strict_auth(self, client):
        """Portfolio endpoints use get_optional_user, so they work without auth too."""
        resp = client.get("/api/portfolio")
        assert resp.status_code == 200

    @patch("app.services.portfolio_service.data_fetcher")
    def test_portfolio_summary_unauthenticated(self, mock_fetcher, client):
        mock_fetcher.get_bulk_quotes.return_value = []
        resp = client.get("/api/portfolio/summary")
        assert resp.status_code == 200
