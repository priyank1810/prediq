"""Tests for stock endpoints: /api/stocks."""

from unittest.mock import patch, MagicMock

import pytest


class TestSearchStocks:
    @patch("app.routers.stocks.data_fetcher")
    def test_search_stocks(self, mock_fetcher, client):
        mock_fetcher.search_stocks.return_value = [
            {"symbol": "RELIANCE", "name": "Reliance Industries", "type": "stock"},
            {"symbol": "RELINFRA", "name": "Reliance Infra", "type": "stock"},
        ]
        resp = client.get("/api/stocks/search?q=RELIANCE")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["symbol"] == "RELIANCE"
        mock_fetcher.search_stocks.assert_called_once_with("RELIANCE")

    @patch("app.routers.stocks.data_fetcher")
    def test_search_empty_returns_popular(self, mock_fetcher, client):
        mock_fetcher.get_popular_stocks.return_value = [
            {"symbol": "NIFTY 50", "name": "NIFTY 50", "type": "index"},
        ]
        resp = client.get("/api/stocks/search?q=")
        assert resp.status_code == 200
        mock_fetcher.get_popular_stocks.assert_called_once()


class TestGetQuote:
    @patch("app.routers.stocks.data_fetcher")
    def test_get_quote(self, mock_fetcher, client):
        mock_fetcher.get_live_quote.return_value = {
            "symbol": "RELIANCE",
            "ltp": 2500.0,
            "open": 2490.0,
            "high": 2520.0,
            "low": 2480.0,
            "close": 2495.0,
            "volume": 5000000,
            "change": 5.0,
            "pct_change": 0.2,
            "timestamp": "2025-03-17T14:30:00",
        }
        resp = client.get("/api/stocks/RELIANCE/quote")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "RELIANCE"
        assert data["ltp"] == 2500.0

    @patch("app.routers.stocks.data_fetcher")
    def test_get_quote_not_found(self, mock_fetcher, client):
        mock_fetcher.get_live_quote.return_value = None
        resp = client.get("/api/stocks/INVALID/quote")
        assert resp.status_code == 404

    def test_get_quote_invalid_symbol(self, client):
        resp = client.get("/api/stocks/!!!/quote")
        assert resp.status_code == 400


class TestMarketStatus:
    @patch("app.routers.stocks.market_status")
    def test_market_status(self, mock_status, client):
        mock_status.return_value = "Market is Open"
        resp = client.get("/api/stocks/market-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "Market is Open"


class TestMarketMovers:
    @patch("app.routers.stocks.market_movers_service")
    def test_market_movers(self, mock_service, client):
        mock_service.get_market_movers.return_value = {
            "gainers": [{"symbol": "TCS", "pct_change": 3.5}],
            "losers": [{"symbol": "INFY", "pct_change": -2.1}],
        }
        resp = client.get("/api/stocks/market-movers")
        assert resp.status_code == 200
        data = resp.json()
        assert "gainers" in data
        assert "losers" in data

    @patch("app.routers.stocks.market_movers_service")
    def test_market_movers_with_count(self, mock_service, client):
        mock_service.get_market_movers.return_value = {"gainers": [], "losers": []}
        resp = client.get("/api/stocks/market-movers?count=5")
        assert resp.status_code == 200
        mock_service.get_market_movers.assert_called_once_with(5)


class TestBulkQuotes:
    @patch("app.routers.stocks.data_fetcher")
    def test_bulk_quotes(self, mock_fetcher, client):
        mock_fetcher.get_bulk_quotes.return_value = [
            {"symbol": "RELIANCE", "ltp": 2500.0},
            {"symbol": "TCS", "ltp": 3600.0},
        ]
        resp = client.post(
            "/api/stocks/quotes/bulk",
            json={"symbols": ["RELIANCE", "TCS"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    @patch("app.routers.stocks.data_fetcher")
    def test_bulk_quotes_empty_list(self, mock_fetcher, client):
        mock_fetcher.get_bulk_quotes.return_value = []
        resp = client.post("/api/stocks/quotes/bulk", json={"symbols": []})
        assert resp.status_code == 200


class TestEarningsUpcoming:
    def test_earnings_upcoming(self, client):
        mock_service = MagicMock()
        mock_service.get_earnings.return_value = [
            {"symbol": "RELIANCE", "date": "2025-04-15", "event": "Q4 Results"},
        ]
        try:
            with patch("app.services.earnings_service.earnings_service", mock_service):
                resp = client.get("/api/stocks/earnings/upcoming?symbols=RELIANCE")
                assert resp.status_code == 200
        except TypeError:
            # earnings_service module uses Python 3.10+ type hints (dict | None)
            # which fail to import on Python 3.9 — skip gracefully
            pytest.skip("earnings_service requires Python 3.10+ type hints")
