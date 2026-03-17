"""Tests for watchlist endpoints: /api/watchlist."""

from unittest.mock import patch

import pytest


class TestAddToWatchlist:
    def test_add_to_watchlist(self, client, test_user):
        payload = {"symbol": "RELIANCE", "item_type": "stock"}
        resp = client.post("/api/watchlist", json=payload, headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "RELIANCE"
        assert "id" in data

    def test_add_index_to_watchlist(self, client, test_user):
        payload = {"symbol": "NIFTY 50", "item_type": "stock"}
        resp = client.post("/api/watchlist", json=payload, headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "NIFTY 50"
        # item_type should be auto-detected as "index"
        assert data["item_type"] == "index"

    def test_add_to_watchlist_unauthenticated(self, client):
        payload = {"symbol": "TCS", "item_type": "stock"}
        resp = client.post("/api/watchlist", json=payload)
        assert resp.status_code == 200


class TestListWatchlist:
    def test_list_watchlist_empty(self, client, test_user):
        resp = client.get("/api/watchlist", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_watchlist_with_items(self, client, test_user):
        client.post("/api/watchlist", json={"symbol": "RELIANCE"}, headers=test_user["headers"])
        client.post("/api/watchlist", json={"symbol": "TCS"}, headers=test_user["headers"])

        resp = client.get("/api/watchlist", headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        symbols = {item["symbol"] for item in data}
        assert symbols == {"RELIANCE", "TCS"}


class TestRemoveFromWatchlist:
    def test_remove_from_watchlist(self, client, test_user):
        client.post("/api/watchlist", json={"symbol": "TCS"}, headers=test_user["headers"])

        resp = client.delete("/api/watchlist/TCS", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Verify it's gone
        list_resp = client.get("/api/watchlist", headers=test_user["headers"])
        assert len(list_resp.json()) == 0

    def test_remove_nonexistent(self, client, test_user):
        resp = client.delete("/api/watchlist/DOESNOTEXIST", headers=test_user["headers"])
        assert resp.status_code == 404


class TestDuplicateAdd:
    def test_duplicate_add(self, client, test_user):
        payload = {"symbol": "RELIANCE", "item_type": "stock"}
        resp1 = client.post("/api/watchlist", json=payload, headers=test_user["headers"])
        assert resp1.status_code == 200

        resp2 = client.post("/api/watchlist", json=payload, headers=test_user["headers"])
        assert resp2.status_code == 400
        assert "already in watchlist" in resp2.json()["detail"]


class TestWatchlistOverview:
    @patch("app.routers.watchlist.data_fetcher")
    def test_watchlist_overview(self, mock_fetcher, client, test_user):
        mock_fetcher.get_bulk_quotes.return_value = [
            {
                "symbol": "RELIANCE", "ltp": 2500.0, "change": 10.0,
                "pct_change": 0.4, "open": 2490.0, "high": 2520.0,
                "low": 2480.0, "volume": 5000000, "avg_volume": 3000000,
            },
        ]
        client.post("/api/watchlist", json={"symbol": "RELIANCE"}, headers=test_user["headers"])

        resp = client.get("/api/watchlist/overview", headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "RELIANCE"
        assert data[0]["ltp"] == 2500.0

    @patch("app.routers.watchlist.data_fetcher")
    def test_watchlist_overview_empty(self, mock_fetcher, client, test_user):
        resp = client.get("/api/watchlist/overview", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json() == []


class TestWatchlistUserIsolation:
    def test_user_isolation(self, client, test_user, second_user):
        client.post("/api/watchlist", json={"symbol": "RELIANCE"}, headers=test_user["headers"])
        client.post("/api/watchlist", json={"symbol": "TCS"}, headers=second_user["headers"])

        a_list = client.get("/api/watchlist", headers=test_user["headers"]).json()
        b_list = client.get("/api/watchlist", headers=second_user["headers"]).json()

        assert {item["symbol"] for item in a_list} == {"RELIANCE"}
        assert {item["symbol"] for item in b_list} == {"TCS"}
