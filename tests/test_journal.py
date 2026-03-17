"""Tests for trade journal endpoints: /api/journal."""

from unittest.mock import patch

import pytest


class TestCreateTrade:
    def test_create_trade(self, client):
        payload = {
            "symbol": "RELIANCE",
            "action": "buy",
            "price": 2500.0,
            "quantity": 10,
            "notes": "Bought on dip",
        }
        resp = client.post("/api/journal/", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["symbol"] == "RELIANCE"
        assert data["action"] == "buy"

    def test_create_sell_trade_with_pnl(self, client):
        payload = {
            "symbol": "TCS",
            "action": "sell",
            "price": 3700.0,
            "quantity": 5,
            "pnl": 1000.0,
            "pnl_pct": 5.7,
            "tags": "momentum,breakout",
        }
        resp = client.post("/api/journal/", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "sell"

    def test_create_trade_with_signal_info(self, client):
        payload = {
            "symbol": "INFY",
            "action": "buy",
            "price": 1500.0,
            "quantity": 20,
            "signal_direction": "BULLISH",
            "signal_confidence": 75.0,
        }
        resp = client.post("/api/journal/", json=payload)
        assert resp.status_code == 200


class TestInvalidAction:
    def test_invalid_action(self, client):
        payload = {
            "symbol": "RELIANCE",
            "action": "hold",
            "price": 2500.0,
            "quantity": 10,
        }
        resp = client.post("/api/journal/", json=payload)
        assert resp.status_code == 400
        assert "buy" in resp.json()["detail"] or "sell" in resp.json()["detail"]

    def test_invalid_symbol(self, client):
        payload = {
            "symbol": "!!!",
            "action": "buy",
            "price": 100.0,
            "quantity": 1,
        }
        resp = client.post("/api/journal/", json=payload)
        assert resp.status_code == 400


class TestListTrades:
    def test_list_trades_empty(self, client):
        resp = client.get("/api/journal/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_trades_with_data(self, client):
        client.post("/api/journal/", json={
            "symbol": "RELIANCE", "action": "buy", "price": 2500.0, "quantity": 10,
        })
        client.post("/api/journal/", json={
            "symbol": "TCS", "action": "sell", "price": 3700.0, "quantity": 5,
        })

        resp = client.get("/api/journal/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_list_trades_pagination(self, client):
        for i in range(5):
            client.post("/api/journal/", json={
                "symbol": "RELIANCE", "action": "buy", "price": 2500.0 + i, "quantity": 1,
            })

        resp = client.get("/api/journal/?limit=2&offset=0")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        resp2 = client.get("/api/journal/?limit=2&offset=2")
        assert resp2.status_code == 200
        assert len(resp2.json()) == 2


class TestDeleteTrade:
    def test_delete_trade(self, client):
        create_resp = client.post("/api/journal/", json={
            "symbol": "RELIANCE", "action": "buy", "price": 2500.0, "quantity": 10,
        })
        trade_id = create_resp.json()["id"]

        resp = client.delete(f"/api/journal/{trade_id}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_delete_nonexistent_trade(self, client):
        resp = client.delete("/api/journal/9999")
        assert resp.status_code == 404


class TestTradeStats:
    def test_trade_stats_empty(self, client):
        resp = client.get("/api/journal/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_trades"] == 0
        assert data["win_rate"] == 0

    def test_trade_stats_with_data(self, client):
        # Add some buy and sell trades
        client.post("/api/journal/", json={
            "symbol": "RELIANCE", "action": "buy", "price": 2500.0, "quantity": 10,
        })
        client.post("/api/journal/", json={
            "symbol": "RELIANCE", "action": "sell", "price": 2600.0, "quantity": 10,
            "pnl": 1000.0, "pnl_pct": 4.0,
        })
        client.post("/api/journal/", json={
            "symbol": "TCS", "action": "sell", "price": 3400.0, "quantity": 5,
            "pnl": -500.0, "pnl_pct": -2.8,
        })

        resp = client.get("/api/journal/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_trades"] == 3
        assert data["sell_trades"] == 2
        assert data["wins"] == 1
        assert data["win_rate"] == 50.0
        assert data["total_pnl"] == 500.0
