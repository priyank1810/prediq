"""Tests for alert endpoints: /api/alerts."""

from unittest.mock import patch

import pytest


class TestCreateAlert:
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

    def test_create_alert_below(self, client, test_user):
        payload = {
            "symbol": "TCS",
            "target_price": 3000.0,
            "condition": "below",
        }
        resp = client.post("/api/alerts", json=payload, headers=test_user["headers"])
        assert resp.status_code == 200
        assert "id" in resp.json()

    def test_create_alert_invalid_condition(self, client, test_user):
        payload = {
            "symbol": "RELIANCE",
            "target_price": 3000.0,
            "condition": "invalid",
        }
        resp = client.post("/api/alerts", json=payload, headers=test_user["headers"])
        assert resp.status_code == 422

    def test_create_alert_invalid_symbol(self, client, test_user):
        payload = {
            "symbol": "!!!",
            "target_price": 3000.0,
            "condition": "above",
        }
        resp = client.post("/api/alerts", json=payload, headers=test_user["headers"])
        assert resp.status_code == 422

    def test_create_alert_unauthenticated(self, client):
        payload = {
            "symbol": "RELIANCE",
            "target_price": 3000.0,
            "condition": "above",
        }
        resp = client.post("/api/alerts", json=payload)
        assert resp.status_code == 200  # auth is optional


class TestListAlerts:
    def test_list_alerts_empty(self, client, test_user):
        resp = client.get("/api/alerts", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_alerts_with_data(self, client, test_user):
        client.post("/api/alerts", json={
            "symbol": "RELIANCE", "target_price": 3000.0, "condition": "above",
        }, headers=test_user["headers"])
        client.post("/api/alerts", json={
            "symbol": "TCS", "target_price": 3500.0, "condition": "below",
        }, headers=test_user["headers"])

        resp = client.get("/api/alerts", headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_list_alerts_user_isolation(self, client, test_user, second_user):
        client.post("/api/alerts", json={
            "symbol": "RELIANCE", "target_price": 3000.0, "condition": "above",
        }, headers=test_user["headers"])
        client.post("/api/alerts", json={
            "symbol": "TCS", "target_price": 3500.0, "condition": "below",
        }, headers=second_user["headers"])

        a_alerts = client.get("/api/alerts", headers=test_user["headers"]).json()
        b_alerts = client.get("/api/alerts", headers=second_user["headers"]).json()

        assert len(a_alerts) == 1
        assert a_alerts[0]["symbol"] == "RELIANCE"
        assert len(b_alerts) == 1
        assert b_alerts[0]["symbol"] == "TCS"


class TestDeleteAlert:
    def test_delete_alert(self, client, test_user):
        create_resp = client.post("/api/alerts", json={
            "symbol": "RELIANCE", "target_price": 3000.0, "condition": "above",
        }, headers=test_user["headers"])
        alert_id = create_resp.json()["id"]

        resp = client.delete(f"/api/alerts/{alert_id}", headers=test_user["headers"])
        assert resp.status_code == 200
        assert resp.json()["message"] == "Alert deleted"

        # Verify it's gone
        alerts = client.get("/api/alerts", headers=test_user["headers"]).json()
        assert len(alerts) == 0

    def test_delete_nonexistent_alert(self, client, test_user):
        resp = client.delete("/api/alerts/9999", headers=test_user["headers"])
        assert resp.status_code == 404

    def test_cannot_delete_other_users_alert(self, client, test_user, second_user):
        create_resp = client.post("/api/alerts", json={
            "symbol": "INFY", "target_price": 2000.0, "condition": "above",
        }, headers=test_user["headers"])
        alert_id = create_resp.json()["id"]

        resp = client.delete(f"/api/alerts/{alert_id}", headers=second_user["headers"])
        assert resp.status_code == 404
